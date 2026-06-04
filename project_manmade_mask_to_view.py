import json
import os
import re
from argparse import ArgumentParser
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from arguments import ModelParams, get_combined_args
from desk_atlas import _deserialize_desk_atlas_state, _uv_to_pixel_coords
from scene.dataset_readers import sceneLoadTypeCallbacks
from utils.camera_utils import cameraList_from_camInfos
from utils.projection_utils import project_xyz_to_plane_uv


def _resolve_atlas_dir(model_path: str, desk_atlas_dir: str) -> str:
    return desk_atlas_dir if os.path.isabs(desk_atlas_dir) else os.path.join(model_path, desk_atlas_dir)


def _safe_name(value: str) -> str:
    value = str(value)
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "camera"


def _optional_int(value, default: int = -1) -> int:
    if value is None:
        return int(default)
    return int(value)


def _numeric_basename_key(value: str):
    stem = os.path.splitext(os.path.basename(str(value).strip()))[0]
    if stem.isdigit():
        return str(int(stem))
    return None


def _parse_camera_queries(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        queries = []
        for item in value:
            queries.extend(_parse_camera_queries(item))
    else:
        text = str(value).strip()
        if not text:
            return []
        if text.startswith("["):
            try:
                return _parse_camera_queries(json.loads(text))
            except json.JSONDecodeError:
                pass
        queries = [item for item in text.replace(",", " ").split() if item]

    seen = set()
    unique_queries = []
    for query in queries:
        if query not in seen:
            unique_queries.append(query)
            seen.add(query)
    return unique_queries


def _load_desk_atlas_state(atlas_dir: str, device: torch.device):
    state_path = os.path.join(atlas_dir, "desk_atlas_state.pt")
    if not os.path.exists(state_path):
        raise FileNotFoundError(f"desk atlas state not found: {state_path}")
    payload = torch.load(state_path, map_location="cpu")
    return _deserialize_desk_atlas_state(payload, device=device), state_path


def _load_binary_mask(path: str, atlas_hw: Tuple[int, int], device: torch.device, threshold: float) -> torch.Tensor:
    if not os.path.exists(path):
        raise FileNotFoundError(f"mask not found: {path}")

    with Image.open(path) as image:
        has_alpha = "A" in image.getbands()
        rgba = image.convert("RGBA")
        arr = np.asarray(rgba, dtype=np.uint8)

    expected_hw = (int(atlas_hw[0]), int(atlas_hw[1]))
    if tuple(arr.shape[:2]) != expected_hw:
        raise RuntimeError(
            f"mask resolution mismatch: expected {expected_hw}, got {tuple(int(v) for v in arr.shape[:2])}."
        )

    threshold_u8 = int(round(float(threshold) * 255.0))
    gray = np.asarray(Image.fromarray(arr[:, :, :3]).convert("L"), dtype=np.uint8)
    alpha = arr[:, :, 3]
    gray_mask = gray > threshold_u8
    alpha_mask = alpha > threshold_u8
    if has_alpha and not gray_mask.any() and alpha_mask.any():
        mask = alpha_mask
    elif has_alpha:
        mask = gray_mask & alpha_mask
    else:
        mask = gray_mask
    return torch.from_numpy(mask).to(device=device)


def _camera_intrinsics(camera) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    device = camera.camera_center.device
    fx = getattr(camera, "fx", None)
    fy = getattr(camera, "fy", None)
    cx = getattr(camera, "cx", None)
    cy = getattr(camera, "cy", None)
    if fx is None:
        fx = getattr(camera, "Fx", None)
    if fy is None:
        fy = getattr(camera, "Fy", None)
    if cx is None:
        cx = getattr(camera, "Cx", None)
    if cy is None:
        cy = getattr(camera, "Cy", None)
    if fx is None or fy is None or cx is None or cy is None:
        intrinsics = getattr(camera, "intrinsics", None)
        if intrinsics is None:
            raise AttributeError("Camera must expose fx/fy/cx/cy, Fx/Fy/Cx/Cy, or intrinsics.")
        fx = intrinsics[0, 0]
        fy = intrinsics[1, 1]
        cx = intrinsics[0, 2]
        cy = intrinsics[1, 2]
    return (
        torch.as_tensor(fx, device=device, dtype=torch.float32),
        torch.as_tensor(fy, device=device, dtype=torch.float32),
        torch.as_tensor(cx, device=device, dtype=torch.float32),
        torch.as_tensor(cy, device=device, dtype=torch.float32),
    )


def _camera_c2w(camera) -> torch.Tensor:
    c2w = getattr(camera, "c2w", None)
    if c2w is not None:
        return c2w
    return torch.inverse(camera.world_view_transform.transpose(0, 1))


def _project_atlas_tensor_to_view(
    camera,
    atlas_tensor: torch.Tensor,
    plane,
    uv_bbox: Tuple[float, float, float, float],
    atlas_hw: Tuple[int, int],
    mode: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = camera.camera_center.device
    atlas = atlas_tensor.to(device=device, dtype=torch.float32)
    image_h = int(camera.image_height)
    image_w = int(camera.image_width)
    channels = int(atlas.shape[0])
    view_out = torch.zeros((channels, image_h, image_w), dtype=atlas.dtype, device=device)
    view_valid = torch.zeros((image_h, image_w), dtype=torch.bool, device=device)
    if image_h <= 0 or image_w <= 0 or atlas.numel() == 0:
        return view_out, view_valid

    atlas_h, atlas_w = (int(atlas_hw[0]), int(atlas_hw[1]))
    if int(atlas.shape[1]) != atlas_h or int(atlas.shape[2]) != atlas_w:
        raise RuntimeError(
            "Atlas tensor shape mismatch inside projection: "
            f"expected (*, {atlas_h}, {atlas_w}), got {tuple(int(v) for v in atlas.shape)}."
        )

    fx, fy, cx, cy = _camera_intrinsics(camera)
    c2w_rot = _camera_c2w(camera).to(device=device, dtype=torch.float32)[:3, :3]
    camera_origin = camera.camera_center
    origin_term = torch.dot(camera_origin, plane.normal) + plane.d
    xs_full = torch.arange(image_w, device=device, dtype=torch.float32)

    for y_start in range(0, image_h, 128):
        y_end = min(y_start + 128, image_h)
        chunk_h = y_end - y_start
        ys = torch.arange(y_start, y_end, device=device, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(ys, xs_full, indexing="ij")
        dirs_cam = torch.stack(
            [
                (grid_x.reshape(-1) - cx) / fx,
                (grid_y.reshape(-1) - cy) / fy,
                torch.ones((chunk_h * image_w,), device=device, dtype=torch.float32),
            ],
            dim=-1,
        )
        dirs_cam = F.normalize(dirs_cam, dim=-1)
        dirs_world = (c2w_rot @ dirs_cam.t()).t()
        denom = dirs_world @ plane.normal
        valid = torch.abs(denom) > 1e-7
        chunk_valid = torch.zeros((chunk_h * image_w,), dtype=torch.bool, device=device)
        chunk_out = torch.zeros((channels, chunk_h * image_w), dtype=atlas.dtype, device=device)

        if valid.any():
            dirs_world_valid = dirs_world[valid]
            denom_valid = denom[valid]
            t = -origin_term / denom_valid
            hit = t > 0
            if hit.any():
                points = camera_origin[None] + dirs_world_valid[hit] * t[hit][:, None]
                uv = project_xyz_to_plane_uv(points, plane)
                pix_x, pix_y = _uv_to_pixel_coords(uv, uv_bbox, atlas_h, atlas_w)
                in_atlas = (
                    (pix_x >= 0.0)
                    & (pix_x <= max(atlas_w - 1, 0))
                    & (pix_y >= 0.0)
                    & (pix_y <= max(atlas_h - 1, 0))
                )
                valid_idx = valid.nonzero(as_tuple=False).squeeze(1)
                hit_idx = valid_idx[hit]
                chunk_valid[hit_idx] = in_atlas

                if in_atlas.any():
                    grid_x_norm = pix_x[in_atlas] / max(atlas_w - 1, 1) * 2.0 - 1.0
                    grid_y_norm = pix_y[in_atlas] / max(atlas_h - 1, 1) * 2.0 - 1.0
                    sample_grid = torch.stack([grid_x_norm, grid_y_norm], dim=-1).view(1, -1, 1, 2)
                    sampled = F.grid_sample(
                        atlas.unsqueeze(0),
                        sample_grid,
                        mode=mode,
                        padding_mode="zeros",
                        align_corners=True,
                    ).squeeze(0).squeeze(-1)
                    chunk_out[:, hit_idx[in_atlas]] = sampled

        view_out[:, y_start:y_end, :] = chunk_out.view(channels, chunk_h, image_w)
        view_valid[y_start:y_end, :] = chunk_valid.view(chunk_h, image_w)

    return view_out, view_valid


def project_binary_atlas_mask_to_view(camera, atlas_mask: torch.Tensor, desk_atlas_state) -> Tuple[torch.Tensor, torch.Tensor]:
    view_mask, view_valid = _project_atlas_tensor_to_view(
        camera=camera,
        atlas_tensor=atlas_mask.float().unsqueeze(0),
        plane=desk_atlas_state.plane,
        uv_bbox=desk_atlas_state.uv_bbox,
        atlas_hw=desk_atlas_state.atlas_hw,
        mode="nearest",
    )
    return view_valid & (view_mask.squeeze(0) > 0.5), view_valid


def _load_cameras(dataset) -> Dict[str, List]:
    if os.path.exists(os.path.join(dataset.source_path, "sparse")):
        scene_info = sceneLoadTypeCallbacks["Colmap"](dataset, dataset.source_path, dataset.images, dataset.eval)
    elif os.path.exists(os.path.join(dataset.source_path, "transforms_train.json")):
        scene_info = sceneLoadTypeCallbacks["Blender"](dataset, dataset.source_path, dataset.eval)
    else:
        raise RuntimeError(f"Could not recognize scene type under source_path: {dataset.source_path}")

    return {
        "train": cameraList_from_camInfos(scene_info.train_cameras, 1.0, dataset),
        "test": cameraList_from_camInfos(scene_info.test_cameras, 1.0, dataset),
    }


def _camera_match_keys(camera, split: str, index: int) -> List[str]:
    image_name = str(getattr(camera, "image_name", ""))
    basename = os.path.basename(image_name)
    stem = os.path.splitext(basename)[0]
    uid = getattr(camera, "uid", None)
    colmap_id = getattr(camera, "colmap_id", None)
    keys = [
        image_name,
        basename,
        stem,
        f"{split}:{index}",
        f"{split}:{stem}",
        f"index:{index}",
    ]
    if uid is not None:
        keys.extend([f"uid:{uid}", f"{split}:uid:{uid}"])
    if colmap_id is not None:
        keys.extend([f"colmap:{colmap_id}", f"{split}:colmap:{colmap_id}"])
    numeric_key = _numeric_basename_key(stem)
    if numeric_key is not None:
        keys.append(numeric_key)
    return [key for key in keys if key != ""]


def _camera_query_keys(camera_query: str) -> List[str]:
    query = str(camera_query).strip()
    basename = os.path.basename(query)
    stem = os.path.splitext(basename)[0]
    keys = [query, basename, stem]
    numeric_key = _numeric_basename_key(stem)
    if numeric_key is not None:
        keys.append(numeric_key)
    return [key for key in keys if key != ""]


def _iter_camera_records(cameras_by_split: Dict[str, List], camera_split: str):
    splits = ("train", "test") if camera_split == "all" else (camera_split,)
    for split in splits:
        for index, camera in enumerate(cameras_by_split.get(split, [])):
            yield split, index, camera


def _select_camera(cameras_by_split: Dict[str, List], camera_query: str, camera_split: str):
    query_keys = {key.lower() for key in _camera_query_keys(camera_query)}
    matches = []
    for split, index, camera in _iter_camera_records(cameras_by_split, camera_split):
        keys = _camera_match_keys(camera, split, index)
        if query_keys & {key.lower() for key in keys}:
            matches.append((split, index, camera, keys))

    if len(matches) == 1:
        return matches[0]
    if len(matches) == 0:
        available = []
        for split, index, camera in _iter_camera_records(cameras_by_split, camera_split):
            image_name = str(getattr(camera, "image_name", ""))
            available.append(f"{split}:{index} image_name={image_name} uid={getattr(camera, 'uid', None)}")
        raise ValueError(
            f"Camera '{camera_query}' was not found in split='{camera_split}'. "
            "Available cameras:\n" + "\n".join(available[:50])
        )

    details = [
        f"{split}:{index} image_name={getattr(camera, 'image_name', '')} uid={getattr(camera, 'uid', None)}"
        for split, index, camera, _ in matches
    ]
    raise ValueError(
        f"Camera query '{camera_query}' matched multiple cameras. "
        "Use --camera_split or a more specific key such as train:<index>.\n" + "\n".join(details)
    )


def _select_cameras(cameras_by_split: Dict[str, List], camera_queries: List[str], camera_split: str):
    selected = []
    seen = set()
    for query in camera_queries:
        split, index, camera, keys = _select_camera(cameras_by_split, query, camera_split)
        camera_key = (split, index)
        if camera_key in seen:
            print(f"[WARN] Duplicate camera query '{query}' maps to {split}:{index}; skipping duplicate.")
            continue
        selected.append((query, split, index, camera, keys))
        seen.add(camera_key)
    return selected


def _tensor_mask_to_pil(mask: torch.Tensor) -> Image.Image:
    arr = (mask.detach().cpu().numpy().astype(np.uint8) * 255)
    return Image.fromarray(arr, mode="L")


def _save_overlay(path: str, camera, mask: torch.Tensor, alpha: float) -> None:
    image = camera.original_image.detach().float().cpu().clamp(0.0, 1.0)
    image_np = image.permute(1, 2, 0).numpy()
    mask_np = mask.detach().cpu().numpy().astype(bool)
    overlay = image_np.copy()
    red = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    overlay[mask_np] = (1.0 - float(alpha)) * overlay[mask_np] + float(alpha) * red
    Image.fromarray((overlay.clip(0.0, 1.0) * 255.0).astype(np.uint8), mode="RGB").save(path)


def _add_args(parser: ArgumentParser) -> None:
    parser.add_argument("--desk_atlas_dir", type=str, default="desk_atlas")
    parser.add_argument("--mask_name", type=str, default="manmade_mask.png")
    parser.add_argument(
        "--camera",
        nargs="+",
        default=None,
        help=(
            "One or more camera basenames/stems, full image paths, uid:<id>, colmap:<id>, or split:index. "
            "Numeric basenames are matched independent of zero padding, so 00064, 064, and 64 are equivalent. "
            "Comma-separated values are also accepted."
        ),
    )
    parser.add_argument("--camera_split", choices=["train", "test", "all"], default="all")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--overlay_alpha", type=float, default=0.45)
    parser.add_argument("--list_cameras", action="store_true", default=False)


def main() -> None:
    parser = ArgumentParser(description="Project model_path/desk_atlas/manmade_mask.png to a selected camera view.")
    lp = ModelParams(parser, sentinel=True)
    _add_args(parser)
    args = get_combined_args(parser)
    dataset = lp.extract(args)
    camera_queries = _parse_camera_queries(getattr(args, "camera", None))
    output_dir_override = getattr(args, "output_dir", None)

    cameras_by_split = _load_cameras(dataset)
    if args.list_cameras:
        for split, index, camera in _iter_camera_records(cameras_by_split, args.camera_split):
            print(
                f"{split}:{index} image_name={getattr(camera, 'image_name', '')} "
                f"uid={getattr(camera, 'uid', None)} colmap={getattr(camera, 'colmap_id', None)} "
                f"size={int(camera.image_width)}x{int(camera.image_height)}"
            )
        return

    if not camera_queries:
        raise ValueError("Please specify --camera, or pass --list_cameras to inspect available cameras.")

    device = torch.device("cuda")
    atlas_dir = _resolve_atlas_dir(dataset.model_path, args.desk_atlas_dir)
    desk_atlas_state, state_path = _load_desk_atlas_state(atlas_dir, device=device)
    mask_path = os.path.join(atlas_dir, args.mask_name)
    atlas_mask = _load_binary_mask(mask_path, desk_atlas_state.atlas_hw, device=device, threshold=args.threshold)

    selected_cameras = _select_cameras(cameras_by_split, camera_queries, args.camera_split)

    output_dir = output_dir_override or os.path.join(dataset.model_path, "debug")
    os.makedirs(output_dir, exist_ok=True)

    summary = {
        "model_path": dataset.model_path,
        "source_path": dataset.source_path,
        "desk_atlas_dir": atlas_dir,
        "desk_atlas_state": state_path,
        "mask_path": mask_path,
        "atlas_hw": [int(desk_atlas_state.atlas_hw[0]), int(desk_atlas_state.atlas_hw[1])],
        "uv_bbox": [float(v) for v in desk_atlas_state.uv_bbox],
        "threshold": float(args.threshold),
        "camera_queries": camera_queries,
        "cameras": [],
    }
    for camera_query, split, index, camera, _ in selected_cameras:
        projected_mask, valid_mask = project_binary_atlas_mask_to_view(camera, atlas_mask, desk_atlas_state)

        camera_name = _safe_name(f"{split}_{index}_{getattr(camera, 'image_name', 'camera')}")
        mask_output_path = os.path.join(output_dir, f"manmade_mask_projected_{camera_name}.png")
        valid_output_path = os.path.join(output_dir, f"manmade_mask_valid_{camera_name}.png")
        overlay_output_path = os.path.join(output_dir, f"manmade_mask_overlay_{camera_name}.png")
        metadata_path = os.path.join(output_dir, f"manmade_mask_projected_{camera_name}.json")

        _tensor_mask_to_pil(projected_mask).save(mask_output_path)
        _tensor_mask_to_pil(valid_mask).save(valid_output_path)
        _save_overlay(overlay_output_path, camera, projected_mask, alpha=args.overlay_alpha)

        metadata = {
            **{key: value for key, value in summary.items() if key != "cameras"},
            "camera_query": camera_query,
            "camera_split": split,
            "camera_index": index,
            "camera_image_name": str(getattr(camera, "image_name", "")),
            "camera_uid": _optional_int(getattr(camera, "uid", None)),
            "camera_colmap_id": _optional_int(getattr(camera, "colmap_id", None)),
            "camera_hw": [int(camera.image_height), int(camera.image_width)],
            "projected_mask_pixels": int(projected_mask.sum().item()),
            "valid_plane_pixels": int(valid_mask.sum().item()),
            "outputs": {
                "projected_mask": mask_output_path,
                "valid_mask": valid_output_path,
                "overlay": overlay_output_path,
            },
        }
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        summary["cameras"].append(
            {
                "camera_query": camera_query,
                "camera_split": split,
                "camera_index": index,
                "camera_image_name": metadata["camera_image_name"],
                "camera_uid": metadata["camera_uid"],
                "camera_colmap_id": metadata["camera_colmap_id"],
                "projected_mask_pixels": metadata["projected_mask_pixels"],
                "valid_plane_pixels": metadata["valid_plane_pixels"],
                "outputs": metadata["outputs"],
                "metadata": metadata_path,
            }
        )
        print(f"[{camera_query}] Saved projected mask to {mask_output_path}")
        print(f"[{camera_query}] Saved valid mask to {valid_output_path}")
        print(f"[{camera_query}] Saved overlay to {overlay_output_path}")
        print(f"[{camera_query}] Saved metadata to {metadata_path}")

    summary_path = os.path.join(output_dir, "manmade_mask_projected_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
