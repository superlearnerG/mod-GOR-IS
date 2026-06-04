import copy
import json
import os
from argparse import ArgumentParser
from collections import deque
from dataclasses import dataclass
from random import randint
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from arguments import ModelParams, OptimizationParams, PipelineParams, get_combined_args
from desk_atlas import DeskAtlasState, PlaneDefinition, _deserialize_desk_atlas_state
from gaussian_renderer import render
from scene import GaussianModel, Scene
from utils.general_utils import inverse_sigmoid, safe_state
from utils.projection_utils import project_xyz_to_plane_uv
from utils.sh_utils import RGB2SH

try:
    from scipy.ndimage import binary_closing, distance_transform_edt, label
except Exception:
    binary_closing = None
    distance_transform_edt = None
    label = None


INPAINT_PHASE = "mod_gor_is_desk_atlas_inpaint"
CHECKPOINT_VERSION = 1
NORMAL_IMAGE_VALID_EPS = 0.1
NORMAL_MAP_VALID_EPS = 1e-6

GAUSSIAN_ATTRS = {
    "xyz": "_xyz",
    "f_dc": "_features_dc",
    "f_rest": "_features_rest",
    "opacity": "_opacity",
    "scaling": "_scaling",
    "rotation": "_rotation",
    "diffuse": "_diffuse",
    "fresnel": "_fresnel",
    "roughness": "_roughness",
    "reflect": "_reflect",
}


@dataclass
class DecoupledSceneState:
    decoupled_gaussians: GaussianModel
    desk_gaussians: GaussianModel
    background_gaussians: GaussianModel
    source_iteration: int
    desk_object_id: int
    decouple_object_ids: List[int]


@dataclass
class MaterialInpaintState:
    hole_start_idx: int
    hole_end_idx: int
    hole_init_uv: torch.Tensor
    hole_init_stride_px: int
    hole_reflection_visible: bool
    completed_diffuse_path: str
    completed_fresnel_path: str
    completed_reflect_path: str
    completed_roughness_path: str
    completed_normal_path: str


@dataclass
class ViewTargetCacheEntry:
    target_diffuse_view: torch.Tensor
    target_fresnel_view: torch.Tensor
    target_reflect_view: torch.Tensor
    target_roughness_view: torch.Tensor
    target_normal_view: torch.Tensor
    merge_mask_view: torch.Tensor
    valid_mask_view: torch.Tensor
    supervision_mask_view: torch.Tensor
    normal_supervision_mask_view: torch.Tensor
    reproj_completed_diffuse_view: torch.Tensor
    reproj_completed_fresnel_view: torch.Tensor
    reproj_completed_reflect_view: torch.Tensor
    reproj_completed_roughness_view: torch.Tensor
    reproj_completed_normal_view: torch.Tensor
    removal_diffuse_view: torch.Tensor
    removal_fresnel_view: torch.Tensor
    removal_reflect_view: torch.Tensor
    removal_roughness_view: torch.Tensor
    removal_normal_view: torch.Tensor


@dataclass
class MaterialTrainingState:
    target_scales_actual: torch.Tensor
    view_target_cache: Dict[int, ViewTargetCacheEntry]
    hole_start_idx: int
    hole_end_idx: int


def _cpu_clone_tree(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_clone_tree(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_cpu_clone_tree(val) for val in value]
    if isinstance(value, tuple):
        return tuple(_cpu_clone_tree(val) for val in value)
    return copy.deepcopy(value)


def _move_tree_to_device(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device=device)
    if isinstance(value, dict):
        return {key: _move_tree_to_device(val, device) for key, val in value.items()}
    if isinstance(value, list):
        return [_move_tree_to_device(val, device) for val in value]
    if isinstance(value, tuple):
        return tuple(_move_tree_to_device(val, device) for val in value)
    return copy.deepcopy(value)


def _gaussian_count(gaussians: GaussianModel) -> int:
    return int(gaussians.get_xyz.shape[0])


def _copy_owner_metadata(target: GaussianModel, source: GaussianModel) -> None:
    target.active_sh_degree = int(source.active_sh_degree)
    target.spatial_lr_scale = float(source.spatial_lr_scale)
    target.percent_dense = float(getattr(source, "percent_dense", 0.0))
    target.use_screen_filter = bool(getattr(source, "use_screen_filter", True))
    target.alpha_min = getattr(source, "alpha_min", target.alpha_min)
    target.env_light.load_state_dict(source.env_light.state_dict())
    target.roughness_net.load_state_dict(source.roughness_net.state_dict())


def _make_gaussian_subset(gaussians: GaussianModel, keep_mask: torch.Tensor) -> GaussianModel:
    keep_mask = keep_mask.bool().reshape(-1).to(device=gaussians.get_xyz.device)
    if keep_mask.numel() != _gaussian_count(gaussians):
        raise ValueError(
            f"Subset mask length mismatch: mask={keep_mask.numel()} gaussians={_gaussian_count(gaussians)}"
        )

    subset = GaussianModel(gaussians.max_sh_degree)
    _copy_owner_metadata(subset, gaussians)

    for _, attr_name in GAUSSIAN_ATTRS.items():
        value = getattr(gaussians, attr_name)[keep_mask].detach().clone()
        setattr(subset, attr_name, nn.Parameter(value.requires_grad_(True)))

    count = int(keep_mask.sum().item())
    device = gaussians.get_xyz.device
    dtype = gaussians.get_xyz.dtype
    if getattr(gaussians, "_mask", None) is not None and gaussians._mask.shape[0] == keep_mask.numel():
        subset._mask = nn.Parameter(gaussians._mask[keep_mask].detach().clone().requires_grad_(True))
    else:
        subset._mask = nn.Parameter(torch.zeros((count,), dtype=dtype, device=device).requires_grad_(True))
    subset.object_id = gaussians.get_object_id[keep_mask].detach().clone()
    subset.object_score = gaussians.get_object_score[keep_mask].detach().clone()
    if hasattr(gaussians, "get_reflection_visible"):
        subset.reflection_visible = gaussians.get_reflection_visible[keep_mask].detach().clone()
    else:
        subset.reflection_visible = torch.ones((count,), dtype=torch.bool, device=device)
    if hasattr(gaussians, "filter_3D") and gaussians.filter_3D.shape[0] == keep_mask.numel():
        subset.filter_3D = gaussians.filter_3D[keep_mask].detach().clone()
    else:
        subset.filter_3D = torch.zeros((count, 1), dtype=dtype, device=device)

    subset.max_radii2D = torch.zeros((count,), dtype=dtype, device=device)
    subset.xyz_gradient_accum = torch.zeros((count, 1), dtype=dtype, device=device)
    subset.xyz_gradient_accum_abs = torch.zeros((count, 1), dtype=dtype, device=device)
    subset.mask_sign_accum = torch.zeros((count, 1), dtype=dtype, device=device)
    subset.mask_val_accum = torch.zeros((count, 2), dtype=dtype, device=device)
    subset.denom = torch.zeros((count, 1), dtype=dtype, device=device)
    subset.optimizer = None
    subset.mask_optimizer = None
    return subset


def _make_gaussian_index_subset(gaussians: GaussianModel, start_idx: int, end_idx: int) -> GaussianModel:
    mask = torch.zeros((_gaussian_count(gaussians),), dtype=torch.bool, device=gaussians.get_xyz.device)
    mask[int(start_idx):int(end_idx)] = True
    return _make_gaussian_subset(gaussians, mask)


def _create_gaussian_union(parts: Sequence[GaussianModel], detach: bool) -> GaussianModel:
    nonempty_parts = [part for part in parts if _gaussian_count(part) > 0]
    if len(nonempty_parts) == 0:
        raise RuntimeError("Cannot materialize gaussian union from an empty owner list.")

    first = nonempty_parts[0]
    combined = GaussianModel(first.max_sh_degree)
    _copy_owner_metadata(combined, first)
    combined.active_sh_degree = max(int(part.active_sh_degree) for part in nonempty_parts)

    for _, attr_name in GAUSSIAN_ATTRS.items():
        chunks = [getattr(part, attr_name) for part in nonempty_parts]
        value = chunks[0] if len(chunks) == 1 else torch.cat(chunks, dim=0)
        if detach:
            value = nn.Parameter(value.detach().clone().requires_grad_(True))
        setattr(combined, attr_name, value)

    mask_chunks = []
    for part in nonempty_parts:
        if getattr(part, "_mask", None) is not None and part._mask.shape[0] == _gaussian_count(part):
            mask_chunks.append(part._mask)
        else:
            mask_chunks.append(torch.zeros((_gaussian_count(part),), dtype=part.get_xyz.dtype, device=part.get_xyz.device))
    mask_value = mask_chunks[0] if len(mask_chunks) == 1 else torch.cat(mask_chunks, dim=0)
    combined._mask = nn.Parameter(mask_value.detach().clone().requires_grad_(True)) if detach else mask_value
    combined.object_id = torch.cat([part.get_object_id for part in nonempty_parts], dim=0).detach().clone()
    combined.object_score = torch.cat([part.get_object_score for part in nonempty_parts], dim=0).detach().clone()
    reflection_chunks = []
    for part in nonempty_parts:
        if hasattr(part, "get_reflection_visible"):
            reflection_chunks.append(part.get_reflection_visible)
        else:
            reflection_chunks.append(torch.ones((_gaussian_count(part),), dtype=torch.bool, device=part.get_xyz.device))
    combined.reflection_visible = torch.cat(reflection_chunks, dim=0).detach().clone()

    filter_chunks = []
    for part in nonempty_parts:
        if hasattr(part, "filter_3D") and part.filter_3D.shape[0] == _gaussian_count(part):
            filter_chunks.append(part.filter_3D)
        else:
            filter_chunks.append(torch.zeros((_gaussian_count(part), 1), dtype=part.get_xyz.dtype, device=part.get_xyz.device))
    combined.filter_3D = torch.cat(filter_chunks, dim=0).detach().clone() if detach else torch.cat(filter_chunks, dim=0)

    n_points = _gaussian_count(combined)
    device = combined.get_xyz.device
    dtype = combined.get_xyz.dtype
    combined.max_radii2D = torch.zeros((n_points,), dtype=dtype, device=device)
    combined.xyz_gradient_accum = torch.zeros((n_points, 1), dtype=dtype, device=device)
    combined.xyz_gradient_accum_abs = torch.zeros((n_points, 1), dtype=dtype, device=device)
    combined.mask_sign_accum = torch.zeros((n_points, 1), dtype=dtype, device=device)
    combined.mask_val_accum = torch.zeros((n_points, 2), dtype=dtype, device=device)
    combined.denom = torch.zeros((n_points, 1), dtype=dtype, device=device)
    combined.optimizer = None
    combined.mask_optimizer = None
    return combined


def materialize_train_scene_gaussians(decoupled_state: DecoupledSceneState) -> GaussianModel:
    return _create_gaussian_union(
        [decoupled_state.background_gaussians, decoupled_state.desk_gaussians],
        detach=False,
    )


def materialize_full_scene_gaussians(decoupled_state: DecoupledSceneState) -> GaussianModel:
    return _create_gaussian_union(
        [
            decoupled_state.background_gaussians,
            decoupled_state.desk_gaussians,
            decoupled_state.decoupled_gaussians,
        ],
        detach=True,
    )


def _create_reference_clean_background_composite(
    background_gaussians: GaussianModel,
    desk_gaussians: GaussianModel,
    desk_prefix_end_idx: int,
) -> GaussianModel:
    parts = []
    if _gaussian_count(background_gaussians) > 0:
        parts.append(background_gaussians)
    if int(desk_prefix_end_idx) > 0:
        parts.append(_make_gaussian_index_subset(desk_gaussians, 0, int(desk_prefix_end_idx)))
    return _create_gaussian_union(parts, detach=True)


def parse_object_id_list(value) -> List[int]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        object_ids: List[int] = []
        for item in value:
            object_ids.extend(parse_object_id_list(item))
        return object_ids

    value = str(value).strip()
    if not value:
        return []

    if value.startswith("["):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid object id list: {value}") from exc
        return parse_object_id_list(parsed)

    return [int(item) for item in value.replace(",", " ").split()]


def parse_single_desk_object_id(raw_value) -> int:
    object_ids = parse_object_id_list(raw_value)
    if len(object_ids) == 0:
        raise ValueError("--desk_object_id is required and must contain exactly one id.")
    if len(object_ids) > 1:
        raise ValueError(f"--desk_object_id supports exactly one id, got {object_ids}.")
    object_id = int(object_ids[0])
    if object_id <= 0:
        raise ValueError(f"--desk_object_id must be > 0, got {object_id}.")
    return object_id


def _label_mask(gaussians: GaussianModel, labels: Sequence[int]) -> torch.Tensor:
    object_ids = gaussians.get_object_id
    mask = torch.zeros_like(object_ids, dtype=torch.bool)
    for current_label in labels:
        mask = torch.logical_or(mask, object_ids == int(current_label))
    return mask


def load_segmentation_checkpoint(gaussians: GaussianModel, checkpoint_path: str) -> Any:
    checkpoint = torch.load(checkpoint_path, map_location="cuda")
    if isinstance(checkpoint, (tuple, list)):
        model_state = checkpoint[0]
        marker = checkpoint[1] if len(checkpoint) > 1 else None
    elif isinstance(checkpoint, dict) and "gaussians" in checkpoint:
        model_state = checkpoint["gaussians"]
        marker = checkpoint.get("iteration")
    else:
        model_state = checkpoint
        marker = None
    if not isinstance(model_state, dict):
        raise ValueError(f"Segmentation checkpoint {checkpoint_path} does not contain a state dict.")
    if "object_id" not in model_state:
        raise ValueError(f"Segmentation checkpoint {checkpoint_path} does not contain object_id.")
    gaussians._restore_from_state_dict(model_state)
    gaussians._ensure_object_state()
    return marker


def build_decoupled_scene_state(
    gaussians: GaussianModel,
    desk_object_id: int,
    decouple_object_ids: Sequence[int],
    source_iteration: int,
) -> DecoupledSceneState:
    labels = sorted({int(label) for label in decouple_object_ids if int(label) > 0})
    if not labels:
        raise ValueError("--decouple_object_id must contain at least one positive object id.")
    if int(desk_object_id) in labels:
        raise ValueError("--decouple_object_id must not contain --desk_object_id.")

    desk_mask = gaussians.get_object_id == int(desk_object_id)
    if int(desk_mask.sum().item()) <= 0:
        raise RuntimeError(f"No desk Gaussians matched desk_object_id={int(desk_object_id)}.")

    decoupled_mask = _label_mask(gaussians, labels)
    if int(decoupled_mask.sum().item()) <= 0:
        raise RuntimeError(f"No Gaussians matched decouple_object_id={labels}.")

    background_mask = torch.logical_not(torch.logical_or(decoupled_mask, desk_mask))
    if int(background_mask.sum().item()) <= 0:
        raise RuntimeError("No background/context Gaussians remain after desk and decoupled object split.")

    return DecoupledSceneState(
        decoupled_gaussians=_make_gaussian_subset(gaussians, decoupled_mask),
        desk_gaussians=_make_gaussian_subset(gaussians, desk_mask),
        background_gaussians=_make_gaussian_subset(gaussians, background_mask),
        source_iteration=int(source_iteration),
        desk_object_id=int(desk_object_id),
        decouple_object_ids=labels,
    )


def _resolve_optional_model_relative_path(path: Optional[str], model_path: str, default_name: str) -> str:
    if path is None:
        return os.path.join(model_path, default_name)
    if os.path.isabs(path):
        return path
    if os.path.exists(path):
        return path
    return os.path.join(model_path, path)


def _resolve_atlas_dir(model_path: str, desk_atlas_dir: str) -> str:
    return desk_atlas_dir if os.path.isabs(desk_atlas_dir) else os.path.join(model_path, desk_atlas_dir)


def _require_existing_file(path: str, desc: str) -> str:
    if path is None or not os.path.exists(path):
        raise FileNotFoundError(f"{desc} '{path}' not found.")
    return path


def _load_rgb_image_tensor(path: str, device: torch.device) -> torch.Tensor:
    with Image.open(path) as image:
        image_np = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(image_np).permute(2, 0, 1).contiguous().to(device=device)


def _load_single_channel_image_tensor(path: str, device: torch.device) -> torch.Tensor:
    with Image.open(path) as image:
        image_np = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    return torch.from_numpy(image_np).unsqueeze(0).contiguous().to(device=device)


def _normal_valid_mask(normal: torch.Tensor, eps: float = NORMAL_MAP_VALID_EPS) -> torch.Tensor:
    if normal.ndim < 1 or int(normal.shape[0]) != 3:
        raise ValueError(f"Expected normal tensor with leading 3 channels, got shape {tuple(normal.shape)}")
    return torch.linalg.norm(normal.float(), dim=0) > float(eps)


def _normalize_normal_image(normal: torch.Tensor, eps: float = NORMAL_MAP_VALID_EPS) -> torch.Tensor:
    if normal.ndim != 3 or int(normal.shape[0]) != 3:
        raise ValueError(f"Expected normal image shape (3, H, W), got {tuple(normal.shape)}")
    norm = torch.linalg.norm(normal.float(), dim=0, keepdim=True)
    valid = norm > float(eps)
    return torch.where(valid, normal.float() / norm.clamp_min(float(eps)), torch.zeros_like(normal.float()))


def _normalize_normal_vectors(vectors: torch.Tensor, eps: float = NORMAL_MAP_VALID_EPS) -> torch.Tensor:
    if vectors.ndim != 2 or int(vectors.shape[1]) != 3:
        raise ValueError(f"Expected normal vectors shape (N, 3), got {tuple(vectors.shape)}")
    norm = torch.linalg.norm(vectors.float(), dim=-1, keepdim=True)
    valid = norm > float(eps)
    return torch.where(valid, vectors.float() / norm.clamp_min(float(eps)), torch.zeros_like(vectors.float()))


def _load_normal_image_tensor(path: str, device: torch.device) -> torch.Tensor:
    normal = _load_rgb_image_tensor(path, device=device) * 2.0 - 1.0
    return _normalize_normal_image(normal, eps=NORMAL_IMAGE_VALID_EPS)


def _load_diffusion_pack(path: str) -> Dict[str, Any]:
    return torch.load(path, map_location="cpu")


def _load_desk_atlas_state_from_dir(
    model_path: str,
    desk_atlas_dir: str,
    device: torch.device,
) -> DeskAtlasState:
    atlas_dir = _resolve_atlas_dir(model_path, desk_atlas_dir)
    state_path = _require_existing_file(os.path.join(atlas_dir, "desk_atlas_state.pt"), "desk atlas state")
    return _deserialize_desk_atlas_state(torch.load(state_path, map_location="cpu"), device=device)


def _warn_support_mask_mismatch(packs: Dict[str, Dict[str, Any]], canonical_name: str = "diffuse") -> None:
    canonical = packs[canonical_name]
    for name, pack in packs.items():
        if name == canonical_name:
            continue
        for key in ("M_support_visible", "M_support_footprint"):
            if key not in canonical or key not in pack:
                continue
            if not torch.equal(canonical[key].bool(), pack[key].bool()):
                print(
                    f"[WARN] {key} differs between {canonical_name}/{name} diffusion packs. "
                    f"Using {canonical_name} diffusion pack mask as the canonical support mask."
                )
                return


def load_material_completion_assets(
    model_path: str,
    desk_atlas_dir: str,
    desk_atlas_state: DeskAtlasState,
    completed_diffuse_path: Optional[str],
    completed_fresnel_path: Optional[str],
    completed_reflect_path: Optional[str],
    completed_roughness_path: Optional[str],
    completed_normal_path: Optional[str],
    device: torch.device,
) -> Dict[str, Any]:
    atlas_dir = _resolve_atlas_dir(model_path, desk_atlas_dir)
    packs = {
        "diffuse": _load_diffusion_pack(
            _require_existing_file(os.path.join(atlas_dir, "desk_diffuse_diffusion_pack.pt"), "diffuse diffusion pack")
        ),
        "fresnel": _load_diffusion_pack(
            _require_existing_file(os.path.join(atlas_dir, "desk_fresnel_diffusion_pack.pt"), "fresnel diffusion pack")
        ),
        "reflect": _load_diffusion_pack(
            _require_existing_file(os.path.join(atlas_dir, "desk_reflect_diffusion_pack.pt"), "reflect diffusion pack")
        ),
        "roughness": _load_diffusion_pack(
            _require_existing_file(os.path.join(atlas_dir, "desk_roughness_diffusion_pack.pt"), "roughness diffusion pack")
        ),
        "normal": _load_diffusion_pack(
            _require_existing_file(os.path.join(atlas_dir, "desk_normal_diffusion_pack.pt"), "normal diffusion pack")
        ),
    }
    _warn_support_mask_mismatch(packs)

    image_paths = {
        "diffuse": _resolve_optional_model_relative_path(completed_diffuse_path, model_path, "diffuse_completed.png"),
        "fresnel": _resolve_optional_model_relative_path(completed_fresnel_path, model_path, "fresnel_completed.png"),
        "reflect": _resolve_optional_model_relative_path(completed_reflect_path, model_path, "reflect_completed.png"),
        "roughness": _resolve_optional_model_relative_path(completed_roughness_path, model_path, "roughness_completed.png"),
        "normal": _resolve_optional_model_relative_path(completed_normal_path, model_path, "normal_completed.png"),
    }
    for name, path in image_paths.items():
        _require_existing_file(path, f"completed {name} image")

    completed = {
        "diffuse": _load_rgb_image_tensor(image_paths["diffuse"], device=device),
        "fresnel": _load_rgb_image_tensor(image_paths["fresnel"], device=device),
        "reflect": _load_single_channel_image_tensor(image_paths["reflect"], device=device),
        "roughness": _load_single_channel_image_tensor(image_paths["roughness"], device=device),
        "normal": _load_normal_image_tensor(image_paths["normal"], device=device),
    }
    atlas_h, atlas_w = (int(desk_atlas_state.atlas_hw[0]), int(desk_atlas_state.atlas_hw[1]))
    for name, tensor in completed.items():
        if tuple(tensor.shape[1:]) != (atlas_h, atlas_w):
            raise RuntimeError(
                f"Completed {name} resolution mismatch: expected ({atlas_h}, {atlas_w}), "
                f"got {tuple(int(v) for v in tensor.shape[1:])}."
            )

    canonical_pack = packs["diffuse"]
    support_visible_mask = canonical_pack["M_support_visible"].bool().to(device=device)
    support_footprint_mask = canonical_pack["M_support_footprint"].bool().to(device=device)
    support_mask_raw = support_visible_mask | support_footprint_mask

    return {
        "completed": completed,
        "completed_paths": {name: os.path.abspath(path) for name, path in image_paths.items()},
        "support_visible_mask": support_visible_mask,
        "support_footprint_mask": support_footprint_mask,
        "support_mask_raw": support_mask_raw,
        "diffusion_packs": packs,
    }


def _uv_to_pixel_coords(
    uv: torch.Tensor,
    bbox: Tuple[float, float, float, float],
    height: int,
    width: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    u_min, u_max, v_min, v_max = bbox
    du = max(u_max - u_min, 1e-6)
    dv = max(v_max - v_min, 1e-6)
    x = (uv[:, 0] - u_min) / du * max(width - 1, 1)
    y = (uv[:, 1] - v_min) / dv * max(height - 1, 1)
    return x, y


def _pixel_coords_to_uv(
    x: torch.Tensor,
    y: torch.Tensor,
    bbox: Tuple[float, float, float, float],
    height: int,
    width: int,
) -> torch.Tensor:
    u_min, u_max, v_min, v_max = bbox
    du = max(u_max - u_min, 1e-6)
    dv = max(v_max - v_min, 1e-6)
    u = u_min + x / max(width - 1, 1) * du
    v = v_min + y / max(height - 1, 1) * dv
    return torch.stack([u, v], dim=-1)


def _compute_pixel_world_size(desk_atlas_state: DeskAtlasState) -> Tuple[float, float]:
    h, w = (int(desk_atlas_state.atlas_hw[0]), int(desk_atlas_state.atlas_hw[1]))
    u_min, u_max, v_min, v_max = desk_atlas_state.uv_bbox
    du = max(float(u_max - u_min), 1e-6)
    dv = max(float(v_max - v_min), 1e-6)
    return du / max(w - 1, 1), dv / max(h - 1, 1)


def _sample_mask_grid_coords(mask: torch.Tensor, stride_px: int) -> torch.Tensor:
    h, w = mask.shape
    stride_px = max(int(stride_px), 1)
    start = stride_px // 2
    ys = torch.arange(start, h, stride_px, device=mask.device)
    xs = torch.arange(start, w, stride_px, device=mask.device)
    if ys.numel() == 0:
        ys = torch.arange(0, h, device=mask.device)
    if xs.numel() == 0:
        xs = torch.arange(0, w, device=mask.device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    coords = torch.stack([grid_y.reshape(-1), grid_x.reshape(-1)], dim=-1)
    coords = coords[mask[coords[:, 0].long(), coords[:, 1].long()]]
    if coords.shape[0] > 0:
        return coords
    fallback = mask.nonzero(as_tuple=False)
    if fallback.shape[0] == 0:
        return fallback
    return fallback[::max(stride_px * stride_px, 1)]


def _dilate_binary_mask(mask: torch.Tensor, radius_px: int) -> torch.Tensor:
    radius_px = max(int(radius_px), 0)
    if radius_px <= 0:
        return mask.bool()
    x = mask.float().unsqueeze(0).unsqueeze(0)
    kernel = radius_px * 2 + 1
    y = F.max_pool2d(x, kernel_size=kernel, stride=1, padding=radius_px)
    return y.squeeze(0).squeeze(0) > 0.5


def _build_init_sampling_mask(
    support_footprint_mask: torch.Tensor,
    support_mask_raw: torch.Tensor,
    boundary_px: int,
) -> torch.Tensor:
    init_core_mask = support_footprint_mask.bool()
    boundary_band = _dilate_binary_mask(init_core_mask, boundary_px) & (~init_core_mask) & support_mask_raw.bool()
    return init_core_mask | boundary_band


def rotation_to_quaternion(rotation: torch.Tensor) -> torch.Tensor:
    r11, r22, r33 = rotation[:, 0, 0], rotation[:, 1, 1], rotation[:, 2, 2]
    qw = torch.sqrt((1.0 + r11 + r22 + r33).clamp_min(1e-7)) * 0.5
    qx = (rotation[:, 2, 1] - rotation[:, 1, 2]) / (4.0 * qw.clamp_min(1e-7))
    qy = (rotation[:, 0, 2] - rotation[:, 2, 0]) / (4.0 * qw.clamp_min(1e-7))
    qz = (rotation[:, 1, 0] - rotation[:, 0, 1]) / (4.0 * qw.clamp_min(1e-7))
    quaternion = torch.stack((qw, qx, qy, qz), dim=-1)
    return F.normalize(quaternion, dim=-1)


def _plane_basis_to_quaternion(
    plane: PlaneDefinition,
    count: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    rotation = torch.stack([plane.e1, plane.e2, plane.normal], dim=1).to(device=device, dtype=dtype)
    rotation = rotation.unsqueeze(0).expand(count, -1, -1).contiguous()
    return rotation_to_quaternion(rotation)


def _normal_vectors_to_plane_aligned_quaternion(
    normals: torch.Tensor,
    plane: PlaneDefinition,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    count = int(normals.shape[0])
    if count == 0:
        return torch.empty((0, 4), dtype=dtype, device=device)

    plane_normal = F.normalize(plane.normal.to(device=device, dtype=torch.float32), dim=0)
    plane_e1 = F.normalize(plane.e1.to(device=device, dtype=torch.float32), dim=0)
    plane_e2 = F.normalize(plane.e2.to(device=device, dtype=torch.float32), dim=0)
    plane_normal_b = plane_normal.unsqueeze(0).expand(count, -1)

    raw_normals = normals.to(device=device, dtype=torch.float32)
    valid = torch.linalg.norm(raw_normals, dim=-1, keepdim=True) > NORMAL_MAP_VALID_EPS
    normal = _normalize_normal_vectors(raw_normals, eps=NORMAL_MAP_VALID_EPS)
    normal = torch.where(valid, normal, plane_normal_b)
    normal = torch.where((normal * plane_normal_b).sum(dim=-1, keepdim=True) < 0.0, -normal, normal)

    t1_seed = plane_e1.unsqueeze(0).expand(count, -1)
    tangent = t1_seed - (t1_seed * normal).sum(dim=-1, keepdim=True) * normal
    tangent_valid = torch.linalg.norm(tangent, dim=-1, keepdim=True) > NORMAL_MAP_VALID_EPS

    fallback_seed = plane_e2.unsqueeze(0).expand(count, -1)
    fallback_tangent = fallback_seed - (fallback_seed * normal).sum(dim=-1, keepdim=True) * normal
    tangent = torch.where(tangent_valid, tangent, fallback_tangent)
    tangent = _normalize_normal_vectors(tangent, eps=NORMAL_MAP_VALID_EPS)

    bitangent = F.normalize(torch.cross(normal, tangent, dim=-1), dim=-1)
    tangent = F.normalize(torch.cross(bitangent, normal, dim=-1), dim=-1)
    rotation = torch.stack([tangent, bitangent, normal], dim=2).to(device=device, dtype=dtype)
    return rotation_to_quaternion(rotation)


def _material_inverse(values: torch.Tensor) -> torch.Tensor:
    return inverse_sigmoid(torch.clamp(values, 1e-6, 1.0 - 1e-6))


def initialize_hole_gaussians_into_desk_owner(
    desk_gaussians: GaussianModel,
    desk_atlas_state: DeskAtlasState,
    completed: Dict[str, torch.Tensor],
    hole_init_stride_px: int,
    boundary_px: int,
    support_footprint_mask: torch.Tensor,
    support_mask_raw: torch.Tensor,
    opt: OptimizationParams,
    desk_object_id: int,
    hole_reflection_visible: bool,
) -> Dict[str, Any]:
    device = desk_gaussians.get_xyz.device
    init_sampling_mask = _build_init_sampling_mask(
        support_footprint_mask=support_footprint_mask.to(device=device).bool(),
        support_mask_raw=support_mask_raw.to(device=device).bool(),
        boundary_px=boundary_px,
    )
    coords = _sample_mask_grid_coords(init_sampling_mask, hole_init_stride_px)
    if coords.shape[0] == 0:
        raise RuntimeError("Initialization support footprint mask is empty; cannot initialize hole gaussians.")

    if desk_gaussians.optimizer is None:
        desk_gaussians.training_setup(opt)

    ys = coords[:, 0].float()
    xs = coords[:, 1].float()
    uv = _pixel_coords_to_uv(
        x=xs,
        y=ys,
        bbox=desk_atlas_state.uv_bbox,
        height=int(desk_atlas_state.atlas_hw[0]),
        width=int(desk_atlas_state.atlas_hw[1]),
    )

    plane = desk_atlas_state.plane
    xyz = plane.origin[None] + uv[:, 0:1] * plane.e1[None] + uv[:, 1:2] * plane.e2[None]
    sampled_diffuse = completed["diffuse"][:, coords[:, 0].long(), coords[:, 1].long()].permute(1, 0).contiguous()
    sampled_fresnel = completed["fresnel"][:, coords[:, 0].long(), coords[:, 1].long()].permute(1, 0).contiguous()
    sampled_reflect = completed["reflect"][:, coords[:, 0].long(), coords[:, 1].long()].permute(1, 0).contiguous()
    sampled_roughness = completed["roughness"][:, coords[:, 0].long(), coords[:, 1].long()].permute(1, 0).contiguous()
    n_new = int(sampled_diffuse.shape[0])

    pixel_world_x, pixel_world_y = _compute_pixel_world_size(desk_atlas_state)
    scale_x = max(0.5 * float(hole_init_stride_px) * pixel_world_x, 1e-5)
    scale_y = max(0.5 * float(hole_init_stride_px) * pixel_world_y, 1e-5)
    scale_z = max(0.1 * min(scale_x, scale_y), 1e-6)
    target_scales_actual = torch.tensor([scale_x, scale_y, scale_z], dtype=desk_gaussians.get_xyz.dtype, device=device)

    scaling = torch.log(target_scales_actual[None].expand(n_new, -1))
    rotation = _plane_basis_to_quaternion(
        plane,
        n_new,
        device=device,
        dtype=desk_gaussians.get_xyz.dtype,
    )
    opacity = inverse_sigmoid(torch.full((n_new, 1), 0.01, dtype=desk_gaussians.get_xyz.dtype, device=device))
    features_dc = RGB2SH(sampled_diffuse).unsqueeze(1)
    features_rest = torch.zeros(
        (n_new, (desk_gaussians.max_sh_degree + 1) ** 2 - 1, 3),
        dtype=desk_gaussians.get_xyz.dtype,
        device=device,
    )

    old_count = _gaussian_count(desk_gaussians)
    desk_gaussians.densification_postfix(
        xyz,
        features_dc,
        features_rest,
        opacity,
        scaling,
        rotation,
        _material_inverse(sampled_diffuse),
        _material_inverse(sampled_fresnel),
        _material_inverse(sampled_roughness),
        _material_inverse(sampled_reflect),
    )
    new_count = _gaussian_count(desk_gaussians)

    with torch.no_grad():
        desk_gaussians.object_id[old_count:new_count] = int(desk_object_id)
        desk_gaussians.object_score[old_count:new_count] = 1.0
        if desk_gaussians._mask is not None and desk_gaussians._mask.shape[0] == new_count:
            desk_gaussians._mask.data[old_count:new_count] = 0.0
        if hasattr(desk_gaussians, "_ensure_reflection_visible_state"):
            desk_gaussians._ensure_reflection_visible_state()
            desk_gaussians.reflection_visible[old_count:new_count] = bool(hole_reflection_visible)

    return {
        "hole_start_idx": old_count,
        "hole_end_idx": new_count,
        "hole_init_uv": uv.detach().clone(),
        "target_scales_actual": target_scales_actual,
        "hole_init_count": n_new,
    }


def _build_elliptical_structure(kernel_size: int) -> np.ndarray:
    kernel_size = max(int(kernel_size), 1)
    if kernel_size % 2 == 0:
        kernel_size += 1
    if kernel_size == 1:
        return np.ones((1, 1), dtype=bool)
    radius = 0.5 * float(kernel_size - 1)
    yy, xx = np.mgrid[0:kernel_size, 0:kernel_size].astype(np.float32)
    ellipse = ((yy - radius) / max(radius, 1e-6)) ** 2 + ((xx - radius) / max(radius, 1e-6)) ** 2 <= 1.0
    return ellipse.astype(bool)


def _binary_dilate_without_scipy(mask_bool: np.ndarray, structure: np.ndarray) -> np.ndarray:
    mask_tensor = torch.from_numpy(mask_bool.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    kernel_tensor = torch.from_numpy(structure.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    pad_y = structure.shape[0] // 2
    pad_x = structure.shape[1] // 2
    response = F.conv2d(F.pad(mask_tensor, (pad_x, pad_x, pad_y, pad_y)), kernel_tensor)
    return response.squeeze(0).squeeze(0).numpy() > 0.0


def _binary_erode_without_scipy(mask_bool: np.ndarray, structure: np.ndarray) -> np.ndarray:
    mask_tensor = torch.from_numpy(mask_bool.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    kernel_tensor = torch.from_numpy(structure.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    pad_y = structure.shape[0] // 2
    pad_x = structure.shape[1] // 2
    response = F.conv2d(F.pad(mask_tensor, (pad_x, pad_x, pad_y, pad_y)), kernel_tensor)
    required = float(structure.astype(np.float32).sum())
    return response.squeeze(0).squeeze(0).numpy() >= required - 1e-6


def _binary_close_without_scipy(mask_bool: np.ndarray, kernel_size: int) -> np.ndarray:
    structure = _build_elliptical_structure(kernel_size)
    return _binary_erode_without_scipy(_binary_dilate_without_scipy(mask_bool, structure), structure)


def _fill_internal_holes_without_scipy(mask_bool: np.ndarray) -> Tuple[np.ndarray, int, int]:
    h, w = mask_bool.shape
    exterior = np.zeros((h, w), dtype=bool)
    queue: deque[Tuple[int, int]] = deque()

    def _try_push(y: int, x: int) -> None:
        if 0 <= y < h and 0 <= x < w and (not mask_bool[y, x]) and (not exterior[y, x]):
            exterior[y, x] = True
            queue.append((y, x))

    for x in range(w):
        _try_push(0, x)
        _try_push(h - 1, x)
    for y in range(h):
        _try_push(y, 0)
        _try_push(y, w - 1)

    while queue:
        y, x = queue.popleft()
        for dy, dx in ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)):
            _try_push(y + dy, x + dx)

    holes = (~mask_bool) & (~exterior)
    if not holes.any():
        return mask_bool.copy(), 0, 0

    visited = np.zeros_like(holes, dtype=bool)
    filled = mask_bool.copy()
    hole_count = 0
    filled_area = 0
    for start_y, start_x in np.argwhere(holes):
        if visited[start_y, start_x]:
            continue
        hole_count += 1
        area = 0
        queue.append((int(start_y), int(start_x)))
        visited[start_y, start_x] = True
        while queue:
            y, x = queue.popleft()
            filled[y, x] = True
            area += 1
            for dy, dx in ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and holes[ny, nx] and (not visited[ny, nx]):
                    visited[ny, nx] = True
                    queue.append((ny, nx))
        filled_area += area
    return filled, hole_count, filled_area


def _fill_internal_holes(mask_bool: np.ndarray) -> Tuple[np.ndarray, int, int]:
    if label is None:
        return _fill_internal_holes_without_scipy(mask_bool)
    connectivity = np.ones((3, 3), dtype=np.uint8)
    labels, num_labels = label(~mask_bool, structure=connectivity)
    if num_labels <= 0:
        return mask_bool, 0, 0
    border_labels = np.unique(np.concatenate([labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]], axis=0))
    filled = mask_bool.copy()
    filled_hole_count = 0
    filled_area = 0
    for label_idx in range(1, int(num_labels) + 1):
        if np.any(border_labels == label_idx):
            continue
        hole_mask = labels == label_idx
        area = int(hole_mask.sum())
        if area <= 0:
            continue
        filled[hole_mask] = True
        filled_hole_count += 1
        filled_area += area
    return filled, filled_hole_count, filled_area


def repair_and_shrink_binary_mask(
    mask: torch.Tensor,
    shrink_px: float,
    close_kernel_size: int = 0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    mask_bool = mask.bool()
    if not bool(mask_bool.any().item()):
        raise RuntimeError("Support mask is empty.")

    mask_np = mask_bool.detach().cpu().numpy().astype(bool)
    original_pixels = int(mask_np.sum())
    close_kernel_size = int(close_kernel_size)
    if close_kernel_size > 1:
        if binary_closing is not None:
            mask_closed = binary_closing(mask_np, structure=_build_elliptical_structure(close_kernel_size))
        else:
            print("[WARN] scipy.ndimage.binary_closing unavailable; using torch fallback.")
            mask_closed = _binary_close_without_scipy(mask_np, close_kernel_size)
    else:
        mask_closed = mask_np.copy()
    closed_pixels = int(mask_closed.sum())

    mask_filled, filled_hole_count, filled_hole_area = _fill_internal_holes(mask_closed)
    filled_pixels = int(mask_filled.sum())

    shrink_px = float(shrink_px)
    if shrink_px > 0.0:
        if distance_transform_edt is not None:
            shrunk_np = distance_transform_edt(mask_filled.astype(np.uint8)) > shrink_px
        else:
            radius = max(int(np.ceil(shrink_px)), 1)
            shrunk_np = _binary_erode_without_scipy(mask_filled, _build_elliptical_structure(radius * 2 + 1))
    else:
        shrunk_np = mask_filled

    shrunk_mask = torch.from_numpy(shrunk_np).to(device=mask_bool.device).bool()
    if not bool(shrunk_mask.any().item()):
        raise RuntimeError(
            "Shrunk support mask is empty. "
            f"Reduce --desk_support_shrink_px (current value: {shrink_px})."
        )
    return shrunk_mask, {
        "original_pixels": float(original_pixels),
        "closed_pixels": float(closed_pixels),
        "filled_pixels": float(filled_pixels),
        "shrunk_pixels": float(int(shrunk_mask.sum().item())),
        "filled_hole_count": float(filled_hole_count),
        "filled_hole_area": float(filled_hole_area),
        "close_kernel_size": float(close_kernel_size),
        "shrink_px": float(shrink_px),
    }


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
    plane: PlaneDefinition,
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


def project_material_atlas_to_view(
    camera,
    atlas_tensor: torch.Tensor,
    plane: PlaneDefinition,
    uv_bbox: Tuple[float, float, float, float],
    atlas_hw: Tuple[int, int],
) -> Dict[str, torch.Tensor]:
    view_value, view_valid = _project_atlas_tensor_to_view(
        camera=camera,
        atlas_tensor=atlas_tensor,
        plane=plane,
        uv_bbox=uv_bbox,
        atlas_hw=atlas_hw,
        mode="bilinear",
    )
    return {"value": view_value, "valid_mask": view_valid}


def project_normal_atlas_to_view(
    camera,
    atlas_tensor: torch.Tensor,
    plane: PlaneDefinition,
    uv_bbox: Tuple[float, float, float, float],
    atlas_hw: Tuple[int, int],
) -> Dict[str, torch.Tensor]:
    view_value, view_valid = _project_atlas_tensor_to_view(
        camera=camera,
        atlas_tensor=atlas_tensor,
        plane=plane,
        uv_bbox=uv_bbox,
        atlas_hw=atlas_hw,
        mode="bilinear",
    )
    raw_valid = _normal_valid_mask(view_value, eps=NORMAL_IMAGE_VALID_EPS)
    view_value = _normalize_normal_image(view_value, eps=NORMAL_IMAGE_VALID_EPS)
    return {"value": view_value, "valid_mask": view_valid & raw_valid}


def project_binary_atlas_mask_to_view(
    camera,
    atlas_mask: torch.Tensor,
    plane: PlaneDefinition,
    uv_bbox: Tuple[float, float, float, float],
    atlas_hw: Tuple[int, int],
) -> torch.Tensor:
    view_mask, view_valid = _project_atlas_tensor_to_view(
        camera=camera,
        atlas_tensor=atlas_mask.float().unsqueeze(0),
        plane=plane,
        uv_bbox=uv_bbox,
        atlas_hw=atlas_hw,
        mode="nearest",
    )
    return view_valid & (view_mask.squeeze(0) > 0.5)


def _view_target_entry_to_device(entry: ViewTargetCacheEntry, device: torch.device) -> ViewTargetCacheEntry:
    return ViewTargetCacheEntry(
        target_diffuse_view=entry.target_diffuse_view.to(device=device),
        target_fresnel_view=entry.target_fresnel_view.to(device=device),
        target_reflect_view=entry.target_reflect_view.to(device=device),
        target_roughness_view=entry.target_roughness_view.to(device=device),
        target_normal_view=entry.target_normal_view.to(device=device),
        merge_mask_view=entry.merge_mask_view.to(device=device),
        valid_mask_view=entry.valid_mask_view.to(device=device),
        supervision_mask_view=entry.supervision_mask_view.to(device=device),
        normal_supervision_mask_view=entry.normal_supervision_mask_view.to(device=device),
        reproj_completed_diffuse_view=entry.reproj_completed_diffuse_view.to(device=device),
        reproj_completed_fresnel_view=entry.reproj_completed_fresnel_view.to(device=device),
        reproj_completed_reflect_view=entry.reproj_completed_reflect_view.to(device=device),
        reproj_completed_roughness_view=entry.reproj_completed_roughness_view.to(device=device),
        reproj_completed_normal_view=entry.reproj_completed_normal_view.to(device=device),
        removal_diffuse_view=entry.removal_diffuse_view.to(device=device),
        removal_fresnel_view=entry.removal_fresnel_view.to(device=device),
        removal_reflect_view=entry.removal_reflect_view.to(device=device),
        removal_roughness_view=entry.removal_roughness_view.to(device=device),
        removal_normal_view=entry.removal_normal_view.to(device=device),
    )


def build_view_target_cache(
    scene: Scene,
    dataset: ModelParams,
    decoupled_state: DecoupledSceneState,
    material_state: MaterialInpaintState,
    desk_atlas_state: DeskAtlasState,
    completed: Dict[str, torch.Tensor],
    merge_atlas_mask: torch.Tensor,
    pipe: PipelineParams,
    background: torch.Tensor,
) -> Dict[int, ViewTargetCacheEntry]:
    clean_background_gaussians = _create_reference_clean_background_composite(
        decoupled_state.background_gaussians,
        decoupled_state.desk_gaussians,
        desk_prefix_end_idx=int(material_state.hole_start_idx),
    )
    cache: Dict[int, ViewTargetCacheEntry] = {}

    with torch.no_grad():
        for cam in tqdm(scene.getTrainCameras(), desc="BuildInpaintTargets"):
            removal_pkg = render(cam, clean_background_gaussians, pipe, background, dataset.kernel_size)
            removal_diffuse = removal_pkg["rend_diffuse"].detach().clamp(0.0, 1.0)
            removal_fresnel = removal_pkg["rend_fresnel"].detach().clamp(0.0, 1.0)
            removal_reflect = removal_pkg["rend_reflect"].detach().clamp(0.0, 1.0)
            removal_roughness = removal_pkg["rend_roughness"].detach().clamp(0.0, 1.0)
            removal_normal = _normalize_normal_image(removal_pkg["rend_normal"].detach(), eps=NORMAL_MAP_VALID_EPS)
            removal_normal_valid = _normal_valid_mask(removal_normal, eps=0.5)

            material_completed = {
                name: completed[name]
                for name in ("diffuse", "fresnel", "reflect", "roughness")
            }
            reproj = {
                name: project_material_atlas_to_view(
                    camera=cam,
                    atlas_tensor=tensor,
                    plane=desk_atlas_state.plane,
                    uv_bbox=desk_atlas_state.uv_bbox,
                    atlas_hw=desk_atlas_state.atlas_hw,
                )
                for name, tensor in material_completed.items()
            }
            reproj_normal = project_normal_atlas_to_view(
                camera=cam,
                atlas_tensor=completed["normal"],
                plane=desk_atlas_state.plane,
                uv_bbox=desk_atlas_state.uv_bbox,
                atlas_hw=desk_atlas_state.atlas_hw,
            )
            merge_mask_view = project_binary_atlas_mask_to_view(
                camera=cam,
                atlas_mask=merge_atlas_mask,
                plane=desk_atlas_state.plane,
                uv_bbox=desk_atlas_state.uv_bbox,
                atlas_hw=desk_atlas_state.atlas_hw,
            )
            valid_mask = (
                merge_mask_view
                & reproj["diffuse"]["valid_mask"]
                & reproj["fresnel"]["valid_mask"]
                & reproj["reflect"]["valid_mask"]
                & reproj["roughness"]["valid_mask"]
            )
            supervision_mask = (~merge_mask_view) | valid_mask
            target_diffuse = torch.where(valid_mask.unsqueeze(0), reproj["diffuse"]["value"], removal_diffuse)
            target_fresnel = torch.where(valid_mask.unsqueeze(0), reproj["fresnel"]["value"], removal_fresnel)
            target_reflect = torch.where(valid_mask.unsqueeze(0), reproj["reflect"]["value"], removal_reflect)
            target_roughness = torch.where(valid_mask.unsqueeze(0), reproj["roughness"]["value"], removal_roughness)
            normal_merge_valid = merge_mask_view & reproj_normal["valid_mask"]
            normal_supervision_mask = normal_merge_valid | ((~merge_mask_view) & removal_normal_valid)
            target_normal = torch.where(normal_merge_valid.unsqueeze(0), reproj_normal["value"], removal_normal)

            cache[int(cam.uid)] = ViewTargetCacheEntry(
                target_diffuse_view=target_diffuse.detach().cpu(),
                target_fresnel_view=target_fresnel.detach().cpu(),
                target_reflect_view=target_reflect.detach().cpu(),
                target_roughness_view=target_roughness.detach().cpu(),
                target_normal_view=target_normal.detach().cpu(),
                merge_mask_view=merge_mask_view.detach().cpu(),
                valid_mask_view=valid_mask.detach().cpu(),
                supervision_mask_view=supervision_mask.detach().cpu(),
                normal_supervision_mask_view=normal_supervision_mask.detach().cpu(),
                reproj_completed_diffuse_view=reproj["diffuse"]["value"].detach().cpu(),
                reproj_completed_fresnel_view=reproj["fresnel"]["value"].detach().cpu(),
                reproj_completed_reflect_view=reproj["reflect"]["value"].detach().cpu(),
                reproj_completed_roughness_view=reproj["roughness"]["value"].detach().cpu(),
                reproj_completed_normal_view=reproj_normal["value"].detach().cpu(),
                removal_diffuse_view=removal_diffuse.detach().cpu(),
                removal_fresnel_view=removal_fresnel.detach().cpu(),
                removal_reflect_view=removal_reflect.detach().cpu(),
                removal_roughness_view=removal_roughness.detach().cpu(),
                removal_normal_view=removal_normal.detach().cpu(),
            )

    del clean_background_gaussians
    return cache


def build_material_training_state(
    scene: Scene,
    dataset: ModelParams,
    decoupled_state: DecoupledSceneState,
    material_state: MaterialInpaintState,
    desk_atlas_state: DeskAtlasState,
    completed: Dict[str, torch.Tensor],
    merge_atlas_mask: torch.Tensor,
    pipe: PipelineParams,
    background: torch.Tensor,
    dtype: torch.dtype,
    device: torch.device,
) -> MaterialTrainingState:
    pixel_world_x, pixel_world_y = _compute_pixel_world_size(desk_atlas_state)
    scale_x = max(0.5 * float(material_state.hole_init_stride_px) * pixel_world_x, 1e-5)
    scale_y = max(0.5 * float(material_state.hole_init_stride_px) * pixel_world_y, 1e-5)
    target_scales_actual = torch.tensor(
        [scale_x, scale_y, max(0.1 * min(scale_x, scale_y), 1e-6)],
        dtype=dtype,
        device=device,
    )
    return MaterialTrainingState(
        target_scales_actual=target_scales_actual,
        view_target_cache=build_view_target_cache(
            scene=scene,
            dataset=dataset,
            decoupled_state=decoupled_state,
            material_state=material_state,
            desk_atlas_state=desk_atlas_state,
            completed=completed,
            merge_atlas_mask=merge_atlas_mask,
            pipe=pipe,
            background=background,
        ),
        hole_start_idx=int(material_state.hole_start_idx),
        hole_end_idx=int(material_state.hole_end_idx),
    )


def _set_gaussian_trainability(gaussians: GaussianModel, enabled: bool) -> None:
    trainable_names = {"_xyz", "_opacity", "_scaling", "_rotation", "_diffuse", "_fresnel", "_roughness", "_reflect"}
    for attr_name in GAUSSIAN_ATTRS.values():
        tensor = getattr(gaussians, attr_name)
        if isinstance(tensor, torch.Tensor):
            tensor.requires_grad_(bool(enabled and attr_name in trainable_names))
    if getattr(gaussians, "_mask", None) is not None:
        gaussians._mask.requires_grad_(False)
    for param in gaussians.env_light.parameters():
        param.requires_grad_(False)
    for param in gaussians.roughness_net.parameters():
        param.requires_grad_(False)


def _freeze_all_gaussian_params(gaussians: GaussianModel) -> None:
    for attr_name in GAUSSIAN_ATTRS.values():
        tensor = getattr(gaussians, attr_name)
        if isinstance(tensor, torch.Tensor):
            tensor.requires_grad_(False)
    if getattr(gaussians, "_mask", None) is not None:
        gaussians._mask.requires_grad_(False)
    for param in gaussians.env_light.parameters():
        param.requires_grad_(False)
    for param in gaussians.roughness_net.parameters():
        param.requires_grad_(False)
    gaussians.optimizer = None
    gaussians.mask_optimizer = None


def _weighted_l1_loss(pred: torch.Tensor, target: torch.Tensor, weight: Optional[torch.Tensor]) -> torch.Tensor:
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch in material L1 loss: pred={pred.shape} target={target.shape}")
    diff = torch.abs(pred - target)
    if weight is None:
        return diff.mean()
    weight = weight.float()
    denom = weight.sum() * pred.shape[0] + 1e-8
    return (diff * weight.unsqueeze(0)).sum() / denom


def compute_material_inpaint_losses(
    decoupled_state: DecoupledSceneState,
    desk_atlas_state: DeskAtlasState,
    training_state: MaterialTrainingState,
    material_state: MaterialInpaintState,
    render_pkg: Dict[str, torch.Tensor],
    view_target_entry: ViewTargetCacheEntry,
    args,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    weight = view_target_entry.supervision_mask_view
    loss_diffuse = _weighted_l1_loss(render_pkg["rend_diffuse"], view_target_entry.target_diffuse_view, weight)
    loss_fresnel = _weighted_l1_loss(render_pkg["rend_fresnel"], view_target_entry.target_fresnel_view, weight)
    loss_reflect = _weighted_l1_loss(render_pkg["rend_reflect"], view_target_entry.target_reflect_view, weight)
    loss_roughness = _weighted_l1_loss(render_pkg["rend_roughness"], view_target_entry.target_roughness_view, weight)
    pred_normal = _normalize_normal_image(render_pkg["rend_normal"], eps=NORMAL_MAP_VALID_EPS)
    loss_normal = _weighted_l1_loss(
        pred_normal,
        view_target_entry.target_normal_view,
        view_target_entry.normal_supervision_mask_view,
    )

    start_idx = int(material_state.hole_start_idx)
    end_idx = int(material_state.hole_end_idx)
    hole_count = max(end_idx - start_idx, 0)
    if hole_count > 0:
        desk_gaussians = decoupled_state.desk_gaussians
        hole_xyz = desk_gaussians.get_xyz[start_idx:end_idx]
        hole_scales = desk_gaussians.get_scaling[start_idx:end_idx]
        plane = desk_atlas_state.plane
        loss_plane = torch.abs(hole_xyz @ plane.normal + plane.d).mean()
        current_uv = project_xyz_to_plane_uv(hole_xyz, plane)
        init_uv = material_state.hole_init_uv.to(device=hole_xyz.device, dtype=hole_xyz.dtype)
        loss_anchor_uv = torch.abs(current_uv - init_uv).mean()
        loss_scale = torch.abs(hole_scales - training_state.target_scales_actual.unsqueeze(0)).mean()
    else:
        zero = loss_diffuse.new_zeros(())
        loss_plane = zero
        loss_anchor_uv = zero
        loss_scale = zero

    total_loss = (
        float(args.lambda_diffuse) * loss_diffuse
        + float(args.lambda_fresnel) * loss_fresnel
        + float(args.lambda_reflect) * loss_reflect
        + float(args.lambda_roughness) * loss_roughness
        + float(args.lambda_normal) * loss_normal
        + float(args.lambda_hole_plane) * loss_plane
        + float(args.lambda_hole_anchor_uv) * loss_anchor_uv
        + float(args.lambda_hole_scale) * loss_scale
    )
    tb_dict = {
        "loss_diffuse": float(loss_diffuse.item()),
        "loss_fresnel": float(loss_fresnel.item()),
        "loss_reflect": float(loss_reflect.item()),
        "loss_roughness": float(loss_roughness.item()),
        "loss_normal": float(loss_normal.item()),
        "loss_hole_plane": float(loss_plane.item()),
        "loss_hole_anchor_uv": float(loss_anchor_uv.item()),
        "loss_hole_scale": float(loss_scale.item()),
        "loss_total": float(total_loss.item()),
        "hole_count": float(hole_count),
        "merge_mask_mean": float(view_target_entry.merge_mask_view.float().mean().item()),
        "valid_mask_mean": float(view_target_entry.valid_mask_view.float().mean().item()),
        "supervision_mean": float(view_target_entry.supervision_mask_view.float().mean().item()),
        "normal_supervision_mean": float(view_target_entry.normal_supervision_mask_view.float().mean().item()),
    }
    return total_loss, tb_dict


def _project_hole_gaussians_back_to_plane(
    desk_gaussians: GaussianModel,
    material_state: MaterialInpaintState,
    desk_atlas_state: DeskAtlasState,
    target_scales_actual: torch.Tensor,
) -> None:
    start_idx = int(material_state.hole_start_idx)
    end_idx = int(material_state.hole_end_idx)
    if end_idx <= start_idx:
        return
    with torch.no_grad():
        plane = desk_atlas_state.plane
        hole_xyz = desk_gaussians._xyz.data[start_idx:end_idx]
        hole_uv = project_xyz_to_plane_uv(hole_xyz, plane)
        desk_gaussians._xyz.data[start_idx:end_idx] = (
            plane.origin[None] + hole_uv[:, 0:1] * plane.e1[None] + hole_uv[:, 1:2] * plane.e2[None]
        )
        hole_scales = desk_gaussians.get_scaling[start_idx:end_idx]
        min_scale = 0.5 * target_scales_actual
        max_scale = 2.0 * target_scales_actual
        clamped_scale = torch.maximum(torch.minimum(hole_scales, max_scale.unsqueeze(0)), min_scale.unsqueeze(0))
        desk_gaussians._scaling.data[start_idx:end_idx] = torch.log(clamped_scale.clamp_min(1e-6))


def _save_training_vis(
    model_path: str,
    iteration: int,
    viewpoint_cam,
    render_pkg: Dict[str, torch.Tensor],
    view_target_entry: ViewTargetCacheEntry,
) -> None:
    def _to_three_channels(image: torch.Tensor) -> torch.Tensor:
        if image.shape[0] == 3:
            return image
        if image.shape[0] == 1:
            return image.repeat(3, 1, 1)
        raise ValueError(f"Expected 1 or 3 channels for visualization, got shape {tuple(image.shape)}")

    def _normal_for_vis(image: torch.Tensor) -> torch.Tensor:
        return (_normalize_normal_image(image, eps=NORMAL_MAP_VALID_EPS) * 0.5 + 0.5).clamp(0.0, 1.0)

    vis_dir = os.path.join(model_path, "visualize_inpaint")
    os.makedirs(vis_dir, exist_ok=True)
    safe_name = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in str(getattr(viewpoint_cam, "image_name", "camera")))
    grid = make_grid(
        torch.stack(
            [
                render_pkg["rend_diffuse"].detach().clamp(0.0, 1.0),
                view_target_entry.target_diffuse_view.detach().clamp(0.0, 1.0),
                view_target_entry.reproj_completed_diffuse_view.detach().clamp(0.0, 1.0),
                view_target_entry.removal_diffuse_view.detach().clamp(0.0, 1.0),
                render_pkg["rend_fresnel"].detach().clamp(0.0, 1.0),
                view_target_entry.target_fresnel_view.detach().clamp(0.0, 1.0),
                view_target_entry.reproj_completed_fresnel_view.detach().clamp(0.0, 1.0),
                view_target_entry.removal_fresnel_view.detach().clamp(0.0, 1.0),
                _to_three_channels(render_pkg["rend_reflect"].detach()).clamp(0.0, 1.0),
                _to_three_channels(view_target_entry.target_reflect_view.detach()).clamp(0.0, 1.0),
                _to_three_channels(view_target_entry.reproj_completed_reflect_view.detach()).clamp(0.0, 1.0),
                _to_three_channels(view_target_entry.removal_reflect_view.detach()).clamp(0.0, 1.0),
                _to_three_channels(render_pkg["rend_roughness"].detach()).clamp(0.0, 1.0),
                _to_three_channels(view_target_entry.target_roughness_view.detach()).clamp(0.0, 1.0),
                _to_three_channels(view_target_entry.reproj_completed_roughness_view.detach()).clamp(0.0, 1.0),
                _to_three_channels(view_target_entry.removal_roughness_view.detach()).clamp(0.0, 1.0),
                _normal_for_vis(render_pkg["rend_normal"].detach()),
                _normal_for_vis(view_target_entry.target_normal_view.detach()),
                _normal_for_vis(view_target_entry.reproj_completed_normal_view.detach()),
                _normal_for_vis(view_target_entry.removal_normal_view.detach()),
                view_target_entry.merge_mask_view.detach().unsqueeze(0).repeat(3, 1, 1).float(),
                view_target_entry.valid_mask_view.detach().unsqueeze(0).repeat(3, 1, 1).float(),
                view_target_entry.supervision_mask_view.detach().unsqueeze(0).repeat(3, 1, 1).float(),
                torch.zeros_like(render_pkg["rend_diffuse"].detach()),
            ],
            dim=0,
        ),
        nrow=4,
    )
    save_image(grid, os.path.join(vis_dir, f"{iteration:06d}_{safe_name}.png"))


def _metadata_payload(
    iteration: int,
    decoupled_state: DecoupledSceneState,
    material_state: MaterialInpaintState,
    desk_atlas_state: DeskAtlasState,
    support_stats: Dict[str, float],
    init_footprint_stats: Dict[str, float],
) -> Dict[str, Any]:
    return {
        "phase": INPAINT_PHASE,
        "version": CHECKPOINT_VERSION,
        "iteration": int(iteration),
        "source_iteration": int(decoupled_state.source_iteration),
        "desk_object_id": int(decoupled_state.desk_object_id),
        "decouple_object_id": [int(v) for v in decoupled_state.decouple_object_ids],
        "hole_start_idx": int(material_state.hole_start_idx),
        "hole_end_idx": int(material_state.hole_end_idx),
        "hole_init_stride_px": int(material_state.hole_init_stride_px),
        "hole_reflection_visible": bool(material_state.hole_reflection_visible),
        "completed_paths": {
            "diffuse": str(material_state.completed_diffuse_path),
            "fresnel": str(material_state.completed_fresnel_path),
            "reflect": str(material_state.completed_reflect_path),
            "roughness": str(material_state.completed_roughness_path),
            "normal": str(material_state.completed_normal_path),
        },
        "desk_atlas_hw": [int(desk_atlas_state.atlas_hw[0]), int(desk_atlas_state.atlas_hw[1])],
        "desk_atlas_uv_bbox": [float(v) for v in desk_atlas_state.uv_bbox],
        "support_stats": {key: float(value) for key, value in support_stats.items()},
        "init_footprint_stats": {key: float(value) for key, value in init_footprint_stats.items()},
    }


def _save_full_scene_snapshot(
    model_path: str,
    iteration: int,
    decoupled_state: DecoupledSceneState,
    material_state: MaterialInpaintState,
    desk_atlas_state: DeskAtlasState,
    support_stats: Dict[str, float],
    init_footprint_stats: Dict[str, float],
) -> None:
    print(f"\n[ITER {iteration}] Saving inpainted full-scene snapshot")
    full_scene_gaussians = materialize_full_scene_gaussians(decoupled_state)
    point_cloud_path = os.path.join(model_path, "point_cloud", f"iteration_{iteration}")
    os.makedirs(point_cloud_path, exist_ok=True)
    ply_path = os.path.join(point_cloud_path, "point_cloud.ply")
    full_scene_gaussians.save_ply(ply_path, include_mask=True)
    pth_path = os.path.join(point_cloud_path, "point_cloud.pth")
    torch.save(
        {
            "env_net": full_scene_gaussians.env_light.state_dict(),
            "roughness_net": full_scene_gaussians.roughness_net.state_dict(),
            "gaussians": _cpu_clone_tree(full_scene_gaussians.capture(include_mask=True)),
            "inpaint": _metadata_payload(
                iteration,
                decoupled_state,
                material_state,
                desk_atlas_state,
                support_stats,
                init_footprint_stats,
            ),
        },
        pth_path,
    )
    del full_scene_gaussians


def _save_full_scene_checkpoint(
    model_path: str,
    iteration: int,
    decoupled_state: DecoupledSceneState,
    material_state: MaterialInpaintState,
    desk_atlas_state: DeskAtlasState,
    support_stats: Dict[str, float],
    init_footprint_stats: Dict[str, float],
) -> None:
    full_scene_gaussians = materialize_full_scene_gaussians(decoupled_state)
    checkpoint_path = os.path.join(model_path, f"chkpnt{iteration}.pth")
    print(f"\n[ITER {iteration}] Saving inpaint checkpoint to {checkpoint_path}")
    torch.save(
        {
            "phase": INPAINT_PHASE,
            "version": CHECKPOINT_VERSION,
            "iteration": int(iteration),
            "gaussians": _cpu_clone_tree(full_scene_gaussians.capture(include_mask=True)),
            "inpaint": _metadata_payload(
                iteration,
                decoupled_state,
                material_state,
                desk_atlas_state,
                support_stats,
                init_footprint_stats,
            ),
        },
        checkpoint_path,
    )
    del full_scene_gaussians


def _step_owner_optimizer(gaussians: GaussianModel) -> None:
    if gaussians.optimizer is None:
        return
    gaussians.optimizer.step()
    gaussians.optimizer.zero_grad(set_to_none=True)


def training(dataset: ModelParams, opt: OptimizationParams, pipe: PipelineParams, args) -> Dict[str, Any]:
    if args.iteration == 0:
        raise ValueError("--iteration must be -1 or a positive trained iteration.")
    desk_object_id = parse_single_desk_object_id(getattr(args, "desk_object_id", None))
    decouple_object_ids = sorted({int(v) for v in parse_object_id_list(getattr(args, "decouple_object_id", None)) if int(v) > 0})
    if not decouple_object_ids:
        raise ValueError("--decouple_object_id is required and must contain at least one positive id.")

    scene_gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, scene_gaussians, load_iteration=args.iteration, shuffle=False)
    device = scene.gaussians.get_xyz.device
    source_iteration = int(scene.loaded_iter or args.iteration or 0)
    if opt.iterations <= source_iteration:
        raise ValueError(f"--iterations ({opt.iterations}) must be > source iteration ({source_iteration}).")

    checkpoint_path = getattr(args, "segmentation_checkpoint", None)
    if not checkpoint_path:
        checkpoint_path = os.path.join(dataset.model_path, "multi_object", "final_multi_object.pth")
    checkpoint_path = os.path.abspath(checkpoint_path)
    _require_existing_file(checkpoint_path, "segmentation checkpoint")
    checkpoint_marker = load_segmentation_checkpoint(scene.gaussians, checkpoint_path)
    if checkpoint_marker is not None and source_iteration <= 0:
        source_iteration = int(checkpoint_marker)

    if dataset.disable_filter3D:
        scene.gaussians.reset_3D_filter()
    elif (
        not hasattr(scene.gaussians, "filter_3D")
        or not isinstance(scene.gaussians.filter_3D, torch.Tensor)
        or scene.gaussians.filter_3D.shape[0] != scene.gaussians.get_xyz.shape[0]
    ):
        scene.gaussians.compute_3D_filter(cameras=scene.getTrainCameras().copy())

    desk_atlas_state = _load_desk_atlas_state_from_dir(dataset.model_path, args.desk_atlas_dir, device=device)
    completed_assets = load_material_completion_assets(
        model_path=dataset.model_path,
        desk_atlas_dir=args.desk_atlas_dir,
        desk_atlas_state=desk_atlas_state,
        completed_diffuse_path=getattr(args, "completed_diffuse_path", None),
        completed_fresnel_path=getattr(args, "completed_fresnel_path", None),
        completed_reflect_path=getattr(args, "completed_reflect_path", None),
        completed_roughness_path=getattr(args, "completed_roughness_path", None),
        completed_normal_path=getattr(args, "completed_normal_path", None),
        device=device,
    )
    completed = completed_assets["completed"]
    merged_support_mask, merged_support_stats = repair_and_shrink_binary_mask(
        completed_assets["support_mask_raw"],
        float(args.desk_support_shrink_px),
        close_kernel_size=int(args.desk_support_close_kernel_size),
    )
    init_footprint_mask, init_footprint_stats = repair_and_shrink_binary_mask(
        completed_assets["support_footprint_mask"],
        float(args.desk_support_shrink_px),
        close_kernel_size=int(args.desk_support_close_kernel_size),
    )

    decoupled_state = build_decoupled_scene_state(
        gaussians=scene.gaussians,
        desk_object_id=desk_object_id,
        decouple_object_ids=decouple_object_ids,
        source_iteration=source_iteration,
    )

    decoupled_state.desk_gaussians.training_setup(opt)
    decoupled_state.background_gaussians.training_setup(opt)
    _freeze_all_gaussian_params(decoupled_state.decoupled_gaussians)

    hole_init_info = initialize_hole_gaussians_into_desk_owner(
        desk_gaussians=decoupled_state.desk_gaussians,
        desk_atlas_state=desk_atlas_state,
        completed=completed,
        hole_init_stride_px=int(args.hole_init_stride_px),
        boundary_px=int(args.view_supervision_boundary_px),
        support_footprint_mask=init_footprint_mask,
        support_mask_raw=completed_assets["support_mask_raw"],
        opt=opt,
        desk_object_id=desk_object_id,
        hole_reflection_visible=bool(args.hole_reflection_visible),
    )
    material_state = MaterialInpaintState(
        hole_start_idx=int(hole_init_info["hole_start_idx"]),
        hole_end_idx=int(hole_init_info["hole_end_idx"]),
        hole_init_uv=hole_init_info["hole_init_uv"],
        hole_init_stride_px=int(args.hole_init_stride_px),
        hole_reflection_visible=bool(args.hole_reflection_visible),
        completed_diffuse_path=str(completed_assets["completed_paths"]["diffuse"]),
        completed_fresnel_path=str(completed_assets["completed_paths"]["fresnel"]),
        completed_reflect_path=str(completed_assets["completed_paths"]["reflect"]),
        completed_roughness_path=str(completed_assets["completed_paths"]["roughness"]),
        completed_normal_path=str(completed_assets["completed_paths"]["normal"]),
    )

    _set_gaussian_trainability(decoupled_state.desk_gaussians, True)
    _set_gaussian_trainability(decoupled_state.background_gaussians, True)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device=device)
    training_state = build_material_training_state(
        scene=scene,
        dataset=dataset,
        decoupled_state=decoupled_state,
        material_state=material_state,
        desk_atlas_state=desk_atlas_state,
        completed=completed,
        merge_atlas_mask=merged_support_mask,
        pipe=pipe,
        background=background,
        dtype=decoupled_state.desk_gaussians.get_xyz.dtype,
        device=decoupled_state.desk_gaussians.get_xyz.device,
    )

    print("[MOD_GOR_INPAINT][SETUP] Restoration complete.")
    print(f"[MOD_GOR_INPAINT][SETUP] model_path={dataset.model_path}")
    print(f"[MOD_GOR_INPAINT][SETUP] source_iteration={source_iteration} target_iteration={opt.iterations}")
    print(
        "[MOD_GOR_INPAINT][SETUP] "
        f"desk_object_id={desk_object_id} decouple_object_id={decouple_object_ids}"
    )
    print(
        "[MOD_GOR_INPAINT][MERGE_MASK] "
        f"support_pixels={int(merged_support_stats['original_pixels'])} "
        f"closed_pixels={int(merged_support_stats['closed_pixels'])} "
        f"filled_pixels={int(merged_support_stats['filled_pixels'])} "
        f"shrunk_pixels={int(merged_support_stats['shrunk_pixels'])} "
        f"filled_holes={int(merged_support_stats['filled_hole_count'])} "
        f"filled_hole_area={int(merged_support_stats['filled_hole_area'])} "
        f"close_kernel={int(merged_support_stats['close_kernel_size'])} "
        f"shrink_px={float(merged_support_stats['shrink_px']):.3f}"
    )
    print(
        "[MOD_GOR_INPAINT][INIT_FOOTPRINT] "
        f"support_pixels={int(init_footprint_stats['original_pixels'])} "
        f"closed_pixels={int(init_footprint_stats['closed_pixels'])} "
        f"filled_pixels={int(init_footprint_stats['filled_pixels'])} "
        f"shrunk_pixels={int(init_footprint_stats['shrunk_pixels'])} "
        f"filled_holes={int(init_footprint_stats['filled_hole_count'])} "
        f"filled_hole_area={int(init_footprint_stats['filled_hole_area'])}"
    )
    print(
        "[MOD_GOR_INPAINT][SETUP] "
        f"hole_index_range=[{material_state.hole_start_idx}, {material_state.hole_end_idx}) "
        f"hole_reflection_visible={bool(material_state.hole_reflection_visible)} "
        f"desk_count={_gaussian_count(decoupled_state.desk_gaussians)} "
        f"background_count={_gaussian_count(decoupled_state.background_gaussians)} "
        f"decoupled_count={_gaussian_count(decoupled_state.decoupled_gaussians)}"
    )

    viewpoint_stack = None
    progress_bar = tqdm(
        range(source_iteration + 1, opt.iterations + 1),
        desc="ModGORISInpaint",
        initial=source_iteration,
        total=opt.iterations,
        miniters=10,
    )

    for iteration in progress_bar:
        decoupled_state.desk_gaussians.update_learning_rate(iteration)
        decoupled_state.background_gaussians.update_learning_rate(iteration)
        if iteration % 1000 == 0:
            decoupled_state.desk_gaussians.oneupSHdegree()
            decoupled_state.background_gaussians.oneupSHdegree()

        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
        if (iteration - 1) == args.debug_from:
            pipe.debug = True

        if decoupled_state.desk_gaussians.optimizer is not None:
            decoupled_state.desk_gaussians.optimizer.zero_grad(set_to_none=True)
        if decoupled_state.background_gaussians.optimizer is not None:
            decoupled_state.background_gaussians.optimizer.zero_grad(set_to_none=True)

        view_target_entry = _view_target_entry_to_device(
            training_state.view_target_cache[int(viewpoint_cam.uid)],
            decoupled_state.desk_gaussians.get_xyz.device,
        )
        train_composite_gaussians = materialize_train_scene_gaussians(decoupled_state)
        render_pkg = render(viewpoint_cam, train_composite_gaussians, pipe, background, dataset.kernel_size)
        total_loss, tb_dict = compute_material_inpaint_losses(
            decoupled_state=decoupled_state,
            desk_atlas_state=desk_atlas_state,
            training_state=training_state,
            material_state=material_state,
            render_pkg=render_pkg,
            view_target_entry=view_target_entry,
            args=args,
        )
        total_loss.backward()

        with torch.no_grad():
            if getattr(pipe, "save_training_vis", False) and (
                iteration % getattr(pipe, "save_training_vis_iteration", 1000) == 0
                or iteration == source_iteration + 1
            ):
                _save_training_vis(
                    model_path=dataset.model_path,
                    iteration=iteration,
                    viewpoint_cam=viewpoint_cam,
                    render_pkg=render_pkg,
                    view_target_entry=view_target_entry,
                )

            _step_owner_optimizer(decoupled_state.background_gaussians)
            _step_owner_optimizer(decoupled_state.desk_gaussians)
            _project_hole_gaussians_back_to_plane(
                decoupled_state.desk_gaussians,
                material_state,
                desk_atlas_state,
                training_state.target_scales_actual,
            )
            _set_gaussian_trainability(decoupled_state.desk_gaussians, True)
            _set_gaussian_trainability(decoupled_state.background_gaussians, True)

            progress_bar.set_postfix(
                {
                    "loss": f"{tb_dict['loss_total']:.5f}",
                    "diff": f"{tb_dict['loss_diffuse']:.5f}",
                    "fr": f"{tb_dict['loss_fresnel']:.5f}",
                    "refl": f"{tb_dict['loss_reflect']:.5f}",
                    "rough": f"{tb_dict['loss_roughness']:.5f}",
                    "norm": f"{tb_dict['loss_normal']:.5f}",
                    "desk": int(_gaussian_count(decoupled_state.desk_gaussians)),
                    "bg": int(_gaussian_count(decoupled_state.background_gaussians)),
                }
            )

            save_interval = int(args.save_interval)
            if iteration == opt.iterations or (save_interval > 0 and iteration % save_interval == 0):
                _save_full_scene_snapshot(
                    model_path=dataset.model_path,
                    iteration=iteration,
                    decoupled_state=decoupled_state,
                    material_state=material_state,
                    desk_atlas_state=desk_atlas_state,
                    support_stats=merged_support_stats,
                    init_footprint_stats=init_footprint_stats,
                )

            checkpoint_interval = int(args.checkpoint_interval)
            if iteration == opt.iterations or (checkpoint_interval > 0 and iteration % checkpoint_interval == 0):
                _save_full_scene_checkpoint(
                    model_path=dataset.model_path,
                    iteration=iteration,
                    decoupled_state=decoupled_state,
                    material_state=material_state,
                    desk_atlas_state=desk_atlas_state,
                    support_stats=merged_support_stats,
                    init_footprint_stats=init_footprint_stats,
                )

        del train_composite_gaussians

    return {
        "scene": scene,
        "decoupled_state": decoupled_state,
        "desk_atlas_state": desk_atlas_state,
        "material_state": material_state,
        "training_state": training_state,
        "completed": completed,
        "hole_init_info": hole_init_info,
    }


def _add_inpaint_args(parser: ArgumentParser) -> None:
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--segmentation_checkpoint", default=None, type=str)
    parser.add_argument("--decouple_object_id", nargs="+", default=None)
    parser.add_argument("--completed_diffuse_path", type=str, default=None)
    parser.add_argument("--completed_fresnel_path", type=str, default=None)
    parser.add_argument("--completed_reflect_path", type=str, default=None)
    parser.add_argument("--completed_roughness_path", type=str, default=None)
    parser.add_argument("--completed_normal_path", type=str, default=None)
    parser.add_argument("--desk_atlas_dir", type=str, default="desk_atlas")
    parser.add_argument("--hole_init_stride_px", type=int, default=4)
    parser.add_argument(
        "--hole_reflection_visible",
        "--reflection_visible",
        action="store_true",
        default=False,
        help=(
            "If set, newly initialized hole Gaussians participate in PBR reflection tracing. "
            "By default they remain camera-visible for inpainting but are invisible to reflection visibility tracing."
        ),
    )
    parser.add_argument("--view_supervision_boundary_px", type=int, default=16)
    parser.add_argument("--desk_support_shrink_px", type=float, default=5.0)
    parser.add_argument("--desk_support_close_kernel_size", type=int, default=0)
    parser.add_argument("--lambda_diffuse", type=float, default=1.0)
    parser.add_argument("--lambda_fresnel", type=float, default=1.0)
    parser.add_argument("--lambda_roughness", type=float, default=1.0)
    parser.add_argument("--lambda_hole_plane", type=float, default=10.0)
    parser.add_argument("--lambda_hole_anchor_uv", type=float, default=1.0)
    parser.add_argument("--lambda_hole_scale", type=float, default=0.2)
    parser.add_argument("--save_interval", type=int, default=5000)
    parser.add_argument("--checkpoint_interval", type=int, default=5000)
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--save_training_vis", action="store_true", default=False)
    parser.add_argument("--save_training_vis_iteration", type=int, default=1000)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--quiet", action="store_true", default=False)


if __name__ == "__main__":
    parser = ArgumentParser(description="mod-GOR-IS desk-atlas material inpaint training script")
    lp = ModelParams(parser, sentinel=True)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    _add_inpaint_args(parser)
    args = get_combined_args(parser)

    print("Optimizing " + args.model_path)
    safe_state(args.quiet, args.seed)
    pipe = pp.extract(args)
    pipe.save_training_vis = args.save_training_vis
    pipe.save_training_vis_iteration = args.save_training_vis_iteration

    training(
        lp.extract(args),
        op.extract(args),
        pipe,
        args,
    )
