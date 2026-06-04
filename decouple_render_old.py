#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# Modifications Copyright (C) 2026, [Yonghao Zhao / Nankai University]
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE file.
#
# For inquiries contact:
# - Original: george.drettakis@inria.fr
# - Modified version: applezyh@outlook.com
#

import json
import os
from argparse import ArgumentParser

import numpy as np
import torch
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import render
from scene import GaussianModel, Scene
from utils.general_utils import safe_state
from utils.image_utils import mapping
from utils.render_utils import save_img_f32, save_img_u8


def parse_object_id_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        object_ids = []
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


def safe_image_stem(viewpoint_cam, fallback_idx):
    image_name = getattr(viewpoint_cam, "image_name", None)
    if not image_name:
        return f"{fallback_idx:05d}"
    return os.path.splitext(os.path.basename(image_name))[0]


def to_numpy_image(tensor):
    array = tensor.detach().float().cpu().numpy()
    if array.ndim == 3 and array.shape[0] in (1, 3):
        array = np.transpose(array, (1, 2, 0))
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    return array


def save_png(path, tensor):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_img_u8(to_numpy_image(torch.clamp(tensor, 0.0, 1.0)), path)


def save_tiff(path, tensor):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    array = tensor.detach().float().cpu().numpy()
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    save_img_f32(array, path)


def clone_filtered_gaussians(source, gaussian_filter):
    gaussian_filter = gaussian_filter.detach().bool()
    if int(gaussian_filter.sum().item()) == 0:
        return None

    subset = GaussianModel(source.max_sh_degree)
    subset.active_sh_degree = source.active_sh_degree
    subset.spatial_lr_scale = source.spatial_lr_scale
    subset.use_screen_filter = source.use_screen_filter

    subset._xyz = source._xyz[gaussian_filter].detach()
    subset._features_dc = source._features_dc[gaussian_filter].detach()
    subset._features_rest = source._features_rest[gaussian_filter].detach()
    subset._scaling = source._scaling[gaussian_filter].detach()
    subset._rotation = source._rotation[gaussian_filter].detach()
    subset._opacity = source._opacity[gaussian_filter].detach()
    subset._diffuse = source._diffuse[gaussian_filter].detach()
    subset._fresnel = source._fresnel[gaussian_filter].detach()
    subset._roughness = source._roughness[gaussian_filter].detach()
    subset._reflect = source._reflect[gaussian_filter].detach()
    subset.filter_3D = source.filter_3D[gaussian_filter].detach()
    subset.max_radii2D = torch.zeros((subset.get_xyz.shape[0]), device=subset.get_xyz.device)
    subset.denom = torch.zeros((subset.get_xyz.shape[0], 1), device=subset.get_xyz.device)
    subset.object_id = source.get_object_id[gaussian_filter].detach()
    subset.object_score = source.get_object_score[gaussian_filter].detach()
    if hasattr(source, "get_reflection_visible"):
        subset.reflection_visible = source.get_reflection_visible[gaussian_filter].detach().clone()
    else:
        subset.reflection_visible = torch.ones((subset.get_xyz.shape[0],), dtype=torch.bool, device=subset.get_xyz.device)
    if hasattr(source, "get_specular_id"):
        subset.specular_id = source.get_specular_id[gaussian_filter].detach()
    else:
        subset.specular_id = torch.zeros((subset.get_xyz.shape[0],), dtype=torch.int32, device=subset.get_xyz.device)

    if source._mask is not None and source._mask.shape[0] == source.get_xyz.shape[0]:
        subset._mask = source._mask[gaussian_filter].detach()
    else:
        subset._mask = None

    subset.env_light = source.env_light
    subset.roughness_net = source.roughness_net
    return subset


def extract_segmentation_state(checkpoint, checkpoint_path):
    if isinstance(checkpoint, (tuple, list)):
        model_state = checkpoint[0]
    elif isinstance(checkpoint, dict) and "gaussians" in checkpoint:
        model_state = checkpoint["gaussians"]
    else:
        model_state = checkpoint

    if not isinstance(model_state, dict):
        raise ValueError(
            f"Segmentation checkpoint {checkpoint_path} does not contain the multi-object state dict."
        )
    if "object_id" not in model_state:
        raise ValueError(f"Segmentation checkpoint {checkpoint_path} does not contain object_id.")
    return model_state


def _flatten_segmentation_tensor(tensor, name, num_points, device, dtype):
    if tensor is None:
        return None
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.as_tensor(tensor)
    tensor = tensor.detach().to(device=device, dtype=dtype)
    if tensor.numel() != num_points:
        raise ValueError(f"Segmentation field {name} has {tensor.numel()} values, expected {num_points}.")
    return tensor.reshape(num_points)


def load_segmentation_fields(gaussians, checkpoint_path):
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        return False

    checkpoint = torch.load(checkpoint_path)
    model_state = extract_segmentation_state(checkpoint, checkpoint_path)

    num_points = gaussians.get_xyz.shape[0]
    device = gaussians.get_xyz.device
    gaussians.object_id = _flatten_segmentation_tensor(
        model_state["object_id"], "object_id", num_points, device, torch.int32
    )
    object_score = _flatten_segmentation_tensor(
        model_state.get("object_score"), "object_score", num_points, device, gaussians.get_xyz.dtype
    )
    if object_score is None:
        object_score = torch.zeros((num_points,), dtype=gaussians.get_xyz.dtype, device=device)
    gaussians.object_score = object_score
    specular_id = _flatten_segmentation_tensor(
        model_state.get("specular_id"), "specular_id", num_points, device, torch.int32
    )
    if specular_id is None:
        specular_id = torch.zeros((num_points,), dtype=torch.int32, device=device)
    gaussians.specular_id = specular_id
    mask = _flatten_segmentation_tensor(
        model_state.get("mask"), "mask", num_points, device, gaussians.get_xyz.dtype
    )
    gaussians._mask = mask
    gaussians._ensure_object_state()
    if hasattr(gaussians, "_ensure_specular_state"):
        gaussians._ensure_specular_state()
    return True


def gaussians_have_segmentation_fields(gaussians):
    num_points = gaussians.get_xyz.shape[0]
    object_id = getattr(gaussians, "object_id", None)
    object_score = getattr(gaussians, "object_score", None)
    mask = getattr(gaussians, "_mask", None)

    if isinstance(mask, torch.Tensor) and mask.numel() == num_points:
        return True
    if isinstance(object_id, torch.Tensor) and object_id.numel() == num_points:
        if bool(torch.any(object_id.detach() != 0).item()):
            return True
    if isinstance(object_score, torch.Tensor) and object_score.numel() == num_points:
        if bool(torch.any(object_score.detach() > 0).item()):
            return True
    return False


def _match_ids(labels, ids):
    matched = torch.zeros_like(labels, dtype=torch.bool)
    for label_id in ids:
        matched |= labels == int(label_id)
    return matched


def get_specular_filter(gaussians, specular_ids, specular_object_ids):
    num_points = gaussians.get_xyz.shape[0]
    device = gaussians.get_xyz.device
    specular_ids = sorted({int(value) for value in specular_ids})
    specular_object_ids = sorted({int(value) for value in specular_object_ids})

    specular_id = getattr(gaussians, "specular_id", None)
    if isinstance(specular_id, torch.Tensor) and specular_id.numel() == num_points:
        specular_id = specular_id.detach().to(device=device, dtype=torch.int32).reshape(num_points)
        if bool(torch.any(specular_id != 0).item()):
            if specular_ids:
                return _match_ids(specular_id, specular_ids), "specular_id"
            return specular_id != 0, "specular_id_nonzero"

    if specular_object_ids:
        return _match_ids(gaussians.get_object_id, specular_object_ids), "object_id_from_specular_object_id"
    if specular_ids:
        return _match_ids(gaussians.get_object_id, specular_ids), "object_id_from_specular_id"

    return torch.zeros((num_points,), dtype=torch.bool, device=device), "none"


def empty_pbr_package(render_pkg):
    zeros_rgb = torch.zeros_like(render_pkg["render"])
    zeros_scalar = torch.zeros_like(render_pkg["rend_alpha"])
    return {
        "render_color": render_pkg["render"],
        "diffuse": zeros_rgb,
        "specular": zeros_rgb,
        "visibility": zeros_scalar,
        "screen_roughness": zeros_scalar,
        "specular_mask": zeros_scalar,
    }


def pbr_blend_with_3dgs(render_pkg, pbr_pkg):
    reflect = render_pkg["rend_reflect"]
    return pbr_pkg["render_color"] * (1 - reflect) + reflect * render_pkg["render"]


def specular_aware_pbr(gaussians, viewpoint_cam, render_pkg, pipe, background, kernel_size, specular_filter):
    specular_filter = specular_filter.detach().bool()
    specular_count = int(specular_filter.sum().item())
    num_points = int(specular_filter.numel())

    if specular_count == 0:
        return empty_pbr_package(render_pkg)

    pbr_pkg = gaussians.pbr(
        viewpoint_cam,
        render_pkg["rend_alpha"],
        render_pkg["rend_normal"],
        render_pkg["surf_depth"],
        render_pkg["rend_diffuse"],
        render_pkg["rend_fresnel"],
        render_pkg["rend_roughness"],
        background,
    )
    full_pbr_render = pbr_blend_with_3dgs(render_pkg, pbr_pkg)

    if specular_count == num_points:
        pbr_pkg["render_color"] = full_pbr_render
        pbr_pkg["specular_mask"] = render_pkg["rend_alpha"].detach().clamp(0.0, 1.0)
        return pbr_pkg

    specular_mask_values = specular_filter.to(device=gaussians.get_xyz.device, dtype=render_pkg["render"].dtype)
    specular_mask_pkg = render(
        viewpoint_cam,
        gaussians,
        pipe,
        background,
        kernel_size,
        include_mask=True,
        mask_override=specular_mask_values,
    )
    specular_mask = specular_mask_pkg["mask"].detach().clamp(0.0, 1.0)
    pbr_pkg["render_color"] = full_pbr_render * specular_mask + render_pkg["render"] * (1 - specular_mask)
    pbr_pkg["diffuse"] = pbr_pkg["diffuse"] * specular_mask
    pbr_pkg["specular"] = pbr_pkg["specular"] * specular_mask
    pbr_pkg["visibility"] = pbr_pkg["visibility"] * specular_mask
    pbr_pkg["screen_roughness"] = pbr_pkg["screen_roughness"] * specular_mask
    pbr_pkg["specular_mask"] = specular_mask
    return pbr_pkg


def get_component_filters(object_ids, object_labels):
    components = []
    selected_filter = torch.zeros_like(object_labels, dtype=torch.bool)

    for object_id in object_ids:
        object_filter = object_labels == int(object_id)
        count = int(object_filter.sum().item())
        if count == 0:
            raise ValueError(f"--decouple_object_id contains {object_id}, but no Gaussian has this object_id.")
        components.append((f"object_{object_id}", object_filter, int(object_id), count))
        selected_filter |= object_filter

    background_filter = ~selected_filter
    components.append(("background", background_filter, "background", int(background_filter.sum().item())))
    return components


def save_render_outputs(root, stem, render_pkg, pbr_pkg):
    save_png(os.path.join(root, "render", "3dgs", f"{stem}.png"), mapping(render_pkg["render"]))
    save_png(os.path.join(root, "render", "pbr", f"{stem}.png"), mapping(pbr_pkg["render_color"]))

    intrinsic_root = os.path.join(root, "intrinsic")
    save_png(os.path.join(intrinsic_root, "diffuse", f"{stem}.png"), render_pkg["rend_diffuse"])
    save_png(os.path.join(intrinsic_root, "fresnel", f"{stem}.png"), render_pkg["rend_fresnel"])
    save_png(os.path.join(intrinsic_root, "roughness", f"{stem}.png"), render_pkg["rend_roughness"])
    save_png(os.path.join(intrinsic_root, "reflect", f"{stem}.png"), render_pkg["rend_reflect"])
    save_png(os.path.join(intrinsic_root, "alpha", f"{stem}.png"), render_pkg["rend_alpha"])
    save_png(os.path.join(intrinsic_root, "normal", f"{stem}.png"), render_pkg["rend_normal"] * 0.5 + 0.5)
    save_png(os.path.join(intrinsic_root, "world_normal", f"{stem}.png"), render_pkg["rend_normal_w"] * 0.5 + 0.5)
    save_png(os.path.join(intrinsic_root, "depth_normal", f"{stem}.png"), render_pkg["surf_normal"] * 0.5 + 0.5)
    save_tiff(os.path.join(intrinsic_root, "depth", f"{stem}.tiff"), render_pkg["surf_depth"])

    pbr_root = os.path.join(root, "pbr_components")
    save_png(os.path.join(pbr_root, "diffuse", f"{stem}.png"), pbr_pkg["diffuse"])
    save_png(os.path.join(pbr_root, "specular", f"{stem}.png"), mapping(pbr_pkg["specular"]))
    save_png(os.path.join(pbr_root, "visibility", f"{stem}.png"), pbr_pkg["visibility"])
    save_png(os.path.join(pbr_root, "screen_roughness", f"{stem}.png"), pbr_pkg["screen_roughness"])
    if "specular_mask" in pbr_pkg:
        save_png(os.path.join(pbr_root, "specular_mask", f"{stem}.png"), pbr_pkg["specular_mask"])


@torch.no_grad()
def render_component(
    component_name,
    gaussians,
    cameras,
    pipe,
    background,
    kernel_size,
    output_root,
    max_views=-1,
    specular_ids=None,
    specular_object_ids=None,
):
    specular_filter, specular_source = get_specular_filter(
        gaussians,
        specular_ids or [],
        specular_object_ids or [],
    )
    specular_count = int(specular_filter.sum().item())
    print(
        f"[{component_name}] specular_source={specular_source} "
        f"specular_count={specular_count}/{gaussians.get_xyz.shape[0]}"
    )
    num_views = len(cameras) if max_views is None or max_views < 0 else min(len(cameras), max_views)
    for idx, viewpoint_cam in tqdm(
        enumerate(cameras[:num_views]),
        total=num_views,
        desc=f"Render {component_name}",
    ):
        stem = safe_image_stem(viewpoint_cam, idx)
        render_pkg = render(viewpoint_cam, gaussians, pipe, background, kernel_size)
        pbr_pkg = specular_aware_pbr(
            gaussians,
            viewpoint_cam,
            render_pkg,
            pipe,
            background,
            kernel_size,
            specular_filter,
        )
        save_render_outputs(output_root, stem, render_pkg, pbr_pkg)
    return {
        "specular_source": specular_source,
        "specular_count": specular_count,
    }


def write_metadata(path, metadata):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as file:
        json.dump(metadata, file, indent=2)


def decouple_render(dataset, pipe, args):
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)

    checkpoint_path = getattr(args, "segmentation_checkpoint", None)
    loaded_checkpoint = False
    segmentation_source = "loaded_iteration"
    if checkpoint_path:
        checkpoint_path = os.path.abspath(checkpoint_path)
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Segmentation checkpoint not found: {checkpoint_path}.")
        loaded_checkpoint = load_segmentation_fields(scene.gaussians, checkpoint_path)
        segmentation_source = "explicit_checkpoint"
    elif gaussians_have_segmentation_fields(scene.gaussians):
        print("Using segmentation fields from loaded point cloud.")
    else:
        fallback_checkpoint_path = os.path.join(dataset.model_path, "multi_object", "final_multi_object.pth")
        if os.path.exists(fallback_checkpoint_path):
            checkpoint_path = fallback_checkpoint_path
            loaded_checkpoint = load_segmentation_fields(scene.gaussians, checkpoint_path)
            segmentation_source = "fallback_checkpoint"
        else:
            checkpoint_path = None
            segmentation_source = "none"
            scene.gaussians._ensure_object_state()
            if hasattr(scene.gaussians, "_ensure_specular_state"):
                scene.gaussians._ensure_specular_state()

    train_cameras = scene.getTrainCameras().copy()
    if dataset.disable_filter3D:
        scene.gaussians.reset_3D_filter()
    elif (
        not hasattr(scene.gaussians, "filter_3D")
        or not isinstance(scene.gaussians.filter_3D, torch.Tensor)
        or scene.gaussians.filter_3D.shape[0] != scene.gaussians.get_xyz.shape[0]
    ):
        scene.gaussians.compute_3D_filter(cameras=train_cameras)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    render_mode = getattr(args, "render_mode", "decouple")
    output_root = getattr(args, "output_dir", None) or os.path.join(dataset.model_path, render_mode)
    render_isolated = bool(getattr(args, "render_isolated", False))
    specular_ids = parse_object_id_list(getattr(args, "specular_id", None))
    specular_object_ids = parse_object_id_list(getattr(dataset, "specular_object_id", ""))
    split_configs = []
    if not args.skip_train:
        split_configs.append(("train", scene.getTrainCameras()))
    if not args.skip_test:
        split_configs.append(("test", scene.getTestCameras()))
    if not split_configs:
        raise ValueError("Both --skip_train and --skip_test were set; nothing to render.")

    metadata = {
        "model_path": dataset.model_path,
        "load_iteration": int(scene.loaded_iter) if scene.loaded_iter is not None else None,
        "segmentation_checkpoint": checkpoint_path if loaded_checkpoint else None,
        "segmentation_source": segmentation_source,
        "render_mode": render_mode,
        "render_isolated": render_isolated,
        "decouple_object_id": [],
        "specular_id": specular_ids,
        "specular_object_id": specular_object_ids,
        "components": {
            "original": {
                "label": "original",
                "gaussian_count": int(scene.gaussians.get_xyz.shape[0]),
            },
        },
        "splits": {},
    }

    print(f"Writing decoupled renders to {output_root}")
    for split_name, cameras in split_configs:
        if len(cameras) == 0:
            print(f"[original/{split_name}] No cameras; skipping.")
            continue
        if split_name not in metadata["splits"]:
            metadata["splits"][split_name] = [safe_image_stem(cam, idx) for idx, cam in enumerate(cameras)]

        render_stats = render_component(
            component_name=f"{split_name}/original",
            gaussians=scene.gaussians,
            cameras=cameras,
            pipe=pipe,
            background=background,
            kernel_size=dataset.kernel_size,
            output_root=os.path.join(output_root, split_name, "original"),
            max_views=args.max_views,
            specular_ids=specular_ids,
            specular_object_ids=specular_object_ids,
        )
        metadata["components"]["original"].update(render_stats)
        torch.cuda.empty_cache()

    if not render_isolated:
        write_metadata(os.path.join(output_root, "metadata.json"), metadata)
        return

    if not gaussians_have_segmentation_fields(scene.gaussians):
        raise ValueError(
            "No segmentation fields are available for isolated rendering. "
            "Load an inpainted iteration saved with mask/object_id/object_score, "
            "or pass --segmentation_checkpoint explicitly."
        )

    object_ids = sorted(set(parse_object_id_list(args.decouple_object_id)))
    if not object_ids:
        raise ValueError("Please pass at least one object id with --decouple_object_id when using --render_isolated.")

    metadata["decouple_object_id"] = object_ids
    object_labels = scene.gaussians.get_object_id
    components = get_component_filters(object_ids, object_labels)

    for component_name, component_filter, label, count in components:
        if count == 0:
            print(f"[{component_name}] Empty component; skipping.")
            continue

        metadata["components"][component_name] = {
            "label": label,
            "gaussian_count": count,
        }
        component_gaussians = clone_filtered_gaussians(scene.gaussians, component_filter)

        for split_name, cameras in split_configs:
            if len(cameras) == 0:
                print(f"[{component_name}/{split_name}] No cameras; skipping.")
                continue
            if split_name not in metadata["splits"]:
                metadata["splits"][split_name] = [safe_image_stem(cam, idx) for idx, cam in enumerate(cameras)]

            component_output_root = os.path.join(output_root, split_name, component_name)
            render_stats = render_component(
                component_name=f"{split_name}/{component_name}",
                gaussians=component_gaussians,
                cameras=cameras,
                pipe=pipe,
                background=background,
                kernel_size=dataset.kernel_size,
                output_root=component_output_root,
                max_views=args.max_views,
                specular_ids=specular_ids,
                specular_object_ids=specular_object_ids,
            )
            metadata["components"][component_name].update(render_stats)
            torch.cuda.empty_cache()

        del component_gaussians
        torch.cuda.empty_cache()

    write_metadata(os.path.join(output_root, "metadata.json"), metadata)


if __name__ == "__main__":
    parser = ArgumentParser(description="Render decoupled intrinsic and PBR images for selected object ids.")
    lp = ModelParams(parser, sentinel=True)
    pp = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument(
        "--decouple_object_id",
        nargs="+",
        default=[],
        help="Object ids to render independently. Supports: 1 2, 1,2, or [1,2].",
    )
    parser.add_argument(
        "--specular_id",
        nargs="+",
        default=[],
        help=(
            "Values in the per-Gaussian specular_id field to render with PBR. "
            "If omitted, non-zero specular_id values are used when present; otherwise "
            "--specular_object_id from cfg_args is matched against object_id."
        ),
    )
    parser.add_argument("--segmentation_checkpoint", default=None, type=str)
    parser.add_argument("--render_mode", choices=["decouple", "decouple+inpaint"], default="decouple")
    parser.add_argument("--output_dir", default=None, type=str)
    parser.add_argument("--skip_train", action="store_true", default=False)
    parser.add_argument("--skip_test", action="store_true", default=False)
    parser.add_argument("--max_views", default=-1, type=int)
    parser.add_argument("--render_isolated", action="store_true", default=False)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--seed", default=0, type=int)
    args = get_combined_args(parser)

    safe_state(args.quiet, args.seed)
    decouple_render(lp.extract(args), pp.extract(args), args)

# 目前读取checkpoint的逻辑是：
# 1. 先用 --iteration 加载 point_cloud/iteration_xxx
# 2. 如果用户显式传了 --segmentation_checkpoint：
#    只从里面加载 mask/object_id/object_score
# 3. 如果用户没有传 --segmentation_checkpoint：
#    如果当前 point_cloud.ply 已经带 object_id，则直接用
#    否则才 fallback 到 multi_object/final_multi_object.pth
