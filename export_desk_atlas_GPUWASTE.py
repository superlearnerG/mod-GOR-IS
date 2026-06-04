import json
import os
import random
from argparse import ArgumentParser, Namespace

import numpy as np
import torch
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, get_combined_args
from desk_atlas_GPUWASTE import (
    available_object_ids,
    build_desk_atlas_state,
    export_desk_atlas_modalities,
    infer_support_object_ids,
    validate_desk_object_id,
)
from gaussian_renderer import render
from scene import GaussianModel, Scene
from utils.mask_provider import MultiLabelMaskProvider


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


def parse_single_desk_object_id(raw_value) -> int:
    object_ids = parse_object_id_list(raw_value)
    if len(object_ids) == 0:
        raise ValueError("--desk_object_id is required and must contain exactly one id.")
    if len(object_ids) > 1:
        raise ValueError(
            f"--desk_object_id supports exactly one id for atlas export, got {object_ids}."
        )
    return int(object_ids[0])


def _add_desk_atlas_args(parser: ArgumentParser) -> None:
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--segmentation_checkpoint", default=None, type=str)
    parser.add_argument("--support_object_ids", nargs="+", default=None)
    parser.add_argument("--output_subdir", type=str, default="desk_atlas")
    parser.add_argument("--desk_atlas_long_side", type=int, default=1024)
    parser.add_argument("--desk_atlas_size_multiple", type=int, default=8)
    parser.add_argument("--footprint_bottom_quantile", type=float, default=1.0)
    parser.add_argument("--footprint_min_opacity", type=float, default=0.05)
    parser.add_argument("--footprint_uv_outlier_iqr", type=float, default=1.5)
    parser.add_argument("--ccm_contact_kernel", type=int, default=15)
    parser.add_argument("--ccm_max_mask_samples", type=int, default=80000)
    parser.add_argument("--ccm_plane_ransac_iters", type=int, default=256)
    parser.add_argument("--ccm_plane_ransac_thresh", type=float, default=0.01)
    parser.add_argument("--desk_plane_down_offset", type=float, default=0.0)
    parser.add_argument("--desk_pack_known_strong_quantile", type=float, default=0.0)
    parser.add_argument("--desk_pack_hole_observed_dilate_kernel", type=int, default=5) # 这个参数用于抑制 M_hole 在 desk边缘一圈出现的小floater，如果floater在默认值还存在的话，尝试着将它调大
    parser.add_argument("--seed", default=0, type=int)


def _extract_desk_atlas_options(args) -> Namespace:
    return Namespace(
        desk_atlas_long_side=int(args.desk_atlas_long_side),
        desk_atlas_size_multiple=int(args.desk_atlas_size_multiple),
        footprint_bottom_quantile=float(args.footprint_bottom_quantile),
        footprint_min_opacity=float(args.footprint_min_opacity),
        footprint_uv_outlier_iqr=float(args.footprint_uv_outlier_iqr),
        ccm_contact_kernel=int(args.ccm_contact_kernel),
        ccm_max_mask_samples=int(args.ccm_max_mask_samples),
        ccm_plane_ransac_iters=int(args.ccm_plane_ransac_iters),
        ccm_plane_ransac_thresh=float(args.ccm_plane_ransac_thresh),
        desk_plane_down_offset=float(args.desk_plane_down_offset),
        desk_pack_known_strong_quantile=float(args.desk_pack_known_strong_quantile),
        desk_pack_hole_observed_dilate_kernel=int(args.desk_pack_hole_observed_dilate_kernel),
    )


def load_segmentation_checkpoint(gaussians, checkpoint_path):
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
        raise ValueError(
            f"Segmentation checkpoint {checkpoint_path} does not contain the multi-object state dict."
        )
    if "object_id" not in model_state:
        raise ValueError(f"Segmentation checkpoint {checkpoint_path} does not contain object_id.")

    gaussians._restore_from_state_dict(model_state)
    gaussians._ensure_object_state()
    return marker


def _ensure_filter_3d(dataset, scene) -> None:
    train_cameras = scene.getTrainCameras().copy()
    if dataset.disable_filter3D:
        scene.gaussians.reset_3D_filter()
        return

    filter_3d = getattr(scene.gaussians, "filter_3D", None)
    if (
        not isinstance(filter_3d, torch.Tensor)
        or filter_3d.shape[0] != scene.gaussians.get_xyz.shape[0]
    ):
        scene.gaussians.compute_3D_filter(cameras=train_cameras)


def _camera_source_key(camera) -> str:
    image_name = getattr(camera, "image_name", None)
    if image_name is not None:
        return str(image_name)
    return str(getattr(camera, "uid", id(camera)))


@torch.no_grad()
def build_render_source_maps(scene, gaussians, pipe, dataset):
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    source_maps = {}

    for cam in tqdm(scene.getTrainCameras(), desc="Render atlas intrinsic sources"):
        render_pkg = render(cam, gaussians, pipe, background, dataset.kernel_size)
        source_maps[_camera_source_key(cam)] = {
            "rgb": cam.original_image.detach(),
            "normal": render_pkg["rend_normal"].detach(),
            "diffuse": render_pkg["rend_diffuse"].detach(),
            "roughness": render_pkg["rend_roughness"].detach(),
            "fresnel": render_pkg["rend_fresnel"].detach(),
            "reflect": render_pkg["rend_reflect"].detach(),
        }

    return source_maps


def export_desk_atlas(dataset, pipe, args) -> None:
    desk_object_id = parse_single_desk_object_id(getattr(args, "desk_object_id", None))
    opt = _extract_desk_atlas_options(args)
    if not getattr(dataset, "object_mask", None):
        dataset.object_mask = "object_mask"
    if not getattr(dataset, "object_order", None):
        dataset.object_order = "label_asc"
    if not getattr(dataset, "images", None):
        dataset.images = "images"
    if getattr(dataset, "sh_degree", None) is None:
        dataset.sh_degree = 3
    if getattr(dataset, "resolution", None) is None:
        dataset.resolution = -1
    if getattr(dataset, "data_device", None) is None:
        dataset.data_device = "cuda"
    if getattr(dataset, "eval", None) is None:
        dataset.eval = False
    if getattr(dataset, "kernel_size", None) is None:
        dataset.kernel_size = 0.0
    if getattr(dataset, "white_background", None) is None:
        dataset.white_background = False
    if getattr(dataset, "disable_filter3D", None) is None:
        dataset.disable_filter3D = False

    if not os.path.isdir(dataset.model_path):
        raise FileNotFoundError(f"Model path does not exist: {dataset.model_path}")

    mask_root = os.path.join(dataset.source_path, dataset.object_mask)
    if not os.path.isdir(mask_root):
        raise FileNotFoundError(f"Mask root does not exist: {mask_root}")

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)

    checkpoint_path = getattr(args, "segmentation_checkpoint", None)
    if not checkpoint_path:
        checkpoint_path = os.path.join(dataset.model_path, "multi_object", "final_multi_object.pth")
    checkpoint_path = os.path.abspath(checkpoint_path)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Segmentation checkpoint not found: {checkpoint_path}. "
            "Run multi-object segmentation first, or pass --segmentation_checkpoint explicitly."
        )

    checkpoint_marker = load_segmentation_checkpoint(scene.gaussians, checkpoint_path)
    _ensure_filter_3d(dataset, scene)

    desk_object_id = validate_desk_object_id(scene.gaussians, desk_object_id)
    explicit_support_ids = parse_object_id_list(getattr(args, "support_object_ids", None))
    support_object_ids = infer_support_object_ids(
        scene.gaussians,
        desk_object_id=desk_object_id,
        explicit_labels=explicit_support_ids if explicit_support_ids else None,
    )

    mask_provider = MultiLabelMaskProvider(
        mask_root,
        scene.getTrainCameras(),
        target_labels="",
        object_order=dataset.object_order,
    )
    visible_mask_labels = set(mask_provider.get_object_labels())
    if desk_object_id not in visible_mask_labels:
        print(
            f"[DeskAtlas] Warning: desk_object_id={desk_object_id} is not present "
            f"in 2D masks under '{mask_root}'."
        )

    source_maps_by_camera = build_render_source_maps(scene, scene.gaussians, pipe, dataset)
    build_iteration = int(scene.loaded_iter or checkpoint_marker or 0)
    desk_atlas_state = build_desk_atlas_state(
        scene=scene,
        gaussians=scene.gaussians,
        mask_provider=mask_provider,
        opt=opt,
        desk_object_id=desk_object_id,
        support_object_ids=support_object_ids,
        iteration=build_iteration,
        source_maps_by_camera=source_maps_by_camera,
    )
    export_desk_atlas_modalities(
        scene=scene,
        mask_provider=mask_provider,
        model_path=dataset.model_path,
        desk_atlas_state=desk_atlas_state,
        desk_object_id=desk_object_id,
        opt=opt,
        output_subdir=args.output_subdir,
        source_maps_by_camera=source_maps_by_camera,
    )

    output_dir = args.output_subdir if os.path.isabs(args.output_subdir) else os.path.join(dataset.model_path, args.output_subdir)
    print(
        "[DeskAtlas] Export complete "
        f"segmentation_checkpoint='{checkpoint_path}' "
        f"desk_object_id={desk_object_id} "
        f"support_object_ids={support_object_ids} "
        f"available_object_ids={available_object_ids(scene.gaussians)} "
        f"output_dir='{output_dir}'"
    )


def main() -> None:
    parser = ArgumentParser(description="Export desk atlas artifacts from a trained mod-GOR-IS scene.")
    lp = ModelParams(parser, sentinel=True)
    pp = PipelineParams(parser)
    _add_desk_atlas_args(parser)
    args = get_combined_args(parser)

    if not getattr(args, "source_path", None):
        parser.error("--source_path is required")
    if not getattr(args, "model_path", None):
        parser.error("--model_path is required")

    random_seed = int(getattr(args, "seed", 0) or 0)
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.manual_seed(random_seed)

    export_desk_atlas(lp.extract(args), pp.extract(args), args)


if __name__ == "__main__":
    main()
