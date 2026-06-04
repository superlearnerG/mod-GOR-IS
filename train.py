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

import os
import sys
import uuid
import json
import math
import tempfile
from argparse import ArgumentParser, Namespace
from random import randint

import torch
from tqdm import tqdm

from arguments import ModelParams, OptimizationParams, PipelineParams
from gaussian_renderer import network_gui, render
from scene import GaussianModel, Scene
from utils.general_utils import safe_state, linear_decay
from utils.image_utils import crop_using_bbox, inverse_mapping, mapping, mask_to_bbox, psnr, render_net_image
from utils.log_utils import Logger
from utils.loss_utils import (bilateral_smooth_loss, l1_loss, ssim)
from utils.mask_provider import MultiLabelMaskProvider
from utils.multi_object_postprocess import postprocess_committed_object


def _write_json(path, data):
    with open(path, "w") as file:
        json.dump(data, file, indent=2)


def _mean_or_nan(total, count):
    return total / count if count > 0 else float("nan")


def _format_metric(value):
    if value is None or not math.isfinite(value):
        return "nan"
    return f"{value:.6f}"


def _metric_to_pil(image):
    from PIL import Image

    image = torch.clamp(image.detach().cpu(), 0.0, 1.0)
    image = (image * 255).byte().permute(1, 2, 0).contiguous().numpy()
    return Image.fromarray(image)


def _calculate_fid(gt_images, pred_images):
    if len(gt_images) == 0 or len(pred_images) == 0:
        return float("nan"), "no valid image pairs"

    try:
        from pytorch_fid.fid_score import calculate_fid_given_paths
    except Exception as exc:
        return float("nan"), f"failed to import pytorch_fid: {exc}"

    with tempfile.TemporaryDirectory() as tmpdir:
        gt_dir = os.path.join(tmpdir, "gt")
        pred_dir = os.path.join(tmpdir, "pred")
        os.makedirs(gt_dir)
        os.makedirs(pred_dir)

        for idx, (gt_image, pred_image) in enumerate(zip(gt_images, pred_images)):
            gt_image.save(os.path.join(gt_dir, f"{idx:05d}.png"))
            pred_image.save(os.path.join(pred_dir, f"{idx:05d}.png"))

        try:
            value = calculate_fid_given_paths([gt_dir, pred_dir], 1, "cuda", 2048, 8)
        except Exception as exc:
            return float("nan"), f"failed to compute FID: {exc}"

    return float(value), None


def _render_eval_image(viewpoint, scene, renderFunc, renderArgs):
    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
    image = render_pkg["render"]
    reflect = render_pkg["rend_reflect"]
    alpha = render_pkg["rend_alpha"]
    rend_normal = render_pkg["rend_normal"]
    depth = render_pkg["surf_depth"]
    diffuse = render_pkg["rend_diffuse"]
    fresnel = render_pkg["rend_fresnel"]
    roughness = render_pkg["rend_roughness"]
    outputs = scene.gaussians.pbr(
        viewpoint, alpha, rend_normal, depth, diffuse, fresnel, roughness, renderArgs[1]
    )
    render_pkg.update(outputs)
    image = render_pkg["render_color"] * (1 - reflect) + reflect * image
    return torch.clamp(mapping(image), 0.0, 1.0)


def _get_masked_pair(pred_image, gt_image, viewpoint):
    mask = getattr(viewpoint, "obj_mask", None)
    if mask is None:
        return None

    mask = mask.to(gt_image.device)
    if mask.ndim == 3:
        mask = mask.squeeze(0)
    if mask.shape != gt_image.shape[-2:]:
        mask = torch.nn.functional.interpolate(
            mask[None, None].float(),
            size=gt_image.shape[-2:],
            mode="nearest",
        )[0, 0]
    mask = mask > 0
    if not mask.any():
        return None

    bbox = mask_to_bbox(mask)
    masked_gt = crop_using_bbox(gt_image, bbox).unsqueeze(0)
    masked_pred = crop_using_bbox(pred_image, bbox).unsqueeze(0)
    if masked_gt.shape[2] < 32 or masked_gt.shape[3] < 32:
        return None

    return masked_pred, masked_gt


@torch.no_grad()
def _evaluate_metric_split(name, cameras, scene, renderFunc, renderArgs, lpips_func):
    metrics = {
        "psnr": float("nan"),
        "ssim": float("nan"),
        "lpips": float("nan"),
        "m_lpips": float("nan"),
        "fid": float("nan"),
        "m_fid": float("nan"),
    }
    notes = []
    if not cameras:
        notes.append(f"{name}: no cameras")
        return metrics, notes

    psnr_total = 0.0
    ssim_total = 0.0
    lpips_total = 0.0
    count = 0
    masked_lpips_total = 0.0
    masked_count = 0
    gt_images = []
    pred_images = []
    masked_gt_images = []
    masked_pred_images = []

    for viewpoint in tqdm(cameras, desc=f"Final {name} metrics"):
        pred_image = _render_eval_image(viewpoint, scene, renderFunc, renderArgs)
        gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
        pred_batch = pred_image.unsqueeze(0)
        gt_batch = gt_image.unsqueeze(0)

        psnr_total += psnr(pred_batch, gt_batch).mean().item()
        ssim_total += ssim(pred_batch, gt_batch).mean().item()
        if lpips_func is not None:
            lpips_total += lpips_func(pred_batch, gt_batch).item()
        gt_images.append(_metric_to_pil(gt_image))
        pred_images.append(_metric_to_pil(pred_image))
        count += 1

        masked_pair = _get_masked_pair(pred_image, gt_image, viewpoint)
        if masked_pair is None:
            continue

        masked_pred, masked_gt = masked_pair
        if lpips_func is not None:
            masked_lpips_total += lpips_func(masked_pred, masked_gt).item()
        masked_gt_images.append(_metric_to_pil(masked_gt[0]))
        masked_pred_images.append(_metric_to_pil(masked_pred[0]))
        masked_count += 1

    metrics["psnr"] = _mean_or_nan(psnr_total, count)
    metrics["ssim"] = _mean_or_nan(ssim_total, count)
    if lpips_func is not None:
        metrics["lpips"] = _mean_or_nan(lpips_total, count)
        metrics["m_lpips"] = _mean_or_nan(masked_lpips_total, masked_count)
    else:
        notes.append(f"{name}: LPIPS unavailable")

    torch.cuda.empty_cache()
    metrics["fid"], fid_error = _calculate_fid(gt_images, pred_images)
    if fid_error is not None:
        notes.append(f"{name} FID: {fid_error}")

    metrics["m_fid"], masked_fid_error = _calculate_fid(masked_gt_images, masked_pred_images)
    if masked_fid_error is not None:
        notes.append(f"{name} M-FID: {masked_fid_error}")
    torch.cuda.empty_cache()

    return metrics, notes


@torch.no_grad()
def write_qualitative_comparison(scene, renderFunc, renderArgs):
    output_path = os.path.join(scene.model_path, "qualitative_comparison.txt")
    print(f"\nWriting final qualitative comparison metrics to {output_path}")

    notes = []
    try:
        from lpipsPyTorch.modules.lpips import LPIPS

        lpips_func = LPIPS("alex").cuda().eval()
    except Exception as exc:
        lpips_func = None
        notes.append(f"LPIPS unavailable: {exc}")

    splits = {
        "train": scene.getTrainCameras(),
        "test": scene.getTestCameras(),
    }
    results = {}
    for split_name, cameras in splits.items():
        split_metrics, split_notes = _evaluate_metric_split(
            split_name, cameras, scene, renderFunc, renderArgs, lpips_func
        )
        results[split_name] = split_metrics
        notes.extend(split_notes)

    with open(output_path, "w") as file:
        file.write("Qualitative Comparison Metrics\n")
        file.write(f"model_path: {scene.model_path}\n")
        file.write("masked metrics: obj_mask bounding-box crops, matching metrics.py M-* behavior\n\n")
        file.write("Split,PSNR,SSIM,LPIPS,M-LPIPS,FID,M-FID\n")
        for split_name in ("train", "test"):
            split_metrics = results[split_name]
            file.write(
                f"{split_name},"
                f"{_format_metric(split_metrics['psnr'])},"
                f"{_format_metric(split_metrics['ssim'])},"
                f"{_format_metric(split_metrics['lpips'])},"
                f"{_format_metric(split_metrics['m_lpips'])},"
                f"{_format_metric(split_metrics['fid'])},"
                f"{_format_metric(split_metrics['m_fid'])}\n"
            )

        if notes:
            file.write("\nNotes:\n")
            for note in notes:
                file.write(f"- {note}\n")

    print(f"Final qualitative comparison metrics written to {output_path}")


def save_segmentation_checkpoint(path, gaussians, marker):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save((gaussians.capture(include_mask=True), marker), path)


def segmentation_iterations(opt, train_cameras):
    if opt.segmentation_iterations > 0:
        return opt.segmentation_iterations
    return max(1, len(train_cameras) * opt.segmentation_n4views)


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


def run_single_object_stage(dataset, opt, pipe, scene, gaussians, mask_provider, label, num_iterations, logger=None, log_offset=0):
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    train_cameras = scene.getTrainCameras().copy()
    base_num = max(1, len(train_cameras) * 2)
    viewpoint_stack = None
    mask_state = False if opt.finetune_mask else True
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(num_iterations), desc=f"Object {label} segmentation")

    for iteration in range(1, num_iterations + 1):
        if opt.finetune_mask and iteration % base_num == 1:
            mask_state = not mask_state
            gaussians.add_training_state(mask_training=mask_state)

        gaussians.update_learning_rate(iteration)

        if not viewpoint_stack:
            viewpoint_stack = train_cameras.copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        bg = torch.rand_like(background) if dataset.random_background else background
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, dataset.kernel_size, include_mask=True)

        if mask_state:
            rendered_mask = render_pkg["mask"]
            mask_signals = render_pkg["mask_signals"]
            gt_mask = mask_provider.get_mask(viewpoint_cam, label)
            total_pixels = gt_mask.numel()
            loss = (-(gt_mask * rendered_mask).sum() + opt.lamb * ((1 - gt_mask) * rendered_mask).sum()) / total_pixels
        else:
            image = render_pkg["render"]
            gt_image = inverse_mapping(viewpoint_cam.original_image.cuda())
            gt_alpha = viewpoint_cam.gt_alpha_mask.cuda() if viewpoint_cam.gt_alpha_mask is not None else torch.ones_like(gt_image[0:1])
            gt_image = gt_image + (1 - gt_alpha) * bg[:, None, None]
            loss = (1.0 - opt.lambda_dssim) * l1_loss(image, gt_image) + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

        loss.backward()

        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.7f}", "Points": f"{len(gaussians.get_xyz)}"})
                progress_bar.update(10)
            if iteration == num_iterations:
                progress_bar.close()

            if logger is not None:
                logger.log("segmentation/loss", loss.item(), log_offset + iteration)
                logger.log("segmentation/points", scene.gaussians.get_xyz.shape[0], log_offset + iteration)

            if mask_state:
                gaussians.add_mask_signal_densification_stats(mask_signals)
                if opt.finetune_mask and iteration % base_num == 0:
                    prune_only = iteration >= num_iterations
                    gaussians.mask_and_split(opt.mask_signals_threshold, scene.cameras_extent, base_num, prune_only=prune_only)
                    if dataset.disable_filter3D:
                        gaussians.reset_3D_filter()
                    else:
                        gaussians.compute_3D_filter(cameras=train_cameras)

            if iteration < num_iterations:
                if mask_state:
                    gaussians.mask_optimizer.step()
                    gaussians.mask_optimizer.zero_grad(set_to_none=True)
                else:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none=True)


def run_multi_object_segmentation(dataset, opt, pipe, logger):
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=dataset.segmentation_load_iteration, shuffle=False)
    train_cameras = scene.getTrainCameras().copy()
    if dataset.disable_filter3D:
        gaussians.reset_3D_filter()
    else:
        gaussians.compute_3D_filter(cameras=train_cameras)

    mask_root = os.path.join(dataset.source_path, dataset.object_mask)
    mask_provider = MultiLabelMaskProvider(
        mask_root,
        train_cameras,
        target_labels=dataset.target_labels,
        object_order=dataset.object_order,
    )
    labels = mask_provider.get_object_labels()
    area_by_label = mask_provider.get_area_by_label()
    num_iterations = segmentation_iterations(opt, train_cameras)
    desk_object_ids = set(parse_object_id_list(getattr(dataset, "desk_object_id", "")))
    specular_object_ids = set(parse_object_id_list(getattr(dataset, "specular_object_id", "")))
    skip_postprocess_ids = desk_object_ids | specular_object_ids

    multi_object_root = os.path.join(dataset.model_path, "multi_object")
    os.makedirs(multi_object_root, exist_ok=True)
    metadata_path = os.path.join(multi_object_root, "metadata.json")
    metadata = {
        "mask_mode": dataset.mask_mode,
        "mask_root": mask_root,
        "ordered_labels": labels,
        "target_labels": labels,
        "object_order": dataset.object_order,
        "area_by_label": {str(label): int(area_by_label.get(label, 0)) for label in labels},
        "iterations_per_object": int(num_iterations),
        "load_iteration": int(scene.loaded_iter) if scene.loaded_iter is not None else None,
        "commit_threshold": float(opt.commit_threshold),
        "committed_count": {},
        "postprocess": {
            "enabled": bool(opt.enable_object_postprocess),
            "skip_desk_object_id": sorted(desk_object_ids),
            "skip_specular_object_id": sorted(specular_object_ids),
            "skipped_labels": {},
            "labels": {},
        },
    }
    _write_json(metadata_path, metadata)

    log_offset = 0
    for label in labels:
        print(f"\n[OBJECT {label}] Resetting segmentation state")
        gaussians.training_setup_segmentation(opt)
        gaussians.reset_mask()

        run_single_object_stage(
            dataset,
            opt,
            pipe,
            scene,
            gaussians,
            mask_provider,
            label,
            num_iterations,
            logger=logger,
            log_offset=log_offset,
        )

        committed = gaussians.commit_current_object(label, commit_thresh=opt.commit_threshold)
        metadata["committed_count"][str(label)] = int(committed)
        print(f"[OBJECT {label}] Committed {committed} Gaussians")

        skip_postprocess_reasons = []
        if label in desk_object_ids:
            skip_postprocess_reasons.append("desk_object_id")
        if label in specular_object_ids:
            skip_postprocess_reasons.append("specular_object_id")

        if opt.enable_object_postprocess and label in skip_postprocess_ids:
            metadata["postprocess"]["skipped_labels"][str(label)] = skip_postprocess_reasons
            print(f"[OBJECT {label}] Skipping postprocess for {', '.join(skip_postprocess_reasons)}")
        elif opt.enable_object_postprocess:
            mask_response = gaussians.get_mask.detach().squeeze().clone()
            postprocess_stats = postprocess_committed_object(
                gaussians,
                label,
                scene.cameras_extent,
                mask_response,
                opt,
            )
            metadata["postprocess"]["labels"][str(label)] = postprocess_stats
            print(
                f"[OBJECT {label}] Postprocess kept {postprocess_stats['object_count_after']} points, "
                f"unassigned {postprocess_stats['floaters_unassigned']}, "
                f"pruned {postprocess_stats['background_pruned']}"
            )
            if dataset.disable_filter3D:
                gaussians.reset_3D_filter()
            else:
                gaussians.compute_3D_filter(cameras=train_cameras)

        _write_json(metadata_path, metadata)
        log_offset += num_iterations

    final_checkpoint = os.path.join(multi_object_root, "final_multi_object.pth")
    save_segmentation_checkpoint(final_checkpoint, gaussians, len(labels))
    _write_json(metadata_path, metadata)
    if logger is not None:
        logger.close()


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, logger):
    if opt.include_mask:
        if dataset.mask_mode != "multi_label":
            raise ValueError("mod-GOR-IS segmentation currently supports --mask_mode multi_label.")
        run_multi_object_segmentation(dataset, opt, pipe, logger)
        return

    first_iter = 0
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    base_background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    trainCameras = scene.getTrainCameras().copy()
    if dataset.disable_filter3D:
        gaussians.reset_3D_filter()
    else:
        gaussians.compute_3D_filter(cameras=trainCameras)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_normal_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        background = torch.rand_like(base_background) if dataset.random_background else base_background
        kernel_size = dataset.kernel_size

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        
        render_pkg = render(viewpoint_cam, gaussians, pipe, background, kernel_size)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        
        gt_image = inverse_mapping(viewpoint_cam.original_image.cuda())
        gt_normal = viewpoint_cam.normal.cuda() if viewpoint_cam.normal is not None else None
        gt_mask = viewpoint_cam.gt_alpha_mask.cuda() if viewpoint_cam.gt_alpha_mask is not None else torch.ones_like(gt_image[0:1])
        gt_spec_mask = viewpoint_cam.spec_mask.cuda() if viewpoint_cam.spec_mask is not None else torch.zeros_like(gt_image[0:1])

        gt_image = gt_image + (1 - gt_mask) * background[:, None, None]

        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        
        # regularization
        lambda_normal = opt.lambda_normal if iteration > opt.geo_reg_from_iter else 0.0

        rend_normal  = render_pkg['rend_normal']
        surf_normal = render_pkg['surf_normal']

        normal_loss = lambda_normal * (1 - (rend_normal * surf_normal).sum(dim=0)).mean()

        ref_normal_loss = torch.tensor(0.0, device="cuda")
        smooth_loss = torch.tensor(0.0, device="cuda")
        reflect_loss = torch.tensor(0.0, device="cuda")

        if iteration > opt.blend_from_iter:
            # ref normal loss
            lambda_ref_normal = linear_decay(
                iteration, 
                opt.blend_from_iter, 
                opt.ref_normal_decay_end, 
                opt.lambda_ref_normal_init, 
                opt.lambda_ref_normal_end
            )
            if gt_normal is not None:
                ref_normal_error = (rend_normal - gt_normal).abs()
                ref_normal_loss = lambda_ref_normal * (ref_normal_error * gt_spec_mask).mean()

            alpha = render_pkg["rend_alpha"]
            depth = render_pkg["surf_depth"]
            diffuse = render_pkg["rend_diffuse"]
            fresnel = render_pkg["rend_fresnel"]
            roughness = render_pkg["rend_roughness"]
            reflect = render_pkg["rend_reflect"]

            # smooth loss
            smooth_loss = opt.lambda_smooth * (
                bilateral_smooth_loss(fresnel, gt_image, gt_spec_mask) +
                bilateral_smooth_loss(roughness, gt_image, gt_spec_mask) +
                bilateral_smooth_loss(rend_normal, gt_image, gt_spec_mask) +
                bilateral_smooth_loss(surf_normal, gt_image, gt_spec_mask)
            )

            # reflect loss
            if gt_spec_mask is not None:
                reflect_error = gt_spec_mask * reflect
                reflect_loss = opt.lambda_reflect * reflect_error.mean()

            # pbr shading
            outputs = gaussians.pbr(
                viewpoint_cam, alpha, rend_normal, depth, diffuse, fresnel, roughness, background
            )
            render_pkg.update(outputs)
            render_color = render_pkg["render_color"] * (1 - reflect) + reflect * image
            Ll1 = l1_loss(render_color, gt_image)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(render_color, gt_image))

        total_loss = loss + normal_loss + ref_normal_loss + smooth_loss + reflect_loss
        total_loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_normal_for_log = 0.4 * normal_loss.item() + 0.6 * ema_normal_for_log

            if iteration % 10 == 0:
                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                    "normal": f"{ema_normal_for_log:.{5}f}",
                    "Points": f"{len(gaussians.get_xyz)}"
                }
                progress_bar.set_postfix(loss_dict)

                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            if logger is not None:
                logger.log('train_loss_patches/main_loss', loss.item(), iteration)
                logger.log('iter_time', iter_start.elapsed_time(iter_end), iteration)
                logger.log('total_points', scene.gaussians.get_xyz.shape[0], iteration)

                logger.log('train_loss_patches/normal_loss', normal_loss, iteration)
                logger.log('train_loss_patches/ref_normal_loss', ref_normal_loss, iteration)

                logger.log('train_loss_patches/smooth_loss', smooth_loss, iteration)
                logger.log('train_loss_patches/reflect_loss', reflect_loss, iteration)

            training_report(logger, iteration, testing_iterations, scene, render, (pipe, base_background, kernel_size))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            if iteration == opt.iterations and logger is not None:
                logger.close()

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.opacity_cull, scene.cameras_extent, size_threshold)
                    if dataset.disable_filter3D:
                        gaussians.reset_3D_filter()
                    else:
                        gaussians.compute_3D_filter(cameras=trainCameras)

                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()
                
            if iteration % 100 == 0 and iteration > opt.densify_until_iter and not dataset.disable_filter3D:
                if iteration < opt.iterations - 100:
                    # don't update in the end of training
                    gaussians.compute_3D_filter(cameras=trainCameras)

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

        with torch.no_grad():        
            if network_gui.conn == None:
                network_gui.try_connect(dataset.render_items)
            while network_gui.conn != None:
                try:
                    net_image_bytes = None
                    custom_cam, do_training, keep_alive, scaling_modifer, render_mode = network_gui.receive()
                    if custom_cam != None:
                        render_pkg = render(custom_cam, gaussians, pipe, background, scaling_modifer)   
                        net_image = render_net_image(render_pkg, dataset.render_items, render_mode, custom_cam)
                        net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                    metrics_dict = {
                        "#": gaussians.get_opacity.shape[0],
                        "loss": ema_loss_for_log
                        # Add more metrics as needed
                    }
                    # Send the data
                    network_gui.send(net_image_bytes, dataset.source_path, metrics_dict)
                    if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                        break
                except Exception as e:
                    # raise e
                    network_gui.conn = None

    write_qualitative_comparison(scene, render, (pipe, base_background, dataset.kernel_size))

def prepare_output_and_logger(args) -> Logger:    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    logger = None
    try:
        logger = Logger(args)
    except Exception as e:
        print(f"Failed to create logger, no logging will be done. Error: {e}")
    return logger

@torch.no_grad()
def training_report(logger: Logger, iteration: int, testing_iterations: int, scene : Scene, renderFunc, renderArgs):
    # Report test and samples of training set
    if iteration % testing_iterations == 0:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                psnr_test = 0.0
                ssim_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = render_pkg["render"]
                    reflect = render_pkg["rend_reflect"]
                    alpha = render_pkg["rend_alpha"]
                    rend_normal = render_pkg["rend_normal"]
                    depth = render_pkg["surf_depth"]
                    diffuse = render_pkg["rend_diffuse"]
                    fresnel = render_pkg["rend_fresnel"]
                    roughness = render_pkg["rend_roughness"]
                    outputs = scene.gaussians.pbr(
                        viewpoint, alpha, rend_normal, depth, diffuse, fresnel, roughness, renderArgs[1]
                    )
                    render_pkg.update(outputs)
                    image = render_pkg["render_color"] * (1 - reflect) + reflect * image
                    image = torch.clamp(mapping(image), 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if logger and (idx < 5):
                        from utils.general_utils import colormap
                        depth = render_pkg["surf_depth"]
                        norm = depth.max()
                        depth = depth / norm
                        image_name = viewpoint.image_name
                        depth = colormap(depth.cpu().numpy()[0], cmap='turbo')
                        logger.log_image(config['name'] + f"_view_{image_name}/depth", depth, step=iteration)
                        logger.log_image(config['name'] + f"_view_{image_name}/render", image, step=iteration)
                        try:
                            diffuse = render_pkg["diffuse"]
                            fresnel = render_pkg["rend_fresnel"]
                            roughness = render_pkg["rend_roughness"]
                            s_roughness = render_pkg["screen_roughness"]
                            visibility = render_pkg["visibility"]
                            specular = torch.clamp(mapping(render_pkg["specular"]), 0.0, 1.0)
                            logger.log_image(config['name'] + f"_view_{image_name}/diffuse", diffuse, step=iteration)
                            logger.log_image(config['name'] + f"_view_{image_name}/specular", specular, step=iteration)
                            logger.log_image(config['name'] + f"_view_{image_name}/fresnel", fresnel, step=iteration)
                            logger.log_image(config['name'] + f"_view_{image_name}/roughness", roughness, step=iteration)
                            logger.log_image(config['name'] + f"_view_{image_name}/s_roughness", s_roughness, step=iteration)
                            logger.log_image(config['name'] + f"_view_{image_name}/reflect", reflect, step=iteration)
                            logger.log_image(config['name'] + f"_view_{image_name}/visibility", visibility, step=iteration)
                        except:
                            pass

                        try:
                            rend_alpha = render_pkg['rend_alpha']
                            rend_normal = render_pkg["rend_normal"] * 0.5 + 0.5
                            surf_normal = render_pkg["surf_normal"] * 0.5 + 0.5
                            logger.log_image(config['name'] + f"_view_{image_name}/rend_normal", rend_normal, step=iteration)
                            logger.log_image(config['name'] + f"_view_{image_name}/surf_normal", surf_normal, step=iteration)
                            logger.log_image(config['name'] + f"_view_{image_name}/rend_alpha", rend_alpha, step=iteration)
                        except:
                            pass

                        if iteration == testing_iterations:
                            logger.log_image(config['name'] + f"_view_{image_name}/ground_truth", gt_image, step=iteration)

                    psnr_test += psnr(image, gt_image).mean().double()
                    ssim_test += ssim(image, gt_image).mean().double()

                psnr_test /= len(config['cameras'])
                ssim_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: PSNR {}, SSIM {}".format(iteration, config['name'], psnr_test, ssim_test))
                if logger:
                    logger.log(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    logger.log(config['name'] + '/loss_viewpoint - ssim', ssim_test, iteration)

        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", type=int, default=2000)
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--logger", type=str, default="tensorboard", choices=["tensorboard", "wandb", "both", "none"])
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    logger = prepare_output_and_logger(args)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet, args.seed)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, logger)

    # All done
    print("\nTraining complete.")
