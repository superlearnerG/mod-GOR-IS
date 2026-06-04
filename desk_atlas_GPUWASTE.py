import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import save_image

from utils.projection_utils import project_mask_pixels_to_plane_samples, project_xyz_to_plane_uv

try:
    from scipy.ndimage import distance_transform_edt
except Exception:
    distance_transform_edt = None

try:
    from scipy.sparse import lil_matrix
    from scipy.sparse.linalg import spsolve
except Exception:
    lil_matrix = None
    spsolve = None

try:
    import cv2
except Exception:
    cv2 = None


@dataclass
class PlaneDefinition:
    normal: torch.Tensor
    d: torch.Tensor
    origin: torch.Tensor
    e1: torch.Tensor
    e2: torch.Tensor


@dataclass
class FootprintMaps:
    uv_points: torch.Tensor


@dataclass
class DeskAtlasState:
    plane: PlaneDefinition
    uv_bbox: Tuple[float, float, float, float]
    atlas_hw: Tuple[int, int]
    support_mask: torch.Tensor
    support_visible_mask: torch.Tensor
    support_footprint_mask: torch.Tensor
    observed_mask: torch.Tensor
    hole_mask: torch.Tensor
    confidence: torch.Tensor
    rgb_observed: torch.Tensor
    build_iteration: int


DESK_ATLAS_MODALITIES = ("rgb", "normal", "diffuse", "roughness", "fresnel", "reflect")
NORMAL_ATLAS_MODALITIES = {"normal"}
ALPHA_MODALITY_FILENAMES = {
    "rgb": "I_obs.png",
    "diffuse": "I_diffuse.png",
    "fresnel": "I_fresnel.png",
    "normal": "I_obs_normal.png",
    "reflect": "I_obs_reflect.png",
    "roughness": "I_obs_roughness.png",
}


def _camera_source_key(camera) -> str:
    image_name = getattr(camera, "image_name", None)
    if image_name is not None:
        return str(image_name)
    return str(getattr(camera, "uid", id(camera)))


def _get_camera_source_map(
    camera,
    source_kind: str,
    source_maps_by_camera: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
) -> Optional[torch.Tensor]:
    if source_maps_by_camera is not None:
        source_maps = source_maps_by_camera.get(_camera_source_key(camera), {})
        if source_kind in source_maps:
            return source_maps[source_kind]
    if source_kind == "rgb":
        return getattr(camera, "original_image", None)
    raise ValueError(f"Unsupported atlas source kind '{source_kind}'")


def _unique_sorted_positive_labels(labels: Sequence[int]) -> List[int]:
    unique = {int(label) for label in labels if int(label) > 0}
    return sorted(unique)


def available_object_ids(gaussians) -> List[int]:
    object_ids = gaussians.get_object_id.detach()
    if object_ids.numel() == 0:
        return []
    labels = torch.unique(object_ids)
    return sorted(int(label.item()) for label in labels if int(label.item()) > 0)


def validate_desk_object_id(gaussians, desk_object_id: int) -> int:
    labels = available_object_ids(gaussians)
    if len(labels) == 0:
        raise RuntimeError("Checkpoint has no object_id > 0; cannot validate desk_object_id.")

    desk_object_id = int(desk_object_id)
    if desk_object_id <= 0:
        raise ValueError(f"desk_object_id must be > 0, got {desk_object_id}")
    if desk_object_id not in labels:
        raise ValueError(
            f"desk_object_id {desk_object_id} is not present in checkpoint object ids {labels}."
        )
    return desk_object_id


def infer_support_object_ids(
    gaussians,
    desk_object_id: int,
    explicit_labels: Optional[Sequence[int]] = None,
) -> List[int]:
    if explicit_labels is not None:
        return [label for label in _unique_sorted_positive_labels(explicit_labels) if int(label) != int(desk_object_id)]

    return [label for label in available_object_ids(gaussians) if int(label) != int(desk_object_id)]


def compute_plane_basis(normal: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    normal = F.normalize(normal, dim=0)
    anchor = torch.tensor([1.0, 0.0, 0.0], device=normal.device, dtype=normal.dtype)
    if torch.abs(torch.dot(normal, anchor)) > 0.9:
        anchor = torch.tensor([0.0, 1.0, 0.0], device=normal.device, dtype=normal.dtype)
    e1 = F.normalize(torch.cross(normal, anchor, dim=0), dim=0)
    e2 = F.normalize(torch.cross(normal, e1, dim=0), dim=0)
    return e1, e2


def fit_support_plane_from_visible_table(
    table_points: torch.Tensor,
    ransac_iters: int = 256,
    inlier_thresh: float = 0.01,
) -> PlaneDefinition:
    if table_points.shape[0] < 3:
        raise RuntimeError("Not enough points to fit support plane")

    n_points = table_points.shape[0]
    best_inlier_mask = None
    best_count = -1

    for _ in range(max(32, int(ransac_iters))):
        sample_idx = torch.randint(0, n_points, (3,), device=table_points.device)
        p0, p1, p2 = table_points[sample_idx]
        normal = torch.cross(p1 - p0, p2 - p0, dim=0)
        norm_val = torch.norm(normal)
        if norm_val < 1e-7:
            continue
        normal = normal / norm_val
        d = -torch.dot(normal, p0)
        dist = torch.abs(table_points @ normal + d)
        inliers = dist < float(inlier_thresh)
        count = int(inliers.sum().item())
        if count > best_count:
            best_count = count
            best_inlier_mask = inliers

    if best_inlier_mask is None or best_inlier_mask.sum() < 3:
        best_inlier_mask = torch.ones((n_points,), dtype=torch.bool, device=table_points.device)

    inlier_points = table_points[best_inlier_mask]
    origin = inlier_points.mean(dim=0)
    centered = inlier_points - origin[None]
    cov = (centered.t() @ centered) / max(1, centered.shape[0])
    _, eigvecs = torch.linalg.eigh(cov)
    normal = F.normalize(eigvecs[:, 0], dim=0)
    if normal[2] < 0:
        normal = -normal
    d = -torch.dot(normal, origin)
    e1, e2 = compute_plane_basis(normal)
    return PlaneDefinition(normal=normal, d=d, origin=origin, e1=e1, e2=e2)


def offset_plane_along_normal(plane: PlaneDefinition, offset: float) -> PlaneDefinition:
    normal = F.normalize(plane.normal, dim=0)
    origin = plane.origin + float(offset) * normal
    return PlaneDefinition(
        normal=normal,
        d=-torch.dot(normal, origin),
        origin=origin,
        e1=plane.e1,
        e2=plane.e2,
    )


def _compute_uv_bbox(uv: torch.Tensor, pad_ratio: float = 0.05) -> Tuple[float, float, float, float]:
    if uv.shape[0] == 0:
        raise RuntimeError("Cannot compute uv bbox from empty samples")
    u_min = float(uv[:, 0].min().item())
    u_max = float(uv[:, 0].max().item())
    v_min = float(uv[:, 1].min().item())
    v_max = float(uv[:, 1].max().item())
    du = max(u_max - u_min, 1e-6)
    dv = max(v_max - v_min, 1e-6)
    pad_u = float(pad_ratio) * du
    pad_v = float(pad_ratio) * dv
    return u_min - pad_u, u_max + pad_u, v_min - pad_v, v_max + pad_v


def _ceil_to_multiple(value: int, multiple: int) -> int:
    multiple = max(int(multiple), 1)
    value = int(value)
    return ((value + multiple - 1) // multiple) * multiple


def _compute_atlas_hw(
    bbox: Tuple[float, float, float, float],
    long_side: int,
    size_multiple: int = 1,
) -> Tuple[int, int]:
    u_min, u_max, v_min, v_max = bbox
    du = max(u_max - u_min, 1e-6)
    dv = max(v_max - v_min, 1e-6)
    size_multiple = max(int(size_multiple), 1)
    long_side = max(int(long_side), size_multiple)
    if long_side % size_multiple != 0:
        raise ValueError(
            f"desk_atlas_long_side={long_side} must be divisible by desk_atlas_size_multiple={size_multiple}"
        )
    if du >= dv:
        width = long_side
        height = _ceil_to_multiple(max(size_multiple, int(round(long_side * dv / du))), size_multiple)
    else:
        height = long_side
        width = _ceil_to_multiple(max(size_multiple, int(round(long_side * du / dv))), size_multiple)
    return height, width


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


def _morph_dilate(mask: torch.Tensor, kernel: int) -> torch.Tensor:
    if kernel <= 1:
        return mask.bool()
    x = mask.float()[None, None]
    y = F.max_pool2d(x, kernel_size=kernel, stride=1, padding=kernel // 2)
    return y.squeeze(0).squeeze(0) > 0.5


def _morph_erode(mask: torch.Tensor, kernel: int) -> torch.Tensor:
    if kernel <= 1:
        return mask.bool()
    x = mask.float()[None, None]
    y = -F.max_pool2d(-x, kernel_size=kernel, stride=1, padding=kernel // 2)
    return y.squeeze(0).squeeze(0) > 0.5


def _rasterize_uv_mask(
    uv: torch.Tensor,
    bbox: Tuple[float, float, float, float],
    height: int,
    width: int,
    kernel: int,
    device: torch.device,
) -> torch.Tensor:
    heat = torch.zeros((height, width), dtype=torch.float32, device=device)
    if uv.shape[0] == 0:
        return heat.bool()

    x, y = _uv_to_pixel_coords(uv, bbox, height, width)
    xi = x.round().long().clamp_(0, width - 1)
    yi = y.round().long().clamp_(0, height - 1)
    heat.index_put_((yi, xi), torch.ones_like(xi, dtype=torch.float32), accumulate=True)
    mask = heat > 0
    close_kernel = max(3, int(kernel) | 1)
    return _morph_erode(_morph_dilate(mask, close_kernel), close_kernel)


def _bilinear_splat_values_to_atlas(
    uv: torch.Tensor,
    values: torch.Tensor,
    weight: torch.Tensor,
    bbox: Tuple[float, float, float, float],
    height: int,
    width: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    channel_count = int(values.shape[1]) if values.ndim == 2 else 3
    value_sum = torch.zeros((channel_count, height, width), dtype=values.dtype, device=device)
    weight_sum = torch.zeros((height, width), dtype=values.dtype, device=device)
    if uv.shape[0] == 0:
        return value_sum, weight_sum

    x, y = _uv_to_pixel_coords(uv, bbox, height, width)
    x0 = torch.floor(x).long().clamp_(0, width - 1)
    y0 = torch.floor(y).long().clamp_(0, height - 1)
    x1 = (x0 + 1).clamp(max=width - 1)
    y1 = (y0 + 1).clamp(max=height - 1)

    wx1 = x - x0.float()
    wy1 = y - y0.float()
    wx0 = 1.0 - wx1
    wy0 = 1.0 - wy1

    corners = (
        (x0, y0, wx0 * wy0),
        (x1, y0, wx1 * wy0),
        (x0, y1, wx0 * wy1),
        (x1, y1, wx1 * wy1),
    )

    for xi, yi, cur_weight in corners:
        corner_weight = weight * cur_weight
        weight_sum.index_put_((yi, xi), corner_weight, accumulate=True)
        for channel in range(channel_count):
            value_sum[channel].index_put_((yi, xi), values[:, channel] * corner_weight, accumulate=True)

    return value_sum, weight_sum


def _filter_uv_outliers_iqr(uv: torch.Tensor, iqr_factor: float) -> torch.Tensor:
    if uv.shape[0] < 4 or iqr_factor <= 0.0:
        return uv

    q1 = torch.quantile(uv, 0.25, dim=0)
    q3 = torch.quantile(uv, 0.75, dim=0)
    iqr = q3 - q1
    lower = q1 - float(iqr_factor) * iqr
    upper = q3 + float(iqr_factor) * iqr
    keep = torch.logical_and(uv >= lower[None], uv <= upper[None]).all(dim=1)
    filtered = uv[keep]
    if filtered.shape[0] < 4:
        return uv
    return filtered


def _gaussian_label_mask(gaussians, labels: Sequence[int]) -> torch.Tensor:
    object_ids = gaussians.get_object_id.detach()
    mask = torch.zeros_like(object_ids, dtype=torch.bool)
    for label in labels:
        mask = torch.logical_or(mask, object_ids == int(label))
    return mask


def _project_label_bottom_gaussians_to_plane_uv(
    gaussians,
    label: int,
    plane: PlaneDefinition,
    bottom_quantile: float,
    min_opacity: float,
    uv_outlier_iqr: float,
) -> torch.Tensor:
    mask3d = gaussians.get_object_id == int(label)
    if mask3d.sum() == 0:
        return torch.empty((0, 2), device=plane.normal.device, dtype=plane.normal.dtype)

    xyz = gaussians._xyz.detach()[mask3d]
    opacity = gaussians.get_opacity.squeeze(-1).detach()[mask3d]
    if xyz.shape[0] == 0:
        return torch.empty((0, 2), device=plane.normal.device, dtype=plane.normal.dtype)

    opacity_keep = opacity >= float(min_opacity)
    if opacity_keep.sum() == 0:
        return torch.empty((0, 2), device=plane.normal.device, dtype=plane.normal.dtype)

    xyz = xyz[opacity_keep]
    dist = torch.abs(xyz @ plane.normal + plane.d)
    keep_ratio = min(max(float(bottom_quantile), 1e-4), 1.0)
    k_quantile = max(1, int(math.ceil(xyz.shape[0] * keep_ratio)))
    k_keep = min(xyz.shape[0], max(8, k_quantile))
    keep_idx = torch.topk(dist, k=k_keep, largest=False).indices
    uv = project_xyz_to_plane_uv(xyz[keep_idx], plane)
    return _filter_uv_outliers_iqr(uv, uv_outlier_iqr)


def build_object_footprint_on_plane(
    gaussians,
    support_object_ids: Sequence[int],
    plane: PlaneDefinition,
    bottom_quantile: float,
    min_opacity: float,
    uv_outlier_iqr: float,
) -> FootprintMaps:
    all_uv: List[torch.Tensor] = []
    for label in _unique_sorted_positive_labels(support_object_ids):
        uv = _project_label_bottom_gaussians_to_plane_uv(
            gaussians=gaussians,
            label=int(label),
            plane=plane,
            bottom_quantile=bottom_quantile,
            min_opacity=min_opacity,
            uv_outlier_iqr=uv_outlier_iqr,
        )
        if uv.shape[0] > 0:
            all_uv.append(uv)

    if len(all_uv) == 0:
        return FootprintMaps(uv_points=torch.empty((0, 2), device=plane.normal.device, dtype=plane.normal.dtype))
    return FootprintMaps(uv_points=torch.cat(all_uv, dim=0))


def _collect_visible_desk_plane_samples(
    scene,
    mask_provider,
    plane: PlaneDefinition,
    desk_object_id: int,
    max_samples: int,
    source_kind: str = "rgb",
    source_maps_by_camera: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    uv_all: List[torch.Tensor] = []
    value_all: List[torch.Tensor] = []
    weight_all: List[torch.Tensor] = []

    for cam in scene.getTrainCameras():
        source_image = _get_camera_source_map(cam, source_kind, source_maps_by_camera=source_maps_by_camera)
        if source_image is None:
            continue

        label_map = mask_provider.get_label_map(cam).squeeze(0).long().to(cam.camera_center.device)
        desk_mask = label_map == int(desk_object_id)
        uv, values, weight = project_mask_pixels_to_plane_samples(
            desk_mask,
            cam,
            plane,
            max_samples=max_samples,
            source_image=source_image,
        )
        if uv.shape[0] == 0:
            continue
        uv_all.append(uv)
        value_all.append(values)
        weight_all.append(weight)

    if len(uv_all) == 0:
        empty_uv = torch.empty((0, 2), device=plane.normal.device, dtype=plane.normal.dtype)
        empty_rgb = torch.empty((0, 3), device=plane.normal.device, dtype=plane.normal.dtype)
        empty_weight = torch.empty((0,), device=plane.normal.device, dtype=plane.normal.dtype)
        return empty_uv, empty_rgb, empty_weight

    return torch.cat(uv_all, dim=0), torch.cat(value_all, dim=0), torch.cat(weight_all, dim=0)


def _normalize_single_channel_map(values: torch.Tensor) -> torch.Tensor:
    values = values.float()
    positive = values[values > 0]
    max_val = positive.max().item() if positive.numel() > 0 else 0.0
    if max_val <= 1e-6:
        return torch.zeros_like(values)
    return torch.clamp(values / max_val, 0.0, 1.0)


def _normalize_modality_observation(
    observed: torch.Tensor,
    observed_mask: torch.Tensor,
    source_kind: str,
) -> torch.Tensor:
    if source_kind not in NORMAL_ATLAS_MODALITIES:
        return observed
    normalized = F.normalize(observed, dim=0)
    return normalized * observed_mask.unsqueeze(0).float()


def _chunked_cdist_min_cpu(
    query_idx: torch.Tensor,
    ref_idx: torch.Tensor,
    max_working_set_mb: int = 256,
) -> Tuple[torch.Tensor, torch.Tensor]:
    query_cpu = query_idx.detach().cpu().float()
    ref_cpu = ref_idx.detach().cpu().float()
    if query_cpu.shape[0] == 0 or ref_cpu.shape[0] == 0:
        empty_dist = torch.empty((query_cpu.shape[0],), dtype=torch.float32)
        empty_idx = torch.empty((query_cpu.shape[0],), dtype=torch.long)
        return empty_dist, empty_idx

    bytes_per_float = 4
    max_bytes = max(int(max_working_set_mb), 1) * 1024 * 1024
    chunk = max(1, min(4096, max_bytes // max(int(ref_cpu.shape[0]), 1) // bytes_per_float))

    min_dist = torch.empty((query_cpu.shape[0],), dtype=torch.float32)
    min_idx = torch.empty((query_cpu.shape[0],), dtype=torch.long)
    for start in range(0, query_cpu.shape[0], chunk):
        current = query_cpu[start:start + chunk]
        dist = torch.cdist(current, ref_cpu)
        current_min_dist, current_min_idx = dist.min(dim=1)
        end = start + current.shape[0]
        min_dist[start:end] = current_min_dist
        min_idx[start:end] = current_min_idx
    return min_dist, min_idx


def _compute_boundary_distance_map(hole_mask: torch.Tensor) -> torch.Tensor:
    if hole_mask.numel() == 0:
        return hole_mask.float()

    hole_np = hole_mask.detach().cpu().numpy().astype(np.uint8)
    if distance_transform_edt is not None:
        dist_np = distance_transform_edt(hole_np > 0)
        return torch.from_numpy(dist_np).to(device=hole_mask.device, dtype=torch.float32)
    if cv2 is not None:
        dist_np = cv2.distanceTransform(hole_np, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        return torch.from_numpy(dist_np).to(device=hole_mask.device, dtype=torch.float32)

    hole_idx = hole_mask.nonzero(as_tuple=False)
    if hole_idx.shape[0] == 0:
        return torch.zeros_like(hole_mask, dtype=torch.float32)

    eroded = _morph_erode(hole_mask, 3)
    boundary = torch.logical_and(hole_mask, ~eroded)
    boundary_idx = boundary.nonzero(as_tuple=False)
    if boundary_idx.shape[0] == 0:
        boundary_idx = hole_idx

    distance_map = torch.zeros_like(hole_mask, dtype=torch.float32)
    min_dist, _ = _chunked_cdist_min_cpu(hole_idx, boundary_idx)
    distance_map[hole_idx[:, 0].long(), hole_idx[:, 1].long()] = min_dist.to(
        device=hole_mask.device,
        dtype=torch.float32,
    )
    return distance_map


def _nearest_neighbor_fill_inside_support(
    image: torch.Tensor,
    support_mask: torch.Tensor,
    observed_mask: torch.Tensor,
) -> torch.Tensor:
    filled = image.clone()
    observed_in_support = torch.logical_and(observed_mask, support_mask)
    fill_mask = torch.logical_and(support_mask, ~observed_in_support)
    if fill_mask.sum() == 0:
        return filled

    if distance_transform_edt is not None:
        observed_np = observed_in_support.detach().cpu().numpy().astype(bool)
        if observed_np.any():
            _, indices = distance_transform_edt(~observed_np, return_indices=True)
            yi = torch.from_numpy(indices[0]).to(device=image.device, dtype=torch.long)
            xi = torch.from_numpy(indices[1]).to(device=image.device, dtype=torch.long)
            fill_idx = fill_mask.nonzero(as_tuple=False)
            src_y = yi[fill_idx[:, 0], fill_idx[:, 1]]
            src_x = xi[fill_idx[:, 0], fill_idx[:, 1]]
            filled[:, fill_idx[:, 0], fill_idx[:, 1]] = image[:, src_y, src_x]
            return filled

    if cv2 is not None:
        observed_np = observed_in_support.detach().cpu().numpy().astype(np.uint8)
        if observed_np.any():
            fill_idx = fill_mask.nonzero(as_tuple=False)
            if fill_idx.shape[0] == 0:
                return filled
            search_np = np.where(observed_np > 0, 0, 1).astype(np.uint8)
            _, labels = cv2.distanceTransformWithLabels(
                search_np,
                cv2.DIST_L2,
                cv2.DIST_MASK_5,
                labelType=cv2.DIST_LABEL_PIXEL,
            )
            observed_coords = np.argwhere(observed_np > 0)
            fill_y = fill_idx[:, 0].detach().cpu().numpy()
            fill_x = fill_idx[:, 1].detach().cpu().numpy()
            label_ids = labels[fill_y, fill_x]
            valid = label_ids > 0
            if valid.any():
                src_coords = observed_coords[label_ids[valid] - 1]
                dst = fill_idx[torch.from_numpy(valid).to(fill_idx.device)]
                src_y = torch.from_numpy(src_coords[:, 0]).to(device=image.device, dtype=torch.long)
                src_x = torch.from_numpy(src_coords[:, 1]).to(device=image.device, dtype=torch.long)
                filled[:, dst[:, 0], dst[:, 1]] = image[:, src_y, src_x]
                return filled

    observed_idx = observed_in_support.nonzero(as_tuple=False)
    fill_idx = fill_mask.nonzero(as_tuple=False)
    if observed_idx.shape[0] == 0 or fill_idx.shape[0] == 0:
        return filled

    _, nearest = _chunked_cdist_min_cpu(fill_idx, observed_idx)
    src = observed_idx[nearest].to(device=image.device, dtype=torch.long)
    dst = fill_idx.to(device=image.device, dtype=torch.long)
    filled[:, dst[:, 0], dst[:, 1]] = image[:, src[:, 0], src[:, 1]]
    return filled


def _poisson_like_prefill(
    image: torch.Tensor,
    support_mask: torch.Tensor,
    observed_mask: torch.Tensor,
    strict_hole_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    filled = image.clone()
    strict_hole = strict_hole_mask if strict_hole_mask is not None else torch.logical_and(support_mask, ~observed_mask)
    observed_in_support = torch.logical_and(observed_mask, support_mask)
    if strict_hole.sum() == 0:
        return filled

    if lil_matrix is None or spsolve is None:
        return _nearest_neighbor_fill_inside_support(image, support_mask, observed_in_support)

    support_np = support_mask.detach().cpu().numpy().astype(bool)
    observed_np = observed_in_support.detach().cpu().numpy().astype(bool)
    hole_np = strict_hole.detach().cpu().numpy().astype(bool)
    if not hole_np.any():
        return filled

    hole_coords = np.argwhere(hole_np)
    coord_to_idx = {tuple(coord.tolist()): idx for idx, coord in enumerate(hole_coords)}
    unknown_count = hole_coords.shape[0]
    if unknown_count == 0:
        return filled

    A = lil_matrix((unknown_count, unknown_count), dtype=np.float32)
    channel_count = int(image.shape[0])
    rhs = np.zeros((unknown_count, channel_count), dtype=np.float32)
    image_np = image.detach().cpu().numpy()
    height, width = support_np.shape
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for row_idx, (y, x) in enumerate(hole_coords):
        degree = 0
        for dy, dx in neighbors:
            ny, nx = int(y + dy), int(x + dx)
            if ny < 0 or ny >= height or nx < 0 or nx >= width or not support_np[ny, nx]:
                continue
            degree += 1
            if hole_np[ny, nx]:
                A[row_idx, coord_to_idx[(ny, nx)]] = -1.0
            else:
                rhs[row_idx] += image_np[:, ny, nx]
        if degree == 0:
            A[row_idx, row_idx] = 1.0
        else:
            A[row_idx, row_idx] = float(degree)

    try:
        A = A.tocsr()
        solved = np.zeros((unknown_count, channel_count), dtype=np.float32)
        for channel in range(channel_count):
            solved[:, channel] = spsolve(A, rhs[:, channel]).astype(np.float32)
    except Exception:
        return _nearest_neighbor_fill_inside_support(image, support_mask, observed_in_support)

    if not np.isfinite(solved).all():
        return _nearest_neighbor_fill_inside_support(image, support_mask, observed_in_support)

    hole_idx = torch.from_numpy(hole_coords).to(device=image.device, dtype=torch.long)
    solved_t = torch.from_numpy(solved.T).to(device=image.device, dtype=image.dtype)
    filled[:, hole_idx[:, 0], hole_idx[:, 1]] = solved_t
    filled[:, observed_mask] = image[:, observed_mask]
    filled[:, ~support_mask] = image[:, ~support_mask]
    return filled


def _observed_mask_for_hole_suppression(
    observed_mask: torch.Tensor,
    support_visible_mask: torch.Tensor,
    support_footprint_mask: torch.Tensor,
    dilate_kernel: int,
) -> torch.Tensor:
    observed_for_hole = observed_mask.bool()
    dilate_kernel = int(dilate_kernel)
    if dilate_kernel <= 1:
        return observed_for_hole

    dilate_kernel = dilate_kernel | 1
    visible_only = torch.logical_and(
        support_visible_mask.bool(),
        ~support_footprint_mask.bool(),
    )
    dilated_observed = _morph_dilate(observed_for_hole, dilate_kernel)
    return torch.logical_or(observed_for_hole, torch.logical_and(dilated_observed, visible_only))


def _compute_known_masks_from_confidence(
    confidence: torch.Tensor,
    support_mask: torch.Tensor,
    quantile: float,
    support_visible_mask: torch.Tensor,
    support_footprint_mask: torch.Tensor,
    hole_observed_dilate_kernel: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    observed_raw = confidence > 0
    observed_for_hole = _observed_mask_for_hole_suppression(
        observed_raw,
        support_visible_mask,
        support_footprint_mask,
        hole_observed_dilate_kernel,
    )
    hole_strict = torch.logical_and(support_mask, ~observed_raw)
    conf_positive = confidence[observed_raw]
    if conf_positive.numel() == 0:
        known_strong = torch.zeros_like(support_mask, dtype=torch.bool)
        known_weak = torch.zeros_like(support_mask, dtype=torch.bool)
        hole_editable = torch.logical_and(support_mask, ~observed_for_hole)
        return known_strong, known_weak, hole_editable, hole_strict

    tau_high = torch.quantile(conf_positive, quantile)
    known_strong = confidence >= tau_high
    known_weak = torch.logical_and(observed_raw, ~known_strong)
    hole_unobserved = torch.logical_and(support_mask, ~observed_for_hole)
    hole_editable = torch.logical_or(
        hole_unobserved,
        torch.logical_and(support_mask, known_weak),
    )
    return known_strong, known_weak, hole_editable, hole_strict


def pack_desk_atlas_for_texture_diffusion(
    desk_atlas_state: DeskAtlasState,
    known_strong_quantile: float,
    hole_observed_dilate_kernel: int = 1,
    source_kind: str = "rgb",
) -> Dict[str, Any]:
    if not (0.0 <= float(known_strong_quantile) <= 1.0):
        raise ValueError(
            f"known_strong_quantile must be in [0, 1], got {known_strong_quantile}"
        )
    hole_observed_dilate_kernel = max(1, int(hole_observed_dilate_kernel))

    confidence_norm = _normalize_single_channel_map(desk_atlas_state.confidence)
    known_strong, known_weak, hole_editable, hole_strict = _compute_known_masks_from_confidence(
        desk_atlas_state.confidence,
        desk_atlas_state.support_mask,
        float(known_strong_quantile),
        desk_atlas_state.support_visible_mask,
        desk_atlas_state.support_footprint_mask,
        hole_observed_dilate_kernel,
    )
    boundary_distance = _compute_boundary_distance_map(hole_editable)
    observed_filled = _poisson_like_prefill(
        desk_atlas_state.rgb_observed,
        desk_atlas_state.support_mask,
        desk_atlas_state.observed_mask,
        strict_hole_mask=hole_strict,
    )
    observed_filled = _normalize_modality_observation(
        observed_filled,
        desk_atlas_state.support_mask,
        source_kind,
    )
    return {
        "plane": _serialize_plane_definition(desk_atlas_state.plane),
        "uv_bbox": tuple(float(v) for v in desk_atlas_state.uv_bbox),
        "atlas_hw": tuple(int(v) for v in desk_atlas_state.atlas_hw),
        "build_iteration": int(desk_atlas_state.build_iteration),
        "known_strong_quantile": float(known_strong_quantile),
        "hole_observed_dilate_kernel": int(hole_observed_dilate_kernel),
        "I_obs": _cpu_clone_tree(desk_atlas_state.rgb_observed),
        "M_known_strong": _cpu_clone_tree(known_strong),
        "M_known_weak": _cpu_clone_tree(known_weak),
        "M_hole": _cpu_clone_tree(hole_editable),
        "M_support_visible": _cpu_clone_tree(desk_atlas_state.support_visible_mask),
        "M_support_footprint": _cpu_clone_tree(desk_atlas_state.support_footprint_mask),
        "C_norm": _cpu_clone_tree(confidence_norm),
        "D_boundary": _cpu_clone_tree(boundary_distance),
        "I_obs_filled": _cpu_clone_tree(observed_filled),
    }


def build_desk_atlas_state(
    scene,
    gaussians,
    mask_provider,
    opt,
    desk_object_id: int,
    support_object_ids: Sequence[int],
    iteration: int,
    source_maps_by_camera: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
) -> DeskAtlasState:
    desk_mask = gaussians.get_object_id == int(desk_object_id)
    if desk_mask.sum() < 3:
        raise RuntimeError(
            f"Not enough desk Gaussians to fit plane for desk_object_id={int(desk_object_id)}; "
            f"found {int(desk_mask.sum().item())}."
        )

    plane = fit_support_plane_from_visible_table(
        gaussians._xyz.detach()[desk_mask],
        ransac_iters=opt.ccm_plane_ransac_iters,
        inlier_thresh=opt.ccm_plane_ransac_thresh,
    )
    plane_down_offset = float(getattr(opt, "desk_plane_down_offset", 0.0))
    if plane_down_offset != 0.0:
        plane = offset_plane_along_normal(plane, -plane_down_offset)

    visible_uv, visible_rgb, visible_weight = _collect_visible_desk_plane_samples(
        scene=scene,
        mask_provider=mask_provider,
        plane=plane,
        desk_object_id=int(desk_object_id),
        max_samples=opt.ccm_max_mask_samples,
        source_maps_by_camera=source_maps_by_camera,
    )
    if visible_uv.shape[0] == 0:
        raise RuntimeError("No visible desk pixels available for atlas construction")

    footprint = build_object_footprint_on_plane(
        gaussians=gaussians,
        support_object_ids=support_object_ids,
        plane=plane,
        bottom_quantile=opt.footprint_bottom_quantile,
        min_opacity=opt.footprint_min_opacity,
        uv_outlier_iqr=opt.footprint_uv_outlier_iqr,
    )

    bbox_samples = [visible_uv]
    if footprint.uv_points.shape[0] > 0:
        bbox_samples.append(footprint.uv_points)
    atlas_uv = torch.cat(bbox_samples, dim=0)
    uv_bbox = _compute_uv_bbox(atlas_uv)
    atlas_h, atlas_w = _compute_atlas_hw(
        uv_bbox,
        opt.desk_atlas_long_side,
        opt.desk_atlas_size_multiple,
    )

    support_visible = _rasterize_uv_mask(
        visible_uv,
        bbox=uv_bbox,
        height=atlas_h,
        width=atlas_w,
        kernel=opt.ccm_contact_kernel,
        device=gaussians.get_xyz.device,
    )
    support_footprint = _rasterize_uv_mask(
        footprint.uv_points,
        bbox=uv_bbox,
        height=atlas_h,
        width=atlas_w,
        kernel=opt.ccm_contact_kernel,
        device=gaussians.get_xyz.device,
    )
    support_mask = torch.logical_or(support_visible, support_footprint)

    rgb_sum, confidence = _bilinear_splat_values_to_atlas(
        visible_uv,
        visible_rgb,
        visible_weight,
        bbox=uv_bbox,
        height=atlas_h,
        width=atlas_w,
        device=gaussians.get_xyz.device,
    )
    rgb_observed = rgb_sum / confidence.clamp_min(1e-6).unsqueeze(0)
    observed_mask = confidence > 0
    rgb_observed = rgb_observed * observed_mask.unsqueeze(0).float()
    hole_mask = torch.logical_and(support_mask, ~observed_mask)

    state = DeskAtlasState(
        plane=plane,
        uv_bbox=uv_bbox,
        atlas_hw=(atlas_h, atlas_w),
        support_mask=support_mask,
        support_visible_mask=support_visible,
        support_footprint_mask=support_footprint,
        observed_mask=observed_mask,
        hole_mask=hole_mask,
        confidence=confidence,
        rgb_observed=rgb_observed,
        build_iteration=int(iteration),
    )
    support_count = int(support_mask.sum().item())
    observed_count = int(observed_mask.sum().item())
    print(
        "[DeskAtlas] "
        f"build_iteration={int(iteration)} "
        f"desk_object_id={int(desk_object_id)} "
        f"support_object_ids={list(int(v) for v in support_object_ids)} "
        f"atlas_hw=({atlas_h}, {atlas_w}) "
        f"desk_plane_down_offset={plane_down_offset:.6g} "
        f"support_pixels={support_count} "
        f"observed_pixels={observed_count}"
    )
    return state


def build_desk_atlas_observation_state(
    scene,
    mask_provider,
    desk_atlas_state: DeskAtlasState,
    desk_object_id: int,
    max_samples: int,
    source_kind: str,
    source_maps_by_camera: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
) -> DeskAtlasState:
    visible_uv, visible_values, visible_weight = _collect_visible_desk_plane_samples(
        scene=scene,
        mask_provider=mask_provider,
        plane=desk_atlas_state.plane,
        desk_object_id=int(desk_object_id),
        max_samples=max_samples,
        source_kind=source_kind,
        source_maps_by_camera=source_maps_by_camera,
    )
    if visible_uv.shape[0] == 0:
        raise RuntimeError(f"No visible desk pixels available for atlas source '{source_kind}'")

    atlas_h, atlas_w = desk_atlas_state.atlas_hw
    rgb_sum, confidence = _bilinear_splat_values_to_atlas(
        visible_uv,
        visible_values,
        visible_weight,
        bbox=desk_atlas_state.uv_bbox,
        height=atlas_h,
        width=atlas_w,
        device=desk_atlas_state.support_mask.device,
    )
    observed = rgb_sum / confidence.clamp_min(1e-6).unsqueeze(0)
    observed_mask = confidence > 0
    observed = observed * observed_mask.unsqueeze(0).float()
    observed = _normalize_modality_observation(observed, observed_mask, source_kind)
    hole_mask = torch.logical_and(desk_atlas_state.support_mask, ~observed_mask)
    return DeskAtlasState(
        plane=desk_atlas_state.plane,
        uv_bbox=desk_atlas_state.uv_bbox,
        atlas_hw=desk_atlas_state.atlas_hw,
        support_mask=desk_atlas_state.support_mask,
        support_visible_mask=desk_atlas_state.support_visible_mask,
        support_footprint_mask=desk_atlas_state.support_footprint_mask,
        observed_mask=observed_mask,
        hole_mask=hole_mask,
        confidence=confidence,
        rgb_observed=observed,
        build_iteration=desk_atlas_state.build_iteration,
    )


def export_desk_atlas_modalities(
    scene,
    mask_provider,
    model_path: str,
    desk_atlas_state: DeskAtlasState,
    desk_object_id: int,
    opt,
    output_subdir: str = "desk_atlas",
    source_maps_by_camera: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
) -> Dict[str, Dict[str, Any]]:
    outputs: Dict[str, Dict[str, Any]] = {}
    for source_kind in DESK_ATLAS_MODALITIES:
        observation_state = build_desk_atlas_observation_state(
            scene=scene,
            mask_provider=mask_provider,
            desk_atlas_state=desk_atlas_state,
            desk_object_id=int(desk_object_id),
            max_samples=opt.ccm_max_mask_samples,
            source_kind=source_kind,
            source_maps_by_camera=source_maps_by_camera,
        )
        diffusion_pack = pack_desk_atlas_for_texture_diffusion(
            observation_state,
            known_strong_quantile=float(opt.desk_pack_known_strong_quantile),
            hole_observed_dilate_kernel=int(
                getattr(opt, "desk_pack_hole_observed_dilate_kernel", 1)
            ),
            source_kind=source_kind,
        )
        outputs[source_kind] = {
            "state": observation_state,
            "pack": diffusion_pack,
        }

    save_desk_atlas_artifacts(model_path, desk_atlas_state, outputs, output_subdir=output_subdir)
    return outputs


def _cpu_clone_tree(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_clone_tree(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_cpu_clone_tree(val) for val in value]
    if isinstance(value, tuple):
        return tuple(_cpu_clone_tree(val) for val in value)
    return value


def _move_tree_to_device(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device=device)
    if isinstance(value, dict):
        return {key: _move_tree_to_device(val, device) for key, val in value.items()}
    if isinstance(value, list):
        return [_move_tree_to_device(val, device) for val in value]
    if isinstance(value, tuple):
        return tuple(_move_tree_to_device(val, device) for val in value)
    return value


def _serialize_plane_definition(plane: PlaneDefinition) -> Dict[str, Any]:
    return {
        "normal": _cpu_clone_tree(plane.normal),
        "d": _cpu_clone_tree(plane.d),
        "origin": _cpu_clone_tree(plane.origin),
        "e1": _cpu_clone_tree(plane.e1),
        "e2": _cpu_clone_tree(plane.e2),
    }


def _deserialize_plane_definition(payload: Dict[str, Any], device: torch.device) -> PlaneDefinition:
    return PlaneDefinition(
        normal=_move_tree_to_device(payload["normal"], device),
        d=_move_tree_to_device(payload["d"], device),
        origin=_move_tree_to_device(payload["origin"], device),
        e1=_move_tree_to_device(payload["e1"], device),
        e2=_move_tree_to_device(payload["e2"], device),
    )


def _serialize_desk_atlas_state(desk_atlas_state: DeskAtlasState) -> Dict[str, Any]:
    return {
        "plane": _serialize_plane_definition(desk_atlas_state.plane),
        "uv_bbox": tuple(float(v) for v in desk_atlas_state.uv_bbox),
        "atlas_hw": tuple(int(v) for v in desk_atlas_state.atlas_hw),
        "support_mask": _cpu_clone_tree(desk_atlas_state.support_mask),
        "support_visible_mask": _cpu_clone_tree(desk_atlas_state.support_visible_mask),
        "support_footprint_mask": _cpu_clone_tree(desk_atlas_state.support_footprint_mask),
        "observed_mask": _cpu_clone_tree(desk_atlas_state.observed_mask),
        "hole_mask": _cpu_clone_tree(desk_atlas_state.hole_mask),
        "confidence": _cpu_clone_tree(desk_atlas_state.confidence),
        "rgb_observed": _cpu_clone_tree(desk_atlas_state.rgb_observed),
        "build_iteration": int(desk_atlas_state.build_iteration),
    }


def _deserialize_desk_atlas_state(payload: Dict[str, Any], device: torch.device) -> DeskAtlasState:
    support_mask = _move_tree_to_device(payload["support_mask"], device)
    support_visible_mask = _move_tree_to_device(payload.get("support_visible_mask", payload["support_mask"]), device)
    support_footprint_mask = _move_tree_to_device(
        payload.get("support_footprint_mask", torch.zeros_like(payload["support_mask"])),
        device,
    )
    return DeskAtlasState(
        plane=_deserialize_plane_definition(payload["plane"], device),
        uv_bbox=tuple(float(v) for v in payload["uv_bbox"]),
        atlas_hw=tuple(int(v) for v in payload["atlas_hw"]),
        support_mask=support_mask,
        support_visible_mask=support_visible_mask,
        support_footprint_mask=support_footprint_mask,
        observed_mask=_move_tree_to_device(payload["observed_mask"], device),
        hole_mask=_move_tree_to_device(payload["hole_mask"], device),
        confidence=_move_tree_to_device(payload["confidence"], device),
        rgb_observed=_move_tree_to_device(payload["rgb_observed"], device),
        build_iteration=int(payload["build_iteration"]),
    )


def _save_mask(mask: torch.Tensor, path: str) -> None:
    save_image(mask.float().unsqueeze(0), path)


def _image_for_png(image: torch.Tensor) -> torch.Tensor:
    if image.ndim == 2:
        image = image.unsqueeze(0)
    if image.ndim == 3 and image.shape[0] == 1:
        return image.repeat(3, 1, 1)
    return image


def _image_for_modality_png(image: torch.Tensor, source_kind: str) -> torch.Tensor:
    if source_kind in NORMAL_ATLAS_MODALITIES:
        image = image * 0.5 + 0.5
    return _image_for_png(image.clamp(0.0, 1.0))


def _image_for_modality_rgba_png(
    image: torch.Tensor,
    alpha_mask: torch.Tensor,
    source_kind: str,
) -> torch.Tensor:
    rgb = _image_for_modality_png(image, source_kind)
    alpha = alpha_mask.float()
    if alpha.ndim == 2:
        alpha = alpha.unsqueeze(0)
    elif alpha.ndim == 3 and alpha.shape[0] != 1:
        alpha = alpha[:1]
    alpha = alpha.to(device=rgb.device, dtype=rgb.dtype).clamp(0.0, 1.0)
    return torch.cat((rgb, alpha), dim=0)


def _resolve_atlas_dir(model_path: str, output_subdir: str) -> str:
    if os.path.isabs(output_subdir):
        return output_subdir
    return os.path.join(model_path, output_subdir)


def save_desk_atlas_artifacts(
    model_path: str,
    desk_atlas_state: DeskAtlasState,
    modality_outputs: Dict[str, Dict[str, Any]],
    output_subdir: str = "desk_atlas",
) -> None:
    atlas_dir = _resolve_atlas_dir(model_path, output_subdir)
    os.makedirs(atlas_dir, exist_ok=True)
    torch.save(_serialize_desk_atlas_state(desk_atlas_state), os.path.join(atlas_dir, "desk_atlas_state.pt"))
    _save_mask(desk_atlas_state.support_visible_mask, os.path.join(atlas_dir, "M_support_visible.png"))
    _save_mask(desk_atlas_state.support_footprint_mask, os.path.join(atlas_dir, "M_support_footprint.png"))

    for source_kind, payload in modality_outputs.items():
        observation_state = payload["state"]
        diffusion_pack = payload["pack"]
        torch.save(_cpu_clone_tree(diffusion_pack), os.path.join(atlas_dir, f"desk_{source_kind}_diffusion_pack.pt"))
        save_image(_image_for_modality_png(observation_state.rgb_observed, source_kind), os.path.join(atlas_dir, f"I_obs_{source_kind}.png"))
        _save_mask(diffusion_pack["M_known_strong"], os.path.join(atlas_dir, f"M_known_strong_{source_kind}.png"))
        _save_mask(diffusion_pack["M_known_weak"], os.path.join(atlas_dir, f"M_known_weak_{source_kind}.png"))
        _save_mask(diffusion_pack["M_hole"], os.path.join(atlas_dir, f"M_hole_{source_kind}.png"))
        save_image(diffusion_pack["C_norm"].float().unsqueeze(0), os.path.join(atlas_dir, f"C_norm_{source_kind}.png"))
        save_image(
            _normalize_single_channel_map(diffusion_pack["D_boundary"]).unsqueeze(0),
            os.path.join(atlas_dir, f"D_boundary_{source_kind}.png"),
        )
        save_image(
            _image_for_modality_png(diffusion_pack["I_obs_filled"], source_kind),
            os.path.join(atlas_dir, f"I_obs_filled_{source_kind}.png"),
        )
        if source_kind == "rgb":
            torch.save(_cpu_clone_tree(diffusion_pack), os.path.join(atlas_dir, "desk_diffusion_pack.pt"))
            save_image(_image_for_modality_png(observation_state.rgb_observed, source_kind), os.path.join(atlas_dir, "I_obs.png"))
        alpha_filename = ALPHA_MODALITY_FILENAMES.get(source_kind)
        if alpha_filename is not None:
            alpha_dir = os.path.join(atlas_dir, "alpha")
            os.makedirs(alpha_dir, exist_ok=True)
            save_image(
                _image_for_modality_rgba_png(
                    observation_state.rgb_observed,
                    observation_state.observed_mask,
                    source_kind,
                ),
                os.path.join(alpha_dir, alpha_filename),
            )


def load_desk_atlas_artifacts(
    model_path: str,
    device: torch.device,
    output_subdir: str = "desk_atlas",
) -> Optional[DeskAtlasState]:
    atlas_dir = _resolve_atlas_dir(model_path, output_subdir)
    state_path = os.path.join(atlas_dir, "desk_atlas_state.pt")
    if not os.path.exists(state_path):
        return None

    return _deserialize_desk_atlas_state(
        torch.load(state_path, map_location="cpu"),
        device=device,
    )
