from typing import Any, Tuple

import torch
import torch.nn.functional as F


def project_xyz_to_plane_uv(xyz: torch.Tensor, plane: Any) -> torch.Tensor:
    vec = xyz - plane.origin[None]
    u = vec @ plane.e1
    v = vec @ plane.e2
    return torch.stack([u, v], dim=-1)


def _sample_mask_indices(mask: torch.Tensor, max_samples: int) -> torch.Tensor:
    idx = mask.nonzero(as_tuple=False)
    if idx.shape[0] == 0:
        return idx
    if idx.shape[0] > max_samples:
        perm = torch.randperm(idx.shape[0], device=idx.device)[:max_samples]
        idx = idx[perm]
    return idx


def _camera_intrinsics(camera) -> Tuple[float, float, float, float]:
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

    return float(fx), float(fy), float(cx), float(cy)


def _camera_c2w(camera) -> torch.Tensor:
    c2w = getattr(camera, "c2w", None)
    if c2w is not None:
        return c2w
    extrinsics = camera.world_view_transform.transpose(0, 1)
    return torch.inverse(extrinsics)


def project_mask_pixels_to_plane_uv(
    mask: torch.Tensor,
    camera,
    plane: Any,
    max_samples: int,
) -> torch.Tensor:
    if mask is None:
        return torch.empty((0, 2), device=plane.normal.device, dtype=plane.normal.dtype)

    if mask.ndim == 3:
        mask = mask.squeeze(0)

    device = camera.camera_center.device
    mask = mask.bool().to(device)
    idx = _sample_mask_indices(mask, max_samples=max_samples)
    if idx.shape[0] == 0:
        return torch.empty((0, 2), device=plane.normal.device, dtype=plane.normal.dtype)

    v = idx[:, 0].float()
    u = idx[:, 1].float()
    fx, fy, cx, cy = _camera_intrinsics(camera)

    dirs_cam = torch.stack([(u - cx) / fx, (v - cy) / fy, torch.ones_like(u)], dim=-1)
    dirs_cam = F.normalize(dirs_cam, dim=-1)
    c2w = _camera_c2w(camera).to(device=device, dtype=dirs_cam.dtype)
    dirs_world = (c2w[:3, :3] @ dirs_cam.t()).t()

    origins = camera.camera_center[None].expand_as(dirs_world)
    denom = dirs_world @ plane.normal
    valid = torch.abs(denom) > 1e-7
    if valid.sum() == 0:
        return torch.empty((0, 2), device=plane.normal.device, dtype=plane.normal.dtype)

    denom = denom[valid]
    origins = origins[valid]
    dirs_world = dirs_world[valid]
    t = -(origins @ plane.normal + plane.d) / denom
    hit = t > 0
    if hit.sum() == 0:
        return torch.empty((0, 2), device=plane.normal.device, dtype=plane.normal.dtype)

    points = origins[hit] + dirs_world[hit] * t[hit][:, None]
    return project_xyz_to_plane_uv(points, plane)


def project_mask_pixels_to_plane_samples(
    mask: torch.Tensor,
    camera,
    plane: Any,
    max_samples: int,
    source_image: torch.Tensor = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    empty_uv = torch.empty((0, 2), device=plane.normal.device, dtype=plane.normal.dtype)
    if source_image is None:
        source_image = camera.original_image
    if source_image is None:
        channel_count = 3
    elif source_image.ndim == 2:
        channel_count = 1
    else:
        channel_count = int(source_image.shape[0])
    empty_values = torch.empty((0, channel_count), device=plane.normal.device, dtype=plane.normal.dtype)
    empty_weight = torch.empty((0,), device=plane.normal.device, dtype=plane.normal.dtype)

    if mask is None or source_image is None:
        return empty_uv, empty_values, empty_weight

    if mask.ndim == 3:
        mask = mask.squeeze(0)
    if source_image.ndim == 2:
        source_image = source_image.unsqueeze(0)

    device = camera.camera_center.device
    source_image = source_image.to(device=device)
    mask = mask.bool().to(device)
    idx = _sample_mask_indices(mask, max_samples=max_samples)
    if idx.shape[0] == 0:
        return empty_uv, empty_values, empty_weight

    v = idx[:, 0].float()
    u = idx[:, 1].float()
    fx, fy, cx, cy = _camera_intrinsics(camera)

    dirs_cam = torch.stack([(u - cx) / fx, (v - cy) / fy, torch.ones_like(u)], dim=-1)
    dirs_cam = F.normalize(dirs_cam, dim=-1)
    c2w = _camera_c2w(camera).to(device=device, dtype=dirs_cam.dtype)
    dirs_world = (c2w[:3, :3] @ dirs_cam.t()).t()

    origins = camera.camera_center[None].expand_as(dirs_world)
    denom = dirs_world @ plane.normal
    valid = torch.abs(denom) > 1e-7
    if valid.sum() == 0:
        return empty_uv, empty_values, empty_weight

    idx = idx[valid]
    denom = denom[valid]
    origins = origins[valid]
    dirs_world = dirs_world[valid]
    t = -(origins @ plane.normal + plane.d) / denom
    hit = t > 0
    if hit.sum() == 0:
        return empty_uv, empty_values, empty_weight

    idx = idx[hit]
    denom = denom[hit]
    points = origins[hit] + dirs_world[hit] * t[hit][:, None]
    uv = project_xyz_to_plane_uv(points, plane)
    values = source_image[:, idx[:, 0].long(), idx[:, 1].long()].permute(1, 0)
    weight = torch.clamp(torch.abs(denom), min=1e-4)
    return uv, values, weight
