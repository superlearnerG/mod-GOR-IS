from collections import deque

import torch


MIN_POSTPROCESS_OBJECT_POINTS = 64
MAX_SUPPORT_RADIUS_VOXELS = 2


def _support_offsets(radius):
    return [(dx, dy, dz)
            for dx in range(-radius, radius + 1)
            for dy in range(-radius, radius + 1)
            for dz in range(-radius, radius + 1)]


def _neighbor_offsets():
    offsets = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                offsets.append((dx, dy, dz))
    return offsets


SUPPORT_OFFSETS = {radius: _support_offsets(radius) for radius in range(MAX_SUPPORT_RADIUS_VOXELS + 1)}
NEIGHBOR_OFFSETS = _neighbor_offsets()


def _to_voxel_tuple(coord):
    return int(coord[0]), int(coord[1]), int(coord[2])


def _compute_voxel_size(object_scales, scene_extent, voxel_scale):
    median_scale = torch.median(object_scales).item()
    extent_floor = max(float(scene_extent) * 1e-3, 1e-6)
    return max(median_scale * float(voxel_scale), extent_floor)


def _build_occupied_voxels(center_voxels, support_radii):
    occupied = set()
    for center, radius in zip(center_voxels, support_radii):
        cx, cy, cz = center
        for dx, dy, dz in SUPPORT_OFFSETS[int(radius)]:
            occupied.add((cx + dx, cy + dy, cz + dz))
    return occupied


def _connected_components(occupied_voxels):
    remaining = set(occupied_voxels)
    components = []
    component_lookup = {}

    while remaining:
        seed = remaining.pop()
        queue = deque([seed])
        component = {seed}
        component_id = len(components)
        component_lookup[seed] = component_id

        while queue:
            voxel = queue.popleft()
            vx, vy, vz = voxel
            for dx, dy, dz in NEIGHBOR_OFFSETS:
                neighbor = (vx + dx, vy + dy, vz + dz)
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    component.add(neighbor)
                    component_lookup[neighbor] = component_id
                    queue.append(neighbor)

        components.append(component)

    return components, component_lookup


def _dilate_voxels(voxels, dilation_steps):
    dilated = set(voxels)
    frontier = set(voxels)

    for _ in range(max(int(dilation_steps), 0)):
        next_frontier = set()
        for vx, vy, vz in frontier:
            for dx, dy, dz in NEIGHBOR_OFFSETS:
                neighbor = (vx + dx, vy + dy, vz + dz)
                if neighbor not in dilated:
                    dilated.add(neighbor)
                    next_frontier.add(neighbor)
        frontier = next_frontier
        if not frontier:
            break

    return dilated


def postprocess_committed_object(gaussians, label, scene_extent, mask_response, config):
    stats = {
        "object_count_before": 0,
        "object_count_after": 0,
        "floaters_unassigned": 0,
        "background_pruned": 0,
        "voxel_size": None,
        "cleanup_skipped_reason": None,
    }

    object_filter = gaussians.get_object_filter(label)
    object_indices = torch.nonzero(object_filter, as_tuple=False).squeeze(1)
    object_count = int(object_indices.numel())
    stats["object_count_before"] = object_count
    stats["object_count_after"] = object_count

    if object_count < MIN_POSTPROCESS_OBJECT_POINTS:
        stats["cleanup_skipped_reason"] = "object_below_min_gaussians"
        return stats

    object_xyz = gaussians.get_xyz[object_indices].detach()
    object_scales = gaussians.get_scaling[object_indices].detach().max(dim=1).values
    voxel_size = _compute_voxel_size(object_scales, scene_extent, config.object_postprocess_voxel_scale)
    stats["voxel_size"] = float(voxel_size)

    center_voxels_tensor = torch.floor(object_xyz / voxel_size).to(torch.int64).cpu()
    center_voxels = [_to_voxel_tuple(coord.tolist()) for coord in center_voxels_tensor]
    support_radii = torch.clamp(
        torch.ceil(object_scales / voxel_size).to(torch.int64),
        min=0,
        max=MAX_SUPPORT_RADIUS_VOXELS,
    ).cpu().tolist()
    occupied_voxels = _build_occupied_voxels(center_voxels, support_radii)
    if not occupied_voxels:
        stats["cleanup_skipped_reason"] = "empty_object_occupancy"
        return stats

    components, component_lookup = _connected_components(occupied_voxels)
    if not components:
        stats["cleanup_skipped_reason"] = "no_connected_component"
        return stats

    component_counts = [0] * len(components)
    center_component_ids = []
    for center_voxel in center_voxels:
        component_id = component_lookup.get(center_voxel)
        center_component_ids.append(component_id)
        if component_id is not None:
            component_counts[component_id] += 1

    if max(component_counts) <= 0:
        stats["cleanup_skipped_reason"] = "no_center_assigned_to_component"
        return stats

    main_component_id = max(range(len(component_counts)), key=lambda idx: component_counts[idx])
    floater_local_indices = [
        local_idx for local_idx, component_id in enumerate(center_component_ids)
        if component_id != main_component_id
    ]

    if floater_local_indices:
        floater_indices = object_indices[torch.tensor(floater_local_indices, device=object_indices.device, dtype=torch.long)]
        gaussians.object_id[floater_indices] = 0
        gaussians.object_score[floater_indices] = 0
        stats["floaters_unassigned"] = int(floater_indices.numel())

    cleanup_domain = _dilate_voxels(components[main_component_id], config.object_postprocess_dilation_voxels)

    mask_response = mask_response.detach().squeeze()
    candidate_filter = torch.logical_and(
        gaussians.get_object_id == 0,
        mask_response >= config.object_postprocess_mask_thresh,
    )
    candidate_indices = torch.nonzero(candidate_filter, as_tuple=False).squeeze(1)
    if candidate_indices.numel() == 0:
        stats["object_count_after"] = int((gaussians.get_object_id == int(label)).sum().item())
        return stats

    candidate_xyz = gaussians.get_xyz[candidate_indices].detach()
    candidate_voxels_tensor = torch.floor(candidate_xyz / voxel_size).to(torch.int64).cpu()
    candidate_prune_local = []
    for local_idx, voxel_coord in enumerate(candidate_voxels_tensor.tolist()):
        if _to_voxel_tuple(voxel_coord) in cleanup_domain:
            candidate_prune_local.append(local_idx)

    if candidate_prune_local:
        prune_mask = torch.zeros((gaussians.get_xyz.shape[0],), dtype=torch.bool, device=gaussians.get_xyz.device)
        prune_indices = candidate_indices[torch.tensor(candidate_prune_local, device=candidate_indices.device, dtype=torch.long)]
        prune_mask[prune_indices] = True
        stats["background_pruned"] = int(prune_indices.numel())
        gaussians.mask_prune_points(prune_mask)

    stats["object_count_after"] = int((gaussians.get_object_id == int(label)).sum().item())
    return stats
