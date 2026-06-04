import os
from pathlib import Path

import cv2
import numpy as np
import torch


def parse_target_labels(target_labels):
    if isinstance(target_labels, (list, tuple)):
        return [int(label) for label in target_labels]
    if target_labels is None:
        return []
    target_labels = str(target_labels).strip()
    if not target_labels:
        return []
    return [int(label.strip()) for label in target_labels.split(",") if label.strip()]


class MultiLabelMaskProvider:
    def __init__(self, mask_root, train_views, target_labels="", object_order="area_desc"):
        self.mask_root = mask_root
        self.object_order = object_order
        self.area_by_label = {}
        self.available_labels = []
        self.object_labels = []
        self._raw_mask_cache = {}
        self._resized_mask_cache = {}

        self._discover_labels(train_views)
        requested_labels = parse_target_labels(target_labels)
        if requested_labels:
            unavailable = sorted(set(requested_labels) - set(self.available_labels))
            if unavailable:
                raise ValueError(f"Requested labels are not present in train masks: {unavailable}")
            labels = requested_labels
        else:
            labels = list(self.available_labels)
        self.object_labels = self._sort_labels(labels)

    def _mask_basename(self, image_name):
        return Path(image_name).stem

    def _resolve_mask_path(self, image_name):
        mask_basename = self._mask_basename(image_name)
        mask_path = os.path.join(self.mask_root, f"{mask_basename}.png")
        if not os.path.exists(mask_path):
            raise FileNotFoundError(
                f"Mask file does not exist for image '{image_name}'. "
                f"Expected basename-matched PNG mask at: {mask_path}"
            )
        return mask_path

    def _read_mask(self, mask_path):
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise ValueError(f"Failed to read mask file: {mask_path}")
        if mask.ndim == 3:
            mask = mask[..., 0]
        return mask

    def _load_raw_mask(self, image_name):
        mask_basename = self._mask_basename(image_name)
        if mask_basename not in self._raw_mask_cache:
            self._raw_mask_cache[mask_basename] = self._read_mask(self._resolve_mask_path(image_name))
        return self._raw_mask_cache[mask_basename]

    def _resize_mask(self, mask, viewpoint):
        target_shape = (int(viewpoint.image_height), int(viewpoint.image_width))
        if mask.shape != target_shape:
            mask = cv2.resize(mask, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)
        return mask

    def _discover_labels(self, train_views):
        area_by_label = {}
        for view in train_views:
            raw_mask = self._load_raw_mask(view.image_name)
            labels, counts = np.unique(raw_mask, return_counts=True)
            for label, count in zip(labels.tolist(), counts.tolist()):
                label = int(label)
                if label == 0:
                    continue
                area_by_label[label] = area_by_label.get(label, 0) + int(count)
        self.area_by_label = area_by_label
        self.available_labels = sorted(area_by_label.keys())
        if not self.available_labels:
            raise ValueError(f"No non-zero labels were found under mask root: {self.mask_root}")

    def _sort_labels(self, labels):
        labels = list(labels)
        if self.object_order == "area_desc":
            return sorted(labels, key=lambda label: (-self.area_by_label.get(int(label), 0), int(label)))
        if self.object_order == "area_asc":
            return sorted(labels, key=lambda label: (self.area_by_label.get(int(label), 0), int(label)))
        if self.object_order == "label_asc":
            return sorted(labels, key=lambda label: int(label))
        raise ValueError(f"Unsupported object order: {self.object_order}")

    def _get_resized_mask(self, viewpoint):
        cache_key = (self._mask_basename(viewpoint.image_name), int(viewpoint.image_height), int(viewpoint.image_width))
        if cache_key not in self._resized_mask_cache:
            self._resized_mask_cache[cache_key] = self._resize_mask(self._load_raw_mask(viewpoint.image_name), viewpoint)
        return self._resized_mask_cache[cache_key]

    def get_mask(self, viewpoint, label):
        binary_mask = self._get_resized_mask(viewpoint) == int(label)
        return torch.from_numpy(binary_mask.astype(np.float32)).unsqueeze(0).cuda()

    def get_label_map(self, viewpoint):
        label_map = self._get_resized_mask(viewpoint).astype(np.int64)
        return torch.from_numpy(label_map).unsqueeze(0).cuda()

    def get_object_labels(self):
        return list(self.object_labels)

    def get_area_by_label(self):
        return dict(self.area_by_label)
