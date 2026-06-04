import argparse
import re
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def natural_sort_key(path):
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Annotate grayscale values on the first masks in a dataset."
    )
    parser.add_argument(
        "dataset_path",
        nargs="?",
        default=None,
        help="Dataset root path. It should contain an object_mask directory.",
    )
    parser.add_argument(
        "--dataset_path",
        "--dataset-path",
        dest="dataset_path_option",
        default=None,
        help="Dataset root path. This overrides the positional dataset_path.",
    )
    parser.add_argument(
        "--num_views",
        "--num-views",
        type=int,
        default=10,
        help="Number of mask views to annotate.",
    )
    parser.add_argument(
        "--mask_dir",
        "--mask-dir",
        default="object_mask",
        help="Mask directory name under dataset_path.",
    )
    parser.add_argument(
        "--output_dir",
        "--output-dir",
        default="mask_value",
        help="Output directory name under dataset_path.",
    )
    parser.add_argument(
        "--skip_background",
        "--skip-background",
        action="store_true",
        help="Do not annotate regions with grayscale value 0.",
    )
    parser.add_argument(
        "--min_area",
        "--min-area",
        type=int,
        default=1,
        help="Ignore connected regions smaller than this many pixels.",
    )
    args = parser.parse_args()

    args.dataset_path = args.dataset_path_option or args.dataset_path
    if args.dataset_path is None:
        parser.error("dataset_path is required.")
    return args


def read_mask(mask_file):
    mask = cv2.imread(str(mask_file), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise ValueError(f"Failed to read mask: {mask_file}")

    if mask.ndim == 3:
        if mask.shape[2] == 4:
            return cv2.cvtColor(mask, cv2.COLOR_BGRA2GRAY)
        return cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    return mask


def make_visual_background(mask):
    if mask.dtype == np.uint8:
        background = mask
    else:
        background = cv2.normalize(mask, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)


def color_for_value(value):
    value = int(value)
    hue = (value * 111) % 180
    hsv = np.array([[[hue, 230, 255]]], dtype=np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def text_origin_for_center(text, center, image_shape, font, font_scale, thickness):
    image_h, image_w = image_shape[:2]
    text_w, text_h = cv2.getTextSize(text, font, font_scale, thickness)[0]
    x = int(round(center[0] - text_w / 2))
    y = int(round(center[1] + text_h / 2))
    x = max(0, min(x, image_w - text_w - 1))
    y = max(text_h + 1, min(y, image_h - 1))
    return x, y


def draw_centered_label(image, text, center, color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    min_side = min(image.shape[:2])
    font_scale = max(0.45, min(1.2, min_side / 900.0))
    thickness = max(1, int(round(font_scale * 2)))
    origin = text_origin_for_center(text, center, image.shape, font, font_scale, thickness)

    cv2.putText(
        image,
        text,
        origin,
        font,
        font_scale,
        (0, 0, 0),
        thickness + 3,
        cv2.LINE_AA,
    )
    cv2.putText(image, text, origin, font, font_scale, color, thickness, cv2.LINE_AA)


def label_point_for_region(binary):
    distance = cv2.distanceTransform(binary.astype(np.uint8), cv2.DIST_L2, 5)
    _, max_value, _, max_location = cv2.minMaxLoc(distance)
    if max_value > 0:
        return max_location

    ys, xs = np.where(binary)
    if len(xs) == 0:
        return None
    return int(round(xs.mean())), int(round(ys.mean()))


def annotate_mask(mask, skip_background=False, min_area=1):
    image = make_visual_background(mask)

    for value in np.unique(mask):
        if int(value) == 0 and skip_background:
            continue

        binary = (mask == value).astype(np.uint8)
        if int(binary.sum()) < min_area:
            continue

        center = label_point_for_region(binary)
        if center is None:
            continue

        draw_centered_label(image, str(int(value)), center, color_for_value(value))

    return image


def mask_files_in(mask_path):
    return sorted(
        (
            path
            for path in mask_path.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=natural_sort_key,
    )


def main():
    args = parse_args()

    dataset_path = Path(args.dataset_path)
    mask_path = dataset_path / args.mask_dir
    mask_value_path = dataset_path / args.output_dir

    if not mask_path.is_dir():
        raise FileNotFoundError(f"Mask directory does not exist: {mask_path}")

    mask_value_path.mkdir(parents=True, exist_ok=True)
    selected_masks = mask_files_in(mask_path)[: args.num_views]

    if not selected_masks:
        raise FileNotFoundError(f"No mask images found in: {mask_path}")

    for mask_file in selected_masks:
        mask = read_mask(mask_file)
        annotated = annotate_mask(
            mask,
            skip_background=args.skip_background,
            min_area=max(1, args.min_area),
        )
        output_file = mask_value_path / mask_file.name
        if not cv2.imwrite(str(output_file), annotated):
            raise ValueError(f"Failed to write annotated mask: {output_file}")
        print(f"Saved {output_file}")


if __name__ == "__main__":
    main()
