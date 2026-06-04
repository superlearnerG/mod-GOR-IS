from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import ssl
import sys
import tempfile
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urlparse

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

GROUPING_ROOT = REPO_ROOT.parent.parent
SHARED_TORCH_HOME_CANDIDATES = [
    GROUPING_ROOT / "compare_methods" / "pretrained_models" / "torch",
    REPO_ROOT.parent / "pretrained_models" / "torch",
]
DEFAULT_TORCH_HOME = next(
    (path for path in SHARED_TORCH_HOME_CANDIDATES if path.exists()),
    SHARED_TORCH_HOME_CANDIDATES[0],
)
os.environ["TORCH_HOME"] = str(DEFAULT_TORCH_HOME)
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import numpy as np
import torch
import torchvision.transforms.functional as tf
from PIL import Image
from tqdm import tqdm

from lpipsPyTorch.modules.lpips import LPIPS
from utils.image_utils import psnr
from utils.loss_utils import ssim


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
DEFAULT_SPLITS = ("test",)
BASE_METRIC_KEYS = ("PSNR", "SSIM", "LPIPS", "FID")
MASKED_METRIC_KEYS = ("PSNR_masked", "SSIM_masked", "LPIPS_masked", "FID_masked")
FID_LOAD_EXCEPTIONS = (
    ImportError,
    FileNotFoundError,
    URLError,
    ssl.SSLError,
    TimeoutError,
    ConnectionError,
    RuntimeError,
    ValueError,
)


def list_images(directory: str | Path) -> list[Path]:
    directory = Path(directory).expanduser().resolve()
    if not directory.is_dir():
        return []
    return sorted(
        [path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS],
        key=lambda path: path.name,
    )


def parse_mask_values(raw_values) -> list[int] | None:
    if raw_values is None:
        return None

    raw_items = raw_values if isinstance(raw_values, list) else [raw_values]
    values: list[int] = []
    for raw_value in raw_items:
        tokens = raw_value.split(",") if isinstance(raw_value, str) else [raw_value]
        for token in tokens:
            if isinstance(token, str):
                token = token.strip()
                if not token:
                    continue
            value = int(token)
            if value < 0 or value > 255:
                raise ValueError(f"Mask gray value must be in [0, 255], got {value}")
            values.append(value)
    if not values:
        raise ValueError("--mask_values was set but no valid gray values were parsed")
    return values


def unique_preserve_order(values: list[str]) -> list[str]:
    seen = set()
    ordered = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def candidate_target_names(render_root: Path, split: str, target_name: str | None) -> list[str]:
    if target_name:
        return [target_name]

    names = ["desk+background", "background", "original"]
    if render_root.is_dir():
        for path in sorted(render_root.glob("desk+background*")):
            if path.is_dir():
                names.append(path.name)
        for path in sorted(render_root.iterdir()):
            if path.is_dir():
                names.append(path.name)

        split_root = render_root / split
        if split_root.is_dir():
            for path in sorted(split_root.iterdir()):
                if path.is_dir():
                    names.append(path.name)
    return unique_preserve_order(names)


def render_kind_names(render_kind: str) -> list[str]:
    if render_kind == "auto":
        return ["pbr", "3dgs", "rgb"]
    return [render_kind]


def build_render_candidates(model_path: Path, split: str, target_name: str | None, render_kind: str) -> list[Path]:
    candidates: list[Path] = []
    for root_name in ("decouple+inpaint", "decouple"):
        render_root = model_path / root_name
        for name in candidate_target_names(render_root, split, target_name):
            candidates.append(render_root / name / split / "render")
            for kind in render_kind_names(render_kind):
                if kind == "rgb":
                    continue
                candidates.append(render_root / split / name / "render" / kind)
            candidates.append(render_root / split / name / "render")
    return unique_preserve_order([str(path) for path in candidates])


def resolve_render_dir(model_path: str | Path, split: str, target_name: str | None, render_kind: str) -> Path:
    model_path = Path(model_path).expanduser().resolve()
    candidates = [Path(path) for path in build_render_candidates(model_path, split, target_name, render_kind)]
    for path in candidates:
        if path.is_dir() and list_images(path):
            return path

    candidate_text = "\n".join(f"  {path}" for path in candidates[:40]) or "  <none>"
    raise FileNotFoundError(
        f"After-inpaint render directory not found for model_path={model_path}, split={split}.\n"
        f"Checked:\n{candidate_text}"
    )


def resolve_gt_path(gt_dir: Path, render_path: Path) -> Path:
    exact = gt_dir / render_path.name
    if exact.is_file():
        return exact

    matches = sorted(
        path
        for path in gt_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS and path.stem == render_path.stem
    )
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise RuntimeError(f"Ambiguous removal_GT match for {render_path.name}: {matches}")
    raise FileNotFoundError(f"removal_GT image not found for render {render_path.name} under {gt_dir}")


def paired_images(render_dir: str | Path, gt_dir: str | Path) -> list[tuple[Path, Path]]:
    render_dir = Path(render_dir).expanduser().resolve()
    gt_dir = Path(gt_dir).expanduser().resolve()
    if not render_dir.is_dir():
        raise FileNotFoundError(f"Render directory not found: {render_dir}")
    if not gt_dir.is_dir():
        raise FileNotFoundError(f"removal_GT directory not found: {gt_dir}")

    render_paths = list_images(render_dir)
    if not render_paths:
        raise RuntimeError(f"No rendered images found under: {render_dir}")
    return [(render_path, resolve_gt_path(gt_dir, render_path)) for render_path in render_paths]


def resolve_mask_path(mask_dir: Path, render_path: Path) -> Path:
    exact = mask_dir / render_path.name
    if exact.is_file():
        return exact

    matches = sorted(
        path
        for path in mask_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS and path.stem == render_path.stem
    )
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise RuntimeError(f"Ambiguous object_mask match for {render_path.name}: {matches}")
    raise FileNotFoundError(f"object_mask image not found for render {render_path.name} under {mask_dir}")


def load_rgb_tensor(path: str | Path, device: torch.device) -> torch.Tensor:
    with Image.open(path) as image:
        return tf.to_tensor(image.convert("RGB")).unsqueeze(0).to(device)


def load_mask_array(mask_path: Path, target_size: tuple[int, int], mask_values: list[int]) -> np.ndarray:
    with Image.open(mask_path) as image:
        mask_image = image.convert("L")
        if mask_image.size != target_size:
            mask_image = mask_image.resize(target_size, Image.Resampling.NEAREST)
        mask_np = np.array(mask_image)

    mask = np.isin(mask_np, np.array(mask_values, dtype=np.uint8))
    if not mask.any():
        raise RuntimeError(f"Mask {mask_path} has no pixels with values: {mask_values}")
    return mask


def load_mask_tensor(mask_path: Path, render_tensor: torch.Tensor, mask_values: list[int]) -> torch.Tensor:
    target_size = (int(render_tensor.shape[-1]), int(render_tensor.shape[-2]))
    mask = load_mask_array(mask_path, target_size, mask_values)
    return torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(render_tensor.device)


def masked_psnr(render_tensor: torch.Tensor, gt_tensor: torch.Tensor, mask_tensor: torch.Tensor) -> float:
    mask_pixels = mask_tensor.sum()
    if mask_pixels.item() <= 0:
        raise RuntimeError("Cannot compute masked PSNR with an empty mask")

    mse = (((render_tensor - gt_tensor) ** 2) * mask_tensor).sum() / (mask_pixels * render_tensor.shape[1])
    if mse.item() <= 0:
        return float("inf")
    return (20 * torch.log10(1.0 / torch.sqrt(mse))).item()


def crop_to_mask_bbox(tensor: torch.Tensor, mask_tensor: torch.Tensor) -> torch.Tensor:
    indices = torch.nonzero(mask_tensor[0, 0] > 0, as_tuple=False)
    if indices.numel() == 0:
        raise RuntimeError("Cannot crop an empty mask")

    y0 = int(indices[:, 0].min().item())
    y1 = int(indices[:, 0].max().item()) + 1
    x0 = int(indices[:, 1].min().item())
    x1 = int(indices[:, 1].max().item()) + 1
    return tensor[..., y0:y1, x0:x1]


def masked_local_tensors(
    render_tensor: torch.Tensor,
    gt_tensor: torch.Tensor,
    mask_tensor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    mask_bool = mask_tensor.bool()
    masked_render = torch.where(mask_bool, render_tensor, gt_tensor)
    return crop_to_mask_bbox(masked_render, mask_tensor), crop_to_mask_bbox(gt_tensor, mask_tensor)


def stage_images_for_fid(paths: list[Path], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for idx, source_path in enumerate(paths):
        suffix = source_path.suffix.lower()
        if suffix not in IMAGE_EXTENSIONS:
            suffix = ".png"
        target_path = output_dir / f"{idx:05d}{suffix}"
        try:
            os.symlink(source_path.resolve(), target_path)
        except OSError:
            shutil.copy2(source_path, target_path)


def stage_masked_images_for_fid(
    pairs: list[tuple[Path, Path]],
    mask_dir: Path,
    mask_values: list[int],
    render_output_dir: Path,
    gt_output_dir: Path,
) -> None:
    render_output_dir.mkdir(parents=True, exist_ok=True)
    gt_output_dir.mkdir(parents=True, exist_ok=True)

    for idx, (render_path, gt_path) in enumerate(pairs):
        with Image.open(render_path) as render_image, Image.open(gt_path) as gt_image:
            render_image = render_image.convert("RGB")
            gt_image = gt_image.convert("RGB")
            if render_image.size != gt_image.size:
                raise RuntimeError(
                    f"Image shape mismatch for {render_path.name}: "
                    f"render={render_image.size} gt={gt_image.size}"
                )
            mask_path = resolve_mask_path(mask_dir, render_path)
            mask = load_mask_array(mask_path, render_image.size, mask_values)
            render_np = np.array(render_image)
            gt_np = np.array(gt_image)
            masked_render_np = np.where(mask[..., None], render_np, gt_np).astype(np.uint8)

        Image.fromarray(masked_render_np).save(render_output_dir / f"{idx:05d}.png")
        Image.fromarray(gt_np).save(gt_output_dir / f"{idx:05d}.png")


def configure_fid_cache(fid_weights_url: str) -> Path:
    weights_name = Path(urlparse(fid_weights_url).path).name
    candidate_hubs = [
        *(path / "hub" for path in SHARED_TORCH_HOME_CANDIDATES),
        Path(torch.hub.get_dir()).expanduser(),
        Path.home() / ".cache" / "torch" / "hub",
    ]

    seen: set[Path] = set()
    checked: list[Path] = []
    for hub_dir in candidate_hubs:
        hub_dir = hub_dir.expanduser().resolve()
        if hub_dir in seen:
            continue
        seen.add(hub_dir)
        checkpoint_path = hub_dir / "checkpoints" / weights_name
        checked.append(checkpoint_path)
        if checkpoint_path.is_file():
            torch.hub.set_dir(str(hub_dir))
            return checkpoint_path

    fallback_hub = (DEFAULT_TORCH_HOME / "hub").expanduser().resolve()
    torch.hub.set_dir(str(fallback_hub))
    raise FileNotFoundError(
        "pytorch-fid Inception checkpoint not found. Checked:\n"
        + "\n".join(f"  {path}" for path in checked)
    )


def calculate_fid_for_pairs(
    render_paths: list[Path],
    gt_paths: list[Path],
    device: torch.device,
    batch_size: int,
    fid_dims: int,
    fid_workers: int,
    label: str,
) -> tuple[float | None, str | None]:
    try:
        from pytorch_fid import fid_score
        from pytorch_fid.inception import FID_WEIGHTS_URL
    except FID_LOAD_EXCEPTIONS as exc:
        return None, f"{type(exc).__name__}: {exc}"

    try:
        fid_checkpoint = configure_fid_cache(FID_WEIGHTS_URL)
        print(f"[{label}] Using FID checkpoint: {fid_checkpoint}")
        with tempfile.TemporaryDirectory(prefix=f"after_inpaint_{label}_fid_") as tmp_dir:
            tmp_root = Path(tmp_dir)
            render_subset = tmp_root / "renders"
            gt_subset = tmp_root / "gt"
            stage_images_for_fid(render_paths, render_subset)
            stage_images_for_fid(gt_paths, gt_subset)
            with open(os.devnull, "w", encoding="utf-8") as devnull, contextlib.redirect_stdout(devnull):
                fid = fid_score.calculate_fid_given_paths(
                    [str(gt_subset), str(render_subset)],
                    batch_size,
                    str(device),
                    fid_dims,
                    fid_workers,
                )
        return float(fid), None
    except FID_LOAD_EXCEPTIONS as exc:
        return None, f"{type(exc).__name__}: {exc}"


def calculate_masked_fid_for_pairs(
    pairs: list[tuple[Path, Path]],
    mask_dir: Path,
    mask_values: list[int],
    device: torch.device,
    batch_size: int,
    fid_dims: int,
    fid_workers: int,
    label: str,
) -> tuple[float | None, str | None]:
    try:
        from pytorch_fid import fid_score
        from pytorch_fid.inception import FID_WEIGHTS_URL
    except FID_LOAD_EXCEPTIONS as exc:
        return None, f"{type(exc).__name__}: {exc}"

    try:
        fid_checkpoint = configure_fid_cache(FID_WEIGHTS_URL)
        print(f"[{label}] Using FID checkpoint for masked metrics: {fid_checkpoint}")
        with tempfile.TemporaryDirectory(prefix=f"after_inpaint_{label}_masked_fid_") as tmp_dir:
            tmp_root = Path(tmp_dir)
            render_subset = tmp_root / "renders"
            gt_subset = tmp_root / "gt"
            stage_masked_images_for_fid(pairs, mask_dir, mask_values, render_subset, gt_subset)
            with open(os.devnull, "w", encoding="utf-8") as devnull, contextlib.redirect_stdout(devnull):
                fid = fid_score.calculate_fid_given_paths(
                    [str(gt_subset), str(render_subset)],
                    batch_size,
                    str(device),
                    fid_dims,
                    fid_workers,
                )
        return float(fid), None
    except FID_LOAD_EXCEPTIONS as exc:
        return None, f"{type(exc).__name__}: {exc}"


def compute_metrics(
    pairs: list[tuple[Path, Path]],
    device: torch.device,
    compute_fid: bool,
    batch_size: int,
    fid_dims: int,
    fid_workers: int,
    label: str,
    mask_dir: Path | None = None,
    mask_values: list[int] | None = None,
) -> tuple[dict, dict]:
    lpips_model = LPIPS("vgg").to(device).eval()
    psnr_values: list[float] = []
    ssim_values: list[float] = []
    lpips_values: list[float] = []
    psnr_masked_values: list[float] = []
    ssim_masked_values: list[float] = []
    lpips_masked_values: list[float] = []
    mask_pixel_values: list[int] = []
    compute_masked = mask_dir is not None and mask_values is not None
    per_view = {
        "PSNR": {},
        "SSIM": {},
        "LPIPS": {},
        "render_path": {},
        "gt_path": {},
    }
    if compute_masked:
        per_view.update({
            "PSNR_masked": {},
            "SSIM_masked": {},
            "LPIPS_masked": {},
            "mask_path": {},
            "mask_pixels": {},
        })

    with torch.no_grad():
        for render_path, gt_path in tqdm(pairs, desc=f"{label} metric evaluation"):
            render_tensor = load_rgb_tensor(render_path, device)
            gt_tensor = load_rgb_tensor(gt_path, device)
            if tuple(render_tensor.shape) != tuple(gt_tensor.shape):
                raise RuntimeError(
                    f"Image shape mismatch for {render_path.name}: "
                    f"render={tuple(render_tensor.shape)} gt={tuple(gt_tensor.shape)}"
                )

            image_psnr = psnr(render_tensor, gt_tensor).mean().item()
            image_ssim = ssim(render_tensor, gt_tensor).item()
            image_lpips = lpips_model(render_tensor, gt_tensor).item()
            psnr_values.append(image_psnr)
            ssim_values.append(image_ssim)
            lpips_values.append(image_lpips)

            key = render_path.name
            per_view["PSNR"][key] = image_psnr
            per_view["SSIM"][key] = image_ssim
            per_view["LPIPS"][key] = image_lpips
            per_view["render_path"][key] = str(render_path)
            per_view["gt_path"][key] = str(gt_path)

            if compute_masked:
                mask_path = resolve_mask_path(mask_dir, render_path)
                mask_tensor = load_mask_tensor(mask_path, render_tensor, mask_values)
                masked_render_tensor, masked_gt_tensor = masked_local_tensors(render_tensor, gt_tensor, mask_tensor)
                image_psnr_masked = masked_psnr(render_tensor, gt_tensor, mask_tensor)
                image_ssim_masked = ssim(masked_render_tensor, masked_gt_tensor).item()
                image_lpips_masked = lpips_model(masked_render_tensor, masked_gt_tensor).item()
                mask_pixels = int(mask_tensor.sum().item())

                psnr_masked_values.append(image_psnr_masked)
                ssim_masked_values.append(image_ssim_masked)
                lpips_masked_values.append(image_lpips_masked)
                mask_pixel_values.append(mask_pixels)
                per_view["PSNR_masked"][key] = image_psnr_masked
                per_view["SSIM_masked"][key] = image_ssim_masked
                per_view["LPIPS_masked"][key] = image_lpips_masked
                per_view["mask_path"][key] = str(mask_path)
                per_view["mask_pixels"][key] = mask_pixels

    render_paths = [render_path for render_path, _ in pairs]
    gt_paths = [gt_path for _, gt_path in pairs]
    fid = None
    fid_error = None
    if compute_fid:
        fid, fid_error = calculate_fid_for_pairs(
            render_paths,
            gt_paths,
            device,
            batch_size,
            fid_dims,
            fid_workers,
            label,
        )
        if fid_error:
            print(f"[{label}] Skipping FID: {fid_error}", file=sys.stderr)

    results = {
        "num_images": len(pairs),
        "PSNR": float(np.mean(psnr_values)),
        "SSIM": float(np.mean(ssim_values)),
        "LPIPS": float(np.mean(lpips_values)),
        "FID": fid,
        "FID_error": fid_error,
    }
    if compute_masked:
        fid_masked = None
        fid_masked_error = None
        if compute_fid:
            fid_masked, fid_masked_error = calculate_masked_fid_for_pairs(
                pairs,
                mask_dir,
                mask_values,
                device,
                batch_size,
                fid_dims,
                fid_workers,
                label,
            )
            if fid_masked_error:
                print(f"[{label}] Skipping masked FID: {fid_masked_error}", file=sys.stderr)

        results.update({
            "mask_dir": str(mask_dir),
            "mask_values": mask_values,
            "mask_num_images": len(psnr_masked_values),
            "mask_pixels_mean": float(np.mean(mask_pixel_values)),
            "PSNR_masked": float(np.mean(psnr_masked_values)),
            "SSIM_masked": float(np.mean(ssim_masked_values)),
            "LPIPS_masked": float(np.mean(lpips_masked_values)),
            "FID_masked": fid_masked,
            "FID_masked_error": fid_masked_error,
        })
    return results, per_view


def format_metric(value: float | None) -> str:
    if value is None:
        return "N/A"
    value = float(value)
    if np.isnan(value):
        return "N/A"
    if np.isposinf(value):
        return "inf"
    if np.isneginf(value):
        return "-inf"
    return f"{value:.7f}"


def write_outputs(
    output_dir: str | Path,
    metadata: dict,
    split_results: dict[str, dict],
    split_per_view: dict[str, dict],
    split_render_dirs: dict[str, Path],
) -> None:
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    json_results = {}
    for split, results in split_results.items():
        json_results[split] = {
            **metadata,
            "split": split,
            "render_dir": str(split_render_dirs[split]),
            **results,
        }

    (output_dir / "results_after_inpaint.json").write_text(
        json.dumps(json_results, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "per_view_after_inpaint.json").write_text(
        json.dumps(split_per_view, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    has_masked = any("PSNR_masked" in results for results in split_results.values())
    if has_masked:
        lines = ["method\tPSNR\tSSIM\tLPIPS\tFID\tPSNR_masked\tSSIM_masked\tLPIPS_masked\tFID_masked"]
    else:
        lines = ["method\tPSNR\tSSIM\tLPIPS\tFID"]

    for split, results in split_results.items():
        if has_masked:
            lines.append(
                f"{split}"
                f"\t{format_metric(results['PSNR'])}"
                f"\t{format_metric(results['SSIM'])}"
                f"\t{format_metric(results['LPIPS'])}"
                f"\t{format_metric(results['FID'])}"
                f"\t{format_metric(results.get('PSNR_masked'))}"
                f"\t{format_metric(results.get('SSIM_masked'))}"
                f"\t{format_metric(results.get('LPIPS_masked'))}"
                f"\t{format_metric(results.get('FID_masked'))}"
            )
        else:
            lines.append(
                f"{split}"
                f"\t{format_metric(results['PSNR'])}"
                f"\t{format_metric(results['SSIM'])}"
                f"\t{format_metric(results['LPIPS'])}"
                f"\t{format_metric(results['FID'])}"
            )

    lines.append("")
    for key, value in metadata.items():
        lines.append(f"{key}: {value}")
    for split, render_dir in split_render_dirs.items():
        lines.append(f"{split}_render_dir: {render_dir}")
        lines.append(f"{split}_num_images: {split_results[split]['num_images']}")
        if "PSNR_masked" in split_results[split]:
            lines.append(f"{split}_mask_dir: {split_results[split]['mask_dir']}")
            lines.append(f"{split}_mask_values: {split_results[split]['mask_values']}")
            lines.append(f"{split}_mask_num_images: {split_results[split]['mask_num_images']}")
            lines.append(f"{split}_mask_pixels_mean: {split_results[split]['mask_pixels_mean']:.2f}")
        if split_results[split].get("FID_error"):
            lines.append(f"{split}_FID_error: {split_results[split]['FID_error']}")
        if split_results[split].get("FID_masked_error"):
            lines.append(f"{split}_FID_masked_error: {split_results[split]['FID_masked_error']}")

    (output_dir / "qualitative_comparison_after_inpaint.txt").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def resolve_manifest_path(raw_path: str, manifest_dir: Path) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = manifest_dir / path
    return path.resolve()


def load_eval_specs(eval_list_path: str | Path, args: argparse.Namespace) -> list[dict]:
    manifest_path = Path(eval_list_path).expanduser().resolve()
    manifest_dir = manifest_path.parent
    global_mask_values = parse_mask_values(args.mask_values)
    with open(manifest_path, "r", encoding="utf-8") as file:
        payload = json.load(file)

    datasets = payload.get("datasets") if isinstance(payload, dict) else payload
    if not isinstance(datasets, list) or not datasets:
        raise RuntimeError(f"Eval list must contain a non-empty datasets list: {manifest_path}")

    specs = []
    for index, item in enumerate(datasets):
        if not isinstance(item, dict):
            raise RuntimeError(f"Dataset entry #{index} must be an object, got: {type(item).__name__}")

        name = str(item.get("name") or item.get("dataset") or f"dataset_{index + 1}")
        split = str(item.get("split") or args.splits[0])
        source_raw = item.get("source_path")
        source_path = resolve_manifest_path(source_raw, manifest_dir) if source_raw else None
        gt_raw = item.get("GTpath") or item.get("GT_path") or item.get("gt_dir") or item.get("gt_path")
        if gt_raw is not None:
            gt_dir = resolve_manifest_path(gt_raw, manifest_dir)
        elif source_path is not None:
            gt_dir = source_path / "removal_GT"
        else:
            raise RuntimeError(f"Dataset '{name}' is missing GTpath/gt_dir or source_path")

        render_raw = item.get("input_path") or item.get("render_dir") or item.get("renders_dir")
        if render_raw is not None:
            render_dir = resolve_manifest_path(render_raw, manifest_dir)
        else:
            model_raw = item.get("model_path")
            if model_raw is None:
                raise RuntimeError(f"Dataset '{name}' is missing input_path/render_dir or model_path")
            model_path = resolve_manifest_path(model_raw, manifest_dir)
            render_dir = resolve_render_dir(
                model_path,
                split,
                item.get("target_name") or args.target_name,
                item.get("render_kind") or args.render_kind,
            )

        mask_raw = item.get("mask_dir") or item.get("object_mask")
        mask_dir = resolve_manifest_path(mask_raw, manifest_dir) if mask_raw else None
        dataset_mask_values = parse_mask_values(item.get("mask_values"))
        effective_mask_values = dataset_mask_values if dataset_mask_values is not None else global_mask_values
        if effective_mask_values is not None and mask_dir is None:
            if source_path is None:
                raise RuntimeError(f"Dataset '{name}' needs mask_dir or source_path when --mask_values is set")
            mask_dir = source_path / "object_mask"
        if effective_mask_values is not None and mask_dir is not None and not mask_dir.is_dir():
            raise FileNotFoundError(f"object_mask directory not found for dataset '{name}': {mask_dir}")

        specs.append({
            "name": name,
            "split": split,
            "render_dir": render_dir,
            "gt_dir": gt_dir,
            "source_path": source_path,
            "mask_dir": mask_dir,
            "mask_values": effective_mask_values,
        })
    return specs


def valid_metric_values(dataset_results: list[dict], key: str) -> list[float]:
    values = []
    for entry in dataset_results:
        value = entry["results"].get(key)
        if value is None:
            continue
        value = float(value)
        if np.isnan(value):
            continue
        values.append(value)
    return values


def print_batch_summary(dataset_results: list[dict]) -> None:
    metric_keys = list(BASE_METRIC_KEYS)
    if any("PSNR_masked" in entry["results"] for entry in dataset_results):
        metric_keys.extend(MASKED_METRIC_KEYS)

    print("\nPer-dataset results")
    print("\t".join(["dataset", "split", "num_images", *metric_keys]))
    for entry in dataset_results:
        results = entry["results"]
        row = [entry["name"], entry["split"], str(results["num_images"])]
        row.extend(format_metric(results.get(key)) for key in metric_keys)
        print("\t".join(row))

    print("\nDataset mean")
    total = len(dataset_results)
    for key in metric_keys:
        values = valid_metric_values(dataset_results, key)
        mean_value = float(np.mean(values)) if values else None
        print(f"{key}: {format_metric(mean_value)} ({len(values)}/{total} valid)")


def run_batch_eval(args: argparse.Namespace) -> None:
    specs = load_eval_specs(args.eval_list, args)
    device = torch.device(args.device)
    dataset_results = []

    for spec in specs:
        print(f"\n[{spec['name']}]")
        print(f"Split: {spec['split']}")
        print(f"Render dir: {spec['render_dir']}")
        print(f"GT dir: {spec['gt_dir']}")
        if spec["mask_values"] is not None:
            print(f"Mask values: {spec['mask_values']}")
        pairs = paired_images(spec["render_dir"], spec["gt_dir"])
        results, _ = compute_metrics(
            pairs,
            device,
            compute_fid=not args.skip_fid,
            batch_size=args.batch_size,
            fid_dims=args.fid_dims,
            fid_workers=args.fid_workers,
            label=spec["name"],
            mask_dir=spec["mask_dir"] if spec["mask_values"] is not None else None,
            mask_values=spec["mask_values"],
        )
        dataset_results.append({**spec, "results": results})

    print_batch_summary(dataset_results)


def default_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        return Path(args.output_dir).expanduser().resolve()
    if args.model_path:
        return Path(args.model_path).expanduser().resolve()
    return Path(args.render_dir).expanduser().resolve() / "metrics_after_inpaint"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate after-inpaint render images against removal_GT using filename/stem matching."
    )
    parser.add_argument("--eval_list", default=None, help="JSON manifest for batch evaluation over multiple datasets.")
    parser.add_argument("-m", "--model_path", default=None, help="Model/output root used for automatic render-dir discovery.")
    parser.add_argument(
        "-r",
        "--render_dir",
        "--input_path",
        dest="render_dir",
        default=None,
        help="Explicit after-inpaint render directory. Overrides --model_path discovery.",
    )
    parser.add_argument("-s", "--source_path", default=None, help="Scene source path containing removal_GT.")
    parser.add_argument("--gt_dir", default=None, help="GT directory override. Default: <source_path>/removal_GT.")
    parser.add_argument("--mask_dir", default=None, help="Optional object mask directory. Default: <source_path>/object_mask.")
    parser.add_argument(
        "--mask_values",
        nargs="+",
        default=None,
        help="Object-mask grayscale values to union for masked metrics, e.g. --mask_values 17 34 or 17,34.",
    )
    parser.add_argument("--target_name", default=None, help="Render target/component name, e.g. desk+background, background, original.")
    parser.add_argument("--render_kind", choices=["auto", "pbr", "3dgs", "rgb"], default="auto")
    parser.add_argument("--splits", nargs="+", default=list(DEFAULT_SPLITS), choices=["train", "test"])
    parser.add_argument("--output_dir", default=None, help="Output directory. Default: <model_path> or <render_dir>/metrics_after_inpaint.")
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--fid_dims", type=int, default=2048)
    parser.add_argument("--fid_workers", type=int, default=0)
    parser.add_argument("--skip_fid", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.eval_list:
        return args
    if args.render_dir is None and args.model_path is None:
        parser.error("Single-dataset mode requires --render_dir/--input_path or --model_path.")
    if args.render_dir is not None and len(args.splits) != 1:
        parser.error("--render_dir can only be used with one split. Use --eval_list for multiple explicit render dirs.")
    if args.gt_dir is None and args.source_path is None:
        parser.error("Either --source_path or --gt_dir must be specified.")
    if args.mask_values is not None and args.mask_dir is None and args.source_path is None:
        parser.error("Either --source_path or --mask_dir must be specified when --mask_values is set.")
    return args


def main() -> None:
    args = parse_args()
    if args.eval_list:
        run_batch_eval(args)
        return

    source_path = Path(args.source_path).expanduser().resolve() if args.source_path else None
    gt_dir = Path(args.gt_dir).expanduser().resolve() if args.gt_dir else source_path / "removal_GT"
    mask_values = parse_mask_values(args.mask_values)
    mask_dir = None
    if mask_values is not None:
        mask_dir = Path(args.mask_dir).expanduser().resolve() if args.mask_dir else source_path / "object_mask"
        if not mask_dir.is_dir():
            raise FileNotFoundError(f"object_mask directory not found: {mask_dir}")

    output_dir = default_output_dir(args)
    device = torch.device(args.device)
    split_results: dict[str, dict] = {}
    split_per_view: dict[str, dict] = {}
    split_render_dirs: dict[str, Path] = {}
    model_path = Path(args.model_path).expanduser().resolve() if args.model_path else None

    for split in args.splits:
        render_dir = (
            Path(args.render_dir).expanduser().resolve()
            if args.render_dir
            else resolve_render_dir(model_path, split, args.target_name, args.render_kind)
        )
        pairs = paired_images(render_dir, gt_dir)
        results, per_view = compute_metrics(
            pairs,
            device,
            compute_fid=not args.skip_fid,
            batch_size=args.batch_size,
            fid_dims=args.fid_dims,
            fid_workers=args.fid_workers,
            label=split,
            mask_dir=mask_dir,
            mask_values=mask_values,
        )
        split_results[split] = results
        split_per_view[split] = per_view
        split_render_dirs[split] = render_dir

    metadata = {
        "model_path": None if model_path is None else str(model_path),
        "source_path": None if source_path is None else str(source_path),
        "gt_dir": str(gt_dir),
        "target_name": args.target_name,
        "render_kind": args.render_kind,
    }
    write_outputs(output_dir, metadata, split_results, split_per_view, split_render_dirs)

    print(f"After-inpaint metrics for splits: {', '.join(args.splits)}")
    print(f"GT dir: {gt_dir}")
    for split, results in split_results.items():
        print(f"[{split}] render dir: {split_render_dirs[split]}")
        print(f"[{split}] num_images: {results['num_images']}")
        print(f"[{split}] PSNR : {format_metric(results['PSNR'])}")
        print(f"[{split}] SSIM : {format_metric(results['SSIM'])}")
        print(f"[{split}] LPIPS: {format_metric(results['LPIPS'])}")
        print(f"[{split}] FID  : {format_metric(results['FID'])}")
        if "PSNR_masked" in results:
            print(f"[{split}] mask dir: {results['mask_dir']}")
            print(f"[{split}] mask values: {results['mask_values']}")
            print(f"[{split}] PSNR_masked : {format_metric(results['PSNR_masked'])}")
            print(f"[{split}] SSIM_masked : {format_metric(results['SSIM_masked'])}")
            print(f"[{split}] LPIPS_masked: {format_metric(results['LPIPS_masked'])}")
            print(f"[{split}] FID_masked  : {format_metric(results['FID_masked'])}")
    print(f"Wrote: {output_dir / 'qualitative_comparison_after_inpaint.txt'}")


if __name__ == "__main__":
    main()
