#!/usr/bin/env python3
"""Local, non-destructive Lalpur building-model audit and AOI fine-tuning.

This tool is for the fixed Lalpur demonstration AOI. It does NOT infer cadastral
parcel boundaries. Fine-tuning uses the supplied Lalpur building labels, so the
resulting full-AOI metrics are explicitly in-sample and must not be presented as
held-out/generalization accuracy.

Run from the project root. Dependencies for audit mode: numpy, rasterio,
shapely, pyproj. Fine-tuning additionally requires torch, torchvision,
segmentation-models-pytorch, timm, huggingface-hub, and (for DeepLab) the pinned
vendor source plus checkpoint described in the project report.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.features import rasterize, shapes
except ImportError as exc:  # pragma: no cover - shown to local user
    raise SystemExit("Missing geospatial dependency. Activate the project's venv and install requirements.txt.") from exc

try:
    import pyproj
    import shapely.geometry as sgeom
    import shapely.ops as sops
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Missing geospatial dependency. Activate the project's venv and install requirements.txt.") from exc

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_RAW_RASTER = ROOT / "data/acquisition/SIH26012_INDIA_CANDIDATE_01/working/lalpur_orthomosaic.tif"
DEFAULT_MODEL_RASTER = ROOT / "data/local_model_run/lalpur_rgb_0.30m.tif"
DEFAULT_LABELS = ROOT / "data/acquisition/SIH26012_INDIA_CANDIDATE_01/working/sanitized_lalpur_buildings_4326.geojson"
DEFAULT_OUTPUT_ROOT = ROOT / "data/local_model_run/lalpur_improvement"
IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


@dataclass
class PreparedInput:
    image_path: Path
    rgb: np.ndarray       # uint8, C x H x W
    valid: np.ndarray     # bool, H x W
    transform: Any
    crs: Any
    width: int
    height: int
    source_path: Path
    source_resolution: list[float]
    target_resolution: list[float]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def project_paths(image_arg: str | None, labels_arg: str | None) -> tuple[Path, Path]:
    if image_arg:
        image_path = Path(image_arg).expanduser().resolve()
    elif DEFAULT_MODEL_RASTER.is_file():
        image_path = DEFAULT_MODEL_RASTER
    else:
        image_path = DEFAULT_RAW_RASTER
    labels_path = Path(labels_arg).expanduser().resolve() if labels_arg else DEFAULT_LABELS
    return image_path, labels_path


def make_model_resolution_copy(src_path: Path, out_path: Path, target_gsd: float, bands: tuple[int, int, int]) -> PreparedInput:
    """Create/validate a 3-band target-GSD working raster; never modify the source."""
    if not src_path.is_file():
        raise FileNotFoundError(f"Input raster not found: {src_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(src_path) as src:
        if src.crs is None:
            raise ValueError("Input raster has no CRS; refusing to train on unreferenced pixels.")
        if not src.crs.is_projected:
            raise ValueError(f"Expected a projected metric Lalpur raster; found {src.crs}.")
        if max(bands) > src.count or min(bands) < 1:
            raise ValueError(f"RGB band mapping {bands} is invalid for a raster with {src.count} bands.")
        # The case study is EPSG:3857; its map units are metres. This guard avoids
        # mistaking degrees or feet for metres when calculating the target image.
        unit_name = (src.crs.linear_units or "").lower()
        if unit_name not in {"metre", "meter", "metres", "meters"}:
            raise ValueError(f"Raster CRS linear units are {src.crs.linear_units!r}, not metres.")
        source_resolution = [abs(float(src.res[0])), abs(float(src.res[1]))]
        bounds_width = float(src.bounds.right - src.bounds.left)
        bounds_height = float(src.bounds.top - src.bounds.bottom)
        target_width = max(1, int(round(bounds_width / target_gsd)))
        target_height = max(1, int(round(bounds_height / target_gsd)))
        target_transform = rasterio.transform.from_bounds(
            src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top,
            target_width, target_height,
        )
        rgb = src.read(
            list(bands),
            out_shape=(3, target_height, target_width),
            resampling=Resampling.average,
        )
        valid = src.dataset_mask(
            out_shape=(target_height, target_width),
            resampling=Resampling.nearest,
        ) > 0
        if src.count >= 4 and 4 not in bands:
            alpha = src.read(4, out_shape=(target_height, target_width), resampling=Resampling.nearest)
            valid &= alpha > 0
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        valid &= np.any(rgb != 0, axis=0)
        rgb[:, ~valid] = 0
        profile = {
            "driver": "GTiff", "height": target_height, "width": target_width,
            "count": 3, "dtype": "uint8", "crs": src.crs, "transform": target_transform,
            "compress": "DEFLATE", "predictor": 2, "tiled": True, "BIGTIFF": "IF_SAFER",
            "nodata": 0,
        }
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(rgb)
            dst.write_mask(valid.astype(np.uint8) * 255)
            dst.update_tags(
                source_raster=str(src_path), target_gsd_m=str(target_gsd),
                rgb_band_mapping=",".join(map(str, bands)),
                purpose="non-destructive Lalpur model-lab working copy",
            )
        return PreparedInput(
            image_path=out_path, rgb=rgb, valid=valid, transform=target_transform, crs=src.crs,
            width=target_width, height=target_height, source_path=src_path,
            source_resolution=source_resolution,
            target_resolution=[abs(float(target_transform.a)), abs(float(target_transform.e))],
        )


def read_geojson(labels_path: Path) -> tuple[list[dict[str, Any]], list[Any]]:
    if not labels_path.is_file():
        raise FileNotFoundError(f"Building reference GeoJSON not found: {labels_path}")
    doc = json.loads(labels_path.read_text(encoding="utf-8"))
    if doc.get("type") != "FeatureCollection" or not isinstance(doc.get("features"), list):
        raise ValueError("Building labels must be a GeoJSON FeatureCollection.")
    features = []
    geometries = []
    for index, feature in enumerate(doc["features"]):
        geom = feature.get("geometry")
        if not geom:
            continue
        shape = sgeom.shape(geom)
        if not shape.is_valid:
            shape = shape.buffer(0)
        if shape.is_empty or shape.geom_type not in {"Polygon", "MultiPolygon"}:
            continue
        features.append(feature)
        geometries.append(shape)
    if not geometries:
        raise ValueError("No valid Polygon/MultiPolygon building labels were found.")
    return features, geometries


def reproject_shapes(geometries: list[Any], source_crs: Any, target_crs: Any) -> list[Any]:
    transformer = pyproj.Transformer.from_crs(source_crs, target_crs, always_xy=True)
    return [sops.transform(transformer.transform, geom) for geom in geometries]


def rasterize_reference(geometries_4326: list[Any], prepared: PreparedInput) -> np.ndarray:
    projected = reproject_shapes(geometries_4326, "EPSG:4326", prepared.crs)
    items = [(geom, 1) for geom in projected if not geom.is_empty]
    mask = rasterize(
        items,
        out_shape=(prepared.height, prepared.width),
        transform=prepared.transform,
        fill=0,
        all_touched=False,
        dtype="uint8",
    ).astype(bool)
    return mask & prepared.valid


def get_local_metric_crs(crs: Any, transform: Any, width: int, height: int) -> str:
    center_x = transform.c + width * transform.a / 2.0
    center_y = transform.f + height * transform.e / 2.0
    lon, lat = pyproj.Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform(center_x, center_y)
    zone = max(1, min(60, int((lon + 180) // 6) + 1))
    return f"EPSG:{32600 + zone if lat >= 0 else 32700 + zone}"


def build_patches(rgb: np.ndarray, mask: np.ndarray, patch_size: int, stride: int, valid: np.ndarray) -> list[tuple[int, int, bool]]:
    height, width = mask.shape
    ys = list(range(0, max(1, height - patch_size + 1), stride))
    xs = list(range(0, max(1, width - patch_size + 1), stride))
    ys += [max(0, height - patch_size)]
    xs += [max(0, width - patch_size)]
    coords = sorted({(y, x) for y in ys for x in xs})
    patches = []
    for y, x in coords:
        y2, x2 = min(height, y + patch_size), min(width, x + patch_size)
        if not valid[y:y2, x:x2].any():
            continue
        has_positive = bool(mask[y:y2, x:x2].any())
        patches.append((y, x, has_positive))
    positives = [p for p in patches if p[2]]
    negatives = [p for p in patches if not p[2]]
    if not positives:
        raise ValueError("No building-label pixels fall inside valid imagery. Check CRS, bounds, and band/nodata settings.")
    # Keep all positive patches and a bounded but broad sample of negative patches.
    rng = random.Random(26012)
    keep_negatives = min(len(negatives), max(12, len(positives)))
    if len(negatives) > keep_negatives:
        negatives = sorted(rng.sample(negatives, keep_negatives))
    return sorted(positives + negatives)


def pad_patch(array: np.ndarray, y: int, x: int, patch_size: int, fill: int | bool = 0) -> np.ndarray:
    if array.ndim == 3:
        out = np.full((array.shape[0], patch_size, patch_size), fill, dtype=array.dtype)
    else:
        out = np.full((patch_size, patch_size), fill, dtype=array.dtype)
    tile = array[..., y:min(y + patch_size, array.shape[-2]), x:min(x + patch_size, array.shape[-1])] if array.ndim == 3 else array[y:min(y + patch_size, array.shape[0]), x:min(x + patch_size, array.shape[1])]
    if array.ndim == 3:
        out[:, :tile.shape[-2], :tile.shape[-1]] = tile
    else:
        out[:tile.shape[0], :tile.shape[1]] = tile
    return out


class PatchDataset:
    def __init__(self, rgb: np.ndarray, mask: np.ndarray, patches: list[tuple[int, int, bool]], patch_size: int):
        self.rgb, self.mask, self.patches, self.patch_size = rgb, mask, patches, patch_size

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, index: int):
        import torch
        y, x, _ = self.patches[index]
        image = pad_patch(self.rgb, y, x, self.patch_size).astype(np.float32) / 255.0
        target = pad_patch(self.mask.astype(np.uint8), y, x, self.patch_size).astype(np.int64)
        if random.random() < 0.5:
            image, target = image[:, :, ::-1].copy(), target[:, ::-1].copy()
        if random.random() < 0.5:
            image, target = image[:, ::-1, :].copy(), target[::-1, :].copy()
        k = random.randint(0, 3)
        if k:
            image = np.rot90(image, k, axes=(1, 2)).copy()
            target = np.rot90(target, k, axes=(0, 1)).copy()
        # Mild radiometric augmentation; keep it deliberately modest for RGB orthomosaics.
        image = np.clip(image * random.uniform(0.9, 1.1) + random.uniform(-0.025, 0.025), 0.0, 1.0)
        return torch.from_numpy(image), torch.from_numpy(target)


def load_model(model_name: str, device: Any):
    import torch
    if model_name == "deeplab":
        from backend.services.model_adapter import DeepLabBuildingModel
        adapter = DeepLabBuildingModel(checkpoint_name="spacenet")
        checkpoint_path = adapter._resolve_checkpoint_path(checkpoint_name="spacenet")
        model = adapter._load_checkpoint_model(checkpoint_path, device)
        return model, "unit", checkpoint_path
    if model_name == "whu":
        import segmentation_models_pytorch as smp
        from backend.services.whu_model import get_whu_model_checkpoint
        checkpoint_path, checkpoint_sha, checkpoint_size = get_whu_model_checkpoint()
        model = smp.UnetPlusPlus(
            encoder_name="efficientnet-b4", encoder_weights=None, in_channels=3, classes=2,
        )
        model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
        model.to(device).eval()
        return model, "imagenet", checkpoint_path
    raise ValueError(f"Unsupported model: {model_name}")


def normalize_batch(tensor: Any, input_scale: str):
    import torch
    if input_scale == "imagenet":
        mean = torch.as_tensor(IMAGENET_MEAN, device=tensor.device).unsqueeze(0)
        std = torch.as_tensor(IMAGENET_STD, device=tensor.device).unsqueeze(0)
        return (tensor - mean) / std
    return tensor


def model_probability(model: Any, batch: Any):
    import torch
    output = model(batch)
    if isinstance(output, (tuple, list)):
        output = output[0]
    if output.ndim != 4 or output.shape[1] < 2:
        raise RuntimeError(f"Expected model logits shaped [B,2,H,W], received {tuple(output.shape)}")
    return torch.softmax(output, dim=1)[:, 1]


def infer_probability(model: Any, input_scale: str, prepared: PreparedInput, device: Any, tile_size: int, stride: int) -> np.ndarray:
    import torch
    model.eval()
    rgb = prepared.rgb
    height, width = prepared.height, prepared.width
    ys = list(range(0, max(1, height - tile_size + 1), stride)) + [max(0, height - tile_size)]
    xs = list(range(0, max(1, width - tile_size + 1), stride)) + [max(0, width - tile_size)]
    ys, xs = sorted(set(ys)), sorted(set(xs))
    if input_scale == "imagenet":
        window = np.outer(np.hanning(tile_size), np.hanning(tile_size)).astype(np.float32) + 1e-5
    else:
        # The pinned DeepLab adapter combines its overlapping tiles uniformly.
        window = np.ones((tile_size, tile_size), dtype=np.float32)
    prob_sum = np.zeros((height, width), dtype=np.float32)
    weight_sum = np.zeros((height, width), dtype=np.float32)
    with torch.no_grad():
        for y in ys:
            for x in xs:
                y2, x2 = min(height, y + tile_size), min(width, x + tile_size)
                image = np.zeros((3, tile_size, tile_size), dtype=np.float32)
                image[:, :y2-y, :x2-x] = rgb[:, y:y2, x:x2].astype(np.float32) / 255.0
                tensor = torch.from_numpy(image).unsqueeze(0).to(device)
                tensor = normalize_batch(tensor, input_scale)
                probability = model_probability(model, tensor)[0].cpu().numpy()[:y2-y, :x2-x]
                weights = window[:y2-y, :x2-x] * prepared.valid[y:y2, x:x2]
                prob_sum[y:y2, x:x2] += probability * weights
                weight_sum[y:y2, x:x2] += weights
    weight_sum[weight_sum == 0] = 1.0
    result = prob_sum / weight_sum
    result[~prepared.valid] = 0.0
    return result


def area_transformers(prepared: PreparedInput):
    metric_crs = get_local_metric_crs(prepared.crs, prepared.transform, prepared.width, prepared.height)
    to_metric = pyproj.Transformer.from_crs(prepared.crs, metric_crs, always_xy=True)
    to_wgs84 = pyproj.Transformer.from_crs(prepared.crs, "EPSG:4326", always_xy=True)
    return metric_crs, to_metric, to_wgs84


def polygonize_probability(probability: np.ndarray, prepared: PreparedInput, threshold: float, min_area_sqm: float, include_features: bool = True) -> tuple[list[dict[str, Any]], list[Any]]:
    metric_crs, to_metric, to_wgs84 = area_transformers(prepared)
    binary = (probability >= threshold) & prepared.valid
    features: list[dict[str, Any]] = []
    metric_geoms: list[Any] = []
    for geom_dict, value in shapes(binary.astype(np.uint8), mask=binary, transform=prepared.transform):
        geom = sgeom.shape(geom_dict)
        if not geom.is_valid:
            geom = geom.buffer(0)
        if geom.is_empty:
            continue
        metric = sops.transform(to_metric.transform, geom)
        area = float(metric.area)
        if area < min_area_sqm:
            continue
        if include_features:
            wgs84 = sops.transform(to_wgs84.transform, geom)
            inverse = ~prepared.transform
            col_a, row_a = inverse * (geom.bounds[0], geom.bounds[3])
            col_b, row_b = inverse * (geom.bounds[2], geom.bounds[1])
            col0 = max(0, int(math.floor(min(col_a, col_b))))
            row0 = max(0, int(math.floor(min(row_a, row_b))))
            col1 = min(prepared.width, int(math.ceil(max(col_a, col_b))))
            row1 = min(prepared.height, int(math.ceil(max(row_a, row_b))))
            confidence = 0.0
            if col1 > col0 and row1 > row0:
                from affine import Affine
                local_transform = prepared.transform * Affine.translation(col0, row0)
                local_mask = rasterize(
                    [(geom, 1)], out_shape=(row1-row0, col1-col0),
                    transform=local_transform, fill=0, dtype="uint8",
                ).astype(bool)
                local_prob = probability[row0:row1, col0:col1]
                if local_mask.any():
                    confidence = float(np.mean(local_prob[local_mask]))
            feat_id = f"AI-LALPUR-{len(features)+1:04d}"
            features.append({
                "type": "Feature",
                "id": feat_id,
                "geometry": sgeom.mapping(wgs84),
                "properties": {
                    "feature_id": feat_id,
                    "feature_type": "building",
                    "source": "lalpur_finetuned_model_candidate",
                    "verification_status": "unverified",
                    "review_status": "unverified",
                    "area_sqm": round(area, 2),
                    "confidence": round(float(np.clip(confidence, 0.0, 1.0)), 4),
                },
            })
        metric_geoms.append(metric)
    return features, metric_geoms


def object_metrics(gt_metric: list[Any], pred_metric: list[Any], iou_threshold: float) -> dict[str, Any]:
    from shapely.strtree import STRtree
    pairs = []
    tree = STRtree(gt_metric) if gt_metric else None
    for pi, pred in enumerate(pred_metric):
        candidate_indices = tree.query(pred) if tree is not None else []
        for raw_gi in candidate_indices:
            gi = int(raw_gi)
            gt = gt_metric[gi]
            if pred.is_empty or gt.is_empty:
                continue
            union = pred.union(gt).area
            iou = float(pred.intersection(gt).area / union) if union > 0 else 0.0
            pairs.append((iou, pi, gi))
    pairs.sort(key=lambda item: (-item[0], item[1], item[2]))
    used_p, used_g = set(), set()
    for iou, pi, gi in pairs:
        if iou < iou_threshold:
            break
        if pi not in used_p and gi not in used_g:
            used_p.add(pi)
            used_g.add(gi)
    tp, fp, fn = len(used_p), len(pred_metric) - len(used_p), len(gt_metric) - len(used_g)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1, "iou_threshold": iou_threshold}


def evaluate_probability(probability: np.ndarray, gt_mask: np.ndarray, gt_metric: list[Any], prepared: PreparedInput, thresholds: list[float], min_area_sqm: float, selected_threshold: float) -> dict[str, Any]:
    results = {}
    for threshold in thresholds:
        pred_mask = (probability >= threshold) & prepared.valid
        ref = gt_mask & prepared.valid
        tp_px = int(np.count_nonzero(pred_mask & ref))
        fp_px = int(np.count_nonzero(pred_mask & ~ref & prepared.valid))
        fn_px = int(np.count_nonzero(~pred_mask & ref))
        precision_px = tp_px / (tp_px + fp_px) if tp_px + fp_px else 0.0
        recall_px = tp_px / (tp_px + fn_px) if tp_px + fn_px else 0.0
        union = tp_px + fp_px + fn_px
        pixel_iou = tp_px / union if union else 0.0
        pixel_f1 = 2 * precision_px * recall_px / (precision_px + recall_px) if precision_px + recall_px else 0.0
        include_features = math.isclose(threshold, selected_threshold, abs_tol=1e-9)
        pred_features, pred_metric = polygonize_probability(probability, prepared, threshold, min_area_sqm, include_features=include_features)
        results[f"{threshold:.2f}"] = {
            "pixel": {"tp": tp_px, "fp": fp_px, "fn": fn_px, "precision": precision_px, "recall": recall_px, "iou": pixel_iou, "f1": pixel_f1},
            "polygon": object_metrics(gt_metric, pred_metric, 0.35) | {"f1_iou_0_50": object_metrics(gt_metric, pred_metric, 0.50)["f1"]},
            "prediction_count": len(pred_metric),
            "features": pred_features,
        }
    return results


def save_probability(path: Path, probability: np.ndarray, prepared: PreparedInput) -> None:
    profile = {
        "driver": "GTiff", "height": prepared.height, "width": prepared.width,
        "count": 1, "dtype": "float32", "crs": prepared.crs,
        "transform": prepared.transform, "compress": "DEFLATE", "predictor": 3,
        "tiled": True, "BIGTIFF": "IF_SAFER", "nodata": 0.0,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(probability.astype(np.float32), 1)
        dst.write_mask(prepared.valid.astype(np.uint8) * 255)


def write_feature_collection(path: Path, features: list[dict[str, Any]], metadata: dict[str, Any]) -> None:
    path.write_text(json.dumps({"type": "FeatureCollection", "name": "lalpur_finetuned_buildings", "model_metadata": metadata, "features": features}, indent=2), encoding="utf-8")


def write_audit(out_dir: Path, prepared: PreparedInput, label_path: Path, features: list[dict[str, Any]], gt_mask: np.ndarray, patches: list[tuple[int, int, bool]], target_gsd: float) -> dict[str, Any]:
    positive_pixels = int(gt_mask.sum())
    valid_pixels = int(prepared.valid.sum())
    audit = {
        "status": "ready_for_fit",
        "scope": "Lalpur building footprints only; no real parcel labels are present in this project copy",
        "source_image": str(prepared.source_path),
        "source_image_sha256": sha256_file(prepared.source_path),
        "model_input_image": str(prepared.image_path),
        "model_input_sha256": sha256_file(prepared.image_path),
        "labels": str(label_path),
        "label_feature_count": len(features),
        "valid_polygon_geometry_count": len(features),
        "crs": str(prepared.crs),
        "source_resolution_map_units": prepared.source_resolution,
        "model_resolution_map_units": prepared.target_resolution,
        "target_gsd_m": target_gsd,
        "dimensions": [prepared.width, prepared.height],
        "valid_pixel_count": valid_pixels,
        "building_label_pixel_count": positive_pixels,
        "building_label_fraction_of_valid_pixels": positive_pixels / valid_pixels if valid_pixels else 0.0,
        "training_patch_count": len(patches),
        "positive_patch_count": sum(1 for p in patches if p[2]),
        "negative_patch_count": sum(1 for p in patches if not p[2]),
        "evaluation_warning": "The local fine-tune uses the same AOI labels used for the full-scene output; report full-scene metrics as in-sample, not held-out accuracy.",
        "parcel_warning": "Building footprints are not cadastral parcels; this run cannot train or score parcel boundaries.",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    with rasterio.open(prepared.image_path) as src:
        profile = src.profile.copy()
    profile.update(driver="GTiff", count=1, dtype="uint8", compress="DEFLATE", nodata=0)
    with rasterio.open(out_dir / "reference_building_mask.tif", "w", **profile) as dst:
        dst.write(gt_mask.astype(np.uint8), 1)
        dst.write_mask(prepared.valid.astype(np.uint8) * 255)
    return audit


def train_model(model: Any, input_scale: str, prepared: PreparedInput, mask: np.ndarray, device: Any, patch_size: int, stride: int, epochs: int, batch_size: int, learning_rate: float, out_dir: Path) -> list[float]:
    import torch
    from torch.utils.data import DataLoader
    patches = build_patches(prepared.rgb, mask, patch_size, stride, prepared.valid)
    dataset = PatchDataset(prepared.rgb, mask, patches, patch_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)
    positive = max(1, int(mask.sum()))
    negative = max(1, int(prepared.valid.sum()) - positive)
    positive_weight = min(5.0, max(1.0, negative / positive))
    class_weights = torch.tensor([1.0, positive_weight], device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    history: list[float] = []
    best_loss = float("inf")
    best_state = None
    model.to(device)
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        steps = 0
        for images, targets in loader:
            images = images.to(device, non_blocking=False)
            targets = targets.to(device, non_blocking=False)
            images = normalize_batch(images, input_scale)
            logits = model(images)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            if logits.ndim != 4 or logits.shape[1] < 2:
                raise RuntimeError(f"Expected two-class logits [B,2,H,W], received {tuple(logits.shape)}")
            cross_entropy = torch.nn.functional.cross_entropy(logits, targets, weight=class_weights)
            probs = torch.softmax(logits, dim=1)[:, 1]
            target_float = targets.float()
            intersection = (probs * target_float).sum(dim=(1, 2))
            denom = probs.sum(dim=(1, 2)) + target_float.sum(dim=(1, 2))
            dice_loss = 1.0 - ((2.0 * intersection + 1.0) / (denom + 1.0)).mean()
            loss = 0.5 * cross_entropy + 0.5 * dice_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += float(loss.detach().cpu())
            steps += 1
        mean_loss = epoch_loss / max(1, steps)
        history.append(mean_loss)
        print(f"epoch {epoch + 1:02d}/{epochs}: in-sample training loss {mean_loss:.5f}", flush=True)
        if mean_loss < best_loss:
            best_loss = mean_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            torch.save({"state_dict": best_state, "epoch": epoch + 1, "training_loss": best_loss}, out_dir / "fine_tuned_checkpoint.pth")
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device).eval()
    (out_dir / "training_history.json").write_text(json.dumps({
        "epochs": epochs, "batch_size": batch_size, "learning_rate": learning_rate,
        "patch_size": patch_size, "stride": stride, "training_patches": len(dataset),
        "loss": history,
        "validation": "none; all Lalpur labels are used to adapt the model for this single AOI",
    }, indent=2), encoding="utf-8")
    return history


def clean_json_metrics(results: dict[str, Any]) -> dict[str, Any]:
    clean = {}
    for threshold, result in results.items():
        clean[threshold] = {key: value for key, value in result.items() if key != "features"}
    return clean


def main() -> int:
    parser = argparse.ArgumentParser(description="Lalpur-only audit and pretrained model fine-tuning")
    parser.add_argument("--mode", choices=("audit", "fit"), default="audit")
    parser.add_argument("--model", choices=("deeplab", "whu"), default="deeplab", help="DeepLab had the stronger recorded baseline; WHU is a fallback.")
    parser.add_argument("--image", help="Input GeoTIFF; default prefers the 0.30 m working TIFF, then the raw Lalpur TIFF.")
    parser.add_argument("--labels", help="317-building GeoJSON in EPSG:4326; defaults to the project working file.")
    parser.add_argument("--output-dir", help="Output directory; defaults to a unique run folder under data/local_model_run/lalpur_improvement.")
    parser.add_argument("--target-gsd", type=float, default=0.30)
    parser.add_argument("--bands", type=int, nargs=3, default=(1, 2, 3), metavar=("R", "G", "B"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--min-area-sqm", type=float, default=10.0)
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--inference-stride", type=int, default=0, help="Default matches provider: DeepLab=512, WHU=256. Training chip stride remains --stride.")
    parser.add_argument("--seed", type=int, default=26012)
    parser.add_argument("--assume-labels-complete", action="store_true", help="Required for fit mode. Acknowledge that training chips have sufficiently complete building labels; unlabeled image areas are treated as background.")
    args = parser.parse_args()

    if args.target_gsd <= 0 or args.epochs < 1 or args.batch_size < 1 or not 0 < args.threshold < 1:
        parser.error("target-gsd, epochs, batch-size, and threshold must be positive and valid.")
    random.seed(args.seed)
    np.random.seed(args.seed)
    image_path, labels_path = project_paths(args.image, args.labels)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else DEFAULT_OUTPUT_ROOT / f"run-{args.model}-{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    prepared_path = out_dir / "input_rgb_0.30m.tif"

    try:
        prepared = make_model_resolution_copy(image_path, prepared_path, args.target_gsd, tuple(args.bands))
        ref_features, ref_geometries = read_geojson(labels_path)
        gt_mask = rasterize_reference(ref_geometries, prepared)
        if not gt_mask.any():
            raise ValueError("Zero label pixels overlap the imagery. Check that the GeoJSON is EPSG:4326 and covers this raster.")
        patches = build_patches(prepared.rgb, gt_mask, args.patch_size, args.stride, prepared.valid)
        audit = write_audit(out_dir, prepared, labels_path, ref_features, gt_mask, patches, args.target_gsd)
        print(json.dumps({key: audit[key] for key in (
            "status", "label_feature_count", "dimensions", "source_resolution_map_units",
            "model_resolution_map_units", "building_label_fraction_of_valid_pixels",
            "training_patch_count", "positive_patch_count", "negative_patch_count",
        )}, indent=2), flush=True)
        print(f"Audit saved: {out_dir / 'audit.json'}", flush=True)
        if args.mode == "audit":
            return 0
        if not args.assume_labels_complete:
            raise ValueError("Fit mode is blocked until you visually confirm the reference labels are sufficiently complete for the training chips. Review audit.json/reference_building_mask.tif, then rerun with --assume-labels-complete. Unlabeled buildings are treated as background.")

        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("Fine-tuning needs PyTorch. Install the model dependencies in the project venv, then rerun. Audit mode succeeded without PyTorch.") from exc
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        if args.model == "deeplab" and not (ROOT / "data/local_model_run/vendor/rgb-footprint-extract").is_dir():
            raise RuntimeError("Pinned DeepLab source is missing: data/local_model_run/vendor/rgb-footprint-extract. Use --model whu or restore the pinned vendor folder and SpaceNet checkpoint.")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cpu":
            print("WARNING: CUDA is unavailable; fine-tuning will run on CPU and may take substantially longer.", flush=True)
        model, input_scale, checkpoint_path = load_model(args.model, device)
        checkpoint_hash = sha256_file(Path(checkpoint_path))
        print(f"Loaded {args.model} checkpoint ({Path(checkpoint_path).stat().st_size:,} bytes, sha256 {checkpoint_hash}) on {device}.", flush=True)

        inference_stride = args.inference_stride or (512 if args.model == "deeplab" else 256)
        if inference_stride < 1:
            parser.error("inference-stride must be a positive number of pixels.")
        thresholds = sorted(set([0.25, 0.35, 0.45, 0.50, 0.65, float(args.threshold)]))
        # Keep reference objects separate for object-level matching; the rasterized mask
        # is used independently for pixel-level metrics and the training loss.
        metric_crs, to_metric, _ = area_transformers(prepared)
        gt_projected = reproject_shapes(ref_geometries, "EPSG:4326", metric_crs)
        gt_metric = [geom for geom in gt_projected if not geom.is_empty and geom.is_valid]

        print("Running unchanged pretrained baseline inference...", flush=True)
        baseline_prob = infer_probability(model, input_scale, prepared, device, args.patch_size, inference_stride)
        baseline_results = evaluate_probability(baseline_prob, gt_mask, gt_metric, prepared, thresholds, args.min_area_sqm, args.threshold)
        save_probability(out_dir / "baseline_probability.tif", baseline_prob, prepared)
        baseline_selected = baseline_results[f"{args.threshold:.2f}"]["features"]
        write_feature_collection(out_dir / "baseline_predictions.geojson", baseline_selected, {
            "provider_id": args.model, "model_stage": "pretrained_baseline", "checkpoint_sha256": checkpoint_hash,
            "input_gsd_m": args.target_gsd, "threshold": args.threshold,
        })

        print(f"Fine-tuning on Lalpur labels ({audit['training_patch_count']} image chips; no independent validation split)...", flush=True)
        history = train_model(model, input_scale, prepared, gt_mask, device, args.patch_size, args.stride, args.epochs, args.batch_size, args.learning_rate, out_dir)
        print("Running fine-tuned full-AOI inference...", flush=True)
        finetuned_prob = infer_probability(model, input_scale, prepared, device, args.patch_size, inference_stride)
        finetuned_results = evaluate_probability(finetuned_prob, gt_mask, gt_metric, prepared, thresholds, args.min_area_sqm, args.threshold)
        save_probability(out_dir / "finetuned_probability.tif", finetuned_prob, prepared)
        selected = finetuned_results[f"{args.threshold:.2f}"]["features"]
        write_feature_collection(out_dir / "finetuned_predictions.geojson", selected, {
            "provider_id": args.model, "model_stage": "lalpur_aoi_finetuned",
            "base_checkpoint_sha256": checkpoint_hash, "fine_tuned_checkpoint": str(out_dir / "fine_tuned_checkpoint.pth"),
            "input_gsd_m": args.target_gsd, "threshold": args.threshold,
            "metric_status": "in_sample_full_AOI_fit_not_held_out",
            "warning": "Fine-tuned using all local reference buildings; metrics are for same-AOI fit only, not generalization.",
        })
        final_report = {
            "status": "completed",
            "model": args.model,
            "base_checkpoint": str(checkpoint_path),
            "base_checkpoint_sha256": checkpoint_hash,
            "input_raster_sha256": sha256_file(prepared.image_path),
            "reference_geojson_sha256": sha256_file(labels_path),
            "device": str(device),
            "target_gsd_m": args.target_gsd,
            "training_patches": audit["training_patch_count"],
            "training_stride": args.stride,
            "inference_stride": inference_stride,
            "epochs": args.epochs,
            "training_loss": history,
            "thresholds": thresholds,
            "minimum_area_sqm": args.min_area_sqm,
            "evaluation_design": "in-sample full-AOI fit; every reference feature was used for fine-tuning; do not claim held-out accuracy",
            "baseline": clean_json_metrics(baseline_results),
            "finetuned": clean_json_metrics(finetuned_results),
            "selected_threshold": args.threshold,
            "baseline_predictions": str(out_dir / "baseline_predictions.geojson"),
            "finetuned_predictions": str(out_dir / "finetuned_predictions.geojson"),
            "parcel_status": "not_evaluated: no real parcel boundaries are present; building footprints are not parcels",
        }
        (out_dir / "comparison_report.json").write_text(json.dumps(final_report, indent=2), encoding="utf-8")
        print(f"\nDone. Results are in: {out_dir}")
        print(f"Comparison report: {out_dir / 'comparison_report.json'}")
        print("Important: this is same-AOI fine-tuning, not an independent accuracy benchmark.")
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        print(f"Partial audit/output files, if any, are in: {out_dir}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
