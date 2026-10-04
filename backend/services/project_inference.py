"""Project-scoped, windowed building-footprint inference."""

from __future__ import annotations

import hashlib
import glob
import importlib.util
import json
import math
import os
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pyproj
import rasterio
import shapely.geometry
import shapely.ops
from affine import Affine
from rasterio.enums import Resampling
from rasterio.features import shapes
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window, from_bounds, transform as window_transform

from backend.services.feature_store import SAVED_FINETUNED_RUN_DIR
from backend.services.mask_cleanup import clean_binary_mask
from backend.services.model_adapter import DeepLabBuildingModel
from backend.services.whu_model import MODEL_COMMIT_SHA, MODEL_REPO_ID

TILE_SIZE = 512
TILE_STRIDE = 256
PROVIDER_SPECS: Dict[str, Dict[str, Any]] = {
    "whu": {
        "label": "WHU U-Net++ (EfficientNet-B4)",
        "description": "Pretrained building-footprint segmentation model.",
        "expected_resolution_m": 0.30,
        "resolution_label": "About 30 cm per pixel",
        "caveat": "Pretrained outside this project; review every footprint against the source imagery.",
        "default_threshold": 0.50,
        "default_min_area_m2": 10.0,
    },
    "deeplab_spacenet": {
        "label": "DeepLab SpaceNet",
        "description": "DeepLabV3+ candidate trained for RGB building footprints.",
        "expected_resolution_m": 0.30,
        "resolution_label": "About 30 cm per pixel",
        "caveat": "SpaceNet-domain output can transfer poorly to other regions and sensors.",
        "default_threshold": 0.50,
        "default_min_area_m2": 10.0,
    },
    "deeplab_lalpur_finetuned": {
        "label": "Lalpur fine-tuned DeepLab (experimental)",
        "description": "DeepLabV3+ fine-tuned on the Lalpur reference footprints.",
        "expected_resolution_m": 0.30,
        "resolution_label": "About 30 cm per pixel",
        "caveat": "Fine-tuned on Lalpur's 317 reference footprints. Metrics shown for Lalpur are in-sample; performance on other imagery is unvalidated.",
        "default_threshold": 0.50,
        "default_min_area_m2": 10.0,
    },
}
FINETUNED_CHECKPOINT = os.path.join(SAVED_FINETUNED_RUN_DIR, "fine_tuned_checkpoint.pth")
VENDOR_ROOT = os.path.abspath(os.path.join("data", "local_model_run", "vendor", "rgb-footprint-extract"))
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def compute_resampling_parameters(
    source_resolution_3857_m: float,
    latitude_deg: float,
    target_resolution_m: float,
) -> Dict[str, float]:
    """Return ground GSD and source-pixel/output-pixel ratio at the raster latitude."""
    values = (source_resolution_3857_m, latitude_deg, target_resolution_m)
    if not all(math.isfinite(value) for value in values) or source_resolution_3857_m <= 0 or target_resolution_m <= 0:
        raise ValueError("Raster and target resolutions must be finite positive values.")
    if not -90 <= latitude_deg <= 90:
        raise ValueError("Raster centre latitude is outside the valid range.")
    # Web Mercator map metres are enlarged by 1/cos(latitude); ground GSD is map GSD * cos(latitude).
    ground_resolution = source_resolution_3857_m * abs(math.cos(math.radians(latitude_deg)))
    resampling_factor = ground_resolution / target_resolution_m
    return {
        "source_ground_resolution_m": ground_resolution,
        "target_resolution_m": target_resolution_m,
        "resampling_factor": resampling_factor,
        "downsample_factor": 1.0 / resampling_factor,
    }


def _cached_whu_checkpoint() -> Optional[str]:
    try:
        from huggingface_hub import try_to_load_from_cache

        path = try_to_load_from_cache(
            repo_id=MODEL_REPO_ID,
            filename="model.pth",
            revision=MODEL_COMMIT_SHA,
        )
    except Exception:
        return None
    return path if isinstance(path, str) and os.path.isfile(path) else None


def _provider_checkpoint(provider_id: str) -> Tuple[Optional[str], Optional[str]]:
    if provider_id == "whu":
        path = _cached_whu_checkpoint()
        return path, None if path else "WHU checkpoint is not present in the local Hugging Face cache."
    if provider_id == "deeplab_spacenet":
        try:
            return DeepLabBuildingModel._resolve_checkpoint_path(checkpoint_name="spacenet"), None
        except FileNotFoundError as exc:
            return None, str(exc)
    if provider_id == "deeplab_lalpur_finetuned":
        if os.path.isfile(FINETUNED_CHECKPOINT):
            return FINETUNED_CHECKPOINT, None
        return None, f"Fine-tuned checkpoint is missing: {os.path.abspath(FINETUNED_CHECKPOINT)}"
    return None, f"Unknown provider '{provider_id}'."


def _saved_runtime_sample(provider_id: str) -> Optional[Dict[str, Any]]:
    run_folder, artifact_name = {
        "whu": ("whu", "predicted_buildings_4326.geojson"),
        "deeplab_spacenet": ("deeplab-spacenet", "predictions.geojson"),
    }.get(provider_id, (None, None))
    if not run_folder:
        return None
    root = os.path.join(WORKSPACE_ROOT, "data", "local_model_run", "runs", run_folder)
    run_dirs = glob.glob(os.path.join(root, "run-*"))
    run_dirs.sort(key=lambda path: int(os.path.basename(path).rsplit("-", 1)[-1]) if os.path.basename(path).rsplit("-", 1)[-1].isdigit() else -1)
    for run_dir in reversed(run_dirs):
        artifact_path = os.path.join(run_dir, artifact_name)
        try:
            with open(artifact_path, "r", encoding="utf-8") as handle:
                metadata = json.load(handle).get("model_metadata", {})
            tile_count = int(metadata.get("tile_count", 0))
            elapsed_seconds = float(metadata.get("elapsed_seconds", 0))
            if tile_count > 0 and math.isfinite(elapsed_seconds) and elapsed_seconds > 0:
                return {
                    "seconds_per_tile": elapsed_seconds / tile_count,
                    "tile_count": tile_count,
                    "elapsed_seconds": elapsed_seconds,
                    "device": metadata.get("device"),
                    "run_id": metadata.get("run_id") or os.path.basename(run_dir),
                }
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
    return None


def get_building_model_providers() -> List[Dict[str, Any]]:
    providers = []
    for provider_id, spec in PROVIDER_SPECS.items():
        checkpoint_path, weight_error = _provider_checkpoint(provider_id)
        missing = []
        if importlib.util.find_spec("torch") is None:
            missing.append("PyTorch is not installed.")
        if provider_id == "whu":
            for module_name in ("huggingface_hub", "segmentation_models_pytorch"):
                if importlib.util.find_spec(module_name) is None:
                    missing.append(f"{module_name} is not installed.")
        else:
            if not os.path.isfile(os.path.join(VENDOR_ROOT, "models", "deeplab", "modeling", "deeplab.py")):
                missing.append(f"Pinned DeepLab model source is missing: {VENDOR_ROOT}")
        if weight_error:
            missing.append(weight_error)
        providers.append({
            "id": provider_id,
            **spec,
            "enabled": not missing,
            "reason": "; ".join(missing) if missing else None,
            "checkpoint_path": os.path.abspath(checkpoint_path) if checkpoint_path else None,
            "runtime_sample": _saved_runtime_sample(provider_id),
        })
    return providers


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_predictor(provider_id: str, checkpoint_path: str) -> Tuple[Callable[[np.ndarray], np.ndarray], str]:
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if provider_id == "whu":
        from backend.services.whu_model import smp

        model = smp.UnetPlusPlus(
            encoder_name="efficientnet-b4", encoder_weights=None, in_channels=3, classes=2
        )
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

        def predict(rgb: np.ndarray) -> np.ndarray:
            tensor = torch.from_numpy(rgb.astype(np.float32) / 255.0).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model((tensor - mean) / std)
                return torch.softmax(output, dim=1)[0, 1].cpu().numpy()
    else:
        model = DeepLabBuildingModel._load_checkpoint_model(checkpoint_path, device)

        def predict(rgb: np.ndarray) -> np.ndarray:
            tensor = torch.from_numpy(rgb.astype(np.float32) / 255.0).unsqueeze(0).to(device)
            with torch.no_grad():
                return torch.softmax(model(tensor), dim=1)[0, 1].cpu().numpy()

    return predict, str(device)


def _tile_starts(length: int) -> List[int]:
    if length <= TILE_SIZE:
        return [0]
    starts = list(range(0, length - TILE_SIZE + 1, TILE_STRIDE))
    if starts[-1] != length - TILE_SIZE:
        starts.append(length - TILE_SIZE)
    return starts


def _core_interval(starts: List[int], index: int, length: int) -> Tuple[int, int]:
    start = starts[index]
    lower = 0 if index == 0 else (starts[index - 1] + TILE_SIZE + start) // 2
    upper = length if index == len(starts) - 1 else (start + TILE_SIZE + starts[index + 1]) // 2
    return lower, min(length, upper)


def _aoi_window(aoi: Optional[Dict[str, Any]], transformer, vrt) -> Window:
    full = Window(0, 0, vrt.width, vrt.height)
    if aoi is None:
        return full
    geometry_doc = aoi.get("geometry") if aoi.get("type") == "Feature" else aoi
    if not isinstance(geometry_doc, dict):
        raise ValueError("AOI must be a GeoJSON Polygon or Feature, or null for the whole image.")
    geometry = shapely.geometry.shape(geometry_doc)
    if geometry.geom_type not in {"Polygon", "MultiPolygon"} or geometry.is_empty or not geometry.is_valid:
        raise ValueError("AOI must be a valid, non-empty GeoJSON polygon.")
    projected = shapely.ops.transform(transformer.transform, geometry)
    bounds = from_bounds(*projected.bounds, transform=vrt.transform).round_offsets().round_lengths()
    try:
        return bounds.intersection(full)
    except Exception as exc:
        raise ValueError("AOI does not intersect the project raster.") from exc


def run_project_building_inference(
    *,
    project_id: str,
    raster_path: str,
    raster_metadata: Dict[str, Any],
    provider_id: str,
    threshold: float,
    aoi: Optional[Dict[str, Any]],
    resolution_m: Optional[float],
    min_area_m2: float,
    run_dir: str,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Dict[str, Any]:
    provider = next((item for item in get_building_model_providers() if item["id"] == provider_id), None)
    if not provider:
        raise ValueError(f"Unknown building-model provider '{provider_id}'.")
    if not provider["enabled"]:
        raise RuntimeError(provider["reason"] or f"Provider '{provider_id}' is unavailable.")
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Threshold must be between 0 and 1.")
    if not math.isfinite(min_area_m2) or min_area_m2 < 0:
        raise ValueError("Minimum building area must be finite and non-negative.")

    target_gsd = float(resolution_m or provider["expected_resolution_m"])
    if not math.isfinite(target_gsd) or target_gsd <= 0:
        raise ValueError("Inference resolution must be finite and greater than zero.")
    checkpoint_path = provider["checkpoint_path"]
    if not checkpoint_path or not os.path.isfile(checkpoint_path):
        raise RuntimeError(provider["reason"] or f"Checkpoint for '{provider_id}' is missing.")

    predict, device = _load_predictor(provider_id, checkpoint_path)
    os.makedirs(run_dir, exist_ok=True)
    started = time.time()
    to_wgs84 = pyproj.Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)

    with rasterio.open(raster_path) as source:
        if source.crs is None or source.crs.to_epsg() != 3857:
            raise ValueError("Project inference requires an EPSG:3857 raster.")
        center_x = (source.bounds.left + source.bounds.right) / 2
        center_y = (source.bounds.bottom + source.bounds.top) / 2
        center_lon, center_lat = to_wgs84.transform(center_x, center_y)
        projected_gsd_x = math.hypot(source.transform.a, source.transform.d)
        projected_gsd_y = math.hypot(source.transform.b, source.transform.e)
        scale = compute_resampling_parameters(
            math.sqrt(projected_gsd_x * projected_gsd_y), center_lat, target_gsd
        )
        cos_lat = abs(math.cos(math.radians(center_lat)))
        target_map_gsd = target_gsd / max(cos_lat, 1e-8)
        vrt_width = max(1, int(math.ceil((source.bounds.right - source.bounds.left) / target_map_gsd)))
        vrt_height = max(1, int(math.ceil((source.bounds.top - source.bounds.bottom) / target_map_gsd)))
        vrt_transform = Affine(target_map_gsd, 0, source.bounds.left, 0, -target_map_gsd, source.bounds.top)
        rgb_indexes = raster_metadata.get("rgb_band_mapping") or [1, 2, 3]
        alpha_index = raster_metadata.get("alpha_band")
        warnings: List[str] = []
        if scale["source_ground_resolution_m"] > target_gsd * 1.5:
            warnings.append(
                f"Source imagery is coarser than the provider target ({scale['source_ground_resolution_m']:.3f} m vs {target_gsd:.3f} m ground GSD); results may be less reliable."
            )

        with WarpedVRT(
            source,
            crs=source.crs,
            transform=vrt_transform,
            width=vrt_width,
            height=vrt_height,
            resampling=Resampling.average,
        ) as vrt:
            aoi_transformer = pyproj.Transformer.from_crs("EPSG:4326", source.crs, always_xy=True)
            selected = _aoi_window(aoi, aoi_transformer, vrt)
            if selected.width <= 0 or selected.height <= 0:
                raise ValueError("Selected area does not overlap the project raster.")
            selected_x = int(selected.col_off)
            selected_y = int(selected.row_off)
            selected_width = int(selected.width)
            selected_height = int(selected.height)
            x_starts = _tile_starts(selected_width)
            y_starts = _tile_starts(selected_height)
            total_tiles = len(x_starts) * len(y_starts)
            utm_zone = max(1, min(60, int((center_lon + 180) // 6) + 1))
            area_crs = f"EPSG:{32600 + utm_zone if center_lat >= 0 else 32700 + utm_zone}"
            to_area = pyproj.Transformer.from_crs(source.crs, area_crs, always_xy=True)
            polygon_parts = []
            done = 0

            for y_index, local_y in enumerate(y_starts):
                core_y0, core_y1 = _core_interval(y_starts, y_index, selected_height)
                for x_index, local_x in enumerate(x_starts):
                    core_x0, core_x1 = _core_interval(x_starts, x_index, selected_width)
                    width = min(TILE_SIZE, selected_width - local_x)
                    height = min(TILE_SIZE, selected_height - local_y)
                    window = Window(selected_x + local_x, selected_y + local_y, width, height)
                    rgb = vrt.read(rgb_indexes, window=window)
                    mask = vrt.dataset_mask(window=window) > 0
                    valid = mask & np.any(rgb != 0, axis=0)
                    if alpha_index:
                        valid &= vrt.read(alpha_index, window=window) > 0
                    tile = np.zeros((3, TILE_SIZE, TILE_SIZE), dtype=np.uint8)
                    tile[:, :height, :width] = np.clip(rgb[:3], 0, 255).astype(np.uint8)
                    valid_padded = np.zeros((TILE_SIZE, TILE_SIZE), dtype=bool)
                    valid_padded[:height, :width] = valid
                    if np.any(valid):
                        probability = np.asarray(predict(tile), dtype=np.float32)
                        if probability.shape != (TILE_SIZE, TILE_SIZE):
                            raise RuntimeError(
                                f"Provider returned probability shape {probability.shape}; expected {(TILE_SIZE, TILE_SIZE)}."
                            )
                        binary = clean_binary_mask(probability >= threshold)
                        binary[~valid_padded] = 0
                        crop_x0 = core_x0 - local_x
                        crop_x1 = min(width, core_x1 - local_x)
                        crop_y0 = core_y0 - local_y
                        crop_y1 = min(height, core_y1 - local_y)
                        cropped = binary[crop_y0:crop_y1, crop_x0:crop_x1]
                        if cropped.size and np.any(cropped):
                            crop_window = Window(
                                selected_x + core_x0,
                                selected_y + core_y0,
                                crop_x1 - crop_x0,
                                crop_y1 - crop_y0,
                            )
                            crop_transform = window_transform(crop_window, vrt.transform)
                            polygon_parts.extend(
                                shapely.geometry.shape(geometry)
                                for geometry, _ in shapes(cropped.astype(np.uint8), mask=cropped.astype(bool), transform=crop_transform)
                            )
                    done += 1
                    if progress_callback:
                        progress_callback(done, total_tiles)

    combined = shapely.ops.unary_union(polygon_parts) if polygon_parts else shapely.geometry.GeometryCollection()
    polygons = list(combined.geoms) if combined.geom_type == "MultiPolygon" else ([combined] if combined.geom_type == "Polygon" else [])
    checkpoint_sha256 = _sha256(checkpoint_path)
    raster_sha256 = _sha256(raster_path)
    created_at = datetime.now(timezone.utc).isoformat()
    features = []
    for polygon in polygons:
        metric_polygon = shapely.ops.transform(to_area.transform, polygon)
        area = float(metric_polygon.area)
        if not math.isfinite(area) or area < min_area_m2:
            continue
        polygon_4326 = shapely.ops.transform(to_wgs84.transform, polygon)
        feature_id = f"AI-{provider_id}-{len(features) + 1:05d}"
        features.append({
            "type": "Feature",
            "id": feature_id,
            "geometry": shapely.geometry.mapping(polygon_4326),
            "properties": {
                "feature_id": feature_id,
                "feature_type": "building",
                "source": "ai",
                "provider": provider_id,
                "provider_label": provider["label"],
                "run_id": os.path.basename(run_dir),
                "confidence_threshold": threshold,
                "min_area_m2": min_area_m2,
                "inference_resolution_m": target_gsd,
                "area_sqm": round(area, 2),
                "created_at": created_at,
                "checkpoint_sha256": checkpoint_sha256,
                "raster_sha256": raster_sha256,
                "review_status": "unverified",
                "verification_status": "unverified",
                "notes": "AI output is a suggestion for human review.",
            },
        })

    run_id = os.path.basename(run_dir)
    metadata = {
        "run_id": run_id,
        "project_id": project_id,
        "layer_id": "ai_predictions",
        "provider_id": provider_id,
        "provider_label": provider["label"],
        "threshold": threshold,
        "aoi": aoi,
        "resolution_m": target_gsd,
        "expected_resolution_m": provider["expected_resolution_m"],
        "source_ground_resolution_m": scale["source_ground_resolution_m"],
        "resampling_factor_source_pixels_per_output_pixel": scale["resampling_factor"],
        "downsample_factor": scale["downsample_factor"],
        "scale_formula": "ground GSD = EPSG:3857 pixel size * cos(raster-centre latitude); resampling factor = source ground GSD / requested model GSD",
        "min_area_m2": min_area_m2,
        "tile_size": TILE_SIZE,
        "tile_overlap": TILE_SIZE - TILE_STRIDE,
        "tile_count": total_tiles,
        "feature_count": len(features),
        "device": device,
        "area_measurement_crs": area_crs,
        "checkpoint_path": checkpoint_path,
        "checkpoint_sha256": checkpoint_sha256,
        "raster_sha256": raster_sha256,
        "created_at": created_at,
        "elapsed_seconds": round(time.time() - started, 2),
        "warnings": warnings,
    }
    artifact = {"type": "FeatureCollection", "name": "project_ai_building_predictions", "model_metadata": metadata, "features": features}
    output_path = os.path.join(run_dir, "predictions.geojson")
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(artifact, handle, indent=2)
    with open(os.path.join(run_dir, "MODEL_RUN.md"), "w", encoding="utf-8") as handle:
        handle.write(
            f"# Project building-model run {run_id}\n\n"
            f"- Project: `{project_id}`\n- Provider: `{provider_id}`\n"
            f"- Source ground resolution: {scale['source_ground_resolution_m']:.6f} m\n"
            f"- Inference resolution: {target_gsd:.6f} m\n"
            f"- Resampling factor: {scale['resampling_factor']:.8f} source pixels per output pixel\n"
            f"- Tile size / overlap: {TILE_SIZE} / {TILE_SIZE - TILE_STRIDE}\n"
            f"- Features: {len(features)}\n- Checkpoint SHA-256: `{checkpoint_sha256}`\n"
            f"- Raster SHA-256: `{raster_sha256}`\n"
            + "".join(f"- Warning: {warning}\n" for warning in warnings)
        )
    return {"features": features, "metadata": metadata, "output_path": output_path}