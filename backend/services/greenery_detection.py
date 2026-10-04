"""No-training RGB greenery candidate extraction using an Excess Green index.

This is a color heuristic on RGB imagery, not a vegetation-health, species, or
land-cover model. Green roofs, crops, shadows, and illumination can cause errors.
"""
from __future__ import annotations

import json
import math
import os
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.enums import ColorInterp, Resampling
from rasterio.features import shapes
from rasterio.transform import Affine
import shapely.geometry
import shapely.ops

from backend.services.mask_cleanup import clean_binary_mask

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_GREENERY_OUTPUT = os.path.join(WORKSPACE_ROOT, "data", "local_model_run", "greenery", "latest.geojson")
MAX_ANALYSIS_PIXELS = 8_000_000


def _analysis_size(width: int, height: int, max_pixels: int = MAX_ANALYSIS_PIXELS) -> Tuple[int, int]:
    scale = min(1.0, math.sqrt(max_pixels / float(width * height)))
    return max(1, int(round(width * scale))), max(1, int(round(height * scale)))


def _rgb_as_unit_float(rgb: np.ndarray) -> np.ndarray:
    dtype = rgb.dtype
    values = rgb.astype(np.float32)
    if dtype.kind == "u":
        values /= float(np.iinfo(dtype).max)
    elif dtype.kind == "i":
        values = np.clip(values, 0, 255) / 255.0
    else:
        finite_max = float(np.nanmax(values)) if values.size else 1.0
        if finite_max > 1.0:
            values /= 255.0 if finite_max <= 255.0 else finite_max
    return np.clip(values, 0.0, 1.0)


def _utm_crs_for(lon: float, lat: float) -> str:
    zone = max(1, min(60, int((lon + 180.0) // 6.0) + 1))
    return f"EPSG:{32600 + zone if lat >= 0 else 32700 + zone}"


def _atomic_write_json(path: str, document: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=os.path.dirname(os.path.abspath(path)), suffix=".tmp", delete=False) as handle:
        temp_path = handle.name
        json.dump(document, handle, indent=2)
    try:
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def detect_rgb_greenery(
    raster_path: str,
    *,
    index_threshold: float = 0.15,
    min_area_sqm: float = 5.0,
    morphology_opening_px: int = 0,
    morphology_closing_px: int = 0,
    output_path: Optional[str] = DEFAULT_GREENERY_OUTPUT,
    max_analysis_pixels: int = MAX_ANALYSIS_PIXELS,
) -> Dict[str, Any]:
    """Extract candidate green pixels as polygons from a georeferenced RGB raster.

    Excess Green is computed from normalized RGB channels as ``2*g - r - b``.
    RGB is area-resampled only when the input exceeds the pixel budget. Output
    polygon coordinates are RFC 7946 longitude/latitude (EPSG:4326).
    """
    if not os.path.isfile(raster_path):
        raise FileNotFoundError(f"RGB orthomosaic not found: {raster_path}")
    if not -1.0 <= float(index_threshold) <= 2.0:
        raise ValueError("index_threshold must be between -1.0 and 2.0.")
    if not 0.0 <= float(min_area_sqm) <= 100_000.0:
        raise ValueError("min_area_sqm must be between 0 and 100000.")

    with rasterio.open(raster_path) as src:
        if src.count < 3:
            raise ValueError("Greenery detection needs at least three RGB bands.")
        if src.crs is None:
            raise ValueError("Greenery detection requires a raster with a known CRS.")
        out_width, out_height = _analysis_size(src.width, src.height, max_analysis_pixels)
        scale_x, scale_y = src.width / out_width, src.height / out_height
        rgb = src.read(
            [1, 2, 3],
            out_shape=(3, out_height, out_width),
            resampling=Resampling.average,
            masked=False,
        )
        analysis_transform = src.transform @ Affine.scale(scale_x, scale_y)
        alpha_band = next((i + 1 for i, ci in enumerate(src.colorinterp) if ci == ColorInterp.alpha), None)
        if alpha_band:
            alpha = src.read(alpha_band, out_shape=(out_height, out_width), resampling=Resampling.nearest)
            valid = alpha > 0
        else:
            valid = np.any(rgb != 0, axis=0)
        source_crs = src.crs
        source_bounds = src.bounds
        input_size = (src.width, src.height)
        input_gsd = (math.hypot(src.transform.a, src.transform.d), math.hypot(src.transform.b, src.transform.e))

    unit_rgb = _rgb_as_unit_float(rgb)
    channel_sum = unit_rgb.sum(axis=0)
    normalized = np.divide(unit_rgb, channel_sum[None, :, :], out=np.zeros_like(unit_rgb), where=channel_sum[None, :, :] > 1e-6)
    excess_green = 2.0 * normalized[1] - normalized[0] - normalized[2]
    raw_mask = (excess_green >= float(index_threshold)) & valid & (channel_sum > 0.04)
    binary_mask = clean_binary_mask(raw_mask, morphology_opening_px, morphology_closing_px)

    to_wgs84 = Transformer.from_crs(source_crs, "EPSG:4326", always_xy=True)
    center_x = (source_bounds.left + source_bounds.right) / 2.0
    center_y = (source_bounds.bottom + source_bounds.top) / 2.0
    center_lon, center_lat = to_wgs84.transform(center_x, center_y)
    area_crs = _utm_crs_for(center_lon, center_lat)
    to_area = Transformer.from_crs(source_crs, area_crs, always_xy=True)

    features: List[Dict[str, Any]] = []
    raw_polygon_count = 0
    for geom_dict, value in shapes(binary_mask, mask=(binary_mask == 1), transform=analysis_transform):
        if int(value) != 1:
            continue
        raw_polygon_count += 1
        polygon = shapely.geometry.shape(geom_dict)
        if polygon.is_empty:
            continue
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        if polygon.is_empty:
            continue
        metric_polygon = shapely.ops.transform(to_area.transform, polygon)
        area_sqm = float(metric_polygon.area)
        if area_sqm < min_area_sqm:
            continue
        green_polygon = shapely.ops.transform(to_wgs84.transform, polygon)
        if green_polygon.is_empty:
            continue
        pixel_mask = rasterio.features.rasterize(
            [geom_dict], out_shape=binary_mask.shape, transform=analysis_transform, fill=0, dtype=np.uint8
        ).astype(bool)
        mean_index = float(np.mean(excess_green[pixel_mask])) if np.any(pixel_mask) else float(index_threshold)
        feature_id = f"GREEN-EXG-{len(features) + 1:04d}"
        features.append({
            "type": "Feature",
            "id": feature_id,
            "geometry": shapely.geometry.mapping(green_polygon),
            "properties": {
                "feature_id": feature_id,
                "feature_type": "greenery_candidate",
                "source": "rgb_excess_green_heuristic",
                "method": "RGB Excess Green (2g-r-b)",
                "training_required": False,
                "area_sqm": round(area_sqm, 2),
                "mean_excess_green": round(mean_index, 4),
                "index_threshold": float(index_threshold),
                "review_status": "unverified",
            },
        })

    document: Dict[str, Any] = {
        "type": "FeatureCollection",
        "name": "rgb_excess_green_candidates",
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
        "features": features,
        "metadata": {
            "method": "RGB Excess Green heuristic (2g-r-b); no model training",
            "source_raster": os.path.basename(raster_path),
            "source_crs": str(source_crs),
            "analysis_crs": area_crs,
            "input_size_px": list(input_size),
            "analysis_size_px": [out_width, out_height],
            "source_gsd": list(input_gsd),
            "index_threshold": float(index_threshold),
            "minimum_area_sqm": float(min_area_sqm),
            "morphology_opening_px": morphology_opening_px,
            "morphology_closing_px": morphology_closing_px,
            "raw_polygon_count": raw_polygon_count,
            "candidate_count": len(features),
            "limitations": [
                "RGB-only heuristic; no near-infrared band or trained land-cover model is used.",
                "Green roofs, crops, shadows, seasonal conditions, and image color balance can produce false positives or misses.",
                "Candidates are not verified vegetation or ecological/health measurements.",
            ],
        },
    }
    if output_path:
        _atomic_write_json(output_path, document)
    return document


def load_saved_greenery(output_path: str = DEFAULT_GREENERY_OUTPUT) -> Dict[str, Any]:
    if not os.path.isfile(output_path):
        return {
            "type": "FeatureCollection",
            "name": "rgb_excess_green_candidates",
            "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
            "features": [],
            "metadata": {"status": "not_run", "candidate_count": 0},
        }
    with open(output_path, "r", encoding="utf-8") as handle:
        return json.load(handle)
