"""
Building Model Interface and Benchmark Dataset Adapter.
Provides:
1. BuildingModel abstract interface
2. MockBuildingModel stub provider with success and failure simulation
3. BenchmarkDatasetAdapter for SpaceNet 2 and Inria Aerial Image Labeling Dataset
   (with terms, download route, attribution, and pending evaluation harness)
4. Deterministic polygon evaluation metrics (IoU, Precision, Recall, F1)
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple, Union
import os
import json
import time
import numpy as np
import shapely.geometry
import shapely.ops
import pyproj

try:
    import rasterio
    from rasterio.features import shapes
except ImportError:  # pragma: no cover - optional dependency for raster processing
    rasterio = None
    shapes = None

class ModelInferenceError(Exception):
    """Raised when model inference fails."""
    pass

class BuildingModel(ABC):
    """Abstract interface for AI building footprint extraction models."""
    
    @abstractmethod
    def predict(
        self,
        image_or_tile: Optional[Union[str, bytes]] = None,
        bounds: Optional[Tuple[float, float, float, float]] = None,
        confidence_threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Executes inference and returns a GeoJSON FeatureCollection of building footprints.
        Features must include: source='ai_building_model', model_name, model_version, confidence.
        """
        pass

class MockBuildingModel(BuildingModel):
    """
    Mock building model provider for UI testing and developer verification.
    Generates realistic building footprint predictions in the Lalpur AOI.
    """
    def __init__(self, model_name: str = "Vaayu-UnetPP-Lite", model_version: str = "0.1.0-mock"):
        self.model_name = model_name
        self.model_version = model_version

    def predict(
        self,
        image_or_tile: Optional[Union[str, bytes]] = None,
        bounds: Optional[Tuple[float, float, float, float]] = None,
        confidence_threshold: float = 0.5,
        simulate_failure: bool = False
    ) -> Dict[str, Any]:
        if simulate_failure:
            raise ModelInferenceError(
                f"Simulated Model Failure: {self.model_name} v{self.model_version} "
                "GPU memory allocation failed or tile preprocessing timeout."
            )
            
        # Sample simulated inference footprints within Lalpur abadi
        simulated_footprints = [
            {
                "type": "Feature",
                "id": "AI-BLD-001",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [72.75685, 23.04105],
                            [72.75705, 23.04105],
                            [72.75705, 23.04122],
                            [72.75685, 23.04122],
                            [72.75685, 23.04105]
                        ]
                    ]
                },
                "properties": {
                    "feature_id": "AI-BLD-001",
                    "feature_type": "building",
                    "source": "ai_building_model",
                    "model_name": self.model_name,
                    "model_version": self.model_version,
                    "confidence": 0.92,
                    "verification_status": "unverified",
                    "review_status": "unverified",
                    "area_sqm": 41.5,
                    "notes": "Automated inference candidate via Unet++"
                }
            },
            {
                "type": "Feature",
                "id": "AI-BLD-002",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [72.75715, 23.04130],
                            [72.75738, 23.04130],
                            [72.75738, 23.04148],
                            [72.75715, 23.04148],
                            [72.75715, 23.04130]
                        ]
                    ]
                },
                "properties": {
                    "feature_id": "AI-BLD-002",
                    "feature_type": "building",
                    "source": "ai_building_model",
                    "model_name": self.model_name,
                    "model_version": self.model_version,
                    "confidence": 0.86,
                    "verification_status": "unverified",
                    "review_status": "unverified",
                    "area_sqm": 53.2,
                    "notes": "Automated inference candidate via Unet++"
                }
            },
            {
                "type": "Feature",
                "id": "AI-BLD-003",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [72.75745, 23.04080],
                            [72.75765, 23.04080],
                            [72.75765, 23.04096],
                            [72.75745, 23.04096],
                            [72.75745, 23.04080]
                        ]
                    ]
                },
                "properties": {
                    "feature_id": "AI-BLD-003",
                    "feature_type": "building",
                    "source": "ai_building_model",
                    "model_name": self.model_name,
                    "model_version": self.model_version,
                    "confidence": 0.64,  # Below 0.70 threshold to test score penalty
                    "verification_status": "unverified",
                    "review_status": "unverified",
                    "area_sqm": 35.8,
                    "notes": "Low-confidence extraction candidate"
                }
            }
        ]
        
        # Filter by confidence
        filtered = [f for f in simulated_footprints if f["properties"]["confidence"] >= confidence_threshold]
        
        return {
            "type": "FeatureCollection",
            "name": "ai_predicted_buildings",
            "model_metadata": {
                "model_name": self.model_name,
                "model_version": self.model_version,
                "inference_status": "success",
                "confidence_threshold": confidence_threshold,
                "detected_count": len(filtered)
            },
            "features": filtered
        }


class WHUBuildingModel(BuildingModel):
    """
    Live inference provider wrapping WHU Building Detection U-Net++ EfficientNet-B4.
    Executes real sliding-window inference over local Lalpur orthomosaic.
    """
    def __init__(
        self,
        model_name: str = "giswqs/whu-building-unetplusplus-efficientnet-b4",
        model_version: str = "09df9efd323bbd3d56b98b4857129eb9b5baa2d3"
    ):
        self.model_name = model_name
        self.model_version = model_version

    def predict(
        self,
        image_or_tile: Optional[Union[str, bytes]] = None,
        bounds: Optional[Tuple[float, float, float, float]] = None,
        confidence_threshold: float = 0.5,
        simulate_failure: bool = False
    ) -> Dict[str, Any]:
        if simulate_failure:
            raise ModelInferenceError(
                f"Model Failure ({self.model_name}): Simulated live model failure request."
            )

        try:
            from backend.services.whu_model import run_whu_live_inference
            res = run_whu_live_inference(confidence_threshold=confidence_threshold)
            return {
                "type": "FeatureCollection",
                "name": res.get("name", "whu_predicted_buildings"),
                "model_metadata": res.get("model_metadata", {}),
                "features": res.get("features", [])
            }
        except Exception as e:
            raise ModelInferenceError(f"Live Model Inference Error: {e}")


def stitch_probability_tiles(
    height: int,
    width: int,
    tiles: List[Tuple[int, int, np.ndarray]],
    tile_size: int = 512,
    valid_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Accumulate overlapping probability tiles while preserving valid-data masks and edge coverage."""
    prob_sum = np.zeros((height, width), dtype=np.float32)
    weight_sum = np.zeros((height, width), dtype=np.float32)
    for y, x, tile_prob in tiles:
        y2 = min(y + tile_size, height)
        x2 = min(x + tile_size, width)
        if y2 <= y or x2 <= x:
            continue
        tile_view = tile_prob[: y2 - y, : x2 - x]
        if valid_mask is not None:
            tile_valid = valid_mask[y:y2, x:x2]
        else:
            tile_valid = np.ones(tile_view.shape, dtype=bool)
        if tile_valid.size == 0:
            continue
        weights = np.where(tile_valid, 1.0, 0.0)
        prob_sum[y:y2, x:x2] += tile_view * weights
        weight_sum[y:y2, x:x2] += weights
    weight_sum[weight_sum == 0] = 1.0
    final_prob = prob_sum / weight_sum
    if valid_mask is not None:
        final_prob[~valid_mask] = 0.0
    return final_prob


def polygonize_probability_mask(
    probability: np.ndarray,
    transform,
    valid_mask: Optional[np.ndarray] = None,
    confidence_threshold: float = 0.5,
    min_area_cutoff_sqm: float = 10.0,
    crs_3857_to_4326=None,
    crs_3857_to_32643=None,
) -> List[Dict[str, Any]]:
    """Convert a binary building probability mask into GeoJSON features with project-safe area filters."""
    if shapes is None:
        return []
    binary_mask = (probability >= confidence_threshold).astype(np.uint8)
    if valid_mask is not None:
        binary_mask[~valid_mask] = 0
    if crs_3857_to_4326 is None:
        crs_3857_to_4326 = pyproj.Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    if crs_3857_to_32643 is None:
        crs_3857_to_32643 = pyproj.Transformer.from_crs("EPSG:3857", "EPSG:32643", always_xy=True)

    features: List[Dict[str, Any]] = []
    for geom_dict, _ in shapes(binary_mask, mask=(binary_mask == 1), transform=transform):
        poly_3857 = shapely.geometry.shape(geom_dict)
        if not poly_3857.is_valid:
            poly_3857 = poly_3857.buffer(0)
        if poly_3857.is_empty:
            continue
        poly_32643 = shapely.ops.transform(crs_3857_to_32643.transform, poly_3857)
        if poly_32643.area < min_area_cutoff_sqm:
            continue
        poly_4326 = shapely.ops.transform(crs_3857_to_4326.transform, poly_3857)
        if poly_4326.is_empty:
            continue
        mask_for_pixels = np.zeros_like(binary_mask, dtype=np.uint8)
        from rasterio.features import rasterize
        mask_for_pixels = rasterize([geom_dict], out_shape=mask_for_pixels.shape, transform=transform, fill=0, dtype=np.uint8)
        poly_pixels = probability[mask_for_pixels == 1]
        confidence = float(np.mean(poly_pixels)) if len(poly_pixels) > 0 else 0.75
        features.append({
            "type": "Feature",
            "geometry": shapely.geometry.mapping(poly_4326),
            "properties": {
                "area_sqm": round(float(poly_32643.area), 1),
                "confidence": round(float(np.clip(confidence, 0.0, 1.0)), 3),
            },
        })
    return features


class DeepLabBuildingModel(BuildingModel):
    """Optional second RGB footprint extraction candidate.

    This adapter is intentionally isolated from the existing WHU path. It accepts the
    model-ready Lalpur RGB GeoTIFF, preserves georeferencing, applies a valid-data mask,
    and polygonizes a probability output using the same project schema. The checkpoint is
    external and must be loaded from the documented model cache / Git LFS location; if not
    present, inference fails explicitly instead of fabricating a result.
    """

    MODEL_ID = "aatifjiwani/rgb-footprint-extract"
    REPO_URL = "https://github.com/aatifjiwani/rgb-footprint-extract"
    REVISION = "418c63b"
    MODEL_CHECKPOINTS = {
        "spacenet": "best_miou_checkpoint.pth.tar",
        "crowdai": "best_miou_checkpoint.pth.tar",
        "urban3d": "best_miou_checkpoint.pth.tar",
    }
    RASTER_PATH_CANDIDATES = [
        os.path.join("data", "local_model_run", "lalpur_rgb_0.30m.tif"),
        os.path.join("data", "acquisition", "SIH26012_INDIA_CANDIDATE_01", "working", "lalpur_orthomosaic.tif"),
    ]

    def __init__(
        self,
        model_name: str = MODEL_ID,
        model_version: str = REVISION,
        checkpoint_name: str = "spacenet",
        raster_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
    ):
        self.model_name = model_name
        self.model_version = model_version
        self.checkpoint_name = checkpoint_name
        self.raster_path = raster_path
        self.checkpoint_path = checkpoint_path

    @staticmethod
    def _resolve_raster_path(raster_path: Optional[str] = None) -> str:
        candidates = [raster_path] if raster_path else []
        candidates.extend(DeepLabBuildingModel.RASTER_PATH_CANDIDATES)
        for candidate in candidates:
            if candidate and os.path.exists(candidate):
                return candidate
        raise FileNotFoundError(
            "RGB Footprint Extract requires a model-ready Lalpur RGB GeoTIFF. "
            "Lookup attempted in data/local_model_run/lalpur_rgb_0.30m.tif and the acquisition working folder."
        )

    @staticmethod
    def _resolve_checkpoint_path(checkpoint_path: Optional[str] = None, checkpoint_name: str = "spacenet") -> str:
        candidates = []
        if checkpoint_path:
            candidates.append(checkpoint_path)

        candidate_dirs = [
            os.path.join("data", "models"),
            os.path.join(".cache", "rgb-footprint-extract"),
            os.path.join("models"),
        ]
        file_name = DeepLabBuildingModel.MODEL_CHECKPOINTS.get(checkpoint_name, DeepLabBuildingModel.MODEL_CHECKPOINTS["spacenet"])
        for root in candidate_dirs:
            for norm in [file_name, f"{checkpoint_name}_{file_name}", f"{file_name}.lfs"]:
                candidates.append(os.path.join(root, norm))

        for candidate in candidates:
            if not os.path.isfile(candidate):
                continue
            file_size = os.path.getsize(candidate)
            with open(candidate, "rb") as handle:
                header = handle.read(256)
            if file_size < 1024 or header.startswith(b"version https://git-lfs.github.com/spec/v1"):
                continue
            return candidate
        raise FileNotFoundError(
            f"DeepLab checkpoint not found for '{checkpoint_name}'. Expected one of: {list(DeepLabBuildingModel.MODEL_CHECKPOINTS.values())}. "
            "Install Git LFS and fetch the released checkpoint into the external model/cache directory."
        )

    @staticmethod
    def _prepare_valid_mask(data: np.ndarray) -> np.ndarray:
        if data.dtype.kind not in {"u", "i"} and data.dtype.kind != "f":
            data = data.astype(np.float32)
        rgb = data
        if rgb.ndim == 3 and rgb.shape[0] >= 3:
            rgb = rgb[:3]
        if rgb.ndim != 3:
            return np.ones(rgb.shape[:2], dtype=bool)
        valid = np.any(rgb != 0, axis=0)
        if rgb.shape[0] == 4:
            valid = np.logical_and(valid, rgb[3] != 0)
        return valid

    @staticmethod
    def _to_uint8_rgb(raster_array: np.ndarray) -> np.ndarray:
        arr = np.asarray(raster_array)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return arr

    @staticmethod
    def _process_raster_to_probability(
        src_path: str,
        tile_size: int = 512,
        stride: int = 512,
        min_area_cutoff_sqm: float = 10.0,
        confidence_threshold: float = 0.5,
    ) -> Tuple[np.ndarray, Any, np.ndarray, Dict[str, Any]]:
        if rasterio is None:
            raise RuntimeError("Rasterio is required for the DeepLab candidate adapter.")

        with rasterio.open(src_path) as src:
            transform = src.transform
            crs = src.crs
            rgb = src.read([1, 2, 3]) if src.count >= 3 else src.read(list(range(1, min(4, src.count) + 1)))
            if rgb.shape[0] == 1:
                rgb = np.repeat(rgb, 3, axis=0)
            rgb = np.ascontiguousarray(rgb)
            height, width = rgb.shape[1], rgb.shape[2]
            valid_mask = np.ones((height, width), dtype=bool)
            if src.count >= 4:
                alpha = src.read(4) if src.count >= 4 else np.ones((height, width), dtype=np.uint8)
                valid_mask = alpha != 0
            valid_mask = valid_mask & np.any(rgb != 0, axis=0)

            if rgb.dtype != np.uint8:
                rgb = np.clip(rgb, 0, 255).astype(np.uint8)

            rgb_uint8 = rgb.astype(np.uint8)
            rgb_uint8[:, ~valid_mask] = 0

            prob_sum = np.zeros((height, width), dtype=np.float32)
            weight_sum = np.zeros((height, width), dtype=np.float32)
            y_steps = list(range(0, max(1, height - tile_size + 1), stride))
            if not y_steps or y_steps[-1] != 0:
                y_steps.append(0)
            x_steps = list(range(0, max(1, width - tile_size + 1), stride))
            if not x_steps or x_steps[-1] != 0:
                x_steps.append(0)

            unique_y = sorted(set(y_steps + [max(0, height - tile_size)]))
            unique_x = sorted(set(x_steps + [max(0, width - tile_size)]))

            for y in unique_y:
                for x in unique_x:
                    y2 = min(y + tile_size, height)
                    x2 = min(x + tile_size, width)
                    if y2 <= y or x2 <= x:
                        continue
                    tile = rgb_uint8[:, y:y2, x:x2]
                    tile_valid = valid_mask[y:y2, x:x2]
                    if tile.size == 0 or not np.any(tile_valid):
                        continue
                    tile_prob = np.zeros((y2 - y, x2 - x), dtype=np.float32)
                    tile_prob[tile_valid] = 0.75
                    tile_prob[~tile_valid] = 0.0
                    prob_sum[y:y2, x:x2] += tile_prob
                    weight_sum[y:y2, x:x2] += np.where(tile_valid, 1.0, 0.0)

            weight_sum[weight_sum == 0] = 1.0
            final_prob = prob_sum / weight_sum
            final_prob[~valid_mask] = 0.0

            binary_mask = (final_prob >= confidence_threshold).astype(np.uint8)
            binary_mask[~valid_mask] = 0
            return final_prob, transform, binary_mask, {"crs": crs, "height": height, "width": width, "valid_mask": valid_mask}

    def predict(
        self,
        image_or_tile: Optional[Union[str, bytes]] = None,
        bounds: Optional[Tuple[float, float, float, float]] = None,
        confidence_threshold: float = 0.5,
        simulate_failure: bool = False
    ) -> Dict[str, Any]:
        if simulate_failure:
            raise ModelInferenceError(
                f"Model Failure ({self.model_name}): Simulated DeepLab candidate failure request."
            )

        try:
            raster_path = self.raster_path or self._resolve_raster_path(image_or_tile if isinstance(image_or_tile, str) else None)
            checkpoint_path = self._resolve_checkpoint_path(
                checkpoint_name=self.checkpoint_name,
                checkpoint_path=self.checkpoint_path,
            )
        except FileNotFoundError as exc:
            raise ModelInferenceError(str(exc))

        raise ModelInferenceError(
            f"DeepLab checkpoint resolved at '{checkpoint_path}', but no verified RGB Footprint Extract "
            "inference loader is installed in this workspace. Refusing to emit synthetic or WHU-derived polygons."
        )

        run_id = f"run-deeplab-{int(time.time())}"
        epsg_3857_to_4326 = pyproj.Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
        epsg_3857_to_32643 = pyproj.Transformer.from_crs("EPSG:3857", "EPSG:32643", always_xy=True)

        try:
            final_prob, transform, binary_mask, raster_info = self._process_raster_to_probability(
                src_path=raster_path,
                confidence_threshold=confidence_threshold,
            )
        except Exception as exc:
            raise ModelInferenceError(f"RGB Footprint Extract preprocessing error: {exc}")

        valid_mask = raster_info["valid_mask"]
        shapes_with_conf = []
        for geom_dict, value in shapes(binary_mask, mask=(binary_mask == 1), transform=transform):
            poly_3857 = shapely.geometry.shape(geom_dict)
            if not poly_3857.is_valid:
                poly_3857 = poly_3857.buffer(0)
            if poly_3857.is_empty:
                continue
            poly_32643 = shapely.ops.transform(epsg_3857_to_32643.transform, poly_3857)
            if poly_32643.area < 10.0:
                continue
            poly_4326 = shapely.ops.transform(epsg_3857_to_4326.transform, poly_3857)
            try:
                poly_mask = np.zeros_like(binary_mask, dtype=np.uint8)
                from rasterio.features import rasterize
                poly_mask = rasterize([geom_dict], out_shape=poly_mask.shape, transform=transform, fill=0, dtype=np.uint8)
                poly_pixels = final_prob[poly_mask == 1]
                confidence_val = float(np.mean(poly_pixels)) if len(poly_pixels) > 0 else 0.75
            except Exception:
                confidence_val = 0.75
            shapes_with_conf.append({
                "geometry": shapely.geometry.mapping(poly_4326),
                "confidence": round(float(np.clip(confidence_val, 0.0, 1.0)), 3),
                "area_sqm": round(float(poly_32643.area), 1),
            })

        features = []
        for index, item in enumerate(shapes_with_conf, start=1):
            feat_id = f"AI-DEEPLAB-{index:03d}"
            features.append({
                "type": "Feature",
                "id": feat_id,
                "geometry": item["geometry"],
                "properties": {
                    "feature_id": feat_id,
                    "feature_type": "building",
                    "source": "ai_building_model",
                    "model_name": self.model_name,
                    "model_version": self.model_version,
                    "run_id": run_id,
                    "confidence": item["confidence"],
                    "area_sqm": item["area_sqm"],
                    "verification_status": "unverified",
                    "review_status": "unverified",
                    "notes": "RGB Footprint Extract candidate output — checkpoint loaded from external cache when available",
                    "checkpoint_path": checkpoint_path,
                    "checkpoint_name": self.checkpoint_name,
                    "repository_url": self.REPO_URL,
                    "repository_revision": self.REVISION,
                    "input_raster": raster_path,
                    "alignment_status": "confirmed by user in QGIS; local reference set, not an official/legal accuracy benchmark"
                }
            })

        return {
            "type": "FeatureCollection",
            "name": "deeplab_predicted_buildings",
            "model_metadata": {
                "model_name": self.model_name,
                "model_version": self.model_version,
                "run_id": run_id,
                "checkpoint_name": self.checkpoint_name,
                "checkpoint_path": checkpoint_path,
                "repository_url": self.REPO_URL,
                "repository_revision": self.REVISION,
                "confidence_threshold": confidence_threshold,
                "input_raster_path": raster_path,
                "detected_count": len(features),
                "alignment_status": "confirmed by user in QGIS; local reference set, not an official/legal accuracy benchmark",
                "output_path": os.path.join("data", "local_model_run", "deeplab_predicted_buildings_4326.geojson"),
                "inference_status": "success" if features else "empty",
            },
            "features": features,
        }


# ==============================================================================
# Benchmark Dataset Adapters & Evaluation Harness
# ==============================================================================

class BenchmarkDatasetAdapter:
    """
    Official benchmark dataset specifications, legal reuse terms, and evaluation harness.
    Implements honest, reproducible evaluation without fabricating benchmark passes.
    """
    SUPPORTED_DATASETS = {
        "inria_aerial_image_labeling": {
            "name": "Inria Aerial Image Labeling Dataset",
            "url": "https://project.inria.fr/aerialimagelabeling/",
            "license": "Inria Non-Commercial Research License / CC BY-NC-ND",
            "resolution": "0.3 m GSD",
            "coverage": "810 km² (Austin, Chicago, Kitsap County, Tyrol, Vienna)",
            "download_method": "Manual registration and multi-GB tarball download (~21 GB total)",
            "status": "PENDING_ACQUISITION",
            "status_details": "Large multi-GB research corpus requiring manual user registration. Adapter ready."
        },
        "spacenet_2_buildings": {
            "name": "SpaceNet 2: Building Detection Dataset",
            "url": "https://spacenet.ai/spacenet-buildings-dataset-v2/",
            "license": "Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)",
            "resolution": "0.3 m WorldView-3 imagery",
            "coverage": "Las Vegas, Paris, Shanghai, Khartoum (>300,000 building footprints)",
            "download_method": "AWS S3 CLI (aws s3 sync s3://spacenet-dataset/spacenet/SN2_buildings/ ...)",
            "status": "PENDING_ACQUISITION",
            "status_details": "Requires configured AWS S3 credentials with Requester Pays. Adapter ready."
        }
    }

    @classmethod
    def get_dataset_manifest(cls) -> Dict[str, Any]:
        """Returns the registered benchmark datasets, terms, and current acquisition status."""
        return {
            "registered_datasets": cls.SUPPORTED_DATASETS,
            "disclaimer": "Benchmark evaluation execution is marked PENDING_ACQUISITION. "
                          "No benchmark pass or accuracy metrics are fabricated."
        }

    @staticmethod
    def compute_polygon_iou(poly1: shapely.geometry.Polygon, poly2: shapely.geometry.Polygon) -> float:
        """Computes geometric Intersection-over-Union (Jaccard Index) between two polygons."""
        if not poly1.is_valid:
            poly1 = poly1.buffer(0)
        if not poly2.is_valid:
            poly2 = poly2.buffer(0)
        if poly1.is_empty or poly2.is_empty:
            return 0.0
            
        inter_area = poly1.intersection(poly2).area
        union_area = poly1.union(poly2).area
        if union_area <= 0:
            return 0.0
        return inter_area / union_area

    @classmethod
    def evaluate_detections(
        cls,
        ground_truth_polygons: List[shapely.geometry.Polygon],
        predicted_polygons: List[shapely.geometry.Polygon],
        iou_threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Deterministic benchmark evaluation computing True Positives, False Positives,
        False Negatives, Precision, Recall, and F1 Score at the specified IoU threshold.
        """
        if not ground_truth_polygons:
            return {
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
                "true_positives": 0,
                "false_positives": len(predicted_polygons),
                "false_negatives": 0,
                "iou_threshold": iou_threshold
            }

        matched_gt = set()
        matched_pred = set()

        for p_idx, pred in enumerate(predicted_polygons):
            best_iou = 0.0
            best_gt_idx = -1
            for g_idx, gt in enumerate(ground_truth_polygons):
                if g_idx in matched_gt:
                    continue
                iou = cls.compute_polygon_iou(gt, pred)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = g_idx
                    
            if best_iou >= iou_threshold and best_gt_idx >= 0:
                matched_gt.add(best_gt_idx)
                matched_pred.add(p_idx)

        tp = len(matched_pred)
        fp = len(predicted_polygons) - tp
        fn = len(ground_truth_polygons) - len(matched_gt)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "total_ground_truth": len(ground_truth_polygons),
            "total_predictions": len(predicted_polygons),
            "iou_threshold": iou_threshold
        }
