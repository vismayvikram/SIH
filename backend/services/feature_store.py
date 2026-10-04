"""
Feature Store and State Manager for SIH26012 Feature Review Platform.
Maintains in-memory layer collections, handles human edits with immutable original_geometry,
dynamic re-calculation of topology warnings, review-priority scoring, and persistent disk storage.
"""
import os
import json
import hashlib
import math
import copy
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple


def scrub_nan_floats(value):
    """
    Recursively scrubs NaN, Infinity, and -Infinity from a data structure.
    Replaces NaN/inf float values with None for JSON-safe serialization.
    """
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    elif isinstance(value, dict):
        return {k: scrub_nan_floats(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [scrub_nan_floats(v) for v in value]
    return value

from backend.services.data_loader import (
    load_lalpur_buildings,
    load_lalpur_roads,
    load_osm_roads,
    load_blank_parcel_template,
    PROVENANCE_DISCLAIMER
)
from backend.services.synthetic import get_synthetic_test_features, SYNTHETIC_LAYER_NAME, SYNTHETIC_LAYER_DISCLAIMER
from backend.services.topology import run_full_topology_validation
from backend.services.scoring import calculate_review_score
from backend.models.schemas import TopologyWarning, ReviewScoreBreakdown

STATE_FILE_PATH = "data/local_model_run/store_state.json"
DEFAULT_LALPUR_PROJECT_ID = "SIH26012_INDIA_CANDIDATE_01_LALPUR"
SAVED_FINETUNED_RUN_DIR = os.path.join(
    "data", "local_model_run", "lalpur_improvement", "run-deeplab-20261004-141612"
)
SAVED_FINETUNED_LAYER = "ai_deeplab_finetuned_predictions"
PROJECT_STORES: Dict[str, "FeatureStore"] = {}


class SavedPredictionLoadError(ValueError):
    """Raised when the exact saved fine-tuned run cannot be loaded safely."""

class FeatureStore:
    """Central state manager for the local prototype with local disk persistence."""
    
    def __init__(
        self,
        state_file_path: Optional[str] = STATE_FILE_PATH,
        saved_finetuned_run_dir: str = SAVED_FINETUNED_RUN_DIR,
        project_id: Optional[str] = None,
    ):
        self.project_id = project_id or DEFAULT_LALPUR_PROJECT_ID
        self.state_file_path = state_file_path
        self.saved_finetuned_run_dir = saved_finetuned_run_dir
        self._saved_finetuned_collection: Optional[Dict[str, Any]] = None
        self.layers: Dict[str, List[Dict[str, Any]]] = self._empty_layers()
        self.warnings: List[TopologyWarning] = []
        self.scores: Dict[str, ReviewScoreBreakdown] = {}
        self.active_prediction_metadata: Optional[Dict[str, Any]] = None
        self.is_initialized = False

    @staticmethod
    def _empty_layers() -> Dict[str, List[Dict[str, Any]]]:
        return {
            "buildings": [],
            "roads": [],
            "osm_roads": [],
            "parcels": [],
            "synthetic": [],
            "drafts": [],
            "ai_predictions": [],
            "ai_whu_predictions": [],
            "ai_deeplab_predictions": [],
            "ai_mock_predictions": []
        }

    @staticmethod
    def _default_state_file_for_project(project_id: str) -> str:
        if not project_id or project_id == DEFAULT_LALPUR_PROJECT_ID:
            return STATE_FILE_PATH
        from backend.services.project_registry import project_registry
        return os.path.join(project_registry.project_storage_dir(project_id), "store_state.json")

    def initialize(self, force_reload: bool = False) -> None:
        """Loads reference layers, synthetic test fixtures, and merges persisted user state."""
        if self.is_initialized and not force_reload:
            return

        self.layers = self._empty_layers()
        self.warnings = []
        self.scores = {}
        self.active_prediction_metadata = None

        if self.project_id == DEFAULT_LALPUR_PROJECT_ID:
            # Load baseline GeoJSON files
            bld_col = load_lalpur_buildings()
            self.layers["buildings"] = bld_col["features"]

            rd_col = load_lalpur_roads()
            self.layers["roads"] = rd_col["features"]

            osm_col = load_osm_roads()
            self.layers["osm_roads"] = osm_col["features"]

            parcel_col = load_blank_parcel_template()
            self.layers["parcels"] = parcel_col["features"]  # 0 features

            # Load synthetic test fixtures
            self.layers["synthetic"] = get_synthetic_test_features()

        self._load_persisted_state()
        self.revalidate_all()
        self.is_initialized = True

    def _save_persisted_state(self) -> None:
        """Saves user edits, drafts, and AI predictions to local disk state file."""
        if not self.state_file_path:
            return
        os.makedirs(os.path.dirname(self.state_file_path), exist_ok=True)

        # Collect modified reference features
        modified_features = {}
        for layer_name in ["buildings", "roads", "osm_roads", "parcels", "synthetic"]:
            for feat in self.layers[layer_name]:
                fid = feat.get("id")
                props = feat.get("properties", {})
                orig_geom = feat.get("original_geometry")
                geom = feat.get("geometry")
                
                if props.get("review_status") != "unverified" or props.get("notes") or (orig_geom and orig_geom != geom):
                    modified_features[fid] = {
                        "properties": props,
                        "geometry": geom,
                        "original_geometry": orig_geom
                    }

        state_doc = {
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "drafts": scrub_nan_floats(self.layers["drafts"]),
            "ai_predictions": scrub_nan_floats(self.layers["ai_predictions"]),
            "active_prediction_metadata": scrub_nan_floats(self.active_prediction_metadata),
            "modified_features": scrub_nan_floats(modified_features)
        }

        with open(self.state_file_path, "w") as f:
            json.dump(state_doc, f, indent=2)

    def _load_persisted_state(self) -> None:
        """Loads and merges saved state from disk into baseline layers."""
        if not self.state_file_path or not os.path.exists(self.state_file_path):
            return

        try:
            with open(self.state_file_path, "r") as f:
                state_doc = json.load(f)

            if "drafts" in state_doc and isinstance(state_doc["drafts"], list):
                self.layers["drafts"] = state_doc["drafts"]

            # Legacy prediction arrays have no trustworthy source/run identity.
            # They must not become active merely because they exist on disk.
            prediction_metadata = state_doc.get("active_prediction_metadata")
            if isinstance(prediction_metadata, dict) and prediction_metadata.get("source_mode"):
                self.layers["ai_predictions"] = scrub_nan_floats(state_doc.get("ai_predictions", []))
                self.active_prediction_metadata = scrub_nan_floats(prediction_metadata)
                provider_layer = {
                    "whu": "ai_whu_predictions",
                    "deeplab": "ai_deeplab_predictions",
                    "mock": "ai_mock_predictions",
                }.get(prediction_metadata.get("provider_id"))
                if provider_layer:
                    self.layers[provider_layer] = copy.deepcopy(self.layers["ai_predictions"])

            modified = state_doc.get("modified_features", {})
            if isinstance(modified, dict):
                for layer_name in ["buildings", "roads", "osm_roads", "parcels", "synthetic"]:
                    for feat in self.layers[layer_name]:
                        fid = feat.get("id")
                        if fid in modified:
                            saved_mod = modified[fid]
                            if "properties" in saved_mod:
                                feat["properties"].update(saved_mod["properties"])
                            if "geometry" in saved_mod:
                                feat["geometry"] = saved_mod["geometry"]
                            if "original_geometry" in saved_mod:
                                feat["original_geometry"] = saved_mod["original_geometry"]
        except Exception as e:
            # Fallback gracefully if state file is corrupted
            pass

    def revalidate_all(self) -> None:
        """Runs topology validation and recalculates scores across all active features."""
        all_buildings = self.layers["buildings"] + self.layers["ai_predictions"]
        
        self.warnings = run_full_topology_validation(
            buildings=all_buildings,
            roads=self.layers["roads"],
            real_parcels=self.layers["parcels"],
            synthetic_features=self.layers["synthetic"]
        )

        feature_warnings: Dict[str, List[str]] = {}
        warning_by_id = {warning.warning_id: warning for warning in self.warnings}
        for warning in self.warnings:
            for feature_id in warning.feature_ids:
                feature_warnings.setdefault(feature_id, []).append(warning.warning_id)

        self.scores.clear()
        for feature_list in self.layers.values():
            for feature in feature_list:
                feature_id = feature["id"]
                warning_ids = feature_warnings.get(feature_id, [])
                feature.setdefault("properties", {})["warning_ids"] = warning_ids
                score = calculate_review_score(
                    feature,
                    associated_warning_ids=warning_ids,
                    associated_warnings=[warning_by_id[warning_id] for warning_id in warning_ids],
                )
                self.scores[feature_id] = score
                feature["properties"]["review_score"] = score.total_score

    def get_layer_manifest(self) -> List[Dict[str, Any]]:
        """Return a project-specific layer manifest for the sidebar UI."""
        self.initialize()

        if self.project_id != DEFAULT_LALPUR_PROJECT_ID and not any(
            self.layers.get(layer_name) for layer_name in [
                "buildings", "roads", "osm_roads", "parcels", "synthetic",
                "drafts", "ai_predictions", "ai_whu_predictions",
                "ai_deeplab_predictions", "ai_mock_predictions"
            ]
        ):
            return [{
                "id": "orthomosaic",
                "label": "Orthomosaic",
                "kind": "reference",
                "feature_count": 1,
                "visible_default": True,
                "read_only": True,
                "style": {"color": "#60a5fa", "fillColor": "#60a5fa", "weight": 1.5, "fillOpacity": 0.12},
                "notes": "Project raster available for review. Add/import layers as needed.",
            }]

        default_layer_specs = [
            {
                "id": "buildings",
                "label": "Building-footprint reference",
                "kind": "reference",
                "visible_default": True,
                "read_only": True,
                "style": {"color": "#b48954", "fillColor": "#b48954", "weight": 1.5, "fillOpacity": 0.35},
                "notes": "Reference building footprints",
            },
            {
                "id": "roads",
                "label": "Village Road Corridors",
                "kind": "reference",
                "visible_default": True,
                "read_only": True,
                "style": {"color": "#f59e0b", "fillColor": "#f59e0b", "weight": 2, "fillOpacity": 0.25},
                "notes": "Road corridor reference data",
            },
            {
                "id": "osm_roads",
                "label": "OSM Road Centerlines",
                "kind": "reference",
                "visible_default": True,
                "read_only": True,
                "style": {"color": "#7c7b5d", "weight": 3, "dashArray": "6, 6"},
                "notes": "OpenStreetMap reference lines",
            },
            {
                "id": "parcels",
                "label": "Cadastral Parcels",
                "kind": "reference",
                "visible_default": False,
                "read_only": True,
                "style": {"color": "#64748b", "fillColor": "#64748b", "weight": 1, "fillOpacity": 0.12},
                "notes": "Empty parcel template for this AOI",
            },
            {
                "id": "synthetic",
                "label": "Synthetic test data",
                "kind": "review",
                "visible_default": True,
                "read_only": False,
                "style": {"color": "#b9895b", "fillColor": "#b9895b", "weight": 2, "fillOpacity": 0.3},
                "notes": "Synthetic test fixtures; not real parcels",
            },
            {
                "id": "drafts",
                "label": "User Review Drafts",
                "kind": "draft",
                "visible_default": True,
                "read_only": False,
                "style": {"color": "#10b981", "fillColor": "#10b981", "weight": 2, "fillOpacity": 0.36},
                "notes": "Digitized draft polygons",
            },
            {
                "id": "ai_predictions",
                "label": "AI Building Model",
                "kind": "ai",
                "visible_default": True,
                "read_only": False,
                "style": {"color": "#38bdf8", "fillColor": "#38bdf8", "weight": 2, "fillOpacity": 0.22},
                "notes": "Current active AI predictions",
            },
            {
                "id": "ai_whu_predictions",
                "label": "WHU U-Net++ Footprints",
                "kind": "ai",
                "visible_default": True,
                "read_only": False,
                "style": {"color": "#38bdf8", "fillColor": "#38bdf8", "weight": 2, "fillOpacity": 0.2},
                "notes": "WHU provider output",
            },
            {
                "id": "ai_deeplab_predictions",
                "label": "DeepLab SpaceNet Footprints",
                "kind": "ai",
                "visible_default": True,
                "read_only": False,
                "style": {"color": "#f59e0b", "fillColor": "#f59e0b", "weight": 2, "fillOpacity": 0.2},
                "notes": "DeepLab provider output",
            },
            {
                "id": "ai_deeplab_finetuned_predictions",
                "label": "DeepLab · Lalpur fine-tuned",
                "kind": "ai",
                "visible_default": False,
                "read_only": True,
                "style": {"color": "#e11d48", "fillColor": "#e11d48", "weight": 2, "fillOpacity": 0.16, "dashArray": "5 4"},
                "notes": "Experimental in-sample fine-tuned layer",
            },
            {
                "id": "ai_mock_predictions",
                "label": "Mock Footprints",
                "kind": "ai",
                "visible_default": False,
                "read_only": False,
                "style": {"color": "#a855f7", "fillColor": "#a855f7", "weight": 2, "fillOpacity": 0.18},
                "notes": "Demo provider output",
            },
        ]

        manifest = []
        for spec in default_layer_specs:
            layer_name = spec["id"]
            if layer_name not in self.layers:
                continue
            if layer_name == "ai_deeplab_finetuned_predictions" and not self.layers.get(layer_name):
                continue
            manifest.append({
                **spec,
                "feature_count": len(self.layers.get(layer_name, [])),
                "visible_default": bool(spec["visible_default"]),
            })
        return manifest

    def get_layer_collection(self, layer_name: str) -> Dict[str, Any]:
        """Returns GeoJSON FeatureCollection representation for the requested layer."""
        self.initialize()
        if layer_name == SAVED_FINETUNED_LAYER:
            return copy.deepcopy(self.load_saved_finetuned_collection())
        if layer_name not in self.layers:
            raise KeyError(f"Unknown layer: {layer_name}")

        features = self.layers[layer_name]
        metadata: Dict[str, Any] = {
            "total_features": len(features),
            "crs": "EPSG:4326 (WGS 84 / RFC 7946)"
        }

        if layer_name == "buildings":
            name = "sanitized_lalpur_buildings_4326"
            metadata["provenance"] = PROVENANCE_DISCLAIMER
            metadata["verification_status"] = "unverified"
        elif layer_name == "roads":
            name = "sanitized_lalpur_road_polygons_4326"
            metadata["provenance"] = PROVENANCE_DISCLAIMER
        elif layer_name == "osm_roads":
            name = "osm_roads_lalpur_4326"
            metadata["attribution"] = "© OpenStreetMap contributors"
            metadata["license"] = "ODbL 1.0"
        elif layer_name == "parcels":
            name = "blank_parcel_template_4326"
            metadata["disposition"] = "NO REAL PARCEL POLYGONS EXIST FOR THIS AOI"
            metadata["comment"] = "Zero features. Empty schema template."
        elif layer_name == "synthetic":
            name = SYNTHETIC_LAYER_NAME
            metadata["disclaimer"] = SYNTHETIC_LAYER_DISCLAIMER
            metadata["source"] = "synthetic_test"
        elif layer_name == "drafts":
            name = "user_review_drafts"
            metadata["source"] = "manual_visual_reference"
        elif layer_name == "ai_predictions":
            name = "ai_predicted_features"
            metadata["source"] = "ai_building_model"
        else:
            name = layer_name

        return {
            "type": "FeatureCollection",
            "name": name,
            "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
            "total_features": len(features),
            "metadata": metadata,
            "features": features
        }

    def load_saved_finetuned_collection(self) -> Dict[str, Any]:
        """Load the pinned Lalpur candidate without adding it to editable state."""
        if self._saved_finetuned_collection is not None:
            return self._saved_finetuned_collection

        run_dir = self.saved_finetuned_run_dir
        if not os.path.isdir(run_dir):
            raise SavedPredictionLoadError(
                f"Saved DeepLab fine-tuned run directory is missing: {run_dir}"
            )

        def read_json(filename: str) -> Dict[str, Any]:
            path = os.path.join(run_dir, filename)
            if not os.path.isfile(path):
                raise SavedPredictionLoadError(
                    f"Required saved DeepLab artifact is missing: {path}"
                )
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    value = json.load(handle)
            except (OSError, json.JSONDecodeError) as exc:
                raise SavedPredictionLoadError(
                    f"Could not read saved DeepLab artifact {path}: {exc}"
                ) from exc
            if not isinstance(value, dict):
                raise SavedPredictionLoadError(
                    f"Saved DeepLab artifact must contain a JSON object: {path}"
                )
            return value

        prediction = read_json("finetuned_predictions.geojson")
        audit = read_json("audit.json")
        report = read_json("comparison_report.json")
        checkpoint_path = os.path.join(run_dir, "fine_tuned_checkpoint.pth")
        if not os.path.isfile(checkpoint_path):
            raise SavedPredictionLoadError(
                f"Required saved DeepLab checkpoint is missing: {checkpoint_path}"
            )
        if prediction.get("type") != "FeatureCollection" or not isinstance(prediction.get("features"), list):
            raise SavedPredictionLoadError(
                f"Malformed saved DeepLab predictions: expected a FeatureCollection in {os.path.join(run_dir, 'finetuned_predictions.geojson')}"
            )
        if any(
            not isinstance(feature, dict)
            or not isinstance(feature.get("geometry"), dict)
            or not isinstance(feature.get("properties"), dict)
            for feature in prediction["features"]
        ):
            raise SavedPredictionLoadError(
                "Malformed saved DeepLab predictions: every feature must have geometry and properties objects."
            )

        try:
            source_metadata = prediction["model_metadata"]
            threshold = float(source_metadata["threshold"])
            threshold_key = f"{threshold:.2f}"
            candidate_metrics = report["finetuned"][threshold_key]
            baseline_metrics = report["baseline"][threshold_key]
            input_sha256 = report["input_raster_sha256"]
            reference_sha256 = report["reference_geojson_sha256"]
            target_gsd_m = float(report["target_gsd_m"])
            training_patches = int(report["training_patches"])
            epochs = int(report["epochs"])
            inference_stride = int(report["inference_stride"])
            reference_count = int(audit["label_feature_count"])
            metric_status = source_metadata["metric_status"]
            if (
                threshold != 0.5
                or metric_status != "in_sample_full_AOI_fit_not_held_out"
                or reference_count != 317
                or audit["model_input_sha256"] != input_sha256
            ):
                raise ValueError("saved run provenance does not match the pinned Lalpur fit")

            checkpoint_digest = hashlib.sha256()
            with open(checkpoint_path, "rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    checkpoint_digest.update(chunk)

            warning = (
                "All 317 reference building footprints were used for fine-tuning. "
                "Metrics are in-sample, object-level recall is low, and no geographic "
                "generalization claim is supported."
            )
            provider_metadata = {
                "provider_id": "deeplab-lalpur-finetuned-20261004-141612",
                "provider_label": "DeepLab · Lalpur fine-tuned (experimental; in-sample)",
                "run_id": os.path.basename(os.path.normpath(run_dir)),
                "checkpoint_sha256": checkpoint_digest.hexdigest(),
                "input_raster_sha256": input_sha256,
                "reference_geojson_sha256": reference_sha256,
                "confidence_threshold": threshold,
                "target_gsd_m": target_gsd_m,
                "training_patch_count": training_patches,
                "epochs": epochs,
                "inference_stride": inference_stride,
                "reference_feature_count": reference_count,
                "metric_status": metric_status,
                "warning": warning,
                "metrics": {
                    "candidate": candidate_metrics,
                    "baseline": baseline_metrics,
                    "pixel_f1_definition": "F1 on thresholded building-mask pixels (not building-detection accuracy).",
                    "object_f1_iou_035_definition": "One-to-one building-object matches at footprint IoU >= 0.35.",
                    "object_f1_iou_050_definition": "One-to-one building-object matches at footprint IoU >= 0.50.",
                },
            }
        except (KeyError, TypeError, ValueError, OSError) as exc:
            if isinstance(exc, SavedPredictionLoadError):
                raise
            raise SavedPredictionLoadError(
                f"Malformed or inconsistent saved DeepLab run metadata in {run_dir}: {exc}"
            ) from exc

        collection = copy.deepcopy(prediction)
        collection["name"] = provider_metadata["provider_id"]
        collection["total_features"] = len(collection["features"])
        collection["metadata"] = {
            **collection.get("metadata", {}),
            "provider": provider_metadata,
            "layer_role": "experimental_building_predictions_read_only",
            "parcel_status": "not_evaluated",
        }
        self._saved_finetuned_collection = collection
        return self._saved_finetuned_collection

    def set_prediction_source(
        self,
        features: List[Dict[str, Any]],
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Replace the active prediction collection and record its provenance."""
        self.initialize()
        self.layers["ai_predictions"] = scrub_nan_floats(copy.deepcopy(features))
        self.active_prediction_metadata = scrub_nan_floats(copy.deepcopy(metadata))
        provider_layer = {
            "whu": "ai_whu_predictions",
            "deeplab": "ai_deeplab_predictions",
            "deeplab_lalpur_finetuned": "ai_deeplab_predictions",
            "mock": "ai_mock_predictions",
        }.get(metadata.get("provider_id"))
        if provider_layer:
            self.layers[provider_layer] = scrub_nan_floats(copy.deepcopy(features))
        self.revalidate_all()
        self._save_persisted_state()
        return self.get_prediction_status()

    def clear_prediction_source(self) -> Dict[str, Any]:
        """Clear active predictions without deleting saved model artifacts."""
        self.initialize()
        self.layers["ai_predictions"] = []
        self.active_prediction_metadata = None
        self.revalidate_all()
        self._save_persisted_state()
        return self.get_prediction_status()

    def get_prediction_status(self) -> Dict[str, Any]:
        """Return explicit active-source state used by the API and UI."""
        self.initialize()
        metadata = copy.deepcopy(self.active_prediction_metadata)
        if not metadata:
            return {
                "project_id": self.project_id,
                "selected": False,
                "source_mode": None,
                "run_id": None,
                "prediction_count": 0,
                "is_mock": False,
            }
        return {
            **metadata,
            "project_id": self.project_id,
            "selected": True,
            "prediction_count": len(self.layers["ai_predictions"]),
        }

    def find_feature(self, feature_id: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        """Finds a feature across all layers by ID. Returns (layer_name, feature_dict)."""
        self.initialize()
        for layer_name, feature_list in self.layers.items():
            for feat in feature_list:
                if feat.get("id") == feature_id:
                    return layer_name, feat
        return None

    def update_feature_status(
        self,
        feature_id: str,
        review_status: str,
        notes: Optional[str] = None,
        reviewer_label: str = "demo-reviewer"
    ) -> Dict[str, Any]:
        """Updates feature review status, notes, and reviewer audit fields."""
        found = self.find_feature(feature_id)
        if not found:
            raise KeyError(f"Feature with ID '{feature_id}' not found.")

        layer_name, feat = found
        props = feat["properties"]

        valid_statuses = ["unverified", "under_review", "approved", "rejected"]
        if review_status not in valid_statuses:
            raise ValueError(f"Invalid review_status '{review_status}'. Valid choices: {valid_statuses}")

        props["review_status"] = review_status
        props["last_reviewed_by"] = reviewer_label
        props["edited_by"] = reviewer_label  # alias for test compatibility
        props["last_reviewed_at"] = datetime.now(timezone.utc).isoformat()
        props["edited_at"] = props["last_reviewed_at"]  # alias for test compatibility
        if notes is not None:
            props["notes"] = notes

        if "audit_history" not in props:
            props["audit_history"] = []

        props["audit_history"].append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": "status_update",
            "reviewer": reviewer_label,
            "new_status": review_status,
            "notes": notes
        })

        self.revalidate_all()
        self._save_persisted_state()
        return feat

    def update_feature_geometry(
        self,
        feature_id: str,
        new_geometry: Dict[str, Any],
        reviewer_label: str = "demo-reviewer",
        edit_reason: Optional[str] = None
    ) -> Dict[str, Any]:
        """Updates feature geometry while preserving original_geometry baseline."""
        found = self.find_feature(feature_id)
        if not found:
            raise KeyError(f"Feature with ID '{feature_id}' not found.")

        layer_name, feat = found

        if "original_geometry" not in feat or feat["original_geometry"] is None:
            feat["original_geometry"] = copy.deepcopy(feat["geometry"])

        feat["geometry"] = copy.deepcopy(new_geometry)
        props = feat["properties"]
        props["is_manually_edited"] = True
        props["last_edited_by"] = reviewer_label
        props["edited_by"] = reviewer_label  # alias for test compatibility
        props["last_edited_at"] = datetime.now(timezone.utc).isoformat()
        props["edited_at"] = props["last_edited_at"]  # alias for test compatibility
        props["edit_reason"] = edit_reason or "Manual geometry refinement"

        if "audit_history" not in props:
            props["audit_history"] = []

        props["audit_history"].append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": "geometry_edit",
            "reviewer": reviewer_label,
            "edit_reason": edit_reason or "Manual geometry refinement"
        })

        self.revalidate_all()
        self._save_persisted_state()
        return feat

    def revert_feature_geometry(self, feature_id: str) -> Dict[str, Any]:
        """Reverts feature geometry back to original baseline."""
        found = self.find_feature(feature_id)
        if not found:
            raise KeyError(f"Feature with ID '{feature_id}' not found.")

        layer_name, feat = found
        if "original_geometry" in feat and feat["original_geometry"] is not None:
            feat["geometry"] = copy.deepcopy(feat["original_geometry"])
            props = feat["properties"]
            props["is_manually_edited"] = False
            revert_reason = "Reverted to unedited original baseline"
            props["edit_reason"] = revert_reason  # test checks "Reverted" in edit_reason

            if "audit_history" not in props:
                props["audit_history"] = []

            props["audit_history"].append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "action": "revert_geometry",
                "notes": revert_reason
            })

        self.revalidate_all()
        self._save_persisted_state()
        return feat

    def add_draft_feature(
        self,
        geometry: Dict[str, Any],
        feature_type: str = "draft_polygon",
        notes: str = "",
        reviewer_label: str = "demo-reviewer"
    ) -> Dict[str, Any]:
        """Adds a newly drawn draft feature tagged manual_visual_reference."""
        self.initialize()
        draft_id = f"DRAFT-{len(self.layers['drafts']) + 1:03d}"
        now_iso = datetime.now(timezone.utc).isoformat()

        new_feat = {
            "type": "Feature",
            "id": draft_id,
            "geometry": copy.deepcopy(geometry),
            "properties": {
                "feature_id": draft_id,
                "feature_type": feature_type,
                "source": "manual_visual_reference",
                "verification_status": "unverified",
                "review_status": "under_review",
                "notes": notes or "User-created manual reference boundary",
                "edited_by": reviewer_label,
                "edited_at": now_iso,
                "locality": "Lalpur",
                "provenance_citation": "Manual human visual trace created in local reviewer UI."
            },
            "original_geometry": copy.deepcopy(geometry)
        }

        self.layers["drafts"].append(new_feat)
        self.revalidate_all()
        self._save_persisted_state()
        return new_feat

    def add_ai_predicted_features(self, features: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Appends features produced by AI Building Model inference adapter."""
        self.initialize()
        added = []
        for feat in features:
            f_copy = scrub_nan_floats(copy.deepcopy(feat))
            if "original_geometry" not in f_copy:
                f_copy["original_geometry"] = copy.deepcopy(f_copy.get("geometry"))
            self.layers["ai_predictions"].append(f_copy)
            added.append(f_copy)
            
        self.revalidate_all()
        self._save_persisted_state()
        return added

    def export_reviewed_bundle(self) -> Dict[str, Any]:
        """Exports all reviewed and active features into a single RFC 7946 GeoJSON bundle."""
        self.initialize()
        all_features = []
        for layer_name in ["buildings", "roads", "osm_roads", "synthetic", "drafts", "ai_predictions"]:
            for feat in self.layers[layer_name]:
                f_export = copy.deepcopy(feat)
                f_export["properties"]["layer"] = layer_name
                all_features.append(f_export)

        now_iso = datetime.now(timezone.utc).isoformat()
        bundle = {
            "type": "FeatureCollection",
            "name": "SIH26012_Reviewed_Features_Export",
            "crs": {
                "type": "name",
                "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}
            },
            "export_metadata": {
                "exported_at": now_iso,
                "project_id": self.project_id,
                "locality": "Lalpur Village, Gujarat (LGD 511638)" if self.project_id == DEFAULT_LALPUR_PROJECT_ID else "Project locality recorded in project metadata",
                "target_crs": "EPSG:4326 (WGS 84 / RFC 7946)",
                "native_analysis_crs": "EPSG:3857 (Web Mercator)",
                "provenance_disclaimer": PROVENANCE_DISCLAIMER,
                "total_features": len(all_features),
                "open_warnings_count": len(self.warnings),
                "cadastral_status": "PROTOTYPE ONLY. Zero real cadastral parcels exist. Not official parcel ground truth."
            },
            "features": all_features
        }
        return bundle

    def import_reviewed_bundle(self, bundle: Dict[str, Any]) -> Dict[str, Any]:
        """Imports and restores features from a previously exported GeoJSON bundle."""
        if not isinstance(bundle, dict) or bundle.get("type") != "FeatureCollection":
            raise ValueError("Import payload must be a valid GeoJSON FeatureCollection.")

        imported_features = bundle.get("features", [])
        if not isinstance(imported_features, list):
            raise ValueError("Malformed GeoJSON: missing features list.")

        layer_buckets: Dict[str, List[Dict[str, Any]]] = {
            "buildings": [],
            "roads": [],
            "osm_roads": [],
            "parcels": [],
            "synthetic": [],
            "drafts": [],
            "ai_predictions": []
        }

        for feat in imported_features:
            props = feat.get("properties", {})
            layer = props.get("layer", "buildings")
            if layer not in layer_buckets:
                layer = "buildings"
            layer_buckets[layer].append(feat)

        for layer, feats in layer_buckets.items():
            if feats:
                self.layers[layer] = feats

        self.revalidate_all()
        self._save_persisted_state()
        return {
            "status": "success",
            "imported_count": len(imported_features),
            "layers_updated": [k for k, v in layer_buckets.items() if v]
        }


def get_project_store(project_id: Optional[str] = None, *, state_file_path: Optional[str] = None) -> FeatureStore:
    """Returns a project-scoped FeatureStore; the Lalpur demo remains the default singleton."""
    resolved_id = project_id or DEFAULT_LALPUR_PROJECT_ID
    existing = PROJECT_STORES.get(resolved_id)
    if existing is not None:
        if state_file_path:
            existing.state_file_path = state_file_path
        return existing
    resolved_state_path = state_file_path or (STATE_FILE_PATH if resolved_id == DEFAULT_LALPUR_PROJECT_ID else FeatureStore._default_state_file_for_project(resolved_id))
    store_obj = FeatureStore(project_id=resolved_id, state_file_path=resolved_state_path)
    PROJECT_STORES[resolved_id] = store_obj
    return store_obj


# Global singleton instance
store = FeatureStore(project_id=DEFAULT_LALPUR_PROJECT_ID)
