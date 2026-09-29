"""
Feature Store and State Manager for SIH26012 Feature Review Platform.
Maintains in-memory layer collections, handles human edits with immutable original_geometry,
dynamic re-calculation of topology warnings, review-priority scoring, and persistent disk storage.
"""
import os
import json
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

class FeatureStore:
    """Central state manager for the local prototype with local disk persistence."""
    
    def __init__(self, state_file_path: str = STATE_FILE_PATH):
        self.state_file_path = state_file_path
        self.layers: Dict[str, List[Dict[str, Any]]] = {
            "buildings": [],
            "roads": [],
            "osm_roads": [],
            "parcels": [],
            "synthetic": [],
            "drafts": [],
            "ai_predictions": []
        }
        self.warnings: List[TopologyWarning] = []
        self.scores: Dict[str, ReviewScoreBreakdown] = {}
        self.active_prediction_metadata: Optional[Dict[str, Any]] = None
        self.is_initialized = False

    def initialize(self, force_reload: bool = False) -> None:
        """Loads reference layers, synthetic test fixtures, and merges persisted user state."""
        if self.is_initialized and not force_reload:
            return

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
        self.layers["drafts"] = []
        self.layers["ai_predictions"] = []
        self.active_prediction_metadata = None

        # Load persisted local edits & predictions if available
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

        feat_warnings_map: Dict[str, List[str]] = {}
        for w in self.warnings:
            for fid in w.feature_ids:
                if fid not in feat_warnings_map:
                    feat_warnings_map[fid] = []
                feat_warnings_map[fid].append(w.warning_id)

        self.scores.clear()
        for layer_name, feature_list in self.layers.items():
            for feat in feature_list:
                fid = feat["id"]
                w_ids = feat_warnings_map.get(fid, [])
                feat["properties"]["warning_ids"] = w_ids
                
                score_breakdown = calculate_review_score(feat, associated_warning_ids=w_ids)
                self.scores[fid] = score_breakdown
                feat["properties"]["review_score"] = score_breakdown.total_score

    def get_layer_collection(self, layer_name: str) -> Dict[str, Any]:
        """Returns GeoJSON FeatureCollection representation for the requested layer."""
        self.initialize()
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

    def set_prediction_source(
        self,
        features: List[Dict[str, Any]],
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Replace the active prediction collection and record its provenance."""
        self.initialize()
        self.layers["ai_predictions"] = scrub_nan_floats(copy.deepcopy(features))
        self.active_prediction_metadata = scrub_nan_floats(copy.deepcopy(metadata))
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
                "selected": False,
                "source_mode": None,
                "run_id": None,
                "prediction_count": 0,
                "is_mock": False,
            }
        return {
            **metadata,
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
                "project_id": "SIH26012_INDIA_CANDIDATE_01_LALPUR",
                "locality": "Lalpur Village, Gujarat (LGD 511638)",
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

# Global singleton instance
store = FeatureStore()
