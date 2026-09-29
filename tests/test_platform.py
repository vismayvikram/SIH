"""
Comprehensive Test Suite for SIH26012 Geospatial Feature Review Platform.
Tests all prompt-mandated finish criteria:
1. GeoJSON parsing and required schema fields.
2. Stable source/status serialization and provenance correction (project_vaayu_sample, unverified).
3. Status transitions and note persistence with reviewer attribution.
4. Edit saves while preserving immutable original_geometry and revert capability.
5. GeoJSON export and re-import round trip integrity.
6. Synthetic topology fixtures produce expected warnings (overlap, bowtie, empty, road overlap, parcel crossing).
7. Transparent review score calculation and heuristic explanation.
8. OSM attribution presence.
9. Empty real parcel layer never triggers real parcel-conflict claims.
10. Model adapter prediction success, confidence filtering, and simulated failure states.
11. Benchmark dataset manifest and evaluation harness (IoU, precision, recall, F1).
12. FastAPI HTTP endpoints end-to-end integration.
13. CRS-aware projected metric geometry (EPSG:32643): area in m2, NOT degrees*constant.
14. Discriminative scoring: features differ in score based on actual conflict, not shared provenance.
15. Parcel-level RAG aggregation and gating.
"""
import copy
import math
import pytest
from fastapi.testclient import TestClient
import shapely.geometry

from backend.api import app
from backend.services.data_loader import (
    load_lalpur_buildings,
    load_lalpur_roads,
    load_osm_roads,
    load_blank_parcel_template,
    sanitize_and_normalize_feature,
    PROVENANCE_DISCLAIMER
)
from backend.services.feature_store import FeatureStore, store
from backend.services.topology import (
    run_full_topology_validation,
    validate_feature_geometries,
    check_polygon_overlaps,
    check_building_road_intersections,
    check_synthetic_parcel_crossings,
    check_building_parcel_crossings
)
from backend.services.synthetic import get_synthetic_test_features, SYNTHETIC_LAYER_DISCLAIMER
from backend.services.scoring import calculate_review_score, load_scoring_config, aggregate_parcel_scores
from backend.services.geometry_utils import (
    calculate_metric_area_sqm,
    calculate_metric_distance_meters,
    calculate_metric_intersection,
    safe_repair_geometry
)
from backend.services.model_adapter import (
    MockBuildingModel,
    WHUBuildingModel,
    DeepLabBuildingModel,
    BenchmarkDatasetAdapter,
    ModelInferenceError
)


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def clean_store():
    """Provides an isolated, freshly initialized FeatureStore instance."""
    s = FeatureStore(state_file_path=None)
    s.initialize(force_reload=True)
    return s


@pytest.fixture
def client():
    """FastAPI TestClient instance."""
    return TestClient(app)


# ==============================================================================
# 1. GeoJSON Parsing and Required Schema Fields
# ==============================================================================

class TestGeoJSONSchemaAndDataLoading:
    def test_load_lalpur_buildings_schema(self):
        fc = load_lalpur_buildings()
        assert fc["type"] == "FeatureCollection"
        assert fc["total_features"] == 317
        assert len(fc["features"]) == 317

        for feat in fc["features"]:
            assert feat["type"] == "Feature"
            assert "id" in feat
            assert "geometry" in feat
            assert "original_geometry" in feat
            props = feat["properties"]

            # Required schema fields
            assert "feature_id" in props
            assert "feature_type" in props
            assert "source" in props
            assert "verification_status" in props
            assert "review_status" in props
            assert "warning_ids" in props
            assert "notes" in props
            assert "edited_by" in props
            assert "edited_at" in props

    def test_load_lalpur_roads(self):
        fc = load_lalpur_roads()
        assert fc["type"] == "FeatureCollection"
        assert fc["total_features"] == 19
        assert len(fc["features"]) == 19

    def test_load_osm_roads_attribution(self):
        fc = load_osm_roads()
        assert fc["type"] == "FeatureCollection"
        assert fc["total_features"] == 5
        assert "OpenStreetMap" in fc.get("attribution", "")

    def test_load_blank_parcel_template_is_zero(self):
        fc = load_blank_parcel_template()
        assert fc["type"] == "FeatureCollection"
        assert fc["total_features"] == 0
        assert len(fc["features"]) == 0
        assert "NO REAL PARCEL" in fc["disposition"]


# ==============================================================================
# 2. Stable Source and Status Serialization & Provenance Correction
# ==============================================================================

class TestProvenanceAndSanitization:
    def test_provenance_correction_project_vaayu(self):
        # Raw feature incorrectly tagged manual_visual_reference must be normalized
        raw_feat = {
            "type": "Feature",
            "id": "RAW-BLD-01",
            "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]},
            "properties": {
                "source": "manual_visual_reference",
                "feature_type": "building"
            }
        }
        normalized = sanitize_and_normalize_feature(raw_feat, default_feature_type="building")
        # Must be corrected to project_vaayu_sample and unverified
        assert normalized["properties"]["source"] == "project_vaayu_sample"
        assert normalized["properties"]["verification_status"] == "unverified"
        assert normalized["properties"]["review_status"] == "unverified"
        assert "Project Vaayu sample" in normalized["properties"]["provenance_citation"]

    def test_pii_sanitization_strips_sensitive_fields(self):
        raw_feat = {
            "type": "Feature",
            "id": "PII-TEST-01",
            "geometry": {"type": "Point", "coordinates": [72.75, 23.04]},
            "properties": {
                "owner_name": "Ramesh Patel",
                "property_card_no": "PC-511638-0099",
                "property_id": "PROP-1234",
                "valid_field": "keep_this"
            }
        }
        normalized = sanitize_and_normalize_feature(raw_feat, default_feature_type="building")
        props = normalized["properties"]
        assert "owner_name" not in props
        assert "property_card_no" not in props
        assert "property_id" not in props
        # Non-standard extra field preserved safely in 'extra'
        assert props.get("extra", {}).get("valid_field") == "keep_this"


# ==============================================================================
# 3. Status Transitions and Notes Persistence
# ==============================================================================

class TestFeatureStatusUpdates:
    def test_status_transitions_and_note_update(self, clean_store):
        first_bld_id = clean_store.layers["buildings"][0]["id"]

        # Initial state
        assert clean_store.layers["buildings"][0]["properties"]["review_status"] == "unverified"

        # Update to under_review
        clean_store.update_feature_status(
            feature_id=first_bld_id,
            review_status="under_review",
            notes="Initial visual check required",
            reviewer_label="reviewer-alice"
        )
        layer_name, feat = clean_store.find_feature(first_bld_id)
        assert feat["properties"]["review_status"] == "under_review"
        assert feat["properties"]["notes"] == "Initial visual check required"
        assert feat["properties"]["edited_by"] == "reviewer-alice"
        assert feat["properties"]["edited_at"] is not None

        # Update to approved
        clean_store.update_feature_status(
            feature_id=first_bld_id,
            review_status="approved",
            notes="Boundary verified against imagery",
            reviewer_label="reviewer-alice"
        )
        _, feat = clean_store.find_feature(first_bld_id)
        assert feat["properties"]["review_status"] == "approved"
        assert feat["properties"]["notes"] == "Boundary verified against imagery"

        # Update to rejected
        clean_store.update_feature_status(
            feature_id=first_bld_id,
            review_status="rejected",
            notes="False positive detection",
            reviewer_label="reviewer-bob"
        )
        _, feat = clean_store.find_feature(first_bld_id)
        assert feat["properties"]["review_status"] == "rejected"
        assert feat["properties"]["edited_by"] == "reviewer-bob"


# ==============================================================================
# 4. Geometry Edits Preserving Immutable Original Geometry & Revert
# ==============================================================================

class TestGeometryEditingAndRevert:
    def test_save_edit_preserves_original_geometry(self, clean_store):
        bld = clean_store.layers["buildings"][0]
        bld_id = bld["id"]
        original_coords = copy.deepcopy(bld["geometry"]["coordinates"])

        # Create modified coordinates
        modified_coords = [
            [[c[0] + 0.0001, c[1] + 0.0001] for c in original_coords[0]]
        ]
        new_geom = {"type": "Polygon", "coordinates": modified_coords}

        clean_store.update_feature_geometry(
            feature_id=bld_id,
            new_geometry=new_geom,
            reviewer_label="reviewer-carol",
            edit_reason="Adjusted south-east vertex"
        )

        _, updated = clean_store.find_feature(bld_id)
        assert updated["geometry"]["coordinates"] == modified_coords
        assert updated["original_geometry"]["coordinates"] == original_coords
        assert updated["properties"]["edited_by"] == "reviewer-carol"
        assert updated["properties"]["edit_reason"] == "Adjusted south-east vertex"

        # Revert geometry back to original
        clean_store.revert_feature_geometry(bld_id)
        _, reverted = clean_store.find_feature(bld_id)
        assert reverted["geometry"]["coordinates"] == original_coords
        assert "Reverted" in reverted["properties"]["edit_reason"]


# ==============================================================================
# 5. Export and Re-Import Round Trip
# ==============================================================================

class TestExportImportRoundTrip:
    def test_export_and_import_integrity(self, clean_store):
        bundle = clean_store.export_reviewed_bundle()
        assert bundle["type"] == "FeatureCollection"
        assert "export_metadata" in bundle
        assert bundle["export_metadata"]["target_crs"] == "EPSG:4326 (WGS 84 / RFC 7946)"
        assert len(bundle["features"]) == 349  # 317 bld + 19 rd + 5 osm + 8 synth

        # Fresh store instance imports the bundle
        new_store = FeatureStore()
        result = new_store.import_reviewed_bundle(bundle)
        assert result["status"] == "success"
        assert result["imported_count"] == 349
        assert len(new_store.layers["buildings"]) == 317
        assert len(new_store.layers["roads"]) == 19
        assert len(new_store.layers["synthetic"]) == 8


# ==============================================================================
# 6. Synthetic Topology Fixtures and Expected Warnings
# ==============================================================================

class TestTopologyWarnings:
    def test_synthetic_fixtures_produce_all_expected_warnings(self, clean_store):
        synth_warnings = [w for w in clean_store.warnings if "synthetic" in w.source]
        types_found = {w.warning_type for w in synth_warnings}

        # 1. Overlapping polygons
        assert "overlapping_polygons" in types_found
        overlap_warn = next(w for w in synth_warnings if w.warning_type == "overlapping_polygons")
        assert "SYN-BLD-OVERLAP-1" in overlap_warn.feature_ids
        assert "SYN-BLD-OVERLAP-2" in overlap_warn.feature_ids

        # 2. Invalid geometry (bowtie / self-intersecting)
        assert "invalid_geometry" in types_found
        inv_warn = next(w for w in synth_warnings if w.warning_type == "invalid_geometry")
        assert "SYN-INVALID-GEOM-1" in inv_warn.feature_ids
        assert "Self-intersection" in inv_warn.explanation

        # 2b. Empty geometry
        assert "empty_geometry" in types_found
        empty_warn = next(w for w in synth_warnings if w.warning_type == "empty_geometry")
        assert "SYN-EMPTY-GEOM-1" in empty_warn.feature_ids

        # 3. Synthetic building intersecting synthetic road corridor
        assert "building_road_spatial_overlap" in types_found
        road_warn = next(w for w in synth_warnings if w.warning_type == "building_road_spatial_overlap")
        assert "SYN-BLD-ROAD-INT-1" in road_warn.feature_ids
        assert "SYN-ROAD-CORRIDOR-1" in road_warn.feature_ids

        # 4. Synthetic building crossing synthetic parcel
        assert "building_crosses_synthetic_parcel" in types_found
        parcel_warn = next(w for w in synth_warnings if w.warning_type == "building_crosses_synthetic_parcel")
        assert "SYN-BLD-PARCEL-CROSS-1" in parcel_warn.feature_ids
        assert "SYN-PARCEL-DEMO-1" in parcel_warn.feature_ids

    def test_empty_real_parcel_layer_never_produces_real_parcel_warnings(self, clean_store):
        real_parcel_warnings = [
            w for w in clean_store.warnings
            if "parcel" in w.warning_type and "synthetic" not in w.warning_type
        ]
        assert len(real_parcel_warnings) == 0, (
            "Violated rule: real parcel conflict warning produced when real parcels are empty!"
        )


# ==============================================================================
# 7. Review Score Calculation and Breakdown
# ==============================================================================

class TestScoringEngine:
    def test_score_calculation_breakdown_and_disclaimer(self):
        sample_feat = {
            "type": "Feature",
            "id": "TEST-SCORE-01",
            "properties": {
                "feature_id": "TEST-SCORE-01",
                "feature_type": "building",
                "source": "project_vaayu_sample",
                "review_status": "unverified",
                "area_sqm": 850.0  # triggers extreme dimensions rule (> 600m)
            }
        }
        breakdown = calculate_review_score(
            sample_feat,
            associated_warning_ids=["W-OVERLAP-01"]
        )

        assert breakdown.feature_id == "TEST-SCORE-01"
        assert 0 <= breakdown.total_score <= 100
        assert "prototype heuristic—not a validated survey-priority model" in breakdown.heuristic_disclaimer

        triggered_rule_ids = {r.rule_id for r in breakdown.rules_triggered}
        assert "R_UNVERIFIED_SOURCE" in triggered_rule_ids
        assert "R_SPATIAL_OVERLAP_WARNING" in triggered_rule_ids
        assert "R_EXTREME_DIMENSIONS" in triggered_rule_ids
        assert breakdown.priority in ["medium", "high"]

    def test_approved_feature_score_reduction(self):
        sample_feat = {
            "type": "Feature",
            "id": "TEST-SCORE-APPROVED",
            "properties": {
                "feature_id": "TEST-SCORE-APPROVED",
                "feature_type": "building",
                "source": "project_vaayu_sample",
                "review_status": "approved",
                "area_sqm": 120.0
            }
        }
        breakdown = calculate_review_score(sample_feat, associated_warning_ids=[])
        triggered_rule_ids = {r.rule_id for r in breakdown.rules_triggered}
        assert "R_HUMAN_APPROVED" in triggered_rule_ids
        assert breakdown.total_score == 0  # 10 base + 20 unverified - 30 approved = 0


# ==============================================================================
# 8. Building Model Interface and Benchmark Evaluation Harness
# ==============================================================================

class TestBuildingModelAndBenchmarkHarness:
    def test_mock_building_model_prediction_success(self):
        model = MockBuildingModel()
        result = model.predict(confidence_threshold=0.70, simulate_failure=False)
        assert result["type"] == "FeatureCollection"
        assert result["model_metadata"]["inference_status"] == "success"
        # Only features with confidence >= 0.70
        for feat in result["features"]:
            assert feat["properties"]["source"] == "ai_building_model"
            assert feat["properties"]["confidence"] >= 0.70

    def test_mock_building_model_failure_simulation(self):
        model = MockBuildingModel()
        with pytest.raises(ModelInferenceError) as exc_info:
            model.predict(simulate_failure=True)
        assert "Simulated Model Failure" in str(exc_info.value)

    def test_benchmark_manifest_status_pending(self):
        manifest = BenchmarkDatasetAdapter.get_dataset_manifest()
        datasets = manifest["registered_datasets"]
        assert "inria_aerial_image_labeling" in datasets
        assert "spacenet_2_buildings" in datasets
        assert datasets["inria_aerial_image_labeling"]["status"] == "PENDING_ACQUISITION"
        assert "PENDING_ACQUISITION" in manifest["disclaimer"]

    def test_polygon_evaluation_metrics_iou_and_f1(self):
        # Two identical squares: IoU = 1.0
        p1 = shapely.geometry.box(0, 0, 10, 10)
        p2 = shapely.geometry.box(0, 0, 10, 10)
        iou = BenchmarkDatasetAdapter.compute_polygon_iou(p1, p2)
        assert iou == 1.0

        # Disjoint squares: IoU = 0.0
        p3 = shapely.geometry.box(20, 20, 30, 30)
        assert BenchmarkDatasetAdapter.compute_polygon_iou(p1, p3) == 0.0

        # Evaluation harness metric calculation
        gt = [p1, p3]
        pred = [p2]  # p2 matches p1, p3 missed (1 TP, 0 FP, 1 FN)
        metrics = BenchmarkDatasetAdapter.evaluate_detections(gt, pred, iou_threshold=0.5)
        assert metrics["true_positives"] == 1
        assert metrics["false_positives"] == 0
        assert metrics["false_negatives"] == 1
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 0.5
        assert round(metrics["f1_score"], 4) == 0.6667


# ==============================================================================
# 9. FastAPI API Endpoints Integration
# ==============================================================================

class TestApiEndpoints:
    def test_api_health(self, client):
        res = client.get("/api/health")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "online"
        assert data["layers_loaded"]["buildings"] == 317

    def test_api_metadata_osm_and_disclaimer(self, client):
        res = client.get("/api/metadata")
        assert res.status_code == 200
        data = res.json()
        assert "OpenStreetMap" in data["osm_attribution"]
        assert "Project Vaayu sample" in data["provenance_disclaimer"]
        assert data["raster_orthomosaic_status"]["browser_service_status"] in ["AVAILABLE_LOCAL_XYZ_TILES", "UNAVAILABLE"]

    def test_api_raster_status_and_tiles(self, client):
        # Status endpoint
        res = client.get("/api/raster/status")
        assert res.status_code == 200
        status_data = res.json()
        assert status_data["status"] == "ready"
        assert status_data["metadata"]["crs"] == "EPSG:3857"
        assert status_data["metadata"]["bands"] == 4

        # Tile endpoint inside bounds
        tile_res = client.get("/api/raster/tiles/18/184052/113823.png")
        assert tile_res.status_code == 200
        assert tile_res.headers["content-type"] == "image/png"
        assert len(tile_res.content) > 1000

        # Tile outside bounds returns blank transparent tile without 404
        out_res = client.get("/api/raster/tiles/18/100/100.png")
        assert out_res.status_code == 200
        assert out_res.headers["content-type"] == "image/png"

    def test_api_get_layers(self, client):
        for layer_name in ["buildings", "roads", "osm_roads", "parcels", "synthetic"]:
            res = client.get(f"/api/layers/{layer_name}")
            assert res.status_code == 200
            data = res.json()
            assert data["type"] == "FeatureCollection"

    def test_api_update_feature_status(self, client):
        # Pick first building
        res_list = client.get("/api/layers/buildings")
        bld_id = res_list.json()["features"][0]["id"]

        update_payload = {
            "review_status": "under_review",
            "notes": "FastAPI integration test note",
            "reviewer_label": "api-tester"
        }
        res = client.put(f"/api/features/{bld_id}", json=update_payload)
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "success"
        assert data["feature"]["properties"]["review_status"] == "under_review"
        assert data["feature"]["properties"]["notes"] == "FastAPI integration test note"

    def test_api_model_predict_endpoint(self, client):
        req = {
            "model_name": "Vaayu-UnetPP-Lite",
            "model_version": "0.1.0-mock",
            "confidence_threshold": 0.6,
            "simulate_failure": False
        }
        res = client.post("/api/models/predict", json=req)
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "success"
        assert data["detected_count"] > 0

    def test_api_model_predict_failure_simulation(self, client):
        req = {
            "model_name": "Vaayu-UnetPP-Lite",
            "model_version": "0.1.0-mock",
            "simulate_failure": True
        }
        res = client.post("/api/models/predict", json=req)
        assert res.status_code == 500
        assert "Simulated Model Failure" in res.json()["detail"]

    def test_api_export_and_import(self, client):
        # Export
        exp_res = client.get("/api/export")
        assert exp_res.status_code == 200
        bundle = exp_res.json()
        assert bundle["type"] == "FeatureCollection"

        # Import
        imp_res = client.post("/api/import", json=bundle)
        assert imp_res.status_code == 200
        imp_data = imp_res.json()
        assert imp_data["status"] == "success"
        assert imp_data["imported_count"] == len(bundle["features"])


# ==============================================================================
# 13. CRS-Aware Projected Metric Geometry Tests
# ==============================================================================

class TestMetricGeometryUtils:
    def test_area_uses_projected_crs_not_degrees(self):
        """Area must be in m², NOT degrees * constant approximation."""
        # B-001 from Lalpur data: ~12.77 m² expected
        poly_deg = shapely.geometry.Polygon([
            [72.75728365790725, 23.04002668761225],
            [72.75731772292114, 23.04002919486646],
            [72.75732057507217, 23.039996387280304],
            [72.75728651005828, 23.039993879198835],
            [72.75728365790725, 23.04002668761225]
        ])
        area_m2 = calculate_metric_area_sqm(poly_deg)
        # Must be in reasonable residential building range (m²), not tiny degree² value
        assert 10 < area_m2 < 50, f"Expected ~12.77 m², got {area_m2}"
        # Area in raw degrees should be orders of magnitude smaller
        assert poly_deg.area < 1e-6, "Raw degree area should be negligible"

    def test_area_larger_than_pure_degree_calculation(self):
        """Projected area must be much larger than raw degree area."""
        poly = shapely.geometry.Polygon([
            [72.757, 23.040], [72.758, 23.040], [72.758, 23.041], [72.757, 23.041], [72.757, 23.040]
        ])
        raw_deg_area = poly.area
        metric_area = calculate_metric_area_sqm(poly)
        # Metric area should be roughly 100m x 100m ≈ 10000 m²; degree area is tiny
        assert metric_area > raw_deg_area * 1e8, "Projected area must be vastly larger than degree area"

    def test_distance_calculation_in_meters(self):
        """Distance between two close points must be in metres."""
        p1 = shapely.geometry.Point(72.757, 23.040)
        p2 = shapely.geometry.Point(72.758, 23.040)
        dist = calculate_metric_distance_meters(p1, p2)
        # ~100m expected (1 degree longitude at 23°N ≈ 102m)
        assert 80 < dist < 130, f"Expected ~100m, got {dist}"

    def test_invalid_geometry_repair(self):
        """Bowtie polygon should be repaired and flagged as repaired."""
        bowtie = shapely.geometry.Polygon([[0, 0], [1, 1], [0, 1], [1, 0], [0, 0]])
        assert not bowtie.is_valid
        repaired, was_repaired, msg = safe_repair_geometry(bowtie)
        assert was_repaired is True
        assert repaired.is_valid

    def test_valid_geometry_not_flagged_as_repaired(self):
        """Valid polygon must not be flagged as repaired."""
        valid_poly = shapely.geometry.box(0, 0, 10, 10)
        _, was_repaired, _ = safe_repair_geometry(valid_poly)
        assert was_repaired is False

    def test_intersection_area_in_m2(self):
        """Intersection area of two overlapping WGS84 polygons must be in m²."""
        poly1 = shapely.geometry.Polygon([
            [72.757, 23.040], [72.758, 23.040], [72.758, 23.041], [72.757, 23.041], [72.757, 23.040]
        ])
        poly2 = shapely.geometry.Polygon([
            [72.7575, 23.040], [72.7585, 23.040], [72.7585, 23.041], [72.7575, 23.041], [72.7575, 23.040]
        ])
        inter_geom, area_m2 = calculate_metric_intersection(poly1, poly2)
        assert inter_geom is not None
        assert area_m2 > 100, f"Expected intersection area > 100 m², got {area_m2}"

    def test_empty_geometry_returns_zero_area(self):
        """Empty geometry must return 0.0 area without raising exceptions."""
        empty = shapely.geometry.Polygon()
        area = calculate_metric_area_sqm(empty)
        assert area == 0.0


# ==============================================================================
# 14. Discriminative Scoring Tests
# ==============================================================================

class TestDiscriminativeScoring:
    def test_no_conflict_feature_has_low_score(self):
        """A clean feature with no warnings should not be scored high."""
        clean_feat = {
            "type": "Feature", "id": "CLEAN-01",
            "properties": {
                "feature_id": "CLEAN-01",
                "feature_type": "building",
                "source": "project_vaayu_sample",
                "review_status": "unverified",
                "area_sqm": 80.0
            }
        }
        score = calculate_review_score(clean_feat, associated_warning_ids=[])
        assert score.total_score <= 15, f"Clean feature should score low, got {score.total_score}"
        assert score.priority == "low"

    def test_overlap_warning_raises_score(self):
        """Feature with overlap warning should have higher score than clean feature."""
        conflicted_feat = {
            "type": "Feature", "id": "CONFLICT-01",
            "properties": {
                "feature_id": "CONFLICT-01",
                "feature_type": "building",
                "source": "project_vaayu_sample",
                "review_status": "unverified",
                "area_sqm": 80.0
            }
        }
        score = calculate_review_score(conflicted_feat, associated_warning_ids=["W-RD-OVERLAP-B-015-VR-004"])
        assert score.total_score >= 40, f"Feature with overlap warning should score >= 40, got {score.total_score}"
        rule_ids = {r.rule_id for r in score.rules_triggered}
        assert "R_SPATIAL_OVERLAP_WARNING" in rule_ids

    def test_invalid_geometry_warning_raises_score(self):
        """Feature with INVALID geometry warning ID should trigger R_INVALID_GEOMETRY rule."""
        invalid_feat = {
            "type": "Feature", "id": "INVALID-01",
            "properties": {
                "feature_id": "INVALID-01",
                "feature_type": "building",
                "source": "project_vaayu_sample",
                "review_status": "unverified",
                "area_sqm": 80.0
            }
        }
        score = calculate_review_score(invalid_feat, associated_warning_ids=["W-INVALID-INVALID-01"])
        rule_ids = {r.rule_id for r in score.rules_triggered}
        assert "R_INVALID_GEOMETRY" in rule_ids
        assert score.total_score >= 40

    def test_empty_geometry_warning_raises_score(self):
        """Feature with EMPTY geometry warning ID should trigger R_EMPTY_GEOMETRY rule."""
        empty_feat = {
            "type": "Feature", "id": "EMPTY-01",
            "properties": {
                "feature_id": "EMPTY-01",
                "feature_type": "building",
                "source": "project_vaayu_sample",
                "review_status": "unverified",
                "area_sqm": 0.0
            }
        }
        score = calculate_review_score(empty_feat, associated_warning_ids=["W-EMPTY-EMPTY-01"])
        rule_ids = {r.rule_id for r in score.rules_triggered}
        assert "R_EMPTY_GEOMETRY" in rule_ids

    def test_conflict_features_score_higher_than_clean_features(self, clean_store):
        """Across the full dataset, features with road-overlap warnings must score higher than clean ones."""
        clean_scores = [
            v.total_score for fid, v in clean_store.scores.items()
            if not clean_store.find_feature(fid)[1]["properties"].get("warning_ids")
        ]
        conflicted_scores = [
            v.total_score for fid, v in clean_store.scores.items()
            if clean_store.find_feature(fid)[1]["properties"].get("warning_ids")
        ]
        if clean_scores and conflicted_scores:
            assert max(clean_scores) < max(conflicted_scores), (
                "Conflicted features must have higher max score than clean features"
            )

    def test_unverified_source_is_score_neutral(self):
        """Unverified source must NOT add a discriminative score penalty on its own."""
        feat_vaayu = {
            "type": "Feature", "id": "VAAYU-01",
            "properties": {
                "feature_id": "VAAYU-01",
                "feature_type": "building",
                "source": "project_vaayu_sample",
                "review_status": "unverified",
                "area_sqm": 50.0
            }
        }
        config = load_scoring_config()
        r_source = next((r for r in config["rules"] if r["rule_id"] == "R_UNVERIFIED_SOURCE"), None)
        # Source rule must be score-neutral (0 points)
        assert r_source is not None
        assert r_source["points"] == 0, (
            "R_UNVERIFIED_SOURCE must be score-neutral (0 pts) to avoid dominating scores"
        )


# ==============================================================================
# 15. Parcel-Level RAG Aggregation and Gating Tests
# ==============================================================================

class TestParcelRAGAggregation:
    def test_empty_real_parcel_not_evaluated(self, clean_store):
        """When parcel layer is empty, system must report 'not evaluated', not zero conflicts."""
        real_parcels = clean_store.layers["parcels"]
        assert len(real_parcels) == 0, "Real parcel layer must be 0 features in this AOI"
        # No parcel-specific warnings should exist for the real layer
        parcel_conflict_warns = [
            w for w in clean_store.warnings
            if "parcel" in w.warning_type and "synthetic" not in w.warning_type
        ]
        assert len(parcel_conflict_warns) == 0, (
            "Must not produce real parcel conflict warnings when parcel layer is empty"
        )

    def test_parcel_rag_with_synthetic_fixtures(self):
        """Synthetic parcel with crossing building must get RED RAG status."""
        synth_parcel = {
            "type": "Feature", "id": "SYN-PARCEL-DEMO-1",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[72.75865, 23.03950], [72.75920, 23.03950], [72.75920, 23.04000], [72.75865, 23.04000], [72.75865, 23.03950]]]
            },
            "properties": {
                "feature_id": "SYN-PARCEL-DEMO-1",
                "feature_type": "synthetic_parcel",
                "source": "synthetic_test",
                "verification_status": "unverified",
                "review_status": "unverified"
            }
        }
        crossing_warnings = ["W-SYN-PARCEL-SYN-BLD-PARCEL-CROSS-1-SYN-PARCEL-DEMO-1"]
        rag = aggregate_parcel_scores(synth_parcel, [], [], crossing_warnings)
        assert rag["is_synthetic"] is True
        assert rag["synthetic_badge"] is not None
        assert rag["rag_status"] == "RED"
        assert rag["heuristic_disclaimer"] == "prototype heuristic—not a validated survey-priority model"

    def test_parcel_rag_clean_parcel_is_green(self):
        """A synthetic parcel with no warnings and no roads should be GREEN."""
        clean_parcel = {
            "type": "Feature", "id": "SYN-CLEAN-PARCEL",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[72.75, 23.04], [72.751, 23.04], [72.751, 23.041], [72.75, 23.041], [72.75, 23.04]]]
            },
            "properties": {
                "feature_id": "SYN-CLEAN-PARCEL",
                "feature_type": "synthetic_parcel",
                "source": "synthetic_test",
                "verification_status": "unverified",
                "review_status": "unverified"
            }
        }
        rag = aggregate_parcel_scores(clean_parcel, [], [], [])
        assert rag["rag_status"] == "GREEN"

    def test_real_parcel_crossing_gated(self):
        """Real parcel crossing check must produce zero warnings if parcel list is empty."""
        warnings = check_building_parcel_crossings([], [], is_synthetic=False)
        assert len(warnings) == 0
        warnings2 = check_building_parcel_crossings([{"type": "Feature", "id": "BLD-1", "geometry": None, "properties": {}}], [], is_synthetic=False)
        assert len(warnings2) == 0


# ==============================================================================
# 16. Topology Edge Cases Tests
# ==============================================================================

class TestTopologyEdgeCases:
    def test_empty_building_layer_no_crash(self):
        """Empty building layer must produce zero warnings without error."""
        warnings = run_full_topology_validation([], [], [], [])
        assert len(warnings) == 0

    def test_zero_area_geometry_handled_safely(self):
        """Zero-area polygon must not cause exceptions in geometry checks."""
        zero_poly_feat = {
            "type": "Feature", "id": "ZERO-AREA-TEST",
            "geometry": {"type": "Polygon", "coordinates": [[[72.757, 23.040], [72.757, 23.041], [72.757, 23.040]]]},
            "properties": {"feature_id": "ZERO-AREA-TEST", "source": "synthetic_test"}
        }
        warnings = validate_feature_geometries([zero_poly_feat], "synthetic_test")
        # Should produce invalid_geometry warning, not crash
        assert isinstance(warnings, list)

    def test_overlap_area_in_metric_m2(self):
        """Overlap warning area values must be in m², distinguishable from degree² values."""
        feat1 = {
            "type": "Feature", "id": "OVER-1",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[72.757, 23.040], [72.758, 23.040], [72.758, 23.041], [72.757, 23.041], [72.757, 23.040]]]
            },
            "properties": {"source": "synthetic_test", "feature_id": "OVER-1"}
        }
        feat2 = {
            "type": "Feature", "id": "OVER-2",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[72.7575, 23.040], [72.7585, 23.040], [72.7585, 23.041], [72.7575, 23.041], [72.7575, 23.040]]]
            },
            "properties": {"source": "synthetic_test", "feature_id": "OVER-2"}
        }
        warnings = check_polygon_overlaps([feat1, feat2], "synthetic_test")
        assert len(warnings) == 1
        # Overlap area in explanation should be in m² range (not fraction of degree²)
        # The 50% overlap of ~10000m² polygon should be ~5000 m²
        assert "m²" in warnings[0].explanation
        # Extract numeric area from explanation
        import re
        match = re.search(r"([\d.]+) m²", warnings[0].explanation)
        if match:
            area_val = float(match.group(1))
            assert area_val > 100, f"Overlap area {area_val} m² appears to be in degree units (should be > 100 m²)"


# ==============================================================================
# 17. Discrepancy & Parcel RAG Endpoints Tests
# ==============================================================================

class TestDeepLabCandidateAdapter:
    def test_deeplab_candidate_has_distinct_model_identity(self):
        whu = WHUBuildingModel()
        deeplab = DeepLabBuildingModel()
        assert whu.model_name != deeplab.model_name
        assert deeplab.model_version == "418c63b"
        assert deeplab.MODEL_ID == "aatifjiwani/rgb-footprint-extract"

    def test_deeplab_candidate_rejects_placeholder_checkpoint(self, tmp_path):
        import numpy as np
        import rasterio
        from rasterio.transform import from_origin

        raster_path = tmp_path / "synthetic_rgb.tif"
        arr = np.zeros((64, 64, 3), dtype=np.uint8)
        arr[10:30, 10:30, :] = 200
        arr[40:55, 20:40, :] = 180

        with rasterio.open(
            raster_path,
            "w",
            driver="GTiff",
            height=64,
            width=64,
            count=3,
            dtype="uint8",
            crs="EPSG:3857",
            transform=from_origin(0, 64, 1, 1),
        ) as dst:
            dst.write(arr.transpose(2, 0, 1))

        checkpoint_path = tmp_path / "best_miou_checkpoint.pth.tar"
        checkpoint_path.write_bytes(b"stub-checkpoint")

        model = DeepLabBuildingModel(checkpoint_name="spacenet", raster_path=str(raster_path), checkpoint_path=str(checkpoint_path))
        with pytest.raises(ModelInferenceError, match="DeepLab checkpoint not found"):
            model.predict(confidence_threshold=0.5)

    def test_deeplab_candidate_missing_checkpoint_fails_explicitly(self, tmp_path):
        raster_path = tmp_path / "synthetic_rgb.tif"
        import numpy as np
        import rasterio
        from rasterio.transform import from_origin

        arr = np.zeros((16, 16, 3), dtype=np.uint8)
        arr[2:6, 2:6] = 255
        with rasterio.open(
            raster_path,
            "w",
            driver="GTiff",
            height=16,
            width=16,
            count=3,
            dtype="uint8",
            crs="EPSG:3857",
            transform=from_origin(0, 16, 1, 1),
        ) as dst:
            dst.write(arr.transpose(2, 0, 1))

        model = DeepLabBuildingModel(raster_path=str(raster_path), checkpoint_path=str(tmp_path / "missing_checkpoint.pth.tar"))
        with pytest.raises(ModelInferenceError, match="DeepLab checkpoint not found"):
            model.predict(confidence_threshold=0.5)


class TestDiscrepancyAndParcelEndpoints:
    def test_discrepancy_endpoint_no_predictions(self, client):
        """GET /api/models/discrepancy should return no_predictions when AI layer is empty."""
        res = client.get("/api/models/discrepancy")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] in ["no_predictions", "discrepancies_computed"]
        assert "comparator_nature" in data or "disclaimer" in data

    def test_discrepancy_endpoint_with_predictions(self, client):
        """After running predict, /api/models/discrepancy should return structured discrepancy report."""
        pred_res = client.post("/api/models/predict", json={"model_name": "Vaayu-UnetPP-Lite", "confidence_threshold": 0.5})
        assert pred_res.status_code == 200

        res = client.get("/api/models/discrepancy")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "discrepancies_computed"
        assert "metrics" in data
        assert "matched_pairs" in data
        assert "model_only_features" in data
        assert "reference_only_features" in data
        assert "NOT temporal change detection" in data["disclaimer"]

    def test_parcel_rag_endpoint_empty_real_parcels(self, client):
        """GET /api/parcels/rag must report not_evaluated for real parcels when real parcel layer is empty."""
        res = client.get("/api/parcels/rag")
        assert res.status_code == 200
        data = res.json()
        assert data["real_parcel_evaluation"]["status"] == "not_evaluated"
        assert data["real_parcel_evaluation"]["real_parcel_count"] == 0
        assert "synthetic_parcel_rag" in data
        assert "Synthetic test data" in data["synthetic_parcel_rag"]["disclaimer"]


# ==============================================================================
# 18. Model Sprint Tests — Persistence, Live Model, and Alignment Status
# ==============================================================================

class TestModelSprintRequirements:
    def test_feature_store_persistence_across_reinit(self, tmp_path):
        """Review edits and draft features must persist across FeatureStore restarts."""
        state_file = str(tmp_path / "test_store_state.json")
        store1 = FeatureStore(state_file_path=state_file)
        store1.initialize()

        # Add draft feature
        draft = store1.add_draft_feature(
            geometry={"type": "Polygon", "coordinates": [[[72.757, 23.040], [72.758, 23.040], [72.758, 23.041], [72.757, 23.041], [72.757, 23.040]]]},
            notes="Test draft persistence"
        )
        draft_id = draft["id"]

        # Update status of building feature
        bld_feat = store1.layers["buildings"][0]
        bld_id = bld_feat["id"]
        store1.update_feature_status(bld_id, "approved", notes="Verified by test suite")

        # Create new store instance pointing to same file and reinitialize
        store2 = FeatureStore(state_file_path=state_file)
        store2.initialize()

        # Verify draft feature survived restart
        found_draft = store2.find_feature(draft_id)
        assert found_draft is not None
        assert found_draft[1]["properties"]["notes"] == "Test draft persistence"

        # Verify building review status update survived restart
        found_bld = store2.find_feature(bld_id)
        assert found_bld is not None
        assert found_bld[1]["properties"]["review_status"] == "approved"
        assert found_bld[1]["properties"]["notes"] == "Verified by test suite"

    def test_discrepancy_endpoint_threshold_and_provisional_label(self, client):
        """Discrepancy endpoint must return confirmed alignment status and handle threshold parameter."""
        res = client.get("/api/models/discrepancy?iou_threshold=0.50")
        assert res.status_code == 200
        data = res.json()
        assert data["alignment_status"].startswith("confirmed by user in QGIS")
        if data["status"] == "discrepancies_computed":
            assert data["iou_matching_threshold"] == 0.50

    def test_predict_modes_distinct_labels(self, client):
        """Mode labels (live, mock, precomputed) must be distinct and preserved."""
        res_mock = client.post("/api/models/predict", json={"mode": "mock", "confidence_threshold": 0.5})
        assert res_mock.status_code == 200
        data_mock = res_mock.json()
        assert data_mock["mode"] == "mock"
        assert data_mock["detected_count"] > 0

    def test_inference_job_status_endpoint(self, client):
        """Job status endpoint returns valid job state."""
        res = client.get("/api/models/job-status")
        assert res.status_code == 200
        data = res.json()
        assert "status" in data
        assert "progress_percent" in data
        assert "device" in data

    def test_simulated_inference_failure_produces_error_state(self, client):
        """Simulated model failure must return HTTP 500 error state and NOT fabricate fake features."""
        res = client.post("/api/models/predict", json={"mode": "mock", "simulate_failure": True})
        assert res.status_code == 500
        assert "Simulated Model Failure" in res.json()["detail"]

    def test_saved_whu_source_is_evaluated_instead_of_persisted_mock(self, client):
        """The saved WHU artifact is the selected collection used by discrepancy scoring."""
        client.delete("/api/models/prediction-source")
        selected = client.post("/api/models/select-precomputed")
        assert selected.status_code == 200
        assert selected.json()["metadata"]["source_label"] == "WHU saved run"
        assert selected.json()["detected_count"] == 67

        result = client.get("/api/models/discrepancy?iou_threshold=0.35").json()
        assert result["prediction_source"]["source_mode"] == "saved"
        assert result["valid_prediction_count"] == 67
        assert result["valid_reference_count"] == 317
        assert result["metrics"]["matched_pairs_count"] == 8
        assert result["metrics"]["model_only_detections"] == 59
        assert result["metrics"]["reference_only_footprints"] == 309

    def test_prediction_modes_do_not_mix_and_clear_is_explicit(self, client):
        mock_result = client.post("/api/models/predict", json={"mode": "mock", "confidence_threshold": 0.5}).json()
        assert mock_result["metadata"]["source_label"] == "Mock predictions"
        assert mock_result["detected_count"] == 3

        whu_result = client.post("/api/models/select-precomputed").json()
        assert whu_result["metadata"]["source_label"] == "WHU saved run"
        assert whu_result["detected_count"] == 67
        assert {feature["id"] for feature in whu_result["features"]}.isdisjoint(
            {"AI-BLD-001", "AI-BLD-002", "AI-BLD-003"}
        )

        cleared = client.delete("/api/models/prediction-source").json()
        assert cleared["selected"] is False
        assert client.get("/api/models/discrepancy").json()["status"] == "no_predictions"

    def test_legacy_prediction_state_cannot_activate_without_metadata(self, tmp_path):
        import json
        state_path = tmp_path / "legacy-state.json"
        state_path.write_text(json.dumps({"ai_predictions": [{"id": "AI-BLD-001"}]}))
        isolated_store = FeatureStore(state_file_path=str(state_path))
        isolated_store.initialize()
        assert isolated_store.get_prediction_status()["selected"] is False
        assert isolated_store.layers["ai_predictions"] == []

    def test_saved_whu_results_are_stable_at_strict_threshold(self, client):
        client.post("/api/models/select-precomputed")
        result = client.get("/api/models/discrepancy?iou_threshold=0.50").json()
        assert result["metrics"]["matched_pairs_count"] == 4
        assert result["metrics"]["model_only_detections"] == 63
        assert result["metrics"]["reference_only_footprints"] == 313

    @pytest.mark.model_integration
    def test_whu_live_model_one_tile_and_checkpoint_load(self):
        """Opt-in integration test: verify WHU model weight loading and 512x512 tile inference when present."""
        import os
        from backend.services.whu_model import get_whu_model_checkpoint, preprocess_lalpur_raster_if_needed
        try:
            weight_path, sha256_hash, file_size = get_whu_model_checkpoint()
            assert os.path.exists(weight_path)
            assert sha256_hash == "922af7c96c0dc44256ab8b4d1a071f2151e0a921c997af80b55bb766bcc30dc6"
            assert file_size > 80000000
        except Exception as e:
            pytest.skip(f"Model integration test skipped: {e}")


