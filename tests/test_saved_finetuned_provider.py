import hashlib
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.api import app
from backend.services.feature_store import (
    SAVED_FINETUNED_LAYER,
    FeatureStore,
    SavedPredictionLoadError,
    store,
)

RUN_DIR = Path("data/local_model_run/lalpur_improvement/run-deeplab-20261004-141612")
BASELINE_PATH = RUN_DIR / "baseline_predictions.geojson"


def _geometry_fingerprints(features):
    return {
        hashlib.sha256(
            json.dumps(feature["geometry"], sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        for feature in features
    }


def test_exact_saved_run_loads_with_separate_stable_provider_and_provenance():
    isolated_store = FeatureStore(state_file_path=None)
    active_before = isolated_store.get_prediction_status()
    collection = isolated_store.get_layer_collection(SAVED_FINETUNED_LAYER)
    provider = collection["metadata"]["provider"]
    saved_geojson = json.loads((RUN_DIR / "finetuned_predictions.geojson").read_text(encoding="utf-8"))

    assert Path(isolated_store.saved_finetuned_run_dir).resolve() == RUN_DIR.resolve()
    assert collection["features"] == saved_geojson["features"]
    assert collection["total_features"] == 78
    assert provider["provider_id"] == "deeplab-lalpur-finetuned-20261004-141612"
    assert provider["run_id"] == "run-deeplab-20261004-141612"
    assert provider["confidence_threshold"] == 0.5
    assert provider["target_gsd_m"] == 0.3
    assert provider["training_patch_count"] == 72
    assert provider["epochs"] == 20
    assert provider["inference_stride"] == 512
    assert provider["reference_feature_count"] == 317
    assert provider["metric_status"] == "in_sample_full_AOI_fit_not_held_out"
    assert provider["checkpoint_sha256"] == hashlib.sha256(
        (RUN_DIR / "fine_tuned_checkpoint.pth").read_bytes()
    ).hexdigest()
    assert provider["input_raster_sha256"] == "349a8f20f50ebff3806166a038ae39601b5c65c716d2d062c0d26e839ad76785"
    assert provider["reference_geojson_sha256"] == "12b70bb15992865c2f09be796ce8328fcd1c4aefbaa68e279bcc2f6afc343d76"
    assert "all 317 reference building footprints were used" in provider["warning"].lower()
    assert "in-sample" in provider["warning"]
    assert "generalization claim" in provider["warning"]
    assert provider["metrics"]["candidate"]["pixel"]["f1"] == pytest.approx(0.913, abs=0.0005)
    assert provider["metrics"]["candidate"]["polygon"]["f1"] == pytest.approx(0.314, abs=0.0005)
    assert provider["metrics"]["candidate"]["polygon"]["f1_iou_0_50"] == pytest.approx(0.223, abs=0.0005)
    assert "not building-detection accuracy" in provider["metrics"]["pixel_f1_definition"]
    assert "IoU >= 0.35" in provider["metrics"]["object_f1_iou_035_definition"]
    assert isolated_store.get_prediction_status() == active_before
    assert SAVED_FINETUNED_LAYER not in isolated_store.layers
    assert collection["metadata"]["layer_role"] == "experimental_building_predictions_read_only"


def test_finetuned_candidate_does_not_replace_baseline_or_reference_layers():
    client = TestClient(app)
    candidate_before = json.loads((RUN_DIR / "finetuned_predictions.geojson").read_text(encoding="utf-8"))
    baseline_before = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    baseline_digest = hashlib.sha256(BASELINE_PATH.read_bytes()).hexdigest()

    candidate_response = client.get(f"/api/layers/{SAVED_FINETUNED_LAYER}")
    baseline_layer_response = client.get("/api/layers/ai_deeplab_predictions")
    mock_layer_response = client.get("/api/layers/ai_mock_predictions")
    reference_response = client.get("/api/layers/buildings")

    assert candidate_response.status_code == 200
    assert baseline_layer_response.status_code == 200
    assert mock_layer_response.status_code == 200
    assert reference_response.status_code == 200
    candidate = candidate_response.json()
    assert len(candidate["features"]) == 78
    assert len(baseline_before["features"]) == 81
    assert _geometry_fingerprints(candidate["features"]) == _geometry_fingerprints(candidate_before["features"])
    assert _geometry_fingerprints(candidate["features"]).isdisjoint(
        _geometry_fingerprints(baseline_before["features"])
    )
    assert baseline_layer_response.json()["features"] != candidate["features"]
    assert len(reference_response.json()["features"]) == 317
    assert hashlib.sha256(BASELINE_PATH.read_bytes()).hexdigest() == baseline_digest

    html = Path("frontend/index.html").read_text(encoding="utf-8")
    app_js = Path("frontend/js/app.js").read_text(encoding="utf-8")
    map_js = Path("frontend/js/map.js").read_text(encoding="utf-8")
    for toggle_id in (
        "toggle-ai-whu",
        "toggle-ai-deeplab",
        "toggle-ai-deeplab-finetuned",
        "toggle-ai-mock",
        "toggle-buildings",
    ):
        assert toggle_id in html
        assert toggle_id in app_js
    assert "experimental; in-sample" in html
    assert "not parcels" in html
    assert "ai_deeplab_finetuned_predictions: L.geoJSON" in map_js
    assert "layer: 'ai_deeplab_finetuned_predictions'" in app_js


def test_provider_metadata_and_warning_remain_available_from_loaded_layer():
    client = TestClient(app)
    collection = client.get(f"/api/layers/{SAVED_FINETUNED_LAYER}").json()
    provider = collection["metadata"]["provider"]

    assert provider["provider_id"] == "deeplab-lalpur-finetuned-20261004-141612"
    assert provider["run_id"] == "run-deeplab-20261004-141612"
    assert provider["checkpoint_sha256"]
    assert provider["input_raster_sha256"]
    assert provider["reference_geojson_sha256"]
    assert "All 317 reference building footprints" in provider["warning"]
    assert collection["metadata"]["parcel_status"] == "not_evaluated"


def test_missing_or_malformed_saved_run_fails_without_fallback(tmp_path, monkeypatch):
    missing_store = FeatureStore(state_file_path=None, saved_finetuned_run_dir=str(tmp_path / "missing-run"))
    with pytest.raises(SavedPredictionLoadError, match="run directory is missing"):
        missing_store.get_layer_collection(SAVED_FINETUNED_LAYER)

    incomplete_run = tmp_path / "incomplete-run"
    incomplete_run.mkdir()
    incomplete_store = FeatureStore(state_file_path=None, saved_finetuned_run_dir=str(incomplete_run))
    with pytest.raises(SavedPredictionLoadError, match="finetuned_predictions.geojson"):
        incomplete_store.get_layer_collection(SAVED_FINETUNED_LAYER)

    malformed_run = tmp_path / "malformed-run"
    malformed_run.mkdir()
    (malformed_run / "finetuned_predictions.geojson").write_text("{", encoding="utf-8")
    malformed_store = FeatureStore(state_file_path=None, saved_finetuned_run_dir=str(malformed_run))
    with pytest.raises(SavedPredictionLoadError, match="Could not read saved DeepLab artifact"):
        malformed_store.get_layer_collection(SAVED_FINETUNED_LAYER)

    monkeypatch.setattr(store, "saved_finetuned_run_dir", str(tmp_path / "missing-api-run"))
    monkeypatch.setattr(store, "_saved_finetuned_collection", None)
    response = TestClient(app).get(f"/api/layers/{SAVED_FINETUNED_LAYER}")
    assert response.status_code == 503
    assert "run directory is missing" in response.json()["detail"]


def test_finetuned_buildings_never_enable_parcel_evaluation():
    client = TestClient(app)
    candidate = client.get(f"/api/layers/{SAVED_FINETUNED_LAYER}").json()
    parcel_layer = client.get("/api/layers/parcels").json()
    parcel_status = client.get("/api/parcels/rag").json()["real_parcel_evaluation"]

    assert candidate["metadata"]["parcel_status"] == "not_evaluated"
    assert all("parcel" not in feature.get("properties", {}).get("feature_type", "").lower()
               for feature in candidate["features"])
    assert parcel_layer["total_features"] == 0
    assert parcel_status["status"] == "not_evaluated"
    assert parcel_status["real_parcel_count"] == 0
