import json

import numpy as np
import rasterio
from rasterio.transform import from_origin

from backend.services.greenery_detection import detect_rgb_greenery
from backend.services.mask_cleanup import clean_binary_mask
from backend.services.warning_narrator import narrate_warnings


def test_mask_cleanup_is_noop_by_default_and_removes_isolated_noise():
    mask = np.zeros((11, 11), dtype=np.uint8)
    mask[3:8, 3:8] = 1
    mask[1, 1] = 1

    unchanged = clean_binary_mask(mask)
    opened = clean_binary_mask(mask, opening_px=3)

    assert np.array_equal(unchanged, mask)
    assert opened[1, 1] == 0
    assert opened[4, 4] == 1


def test_mask_cleanup_closing_fills_a_small_gap():
    mask = np.zeros((11, 11), dtype=np.uint8)
    mask[3:8, 3:8] = 1
    mask[5, 5] = 0

    closed = clean_binary_mask(mask, closing_px=3)

    assert closed[5, 5] == 1


def test_mask_cleanup_rejects_even_or_unsupported_kernel():
    with __import__('pytest').raises(ValueError):
        clean_binary_mask(np.zeros((3, 3), dtype=np.uint8), opening_px=4)


def test_rgb_excess_green_returns_georeferenced_candidate_and_saves_geojson(tmp_path):
    raster_path = tmp_path / "rgb.tif"
    output_path = tmp_path / "greenery.geojson"
    data = np.zeros((4, 32, 32), dtype=np.uint8)
    data[0:3] = 100
    data[3] = 255
    data[0, 8:24, 8:24] = 20
    data[1, 8:24, 8:24] = 200
    data[2, 8:24, 8:24] = 20

    with rasterio.open(
        raster_path,
        "w",
        driver="GTiff",
        height=32,
        width=32,
        count=4,
        dtype="uint8",
        crs="EPSG:3857",
        transform=from_origin(8099000, 2637000, 0.5, 0.5),
        colorinterp=(rasterio.enums.ColorInterp.red, rasterio.enums.ColorInterp.green,
                     rasterio.enums.ColorInterp.blue, rasterio.enums.ColorInterp.alpha),
    ) as dst:
        dst.write(data)

    result = detect_rgb_greenery(
        str(raster_path), index_threshold=0.15, min_area_sqm=1.0,
        output_path=str(output_path), max_analysis_pixels=10_000,
    )

    assert result["metadata"]["method"].startswith("RGB Excess Green")
    assert result["metadata"]["candidate_count"] >= 1
    assert result["features"][0]["properties"]["source"] == "rgb_excess_green_heuristic"
    assert result["features"][0]["properties"]["training_required"] is False
    saved = json.loads(output_path.read_text())
    assert saved["type"] == "FeatureCollection"
    coords = saved["features"][0]["geometry"]["coordinates"]
    assert coords


def test_warning_narrator_uses_original_template_when_disabled(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("WARNING_LLM_ENABLED", "true")
    warning = {
        "warning_id": "W-1",
        "warning_type": "overlapping_polygons",
        "severity": "medium",
        "explanation": "Two polygons overlap spatially.",
        "plain_language_rule": "Review polygon boundaries.",
        "feature_ids": ["A", "B"],
    }

    result = narrate_warnings([warning])

    assert result["fallback"] is True
    assert result["narratives"]["W-1"] == {"text": "Two polygons overlap spatially.", "source": "template"}


def test_warning_narrator_is_deterministic_when_llm_environment_is_enabled(monkeypatch):
    monkeypatch.setenv("WARNING_LLM_ENABLED", "true")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    warning = {
        "warning_id": "W-1", "warning_type": "overlap", "severity": "medium",
        "explanation": "Template sentence.", "plain_language_rule": "Review visually.",
        "feature_ids": ["A", "B"],
        "evidence": {"overlap_area_m2": 3.25},
        "tolerance": {"name": "overlap_area_m2", "value": 1.0, "unit": "m2"},
        "suggested_action": {"action_type": "desk_review", "text": "Check both boundaries."},
    }

    result = narrate_warnings([warning])

    narrative = result["narratives"]["W-1"]
    assert result["enabled"] is False
    assert result["mode"] == "deterministic"
    assert narrative["source"] == "template"
    assert "3.25 m2" in narrative["text"]
    assert "is 1 m2" in narrative["text"]
    assert "Check both boundaries" in narrative["text"]


def test_greenery_api_forwards_sidebar_options(monkeypatch):
    from fastapi.testclient import TestClient
    from backend import api

    received = {}
    monkeypatch.setattr(api.raster_tile_service, "is_available", True)
    monkeypatch.setattr(api.raster_tile_service, "tif_path", "/tmp/lalpur-test.tif")

    def fake_detect(path, **options):
        received.update(path=path, **options)
        return {"type": "FeatureCollection", "features": [], "metadata": {"candidate_count": 0}}

    monkeypatch.setattr(api, "detect_rgb_greenery", fake_detect)
    client = TestClient(api.app)
    response = client.post("/api/greenery/detect", json={
        "index_threshold": 0.25,
        "min_area_sqm": 12,
        "morphology_opening_px": 3,
        "morphology_closing_px": 5,
    })

    assert response.status_code == 200
    assert received == {
        "path": "/tmp/lalpur-test.tif",
        "index_threshold": 0.25,
        "min_area_sqm": 12,
        "morphology_opening_px": 3,
        "morphology_closing_px": 5,
    }


def test_warning_narration_api_uses_template_fallback(monkeypatch):
    from fastapi.testclient import TestClient
    from types import SimpleNamespace
    from backend import api

    warning = SimpleNamespace(warning_id="API-W-1", source="verified_reference")
    monkeypatch.setattr(api.store, "initialize", lambda: None)
    monkeypatch.setattr(api.store, "warnings", [warning])
    monkeypatch.setattr(api, "narrate_warnings", lambda warnings: {
        "enabled": False,
        "fallback": True,
        "narratives": {warnings[0].warning_id: {"text": "Original rule text.", "source": "template"}},
    })

    client = TestClient(api.app)
    response = client.post("/api/warnings/narrate", json={"warning_ids": ["API-W-1"]})

    assert response.status_code == 200
    assert response.json()["fallback"] is True
    assert response.json()["narratives"]["API-W-1"]["source"] == "template"
