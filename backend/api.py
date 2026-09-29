"""
FastAPI Application and REST API for SIH26012 Feature Review Platform.
Provides endpoints for layers, inspection, human edits, topology warnings,
transparent scoring, model inference stub, benchmark status, and GeoJSON export/import.
"""
import os
import json
import hashlib
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, status, Query, Body, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse

from backend.services.feature_store import store
from backend.services.data_loader import PROVENANCE_DISCLAIMER
from backend.services.raster_service import raster_tile_service
from backend.services.model_adapter import MockBuildingModel, BenchmarkDatasetAdapter, ModelInferenceError
from backend.services.scoring import aggregate_parcel_scores
from backend.models.schemas import (
    GeometryEditRequest,
    FeatureStatusUpdateRequest,
    DraftFeatureCreateRequest,
    ModelPredictRequest
)

from contextlib import asynccontextmanager

def _file_sha256(path: Optional[str]) -> Optional[str]:
    if not path or not os.path.isfile(path):
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def _persist_model_run(result: Dict[str, Any], mode: str, run_id: str, metadata: Dict[str, Any]) -> str:
    run_dir = os.path.join("data", "local_model_run", "runs", mode, run_id)
    os.makedirs(run_dir, exist_ok=True)
    output_path = os.path.join(run_dir, "predictions.geojson")
    artifact = {
        **result,
        "model_metadata": {
            **metadata,
            "run_id": run_id,
            "output_path": output_path,
            "valid_feature_count": len(result.get("features", [])),
        },
    }
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(artifact, handle, indent=2)
    return output_path

@asynccontextmanager
async def lifespan(app: FastAPI):
    store.initialize()
    yield

app = FastAPI(
    title="SIH26012 Feature Review Platform API",
    description="Geospatial feature-review and topology inspection platform for Lalpur study area.",
    version="0.1.0",
    lifespan=lifespan
)

# CORS middleware for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/api/health")
def get_health():
    """Health check and active layer status."""
    store.initialize()
    return {
        "status": "online",
        "study_area": "Lalpur Village, Gujarat (LGD 511638)",
        "layers_loaded": {k: len(v) for k, v in store.layers.items()},
        "total_warnings": len(store.warnings)
    }

@app.get("/api/metadata")
def get_metadata():
    """System metadata, study area bounds, CRS, and provenance notices."""
    return {
        "aoi_id": "SIH26012_INDIA_CANDIDATE_01_LALPUR",
        "locality": "Lalpur Village, Gujarat, India (LGD Code: 511638)",
        "bounds_epsg_4326": {
            "lon_min": 72.754522,
            "lat_min": 23.037534,
            "lon_max": 72.760638,
            "lat_max": 23.043371,
            "centroid": [72.757580, 23.040453]
        },
        "coordinate_systems": {
            "web_map_input": "EPSG:4326 (WGS 84 / RFC 7946)",
            "native_raster_crs": "EPSG:3857 (Web Mercator)"
        },
        "raster_orthomosaic_status": {
            "filename": "lalpur_orthomosaic.tif",
            "format": "GeoTIFF (4 bands, uint8, EPSG:3857, 20,137 x 20,886 px, 3.38 cm GSD)",
            "bounds_epsg_3857": [8098996.3782, 2636558.4073, 8099677.1552, 2637264.506],
            "file_size_bytes": 1682493007,
            "browser_service_status": "AVAILABLE_LOCAL_XYZ_TILES" if raster_tile_service.is_available else "UNAVAILABLE",
            "tile_url": "/api/raster/tiles/{z}/{x}/{y}.png",
            "min_zoom": 14,
            "max_zoom": 21,
            "details": raster_tile_service.metadata
        },
        "provenance_disclaimer": PROVENANCE_DISCLAIMER,
        "cadastral_notice": (
            "This application is a feature-review prototype and is NOT an official cadastral system. "
            "Building footprints are unverified physical envelopes, not cadastral parcels. "
            "No official parcel ground truth exists for this AOI (parcel template has 0 features)."
        ),
        "osm_attribution": "© OpenStreetMap contributors (ODbL 1.0)"
    }

@app.get("/api/raster/status")
def get_raster_status():
    """Returns technical metadata and availability of the local GeoTIFF orthomosaic."""
    return {
        "status": "ready" if raster_tile_service.is_available else "unavailable",
        "metadata": raster_tile_service.metadata,
        "tile_template": "/api/raster/tiles/{z}/{x}/{y}.png"
    }

@app.get("/api/raster/tiles/{z}/{x}/{y}.png")
def get_raster_tile(z: int, x: int, y: int):
    """Dynamically serves 256x256 Web Mercator PNG tile extracted from local GeoTIFF."""
    tile_bytes = raster_tile_service.get_tile_png(z, x, y)
    return Response(
        content=tile_bytes,
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=3600"}
    )

@app.get("/api/layers/{layer_name}")
def get_layer(layer_name: str):
    """Returns GeoJSON FeatureCollection for specified layer."""
    try:
        return store.get_layer_collection(layer_name)
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Layer '{layer_name}' not found. Available layers: {list(store.layers.keys())}"
        )

@app.get("/api/warnings")
def get_warnings():
    """Returns list of all active topology warnings."""
    store.initialize()
    return {
        "total_warnings": len(store.warnings),
        "synthetic_warning_count": sum(1 for w in store.warnings if "synthetic" in w.source),
        "real_warning_count": sum(1 for w in store.warnings if "synthetic" not in w.source),
        "warnings": store.warnings
    }

@app.get("/api/scores")
def get_scores():
    """Returns review-priority scores for all loaded features."""
    store.initialize()
    return {
        "total_scored_features": len(store.scores),
        "heuristic_disclaimer": "prototype heuristic—not a validated survey-priority model",
        "scores": store.scores
    }

@app.get("/api/features/{feature_id}")
def get_feature_details(feature_id: str):
    """Retrieves full details, audit trail, warnings, and score breakdown for a single feature."""
    found = store.find_feature(feature_id)
    if not found:
        raise HTTPException(status_code=404, detail=f"Feature with ID '{feature_id}' not found.")

    layer_name, feat = found
    score_bd = store.scores.get(feature_id)
    w_ids = feat.get("properties", {}).get("warning_ids", [])
    relevant_warnings = [w for w in store.warnings if w.warning_id in w_ids]

    has_geometry_edits = False
    if feat.get("original_geometry") and feat.get("original_geometry") != feat.get("geometry"):
        has_geometry_edits = True

    return {
        "feature": feat,
        "layer": layer_name,
        "score_breakdown": score_bd,
        "associated_warnings": relevant_warnings,
        "has_geometry_edits": has_geometry_edits
    }

@app.put("/api/features/{feature_id}")
def update_feature(feature_id: str, req: FeatureStatusUpdateRequest):
    """Updates review status (under_review, approved, rejected) and inspector notes."""
    try:
        updated = store.update_feature_status(
            feature_id=feature_id,
            review_status=req.review_status,
            notes=req.notes,
            reviewer_label=req.reviewer_label
        )
        return {
            "status": "success",
            "feature": updated,
            "score": store.scores.get(feature_id)
        }
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.post("/api/features/{feature_id}/edit-geometry")
def edit_geometry(feature_id: str, req: GeometryEditRequest):
    """Saves human geometry edits while preserving original_geometry audit snapshot."""
    try:
        updated = store.update_feature_geometry(
            feature_id=feature_id,
            new_geometry=req.geometry.model_dump(),
            reviewer_label=req.reviewer_label,
            edit_reason=req.edit_reason
        )
        return {
            "status": "success",
            "message": "Geometry updated; original geometry preserved in audit snapshot.",
            "feature": updated
        }
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.post("/api/features/{feature_id}/revert")
def revert_geometry(feature_id: str):
    """Reverts a feature's geometry back to its original unedited geometry."""
    try:
        reverted = store.revert_feature_geometry(feature_id)
        return {
            "status": "success",
            "message": "Geometry reverted to original baseline.",
            "feature": reverted
        }
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.post("/api/features/draft")
def create_draft(req: DraftFeatureCreateRequest):
    """Adds a newly drawn draft feature (tagged manual_visual_reference)."""
    draft = store.add_draft_feature(
        geometry=req.geometry.model_dump(),
        feature_type=req.feature_type,
        notes=req.notes,
        reviewer_label=req.reviewer_label
    )
    return {"status": "success", "feature": draft}

from fastapi import FastAPI, HTTPException, status, Query, Body, Response, BackgroundTasks
from backend.services.model_adapter import MockBuildingModel, WHUBuildingModel, DeepLabBuildingModel, BenchmarkDatasetAdapter, ModelInferenceError
from backend.services.whu_model import job_manager, run_whu_live_inference

@app.post("/api/models/predict")
def predict_building_model(req: ModelPredictRequest):
    """
    Triggers building model inference using specified mode:
    - 'live': Live model inference running WHU U-Net++ EfficientNet-B4 over Lalpur raster
    - 'mock': Mock provider fallback for developer verification
    - 'precomputed': Demo output loaded from local manifest
    """
    try:
        output_path = None
        if req.mode == "precomputed":
            output_path = os.path.join("data", "local_model_run", "predicted_buildings_4326.geojson")
            with open(output_path, "r", encoding="utf-8") as handle:
                result = json.load(handle)
            metadata = result.get("model_metadata", {})
        else:
            if req.mode == "live":
                model = WHUBuildingModel(model_name=req.model_name, model_version=req.model_version)
            elif req.mode == "deeplab":
                model = DeepLabBuildingModel(
                    model_name=req.model_name or "aatifjiwani/rgb-footprint-extract",
                    model_version=req.model_version or "418c63b",
                    checkpoint_name="spacenet",
                )
            elif req.mode == "mock":
                model = MockBuildingModel(model_name=req.model_name, model_version=req.model_version)
            else:
                raise HTTPException(status_code=400, detail=f"Unknown model provider '{req.mode}'.")
            result = model.predict(
                confidence_threshold=req.confidence_threshold,
                simulate_failure=req.simulate_failure
            )
            metadata = result.get("model_metadata", {})

        run_id = metadata.get("run_id") or f"run-{req.mode}-{int(datetime.now(timezone.utc).timestamp())}"
        provider_id = "whu" if req.mode in {"live", "precomputed"} else req.mode
        source_mode = {
            "mock": "mock",
            "precomputed": "saved",
            "live": "fresh",
            "deeplab": "fresh",
        }[req.mode]
        provider_label = {
            "mock": "Mock predictions",
            "precomputed": "WHU saved run",
            "live": "Fresh WHU inference",
            "deeplab": "RGB Footprint Extract — SpaceNet checkpoint",
        }[req.mode]
        if req.mode != "precomputed":
            output_path = _persist_model_run(result, req.mode, run_id, metadata)
        else:
            output_path = output_path or metadata.get("output_path")
        source_metadata = {
            "source_mode": source_mode,
            "request_mode": req.mode,
            "provider_id": provider_id,
            "provider_label": provider_label,
            "source_label": provider_label,
            "model_id": metadata.get("model_name", req.model_name),
            "model_revision": metadata.get("model_version", req.model_version),
            "run_id": run_id,
            "output_path": output_path,
            "checkpoint_sha256": metadata.get("model_weight_sha256") or metadata.get("checkpoint_sha256"),
            "valid_feature_count": len(result.get("features", [])),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "is_mock": req.mode == "mock",
            "alignment_status": metadata.get("alignment_status", "confirmed by user in QGIS; local reference set, not an official/legal accuracy benchmark")
        }
        status_info = store.set_prediction_source(result.get("features", []), source_metadata)
        return {
            "status": "success",
            "mode": req.mode,
            "detected_count": status_info["prediction_count"],
            "features": store.layers["ai_predictions"],
            "metadata": {**metadata, **status_info}
        }
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Saved WHU prediction GeoJSON was not found.")
    except ModelInferenceError as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.get("/api/models/prediction-source")
def get_prediction_source():
    """Return the explicit active prediction source, or an empty selection."""
    return store.get_prediction_status()


@app.post("/api/models/select-precomputed")
def select_precomputed_prediction():
    """Select the saved WHU GeoJSON without modifying that artifact."""
    return predict_building_model(ModelPredictRequest(mode="precomputed"))


@app.post("/api/models/select-deeplab")
def select_deeplab_prediction():
    """Select the DeepLab candidate model if the checkpoint is available."""
    return predict_building_model(ModelPredictRequest(mode="deeplab"))


@app.delete("/api/models/prediction-source")
def clear_prediction_source():
    """Clear the active layer while retaining all saved model artifacts."""
    return store.clear_prediction_source()

@app.post("/api/models/run-inference")
def start_live_model_job(background_tasks: BackgroundTasks, confidence_threshold: float = Body(0.50, embed=True)):
    """
    Launches a local, bounded background job for WHU building model inference
    without freezing the web server request or UI.
    """
    status_info = job_manager.get_status()
    if status_info["status"] == "running":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"An inference job ({status_info['job_id']}) is already in progress ({status_info['progress_percent']}%)."
        )

    def _run_task():
        try:
            res = run_whu_live_inference(confidence_threshold=confidence_threshold)
            metadata = res.get("model_metadata", {})
            store.set_prediction_source(res.get("features", []), {
                "source_mode": "fresh",
                "request_mode": "live",
                "provider_id": "whu",
                "provider_label": "Fresh WHU inference",
                "source_label": "Fresh WHU inference",
                "model_id": metadata.get("model_name", "giswqs/whu-building-unetplusplus-efficientnet-b4"),
                "model_revision": metadata.get("model_version"),
                "run_id": metadata.get("run_id"),
                "output_path": "data/local_model_run/predicted_buildings_4326.geojson",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "is_mock": False,
                "alignment_status": metadata.get("alignment_status", "confirmed by user in QGIS; local reference set, not an official/legal accuracy benchmark")
            })
        except Exception as e:
            job_manager.fail_job(str(e))

    background_tasks.add_task(_run_task)
    return {
        "status": "started",
        "message": "Live model inference job started in background thread.",
        "status_endpoint": "/api/models/job-status"
    }

@app.get("/api/models/job-status")
def get_inference_job_status():
    """Returns status, tile count, duration, device, and result summary of background inference job."""
    return job_manager.get_status()


@app.get("/api/models/benchmarks")
def get_benchmarks():
    """Returns official benchmark datasets manifest and evaluation status."""
    return BenchmarkDatasetAdapter.get_dataset_manifest()

@app.get("/api/export")
def export_bundle():
    """Exports all reviewed features into a single RFC 7946 GeoJSON bundle."""
    bundle = store.export_reviewed_bundle()
    return JSONResponse(
        content=bundle,
        headers={"Content-Disposition": "attachment; filename=SIH26012_Lalpur_Reviewed_Features.geojson"}
    )

@app.post("/api/import")
def import_bundle(bundle: Dict[str, Any] = Body(...)):
    """Imports and validates an exported GeoJSON bundle."""
    try:
        res = store.import_reviewed_bundle(bundle)
        return res
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/parcels/rag")
def get_parcel_rag():
    """
    Returns Red/Amber/Green parcel aggregation status.
    When real parcel layer is empty (0 features), reports 'not_evaluated'.
    When synthetic parcels present, aggregates with 'synthetic' badge.
    """
    store.initialize()
    real_parcels = store.layers["parcels"]
    synthetic_feats = store.layers["synthetic"]
    synth_parcels = [f for f in synthetic_feats if f.get("properties", {}).get("feature_type") == "synthetic_parcel"]

    if len(real_parcels) == 0:
        real_parcel_status = {
            "status": "not_evaluated",
            "reason": "No real parcel polygons loaded for this AOI. "
                      "Official cadastral parcel data is not available for Lalpur (LGD 511638). "
                      "Use QGIS to digitize a manual_visual_reference layer and import it to enable parcel evaluation.",
            "real_parcel_count": 0,
            "disclaimer": "This status is NOT equivalent to 'zero conflicts'. "
                          "It means evaluation was not performed due to absent reference data."
        }
    else:
        real_parcel_status = {"status": "real_parcels_present", "real_parcel_count": len(real_parcels)}

    # Synthetic RAG for demo
    synth_rag_results = []
    for parcel in synth_parcels:
        pid = parcel.get("id") or parcel.get("properties", {}).get("feature_id")
        parcel_warnings = [w.warning_id for w in store.warnings if pid in w.feature_ids]
        intersecting_blds = [f for f in synthetic_feats if f.get("properties", {}).get("feature_type") == "synthetic_building"]
        rag = aggregate_parcel_scores(parcel, intersecting_blds, [], parcel_warnings)
        synth_rag_results.append(rag)

    return {
        "real_parcel_evaluation": real_parcel_status,
        "synthetic_parcel_rag": {
            "disclaimer": "Synthetic test data — not real parcels. Demo only.",
            "parcel_count": len(synth_rag_results),
            "results": synth_rag_results
        },
        "heuristic_disclaimer": "prototype heuristic—not a validated survey-priority model"
    }


@app.get("/api/models/discrepancy")
def get_model_reference_discrepancy(iou_threshold: float = Query(0.35, ge=0.05, le=0.95)):
    """
    Model vs Reference Discrepancy Comparator.
    Compares AI-predicted building footprints against the 317 Vaayu reference footprints
    using spatial IoU matching within the same area. NOT temporal change detection.
    The reference footprints are not an 'old' layer — they are same-area annotations.
    """
    store.initialize()
    prediction_status = store.get_prediction_status()
    ai_preds = store.layers["ai_predictions"] if prediction_status["selected"] else []
    reference = store.layers["buildings"]

    if not ai_preds:
        return {
            "status": "no_predictions",
            "message": "No AI model predictions loaded. Run model inference first via POST /api/models/predict.",
            "disclaimer": "This comparator shows same-area AI-vs-reference disagreement. "
                          "It is NOT temporal change detection. Reference features are unverified annotations.",
            "alignment_status": "confirmed by user in QGIS; local reference set, not an official/legal accuracy benchmark",
            "prediction_source": prediction_status,
            "valid_prediction_count": 0,
            "valid_reference_count": len(store.layers["buildings"]),
            "invalid_prediction_count": 0,
            "invalid_reference_count": 0
        }

    import shapely.geometry
    import shapely.ops
    import pyproj
    from backend.services.model_adapter import BenchmarkDatasetAdapter

    transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:32643", always_xy=True)

    def parse_shape(feat):
        geom = feat.get("geometry")
        if not geom:
            return None
        try:
            s = shapely.geometry.shape(geom)
            if not s.is_valid:
                s = s.buffer(0)
            if s.is_empty:
                return None
            return shapely.ops.transform(transformer.transform, s)
        except Exception:
            return None

    ref_shapes_all = [(f["id"], parse_shape(f)) for f in reference]
    pred_shapes_all = [(f["id"], parse_shape(f)) for f in ai_preds]
    invalid_ref_count = sum(1 for _, shape in ref_shapes_all if shape is None)
    invalid_pred_count = sum(1 for _, shape in pred_shapes_all if shape is None)
    ref_shapes = ref_shapes_all
    ref_shapes = [(fid, s) for fid, s in ref_shapes if s]
    pred_shapes = pred_shapes_all
    pred_shapes = [(fid, s) for fid, s in pred_shapes if s]

    matched_ref = set()
    matched_pred = set()
    matched_pairs = []

    candidates = sorted(
        ((BenchmarkDatasetAdapter.compute_polygon_iou(rs, ps), pid, rid)
         for pid, ps in pred_shapes for rid, rs in ref_shapes),
        key=lambda item: (-item[0], item[1], item[2])
    )
    for iou, pid, rid in candidates:
        if iou < iou_threshold or pid in matched_pred or rid in matched_ref:
            continue
        matched_ref.add(rid)
        matched_pred.add(pid)
        matched_pairs.append({"ai_id": pid, "ref_id": rid, "iou": round(iou, 4)})

    model_only = [pid for pid, _ in pred_shapes if pid not in matched_pred]
    ref_only = [rid for rid, _ in ref_shapes if rid not in matched_ref]

    tp = len(matched_pred)
    fp = len(model_only)
    fn = len(ref_only)
    precision_raw = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall_raw = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_raw = (2 * precision_raw * recall_raw) / (precision_raw + recall_raw) if (precision_raw + recall_raw) > 0 else 0.0
    precision = round(precision_raw, 4)
    recall = round(recall_raw, 4)
    f1 = round(f1_raw, 4)

    return {
        "status": "discrepancies_computed",
        "alignment_status": "confirmed by user in QGIS; local reference set, not an official/legal accuracy benchmark",
        "comparator_nature": "Same-area spatial disagreement check (NOT temporal change detection)",
        "disclaimer": "Model vs Reference Discrepancy: same-area AI-vs-annotation disagreement check. "
                      "NOT temporal change detection. 317 reference footprints are a user-aligned local reference set "
                      "(Project Vaayu sample, not an official or legal accuracy benchmark). This is a prototype heuristic.",
        "iou_matching_threshold": iou_threshold,
        "prediction_source": prediction_status,
        "metrics": {
            "matched_pairs_count": len(matched_pairs),
            "model_only_detections": len(model_only),
            "reference_only_footprints": len(ref_only),
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        },
        "total_reference_features": len(ref_shapes),
        "total_ai_predictions": len(pred_shapes),
        "valid_prediction_count": len(pred_shapes),
        "valid_reference_count": len(ref_shapes),
        "invalid_prediction_count": invalid_pred_count,
        "invalid_reference_count": invalid_ref_count,
        "matched_pairs": matched_pairs[:20],
        "model_only_features": model_only[:20],
        "reference_only_features": ref_only[:20],
        "interpretation": {
            "matched": "AI and reference overlap at IoU >= threshold — spatial agreement",
            "model_only": "AI detected footprint not found in reference — possible false positive or unmapped building",
            "ref_only": "Reference footprint not found in AI predictions — possible missed detection"
        }
    }


# Mount static frontend files if folder exists
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.exists(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
