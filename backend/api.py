"""
FastAPI Application and REST API for SIH26012 Feature Review Platform.
Provides endpoints for layers, inspection, human edits, topology warnings,
transparent scoring, model inference stub, benchmark status, and GeoJSON export/import.
"""
import os
import json
import hashlib
import shutil
import tempfile
import math
import uuid
import threading
import time
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, status, Query, Body, Response, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
import pyproj
import rasterio
from rasterio.enums import ColorInterp

from backend.services.feature_store import (
    DEFAULT_LALPUR_PROJECT_ID,
    SAVED_FINETUNED_LAYER,
    SavedPredictionLoadError,
    get_project_store,
    store,
)
from backend.services.data_loader import PROVENANCE_DISCLAIMER
from backend.services.raster_service import raster_tile_service
from backend.services.greenery_detection import detect_rgb_greenery, load_saved_greenery
from backend.services.warning_narrator import narrate_warnings, narration_status
from backend.services.model_adapter import MockBuildingModel, BenchmarkDatasetAdapter, ModelInferenceError
from backend.services.project_registry import project_registry, WORKSPACE_ROOT
from backend.services.scoring import aggregate_parcel_scores
from backend.services.project_inference import get_building_model_providers, run_project_building_inference
from backend.models.schemas import (
    GeometryEditRequest,
    FeatureStatusUpdateRequest,
    DraftFeatureCreateRequest,
    ModelPredictRequest,
    GreeneryDetectRequest,
    WarningNarrationRequest,
)

from contextlib import asynccontextmanager

MAX_PROJECT_RASTER_BYTES = 2 * 1024 * 1024 * 1024
MAX_PROJECT_RASTER_PIXELS = 500_000_000
MIN_PROJECT_DISK_RESERVE_BYTES = 256 * 1024 * 1024

def _file_sha256(path: Optional[str]) -> Optional[str]:
    if not path or not os.path.isfile(path):
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _wgs84_bounds_from_3857(bounds: Optional[Dict[str, float]]) -> Optional[List[List[float]]]:
    if not isinstance(bounds, dict):
        return None
    try:
        minx = float(bounds["minx"])
        miny = float(bounds["miny"])
        maxx = float(bounds["maxx"])
        maxy = float(bounds["maxy"])
    except (TypeError, KeyError, ValueError):
        return None
    transformer = pyproj.Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    west, south = transformer.transform(minx, miny)
    east, north = transformer.transform(maxx, maxy)
    values = (west, south, east, north)
    if not all(math.isfinite(value) for value in values):
        return None
    result = [[min(south, north), min(west, east)], [max(south, north), max(west, east)]]
    if not (-90 <= result[0][0] <= 90 and -90 <= result[1][0] <= 90):
        return None
    if not (-180 <= result[0][1] <= 180 and -180 <= result[1][1] <= 180):
        return None
    return result


def _native_zoom_for_raster(raster: Dict[str, Any]) -> int:
    resolutions = []
    for key in ("gsd_x", "gsd_y"):
        try:
            resolution = float(raster.get(key))
        except (TypeError, ValueError):
            continue
        if math.isfinite(resolution) and resolution > 0:
            resolutions.append(resolution)
    if not resolutions:
        return 21
    zoom = round(math.log2(156543.03392804097 / min(resolutions)))
    return max(0, min(24, zoom))


def _project_raster_payload(project: Dict[str, Any]) -> Dict[str, Any]:
    payload = dict(project)
    raster = dict(payload.get("raster") or {})
    if raster.get("bounds"):
        bounds_wgs84 = _wgs84_bounds_from_3857(raster.get("bounds"))
        if bounds_wgs84 is None:
            raster.pop("bounds_wgs84", None)
        else:
            raster["bounds_wgs84"] = bounds_wgs84
    raster["max_native_zoom"] = _native_zoom_for_raster(raster)
    payload["raster"] = raster
    return payload


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

JOB_LOCK = threading.Lock()
PROJECT_JOBS: Dict[str, Dict[str, Any]] = {}


def _normalise_project_store(project_id: Optional[str]):
    if not project_id or project_id == DEFAULT_LALPUR_PROJECT_ID:
        return store
    state_path = os.path.join(project_registry.project_storage_dir(project_id), "store_state.json")
    return get_project_store(project_id, state_file_path=state_path)


def _queue_project_model_job(project_id: str, *, model: str, threshold: float, aoi: Optional[Dict[str, Any]], resolution_m: Optional[float], min_area_m2: float) -> Dict[str, Any]:
    with JOB_LOCK:
        active = [job for job in PROJECT_JOBS.values() if job.get("status") in {"queued", "running"}]
        if active:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="A job is already running.",
            )
        job_id = uuid.uuid4().hex
        record = {
            "job_id": job_id,
            "project_id": project_id,
            "model": model,
            "threshold": threshold,
            "aoi": aoi,
            "resolution_m": resolution_m,
            "min_area_m2": min_area_m2,
            "status": "queued",
            "progress": {"done": 0, "total": 1, "percent": 0.0},
            "feature_count": 0,
            "run_id": None,
            "error_message": None,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        PROJECT_JOBS[job_id] = record
        return record


def _finish_project_model_job(job_id: str, *, success: bool, feature_count: int = 0, run_id: Optional[str] = None, error_message: Optional[str] = None, result_summary: Optional[Dict[str, Any]] = None) -> None:
    job = PROJECT_JOBS.get(job_id)
    if not job:
        return
    job["status"] = "completed" if success else "failed"
    job["progress"] = {"done": 1, "total": 1, "percent": 100.0}
    job["feature_count"] = feature_count
    job["run_id"] = run_id
    job["error_message"] = error_message
    job["result_summary"] = result_summary
    job["elapsed_seconds"] = result_summary.get("elapsed_seconds") if result_summary else None
    job["updated_at"] = datetime.now(timezone.utc).isoformat()


@asynccontextmanager
async def lifespan(app: FastAPI):
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

@app.middleware("http")
async def cache_control_middleware(request, call_next):
    response = await call_next(request)
    if request.url.path in {"/", "/index.html"} or request.url.path.endswith((".html", ".js", ".css")):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response

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

@app.get("/api/projects")
def list_projects():
    """Returns the available project records, beginning with the Lalpur demo."""
    projects = project_registry.list_projects()
    for project in projects:
        if project["project_id"] == "SIH26012_INDIA_CANDIDATE_01_LALPUR":
            project["status"] = "demo_ready" if raster_tile_service.is_available else "raster_unavailable"
        project["raster"] = _project_raster_payload(project)["raster"]
    return {
        "projects": projects,
        "active_project_id": projects[0]["project_id"] if projects else None,
        "count": len(projects)
    }

@app.get("/api/projects/{project_id}")
def get_project(project_id: str):
    """Returns a single project record or a 404 if it is not present in the registry."""
    try:
        return _project_raster_payload(project_registry.get_project(project_id))
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))


@app.post("/api/projects", status_code=status.HTTP_201_CREATED)
async def create_project(
    name: str = Form(...),
    file: UploadFile = File(...),
    locality: str = Form(""),
):
    """Create a project from a supported, georeferenced RGB GeoTIFF upload."""
    if not name.strip() or len(name.strip()) > 100:
        raise HTTPException(status_code=400, detail="Project name must contain 1 to 100 characters.")
    if len(locality.strip()) > 120:
        raise HTTPException(status_code=400, detail="Locality must be 120 characters or fewer.")
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file was uploaded.")

    safe_filename = os.path.basename(file.filename.replace("\\", "/"))
    if not safe_filename.lower().endswith((".tif", ".tiff")):
        raise HTTPException(status_code=400, detail="Unsupported format. Upload a GeoTIFF (.tif/.tiff).")

    file.file.seek(0, os.SEEK_END)
    upload_size = file.file.tell()
    file.file.seek(0)
    if upload_size == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if upload_size > MAX_PROJECT_RASTER_BYTES:
        raise HTTPException(status_code=413, detail="GeoTIFF exceeds the 2 GiB project upload limit.")

    project_id = f"project-{uuid.uuid4().hex}"
    project_dir = project_registry.project_storage_dir(project_id)
    raster_dir = os.path.join(project_dir, "rasters")
    if shutil.disk_usage(raster_dir).free < upload_size + MIN_PROJECT_DISK_RESERVE_BYTES:
        raise HTTPException(status_code=507, detail="Not enough disk space to safely store this GeoTIFF.")

    temp_path = None
    target_path = os.path.join(raster_dir, safe_filename)
    try:
        with tempfile.NamedTemporaryFile(dir=raster_dir, suffix=".upload", delete=False) as temp_file:
            temp_path = temp_file.name
            shutil.copyfileobj(file.file, temp_file, length=1024 * 1024)
        if os.path.getsize(temp_path) != upload_size:
            raise ValueError("Upload size changed while receiving the file.")

        with rasterio.open(temp_path) as dataset:
            if dataset.crs is None:
                raise ValueError("GeoTIFF has no CRS. Assign a valid CRS before upload.")
            if dataset.crs.to_epsg() != 3857:
                raise ValueError("This build can display project rasters only in EPSG:3857.")
            if dataset.count < 3:
                raise ValueError("RGB GeoTIFF required: at least three raster bands must be present.")
            if dataset.width <= 0 or dataset.height <= 0 or dataset.width * dataset.height > MAX_PROJECT_RASTER_PIXELS:
                raise ValueError("Raster dimensions exceed the supported 500-million-pixel limit.")
            transform = dataset.transform
            transform_values = (transform.a, transform.b, transform.c, transform.d, transform.e, transform.f)
            determinant = transform.a * transform.e - transform.b * transform.d
            if (
                not all(math.isfinite(value) for value in transform_values)
                or determinant == 0
                or transform_values == (1, 0, 0, 0, 1, 0)
            ):
                raise ValueError("GeoTIFF has no usable spatial transform.")
            if not all(math.isfinite(value) for value in dataset.bounds) or dataset.bounds.right <= dataset.bounds.left or dataset.bounds.top <= dataset.bounds.bottom:
                raise ValueError("GeoTIFF has invalid or empty spatial bounds.")

            color_indexes = {color: index for index, color in enumerate(dataset.colorinterp, start=1)}
            if all(color in color_indexes for color in (ColorInterp.red, ColorInterp.green, ColorInterp.blue)):
                rgb_band_mapping = [color_indexes[ColorInterp.red], color_indexes[ColorInterp.green], color_indexes[ColorInterp.blue]]
            else:
                rgb_band_mapping = [1, 2, 3]
            dtypes = [str(dataset.dtypes[index - 1]) for index in rgb_band_mapping]
            if any(dtype != "uint8" for dtype in dtypes):
                raise ValueError("This build requires uint8 RGB bands; convert other data types before upload.")
            alpha_band = color_indexes.get(ColorInterp.alpha)
            nodata = [value for value in dataset.nodatavals if value is not None]
            nodata_summary = f"Nodata values: {nodata}." if nodata else "No nodata values declared."
            if alpha_band:
                nodata_summary += f" Alpha channel: band {alpha_band}."

            metadata = {
                "relative_path": os.path.join("rasters", safe_filename),
                "original_filename": safe_filename,
                "crs": dataset.crs.to_string(),
                "bounds": {
                    "minx": float(dataset.bounds.left),
                    "miny": float(dataset.bounds.bottom),
                    "maxx": float(dataset.bounds.right),
                    "maxy": float(dataset.bounds.top),
                },
                "width": int(dataset.width),
                "height": int(dataset.height),
                "bands": int(dataset.count),
                "dtypes": [str(dtype) for dtype in dataset.dtypes],
                "gsd_x": math.hypot(transform.a, transform.d),
                "gsd_y": math.hypot(transform.b, transform.e),
                "rgb_band_mapping": rgb_band_mapping,
                "alpha_band": alpha_band,
                "nodata_or_alpha_summary": nodata_summary,
                "file_size_bytes": upload_size,
                "available": True,
            }
            metadata["bounds_wgs84"] = _wgs84_bounds_from_3857(metadata["bounds"])
    except Exception as exc:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=400, detail=f"Invalid GeoTIFF: {exc}")

    try:
        os.replace(temp_path, target_path)
        temp_path = None
        record = project_registry.ensure_project_record(
            project_id,
            name=name.strip(),
            locality=locality.strip() or "Unknown locality",
            raster_meta=metadata,
        )
    except Exception as exc:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        if os.path.exists(target_path):
            os.remove(target_path)
        raise HTTPException(status_code=500, detail=f"Project storage failed: {exc}")

    return {"valid": True, "project_id": project_id, "record": record, "raster": metadata}


@app.post("/api/projects/{project_id}/model-runs", status_code=status.HTTP_202_ACCEPTED)
def start_project_model_run(project_id: str, payload: Dict[str, Any] = Body(default_factory=dict)):
    """Starts a project-specific WHU model run using that project's own raster and isolated run storage."""
    try:
        project_registry.get_project(project_id)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))

    project_raster_path = project_registry.get_project_raster_path(project_id)
    if not project_raster_path or not os.path.isfile(project_raster_path):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"No raster exists for project '{project_id}'.")

    project = project_registry.get_project(project_id)
    raster_meta = project.get("raster") or {}
    run_id = f"run-project-{uuid.uuid4().hex}"
    run_dir = project_registry.project_runs_dir(project_id, run_id)
    result = run_whu_live_inference(
        confidence_threshold=float(payload.get("confidence_threshold", 0.50)),
        min_area_cutoff_sqm=float(payload.get("min_area_cutoff_sqm", 10.0)),
        tile_size=int(payload.get("tile_size", 512)),
        stride=int(payload.get("stride", 256)),
        morphology_opening_px=int(payload.get("morphology_opening_px", 0)),
        morphology_closing_px=int(payload.get("morphology_closing_px", 0)),
        src_tif_path=project_raster_path,
        out_dir=run_dir,
        rgb_band_mapping=payload.get("rgb_band_mapping") or raster_meta.get("rgb_band_mapping", [1, 2, 3]),
        alpha_band=payload.get("alpha_band", raster_meta.get("alpha_band")),
        project_id=project_id,
        run_id=run_id,
    )

    prediction_path = os.path.join(run_dir, "predictions.geojson")
    metadata = dict(result.get("model_metadata", {}))
    metadata.setdefault("project_id", project_id)
    metadata.setdefault("run_id", run_id)
    persisted_result = {**result, "model_metadata": metadata}
    with open(prediction_path, "w", encoding="utf-8") as handle:
        json.dump(persisted_result, handle, indent=2)

    record = {
        "project_id": project_id,
        "run_id": run_id,
        "status": "completed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_metadata": metadata,
        "features": result.get("features", []),
        "result": persisted_result,
        "output_path": prediction_path,
    }
    project_registry.add_model_run(project_id, run_id)
    project_registry.save_model_run(project_id, run_id, record)

    return {
        "status": "accepted",
        "project_id": project_id,
        "run_id": run_id,
        "output_dir": run_dir,
        "message": "Project model run completed and stored in the project-scoped run directory.",
    }


@app.get("/api/projects/{project_id}/model-runs")
def list_project_model_runs(project_id: str):
    """Returns all saved model runs for a given project."""
    try:
        project_registry.get_project(project_id)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))

    runs = project_registry.list_model_runs(project_id)
    return {"project_id": project_id, "runs": runs, "count": len(runs)}


@app.get("/api/projects/{project_id}/model-runs/{run_id}")
def get_project_model_run(project_id: str, run_id: str):
    """Returns the saved project run result for a single run."""
    try:
        run = project_registry.get_model_run(project_id, run_id)
    except (KeyError, FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))

    payload = {"project_id": project_id, "run_id": run_id, **run}
    payload.setdefault("status", "completed")
    payload.setdefault("features", run.get("result", {}).get("features", []))
    payload.setdefault("model_metadata", run.get("result", {}).get("model_metadata", {}))
    return payload


@app.get("/api/projects/{project_id}/layers")
def get_project_layers(project_id: str):
    """Returns all project-scoped layer payloads for a project, leaving the demo store alone."""
    try:
        project_store = _normalise_project_store(project_id)
        project_store.initialize(force_reload=True)
        return {
            "project_id": project_id,
            "layers": {
                name: project_store.get_layer_collection(name)
                for name in [
                    "buildings",
                    "roads",
                    "osm_roads",
                    "parcels",
                    "synthetic",
                    "drafts",
                    "ai_predictions",
                    "ai_whu_predictions",
                    "ai_deeplab_predictions",
                    "ai_mock_predictions",
                ]
            },
            "count": len(project_store.layers),
        }
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))


@app.get("/api/projects/{project_id}/layers/manifest")
def get_project_layer_manifest(project_id: str):
    """Return the sidebar manifest for a project or the demo project."""
    try:
        project = _project_raster_payload(project_registry.get_project(project_id))
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))

    project_store = _normalise_project_store(project_id)
    manifest = project_store.get_layer_manifest()
    has_reference_layer = bool(project_store.layers.get("buildings"))
    if not has_reference_layer:
        manifest = [layer for layer in manifest if layer.get("id") != "buildings"]
    raster = project.get("raster") or {}
    orthomosaic = next((layer for layer in manifest if layer.get("id") == "orthomosaic"), None)
    if orthomosaic is None:
        orthomosaic = {
            "id": "orthomosaic",
            "label": "Orthomosaic",
            "kind": "reference",
            "feature_count": 1,
            "visible_default": True,
            "read_only": True,
            "style": {"color": "#60a5fa", "fillColor": "#60a5fa", "weight": 1.5, "fillOpacity": 0.12},
            "notes": "Project orthomosaic raster.",
        }
        manifest.insert(0, orthomosaic)
    orthomosaic["tile_url_template"] = f"/api/projects/{project_id}/raster/tiles/{{z}}/{{x}}/{{y}}.png"
    orthomosaic["max_native_zoom"] = raster.get("max_native_zoom", _native_zoom_for_raster(raster))
    orthomosaic["max_zoom"] = orthomosaic["max_native_zoom"]
    return {
        "project_id": project_id,
        "layers": manifest,
        "count": len(manifest),
        "has_reference_layer": has_reference_layer,
    }


@app.get("/api/models/providers")
def get_model_providers():
    """Return installed project-capable providers and exact local availability reasons."""
    return {"providers": get_building_model_providers()}


@app.get("/api/projects/{project_id}/layers/{layer_name}")
def get_project_layer(project_id: str, layer_name: str):
    """Returns one project-scoped layer collection, or an empty collection for a new project."""
    try:
        project_store = _normalise_project_store(project_id)
        return project_store.get_layer_collection(layer_name)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))


@app.get("/api/projects/{project_id}/raster/tiles/{z}/{x}/{y}.png")
def get_project_raster_tile(project_id: str, z: int, x: int, y: int):
    """Serves project-scoped raster tiles while leaving the default Lalpur route intact."""
    try:
        service = project_registry.get_project_raster_service(project_id)
    except KeyError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Project '{project_id}' not found.")
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))

    try:
        tile_bytes = service.get_tile_png(z, x, y, strict=True)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not render project raster tile: {exc}",
        ) from exc
    return Response(
        content=tile_bytes,
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=3600"}
    )

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
    except SavedPredictionLoadError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Layer '{layer_name}' not found. Available layers: {[*store.layers.keys(), SAVED_FINETUNED_LAYER]}"
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


@app.get("/api/projects/{project_id}/warnings")
def get_project_warnings(project_id: str):
    """Returns warnings limited to the active project only."""
    try:
        project_store = _normalise_project_store(project_id)
        project_store.initialize(force_reload=True)
        return {
            "project_id": project_id,
            "total_warnings": len(project_store.warnings),
            "synthetic_warning_count": sum(1 for w in project_store.warnings if "synthetic" in w.source),
            "real_warning_count": sum(1 for w in project_store.warnings if "synthetic" not in w.source),
            "warnings": project_store.warnings,
        }
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))

@app.get("/api/warnings/narration-status")
def get_warning_narration_status():
    """Expose only whether optional runtime LLM phrasing is configured."""
    return narration_status()


@app.post("/api/warnings/narrate")
def post_warning_narrations(req: WarningNarrationRequest):
    """Optionally rephrase selected warnings; original template wording is always returned as fallback."""
    store.initialize()
    warnings_by_id = {warning.warning_id: warning for warning in store.warnings}
    missing = [warning_id for warning_id in req.warning_ids if warning_id not in warnings_by_id]
    if missing:
        raise HTTPException(status_code=404, detail=f"Unknown warning IDs: {missing[:5]}")
    selected = [warnings_by_id[warning_id] for warning_id in req.warning_ids]
    return narrate_warnings(selected)


@app.get("/api/greenery/results")
def get_greenery_results():
    """Load the last RGB Excess Green candidate layer, or an empty collection before first run."""
    return load_saved_greenery()


@app.post("/api/greenery/detect")
def run_greenery_detection(req: GreeneryDetectRequest):
    """Run a bounded, no-training RGB greenery heuristic on the configured Lalpur orthomosaic."""
    if not raster_tile_service.is_available:
        raise HTTPException(status_code=503, detail="The configured Lalpur orthomosaic is unavailable.")
    try:
        return detect_rgb_greenery(
            raster_tile_service.tif_path,
            index_threshold=req.index_threshold,
            min_area_sqm=req.min_area_sqm,
            morphology_opening_px=req.morphology_opening_px,
            morphology_closing_px=req.morphology_closing_px,
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="The configured RGB orthomosaic was not found.")
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Greenery extraction failed: {exc}")


@app.get("/api/scores")
def get_scores():
    """Returns review-priority scores for all loaded features."""
    store.initialize()
    return {
        "total_scored_features": len(store.scores),
        "heuristic_disclaimer": "prototype heuristic—not a validated survey-priority model",
        "scores": store.scores
    }


@app.get("/api/projects/{project_id}/scores")
def get_project_scores(project_id: str):
    """Returns the active project's review scores only."""
    try:
        project_store = _normalise_project_store(project_id)
        project_store.initialize(force_reload=True)
        return {
            "project_id": project_id,
            "total_scored_features": len(project_store.scores),
            "heuristic_disclaimer": "prototype heuristic—not a validated survey-priority model",
            "scores": project_store.scores,
        }
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))

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


@app.get("/api/projects/{project_id}/features/{feature_id}")
def get_project_feature_details(project_id: str, feature_id: str):
    try:
        project_registry.get_project(project_id)
        project_store = _normalise_project_store(project_id)
        project_store.initialize(force_reload=True)
        found = project_store.find_feature(feature_id)
        if not found:
            raise KeyError(f"Feature with ID '{feature_id}' not found in project '{project_id}'.")
        layer_name, feature = found
        warning_ids = feature.get("properties", {}).get("warning_ids", [])
        return {
            "project_id": project_id,
            "feature": feature,
            "layer": layer_name,
            "score_breakdown": project_store.scores.get(feature_id),
            "associated_warnings": [warning for warning in project_store.warnings if warning.warning_id in warning_ids],
            "has_geometry_edits": bool(feature.get("original_geometry") and feature.get("original_geometry") != feature.get("geometry")),
        }
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.put("/api/projects/{project_id}/features/{feature_id}")
def update_project_feature(project_id: str, feature_id: str, req: FeatureStatusUpdateRequest):
    try:
        project_registry.get_project(project_id)
        project_store = _normalise_project_store(project_id)
        project_store.initialize(force_reload=True)
        updated = project_store.update_feature_status(
            feature_id=feature_id,
            review_status=req.review_status,
            notes=req.notes,
            reviewer_label=req.reviewer_label,
        )
        return {"status": "success", "project_id": project_id, "feature": updated, "score": project_store.scores.get(feature_id)}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/projects/{project_id}/features/{feature_id}/edit-geometry")
def edit_project_feature_geometry(project_id: str, feature_id: str, req: GeometryEditRequest):
    try:
        project_registry.get_project(project_id)
        project_store = _normalise_project_store(project_id)
        project_store.initialize(force_reload=True)
        updated = project_store.update_feature_geometry(
            feature_id=feature_id,
            new_geometry=req.geometry.model_dump(),
            reviewer_label=req.reviewer_label,
            edit_reason=req.edit_reason,
        )
        return {"status": "success", "project_id": project_id, "feature": updated}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/projects/{project_id}/features/{feature_id}/revert")
def revert_project_feature_geometry(project_id: str, feature_id: str):
    try:
        project_registry.get_project(project_id)
        project_store = _normalise_project_store(project_id)
        project_store.initialize(force_reload=True)
        reverted = project_store.revert_feature_geometry(feature_id)
        return {"status": "success", "project_id": project_id, "feature": reverted}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

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
            predict_options = {
                "confidence_threshold": req.confidence_threshold,
                "simulate_failure": req.simulate_failure,
            }
            if req.mode in {"live", "deeplab"}:
                predict_options.update({
                    "morphology_opening_px": req.morphology_opening_px,
                    "morphology_closing_px": req.morphology_closing_px,
                })
            result = model.predict(**predict_options)
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


@app.get("/api/projects/{project_id}/models/prediction-source")
def get_project_prediction_source(project_id: str):
    """Return the active prediction source for a specific project."""
    try:
        project_store = _normalise_project_store(project_id)
        project_store.initialize(force_reload=True)
        return project_store.get_prediction_status()
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))


@app.post("/api/models/select-precomputed")
def select_precomputed_prediction():
    """Select the saved WHU GeoJSON without modifying that artifact."""
    return predict_building_model(ModelPredictRequest(mode="precomputed"))


@app.post("/api/projects/{project_id}/models/select-precomputed")
def select_project_precomputed_prediction(project_id: str):
    """Select the saved WHU GeoJSON for a project without modifying the artifact."""
    try:
        project_registry.get_project(project_id)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    project_store = _normalise_project_store(project_id)
    project_store.initialize(force_reload=True)
    return {
        "status": "success",
        "project_id": project_id,
        "prediction_source": project_store.get_prediction_status(),
    }


@app.post("/api/models/select-deeplab")
def select_deeplab_prediction():
    """Select the DeepLab candidate model if the checkpoint is available."""
    return predict_building_model(ModelPredictRequest(mode="deeplab"))


@app.delete("/api/models/prediction-source")
def clear_prediction_source():
    """Clear the active layer while retaining all saved model artifacts."""
    return store.clear_prediction_source()


@app.delete("/api/projects/{project_id}/models/prediction-source")
def clear_project_prediction_source(project_id: str):
    """Clear the active layer for a project while retaining all saved model artifacts."""
    try:
        project_registry.get_project(project_id)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    project_store = _normalise_project_store(project_id)
    project_store.initialize(force_reload=True)
    return project_store.clear_prediction_source()

@app.post("/api/models/run-inference")
def start_live_model_job(
    background_tasks: BackgroundTasks,
    confidence_threshold: float = Body(0.50, embed=True),
    morphology_opening_px: int = Body(0, embed=True),
    morphology_closing_px: int = Body(0, embed=True),
):
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
            res = run_whu_live_inference(
                confidence_threshold=confidence_threshold,
                morphology_opening_px=morphology_opening_px,
                morphology_closing_px=morphology_closing_px,
            )
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
                "output_path": metadata.get("output_path"),
                "checkpoint_sha256": metadata.get("model_weight_sha256"),
                "valid_feature_count": len(res.get("features", [])),
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


@app.get("/api/projects/{project_id}/export")
def export_project_bundle(project_id: str):
    """Export the reviewed features belonging to an active project only."""
    try:
        project_registry.get_project(project_id)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    project_store = _normalise_project_store(project_id)
    project_store.initialize(force_reload=True)
    bundle = project_store.export_reviewed_bundle()
    return JSONResponse(
        content=bundle,
        headers={"Content-Disposition": f"attachment; filename={project_id}_Reviewed_Features.geojson"}
    )


@app.post("/api/import")
def import_bundle(bundle: Dict[str, Any] = Body(...)):
    """Imports and validates an exported GeoJSON bundle."""
    try:
        res = store.import_reviewed_bundle(bundle)
        return res
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/projects/{project_id}/import")
def import_project_bundle(project_id: str, bundle: Dict[str, Any] = Body(...)):
    """Import a GeoJSON bundle into the given project-scoped store."""
    try:
        project_registry.get_project(project_id)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    project_store = _normalise_project_store(project_id)
    project_store.initialize(force_reload=True)
    try:
        return project_store.import_reviewed_bundle(bundle)
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
def get_model_reference_discrepancy(iou_threshold: float = Query(0.35, ge=0.05, le=0.95), project_id: Optional[str] = Query(None)):
    """
    Model vs Reference Discrepancy Comparator.
    Compares AI-predicted building footprints against the 317 Vaayu reference footprints
    using spatial IoU matching within the same area. NOT temporal change detection.
    The reference footprints are not an 'old' layer — they are same-area annotations.
    """
    resolved_project_id = project_id or DEFAULT_LALPUR_PROJECT_ID
    if resolved_project_id != DEFAULT_LALPUR_PROJECT_ID:
        try:
            project_registry.get_project(resolved_project_id)
        except KeyError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    project_store = _normalise_project_store(resolved_project_id)
    project_store.initialize(force_reload=True)
    prediction_status = project_store.get_prediction_status()
    ai_preds = project_store.layers["ai_predictions"] if prediction_status["selected"] else []
    reference = project_store.layers["buildings"]

    if not reference:
        raise HTTPException(status_code=422, detail="No reference layer for this project.")

    if not ai_preds:
        return {
            "status": "no_predictions",
            "message": "No AI model predictions loaded. Run model inference first via POST /api/models/predict.",
            "disclaimer": "This comparator shows same-area AI-vs-reference disagreement. "
                          "It is NOT temporal change detection. Reference features are unverified annotations.",
            "alignment_status": "confirmed by user in QGIS; local reference set, not an official/legal accuracy benchmark",
            "prediction_source": prediction_status,
            "valid_prediction_count": 0,
            "valid_reference_count": len(project_store.layers["buildings"]),
            "invalid_prediction_count": 0,
            "invalid_reference_count": 0,
            "project_id": project_id,
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
        },
        "project_id": project_id,
    }


@app.post("/api/projects/{project_id}/process/buildings", status_code=status.HTTP_202_ACCEPTED)
def start_project_building_job(project_id: str, payload: Dict[str, Any] = Body(default_factory=dict)):
    """Run a registered building model against this project's raster in the background."""
    try:
        project = project_registry.get_project(project_id)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))

    provider_id = str(payload.get("provider") or "")
    provider = next((item for item in get_building_model_providers() if item["id"] == provider_id), None)
    if provider is None:
        raise HTTPException(status_code=422, detail=f"Unknown building-model provider '{provider_id}'.")
    if not provider["enabled"]:
        raise HTTPException(status_code=503, detail=provider["reason"] or f"Provider '{provider_id}' is unavailable.")

    try:
        threshold = float(payload.get("threshold", provider["default_threshold"]))
        min_area_m2 = float(payload.get("min_area_m2", provider["default_min_area_m2"]))
        resolution_m = payload.get("resolution_m")
        resolution_m = None if resolution_m in (None, "") else float(resolution_m)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail="Threshold, resolution, and minimum area must be numeric.") from exc
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise HTTPException(status_code=422, detail="Threshold must be between 0 and 1.")
    if not math.isfinite(min_area_m2) or min_area_m2 < 0:
        raise HTTPException(status_code=422, detail="Minimum building area must be finite and non-negative.")
    if resolution_m is not None and (not math.isfinite(resolution_m) or resolution_m <= 0):
        raise HTTPException(status_code=422, detail="Inference resolution must be finite and greater than zero.")

    aoi = payload.get("aoi")
    raster_path = project_registry.get_project_raster_path(project_id)
    if not raster_path or not os.path.isfile(raster_path):
        raise HTTPException(status_code=404, detail=f"No raster exists for project '{project_id}'.")

    raster_metadata = project.get("raster") or {}
    run_id = f"run-project-{uuid.uuid4().hex}"
    try:
        job = _queue_project_model_job(
            project_id,
            model=provider_id,
            threshold=threshold,
            aoi=aoi,
            resolution_m=resolution_m,
            min_area_m2=min_area_m2,
        )
    except HTTPException:
        raise
    job["provider_id"] = provider_id
    job["run_id"] = run_id

    def _run() -> None:
        run_dir = None
        project_store = None
        previous_features = None
        previous_metadata = None
        state_committed = False
        try:
            run_dir = project_registry.project_runs_dir(project_id, run_id)
            project_store = _normalise_project_store(project_id)
            job["status"] = "running"
            job["updated_at"] = datetime.now(timezone.utc).isoformat()

            def update_progress(done: int, total: int) -> None:
                job["progress"] = {
                    "done": done,
                    "total": total,
                    "percent": round(done * 100.0 / total, 1) if total else 0.0,
                }
                job["updated_at"] = datetime.now(timezone.utc).isoformat()

            result = run_project_building_inference(
                project_id=project_id,
                raster_path=raster_path,
                raster_metadata=raster_metadata,
                provider_id=provider_id,
                threshold=threshold,
                aoi=aoi,
                resolution_m=resolution_m,
                min_area_m2=min_area_m2,
                run_dir=run_dir,
                progress_callback=update_progress,
            )
            features = result["features"]
            metadata = result["metadata"]
            project_store.initialize(force_reload=True)
            previous_features = json.loads(json.dumps(project_store.layers["ai_predictions"]))
            previous_metadata = json.loads(json.dumps(project_store.active_prediction_metadata)) if project_store.active_prediction_metadata else None

            source_metadata = {
                **metadata,
                "source_mode": "fresh",
                "request_mode": "project_process",
                "provider_id": provider_id,
                "provider_label": provider["label"],
                "source_label": provider["label"],
                "model_id": provider_id,
                "run_id": run_id,
                "output_path": result["output_path"],
                "valid_feature_count": len(features),
                "is_mock": False,
                "alignment_status": "not evaluated against a project reference; predictions are unreviewed AI suggestions",
            }
            state_committed = True
            project_store.set_prediction_source(features, source_metadata)

            run_record = {
                "project_id": project_id,
                "run_id": run_id,
                "status": "completed",
                "created_at": metadata["created_at"],
                "layer_id": "ai_predictions",
                "feature_count": len(features),
                "model_metadata": source_metadata,
                "features": features,
                "output_path": result["output_path"],
            }
            project_registry.save_model_run(project_id, run_id, run_record)
            project_registry.add_model_run(project_id, run_id)
            _finish_project_model_job(
                job["job_id"],
                success=True,
                feature_count=len(features),
                run_id=run_id,
                result_summary={
                    "layer_id": "ai_predictions",
                    "feature_count": len(features),
                    "provider_id": provider_id,
                    "inference_resolution_m": metadata["resolution_m"],
                    "source_ground_resolution_m": metadata["source_ground_resolution_m"],
                    "resampling_factor_source_pixels_per_output_pixel": metadata["resampling_factor_source_pixels_per_output_pixel"],
                    "tile_count": metadata["tile_count"],
                    "elapsed_seconds": metadata.get("elapsed_seconds"),
                    "warnings": metadata["warnings"],
                    "output_path": result["output_path"],
                },
            )
        except Exception as exc:
            if state_committed and previous_features is not None:
                try:
                    project_store.layers["ai_predictions"] = previous_features
                    project_store.active_prediction_metadata = previous_metadata
                    project_store.revalidate_all()
                    project_store._save_persisted_state()
                except Exception:
                    pass
            if run_dir and os.path.isdir(run_dir):
                shutil.rmtree(run_dir, ignore_errors=True)
            _finish_project_model_job(job["job_id"], success=False, run_id=run_id, error_message=str(exc))

    threading.Thread(target=_run, daemon=True).start()
    return {
        "job_id": job["job_id"],
        "project_id": project_id,
        "provider_id": provider_id,
        "status": "queued",
        "layer_id": "ai_predictions",
    }


@app.get("/api/jobs/{job_id}")
def get_project_job(job_id: str):
    job = PROJECT_JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Job '{job_id}' not found.")
    return job


@app.post("/api/jobs/{job_id}/cancel")
def cancel_project_job(job_id: str):
    job = PROJECT_JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Job '{job_id}' not found.")
    if job["status"] in {"completed", "failed", "cancelled"}:
        return {"job_id": job_id, "status": job["status"]}
    job["status"] = "cancelled"
    job["error_message"] = "Cancelled by user."
    job["updated_at"] = datetime.now(timezone.utc).isoformat()
    return {"job_id": job_id, "status": "cancelled"}


# Mount static frontend files if folder exists
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.exists(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
