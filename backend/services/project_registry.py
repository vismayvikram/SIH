"""Project registry for Lalpur and future project-aware raster workflows."""

from __future__ import annotations

import copy
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from typing import Any, Dict, List

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


class ProjectRegistry:
    """Persistent project records and raster services, isolated from the Lalpur demo service."""

    def __init__(self, projects_root: str | None = None) -> None:
        self.projects_root = os.path.abspath(projects_root or os.path.join(WORKSPACE_ROOT, "data", "projects"))
        self._projects = self._build_default_projects()
        self._raster_services = {}
        self._load_saved_projects()

    def _load_saved_projects(self) -> None:
        if not os.path.isdir(self.projects_root):
            return
        default_id = self._projects[0]["project_id"]
        for entry in os.scandir(self.projects_root):
            if not entry.is_dir():
                continue
            project_file = os.path.join(entry.path, "project.json")
            if not os.path.isfile(project_file):
                continue
            try:
                with open(project_file, "r", encoding="utf-8") as handle:
                    record = json.load(handle)
                project_id = record.get("project_id")
                if not project_id or project_id != self.safe_project_id(project_id) or project_id != entry.name:
                    continue
                if project_id == default_id or not isinstance(record.get("raster"), dict):
                    continue
                self._projects.append(record)
            except (OSError, ValueError, TypeError):
                continue

    def _write_project_record(self, record: Dict[str, Any]) -> None:
        project_dir = self.project_storage_dir(record["project_id"])
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=project_dir, suffix=".tmp", delete=False
        ) as handle:
            temp_path = handle.name
            json.dump(record, handle, indent=2)
        try:
            os.replace(temp_path, os.path.join(project_dir, "project.json"))
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def project_runs_dir(self, project_id: str, run_id: str | None = None) -> str:
        self.get_project(project_id)
        project_dir = self.project_storage_dir(project_id)
        runs_dir = os.path.join(project_dir, "runs")
        os.makedirs(runs_dir, exist_ok=True)
        if run_id is None:
            return runs_dir
        safe_run_id = self.safe_project_id(run_id)
        if safe_run_id != run_id or not run_id.startswith("run-project-"):
            raise ValueError("Invalid project model run ID.")
        run_dir = os.path.abspath(os.path.join(runs_dir, run_id))
        if os.path.commonpath([runs_dir, run_dir]) != os.path.abspath(runs_dir):
            raise ValueError("Invalid project model run path.")
        os.makedirs(run_dir, exist_ok=True)
        return run_dir

    def add_model_run(self, project_id: str, run_id: str) -> None:
        project = self.get_project(project_id)
        if run_id not in project.setdefault("model_run_ids", []):
            project["model_run_ids"].append(run_id)
        self._write_project_record(project)
        for index, stored in enumerate(self._projects):
            if stored["project_id"] == project_id:
                self._projects[index] = project
                break

    def save_model_run(self, project_id: str, run_id: str, record: Dict[str, Any]) -> None:
        run_dir = self.project_runs_dir(project_id, run_id)
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=run_dir, suffix=".tmp", delete=False
        ) as handle:
            temp_path = handle.name
            json.dump(record, handle, indent=2)
        try:
            os.replace(temp_path, os.path.join(run_dir, "run.json"))
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def get_model_run(self, project_id: str, run_id: str) -> Dict[str, Any]:
        run_path = os.path.join(self.project_runs_dir(project_id, run_id), "run.json")
        with open(run_path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def list_model_runs(self, project_id: str) -> List[Dict[str, Any]]:
        project = self.get_project(project_id)
        runs = []
        for run_id in project.get("model_run_ids", []):
            try:
                runs.append(self.get_model_run(project_id, run_id))
            except (FileNotFoundError, ValueError):
                continue
        return sorted(runs, key=lambda run: run.get("created_at", ""), reverse=True)

    @staticmethod
    def _build_default_projects() -> List[Dict[str, Any]]:
        relative_path = os.path.join(
            "data",
            "acquisition",
            "SIH26012_INDIA_CANDIDATE_01",
            "working",
            "lalpur_orthomosaic.tif",
        )
        return [
            {
                "project_id": "SIH26012_INDIA_CANDIDATE_01_LALPUR",
                "name": "Lalpur pilot",
                "locality": "Lalpur, Gujarat",
                "created_at": "2026-09-30T00:00:00Z",
                "status": "active",
                "raster": {
                    "relative_path": relative_path,
                    "original_filename": "lalpur_orthomosaic.tif",
                    "crs": "EPSG:3857",
                    "bounds": {
                        "minx": 8098996.3782,
                        "miny": 2636558.4073,
                        "maxx": 8099677.1552,
                        "maxy": 2637264.506,
                    },
                    "width": 20137,
                    "height": 20886,
                    "bands": 4,
                    "dtypes": ["uint8"],
                    "gsd_x": 0.0338,
                    "gsd_y": 0.0338,
                    "rgb_band_mapping": [1, 2, 3],
                    "nodata_or_alpha_summary": "Alpha channel present; source remains a valid 4-band GeoTIFF.",
                    "available": True,
                },
                "reference_layer_ids": ["lalpur_buildings_reference_v1"],
                "model_run_ids": ["whu_lalpur_baseline", "deeplab_lalpur_baseline"],
            }
        ]

    @staticmethod
    def safe_project_id(project_id: str) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", project_id).strip("-")
        return cleaned or "project"

    def list_projects(self) -> List[Dict[str, Any]]:
        projects = copy.deepcopy(self._projects)
        for project in projects:
            if project["project_id"] == self._projects[0]["project_id"]:
                continue
            raster = project.get("raster") or {}
            raster_path = self.get_project_raster_path(project["project_id"])
            raster["available"] = bool(raster_path and os.path.isfile(raster_path))
        return projects

    def get_project(self, project_id: str) -> Dict[str, Any]:
        for project in self._projects:
            if project["project_id"] == project_id:
                return copy.deepcopy(project)
        raise KeyError(f"Project '{project_id}' not found.")

    def get_active_project(self) -> Dict[str, Any]:
        return copy.deepcopy(self._projects[0])

    def ensure_project_record(self, project_id: str, *, name: str, locality: str, raster_meta: Dict[str, Any]) -> Dict[str, Any]:
        safe_id = self.safe_project_id(project_id)
        for index, project in enumerate(self._projects):
            if project["project_id"] == safe_id:
                updated = {
                    **project,
                    "name": name or project.get("name") or safe_id,
                    "locality": locality or project.get("locality") or "Unknown locality",
                    "raster": {**project.get("raster", {}), **raster_meta},
                }
                self._write_project_record(updated)
                self._projects[index] = updated
                return copy.deepcopy(updated)

        record = {
            "project_id": safe_id,
            "name": name or safe_id,
            "locality": locality or "Unknown locality",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "active",
            "raster": raster_meta,
            "reference_layer_ids": [],
            "model_run_ids": [],
        }
        self._write_project_record(record)
        self._projects.append(record)
        return copy.deepcopy(record)

    def project_storage_dir(self, project_id: str) -> str:
        safe_id = self.safe_project_id(project_id)
        project_dir = os.path.join(self.projects_root, safe_id)
        os.makedirs(project_dir, exist_ok=True)
        os.makedirs(os.path.join(project_dir, "rasters"), exist_ok=True)
        return project_dir

    def get_project_raster_path(self, project_id: str) -> str | None:
        try:
            project = self.get_project(project_id)
        except KeyError:
            return None
        raster = project.get("raster") or {}
        relative_path = raster.get("relative_path")
        if not relative_path:
            return None
        if project_id == self._projects[0]["project_id"]:
            return os.path.abspath(os.path.join(WORKSPACE_ROOT, relative_path))

        project_dir = os.path.abspath(os.path.join(self.projects_root, self.safe_project_id(project_id)))
        raster_path = os.path.abspath(os.path.join(project_dir, relative_path))
        if os.path.commonpath([project_dir, raster_path]) != project_dir:
            return None
        return raster_path

    def get_project_raster_service(self, project_id: str):
        """Return a raster tile service for a project-specific raster when present."""
        from backend.services.raster_service import RasterTileService

        raster_path = self.get_project_raster_path(project_id)
        if not raster_path or not os.path.exists(raster_path):
            raise FileNotFoundError(f"No raster exists for project '{project_id}'.")
        project = self.get_project(project_id)
        raster = project.get("raster") or {}
        service_key = (raster_path, tuple(raster.get("rgb_band_mapping", [])), raster.get("alpha_band"))
        if service_key not in self._raster_services:
            self._raster_services[service_key] = RasterTileService(
                raster_path,
                rgb_band_mapping=raster.get("rgb_band_mapping"),
                alpha_band=raster.get("alpha_band"),
            )
        return self._raster_services[service_key]


project_registry = ProjectRegistry()
