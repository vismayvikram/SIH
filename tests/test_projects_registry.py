from fastapi.testclient import TestClient

from backend import api
from backend.api import app


def test_project_registry_lists_lalpur_project_metadata():
    client = TestClient(app)

    response = client.get('/api/projects')

    assert response.status_code == 200
    payload = response.json()
    assert 'projects' in payload
    projects = payload['projects']
    assert len(projects) >= 1
    project = next((p for p in projects if p['project_id'] == 'SIH26012_INDIA_CANDIDATE_01_LALPUR'), None)
    assert project is not None
    assert project['name'] == 'Lalpur pilot'
    assert project['status'] in {'demo_ready', 'raster_unavailable'}
    assert project['raster']['original_filename'] == 'lalpur_orthomosaic.tif'


def test_project_registry_returns_single_project_record():
    client = TestClient(app)

    response = client.get('/api/projects/SIH26012_INDIA_CANDIDATE_01_LALPUR')

    assert response.status_code == 200
    payload = response.json()
    assert payload['project_id'] == 'SIH26012_INDIA_CANDIDATE_01_LALPUR'
    assert payload['name'] == 'Lalpur pilot'
    assert payload['locality'] == 'Lalpur, Gujarat'
    assert payload['raster']['crs'] == 'EPSG:3857'
    assert 'reference_layer_ids' in payload
    assert 'model_run_ids' in payload


def test_lalpur_dashboard_status_reflects_raster_service(monkeypatch):
    client = TestClient(app)
    monkeypatch.setattr(api.raster_tile_service, 'is_available', False)

    response = client.get('/api/projects')

    assert response.status_code == 200
    lalpur = next(project for project in response.json()['projects'] if project['project_id'] == 'SIH26012_INDIA_CANDIDATE_01_LALPUR')
    assert lalpur['status'] == 'raster_unavailable'
