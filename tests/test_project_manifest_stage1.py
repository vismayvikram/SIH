import numpy as np
import rasterio
from fastapi.testclient import TestClient
from rasterio.transform import from_origin

from backend import api
from backend.api import app
from backend.services.project_registry import ProjectRegistry


def _create_rgb_raster(path):
    with rasterio.open(
        path,
        'w',
        driver='GTiff',
        height=32,
        width=32,
        count=3,
        dtype='uint8',
        crs='EPSG:3857',
        transform=from_origin(0, 1000, 1, 1),
    ) as dataset:
        dataset.write(np.full((3, 32, 32), 80, dtype='uint8'))


def test_project_manifest_is_project_scoped_and_empty_for_new_projects(tmp_path, monkeypatch):
    registry = ProjectRegistry(str(tmp_path / 'projects'))
    monkeypatch.setattr(api, 'project_registry', registry)
    raster_path = tmp_path / 'user.tif'
    _create_rgb_raster(raster_path)
    client = TestClient(app)

    with raster_path.open('rb') as handle:
        created = client.post(
            '/api/projects',
            files={'file': ('user.tif', handle.read(), 'image/tiff')},
            data={'name': 'Iso project', 'locality': 'Remote field'},
        )
    assert created.status_code == 201, created.text
    project_id = created.json()['project_id']

    response = client.get(f'/api/projects/{project_id}/layers/manifest')
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload['project_id'] == project_id
    assert payload['count'] == 1
    assert payload['layers'][0]['id'] == 'orthomosaic'
    assert payload['layers'][0]['feature_count'] == 1

    layer_resp = client.get(f'/api/projects/{project_id}/layers')
    layer_payload = layer_resp.json()
    assert layer_payload['count'] == len(layer_payload['layers'])

    warning_resp = client.get(f'/api/projects/{project_id}/warnings')
    assert warning_resp.json()['total_warnings'] == 0

    score_resp = client.get(f'/api/projects/{project_id}/scores')
    assert score_resp.json()['total_scored_features'] == 0


def test_demo_manifest_keeps_existing_layer_count_and_project_route_is_404_for_unknown_projects():
    client = TestClient(app)
    demo = client.get('/api/projects/SIH26012_INDIA_CANDIDATE_01_LALPUR/layers/manifest')
    assert demo.status_code == 200, demo.text
    manifest = demo.json()
    assert 'layers' in manifest and len(manifest['layers']) > 1
    assert any(layer['id'] == 'buildings' for layer in manifest['layers'])

    missing = client.get('/api/projects/unknown-project/layers/manifest')
    assert missing.status_code == 404, missing.text
