import json
import os

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


def test_new_project_stays_empty_and_isolated_from_lalpur_demo(tmp_path, monkeypatch):
    registry = ProjectRegistry(str(tmp_path / 'projects'))
    monkeypatch.setattr(api, 'project_registry', registry)
    raster_path = tmp_path / 'user.tif'
    _create_rgb_raster(raster_path)
    client = TestClient(app)

    with raster_path.open('rb') as handle:
        created = client.post(
            '/api/projects',
            files={'file': ('user.tif', handle.read(), 'image/tiff')},
            data={'name': 'Isolated project', 'locality': 'Remote field'},
        )
    assert created.status_code == 201, created.text
    project_id = created.json()['project_id']

    for layer_name in ['buildings', 'roads', 'osm_roads', 'parcels', 'synthetic', 'drafts', 'ai_predictions']:
        response = client.get(f'/api/projects/{project_id}/layers/{layer_name}')
        assert response.status_code == 200, response.text
        payload = response.json()
        assert payload['total_features'] == 0, (layer_name, payload)

    warnings_response = client.get(f'/api/projects/{project_id}/warnings')
    assert warnings_response.status_code == 200, warnings_response.text
    assert warnings_response.json()['total_warnings'] == 0

    layer_response = client.get(f'/api/projects/{project_id}/layers')
    assert layer_response.status_code == 200, layer_response.text
    assert layer_response.json()['project_id'] == project_id


def test_project_inference_uses_project_raster_and_isolated_run_storage(tmp_path, monkeypatch):
    registry = ProjectRegistry(str(tmp_path / 'projects'))
    monkeypatch.setattr(api, 'project_registry', registry)
    inference_inputs = {}

    def fake_inference(**kwargs):
        inference_inputs.update(kwargs)
        os.makedirs(kwargs['out_dir'], exist_ok=True)
        for filename in ('building_probability.tif', 'building_mask.tif', 'MODEL_RUN.md'):
            with open(os.path.join(kwargs['out_dir'], filename), 'wb') as handle:
                handle.write(b'test-artifact')
        return {
            'type': 'FeatureCollection',
            'name': 'project_predictions',
            'features': [],
            'model_metadata': {
                'model_name': 'WHU test model',
                'model_version': 'test-revision',
                'run_id': kwargs['run_id'],
                'detected_count': 0,
            },
        }

    monkeypatch.setattr(api, 'run_whu_live_inference', fake_inference)
    raster_path = tmp_path / 'source.tif'
    _create_rgb_raster(raster_path)
    client = TestClient(app)
    with raster_path.open('rb') as handle:
        created = client.post(
            '/api/projects',
            files={'file': ('source.tif', handle.read(), 'image/tiff')},
            data={'name': 'Inference test', 'locality': 'Test area'},
        )
    assert created.status_code == 201, created.text
    project_id = created.json()['project_id']
    project_raster_path = registry.get_project_raster_path(project_id)

    started = client.post(
        f'/api/projects/{project_id}/model-runs',
        json={'confidence_threshold': 0.55},
    )

    assert started.status_code == 202, started.text
    run_id = started.json()['run_id']
    assert inference_inputs['src_tif_path'] == project_raster_path
    assert inference_inputs['project_id'] == project_id
    assert inference_inputs['confidence_threshold'] == 0.55
    expected_runs_dir = os.path.join(str(tmp_path / 'projects'), project_id, 'runs')
    assert os.path.commonpath([inference_inputs['out_dir'], expected_runs_dir]) == expected_runs_dir

    run_response = client.get(f'/api/projects/{project_id}/model-runs/{run_id}')
    assert run_response.status_code == 200
    run = run_response.json()
    assert run['status'] == 'completed'
    assert run['project_id'] == project_id
    assert run['features'] == []
    assert os.path.isfile(os.path.join(inference_inputs['out_dir'], 'predictions.geojson'))
    with open(os.path.join(inference_inputs['out_dir'], 'predictions.geojson'), encoding='utf-8') as handle:
        assert json.load(handle)['model_metadata']['project_id'] == project_id