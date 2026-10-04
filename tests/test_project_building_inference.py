import math
import os
import threading
import time

import numpy as np
import pytest
import rasterio
from fastapi.testclient import TestClient
from rasterio.transform import from_origin

from backend import api
from backend.api import app
from backend.services.project_registry import ProjectRegistry
from backend.services.project_inference import (
    compute_resampling_parameters,
    get_building_model_providers,
)
from backend.services import project_inference


def test_web_mercator_resampling_factor_uses_ground_resolution():
    scale = compute_resampling_parameters(0.05, 23.0, 0.30)

    expected_ground_gsd = 0.05 * math.cos(math.radians(23.0))
    assert math.isclose(scale['source_ground_resolution_m'], expected_ground_gsd, rel_tol=1e-12)
    assert math.isclose(scale['resampling_factor'], expected_ground_gsd / 0.30, rel_tol=1e-12)
    assert math.isclose(scale['downsample_factor'], 0.30 / expected_ground_gsd, rel_tol=1e-12)


def test_provider_registry_lists_all_explicit_providers():
    providers = get_building_model_providers()
    by_id = {provider['id']: provider for provider in providers}

    assert set(by_id) == {'whu', 'deeplab_spacenet', 'deeplab_lalpur_finetuned'}
    assert all(isinstance(provider['enabled'], bool) for provider in providers)
    assert all(provider['reason'] for provider in providers if not provider['enabled'])
    assert by_id['deeplab_lalpur_finetuned']['expected_resolution_m'] == 0.30


def test_provider_endpoint_reports_real_local_availability():
    response = TestClient(app).get('/api/models/providers')
    assert response.status_code == 200, response.text
    providers = {provider['id']: provider for provider in response.json()['providers']}

    assert set(providers) == {'whu', 'deeplab_spacenet', 'deeplab_lalpur_finetuned'}
    for provider in providers.values():
        assert provider['enabled'] == (provider['reason'] is None)
        if provider['enabled']:
            assert provider['checkpoint_path'] and os.path.isfile(provider['checkpoint_path'])
        else:
            assert provider['reason']
    finetuned = providers['deeplab_lalpur_finetuned']
    assert finetuned['checkpoint_path'].endswith(os.path.join('run-deeplab-20261004-141612', 'fine_tuned_checkpoint.pth'))


def test_dashboard_project_list_does_not_initialize_lalpur_store(monkeypatch):
    def unexpected_demo_load(*args, **kwargs):
        raise AssertionError('Dashboard project-list request eagerly loaded Lalpur data.')

    monkeypatch.setattr(api.store, 'initialize', unexpected_demo_load)
    with TestClient(app) as client:
        response = client.get('/api/projects')

    assert response.status_code == 200, response.text


def test_windowed_inference_polygonizes_a_tiny_project_raster(tmp_path, monkeypatch):
    raster_path = tmp_path / 'tiny.tif'
    _create_rgb_raster(raster_path)
    checkpoint = tmp_path / 'checkpoint.pth'
    checkpoint.write_bytes(b'test checkpoint')
    monkeypatch.setattr(project_inference, 'get_building_model_providers', lambda: [{
        'id': 'whu',
        'label': 'WHU test provider',
        'enabled': True,
        'reason': None,
        'expected_resolution_m': 0.30,
        'checkpoint_path': str(checkpoint),
    }])

    def fake_predictor(provider_id, checkpoint_path):
        def predict(tile):
            probabilities = np.zeros((512, 512), dtype=np.float32)
            probabilities[:6, :6] = 0.95
            return probabilities
        return predict, 'cpu'

    monkeypatch.setattr(project_inference, '_load_predictor', fake_predictor)
    result = project_inference.run_project_building_inference(
        project_id='project-test',
        raster_path=str(raster_path),
        raster_metadata={'rgb_band_mapping': [1, 2, 3], 'alpha_band': None},
        provider_id='whu',
        threshold=0.5,
        aoi=None,
        resolution_m=None,
        min_area_m2=0.01,
        run_dir=str(tmp_path / 'run-project-test'),
    )

    assert result['metadata']['tile_count'] == 1
    assert result['metadata']['resolution_m'] == 0.30
    assert result['features']
    assert result['features'][0]['properties']['source'] == 'ai'
    assert result['features'][0]['properties']['checkpoint_sha256']
    assert os.path.isfile(result['output_path'])


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
        transform=from_origin(8099000, 2637000, 0.05, 0.05),
    ) as dataset:
        dataset.write(np.full((3, 32, 32), 120, dtype='uint8'))


def _install_enabled_test_provider(monkeypatch, *, runner=None):
    providers = [{
        'id': 'whu',
        'label': 'WHU test provider',
        'enabled': True,
        'reason': None,
        'default_threshold': 0.5,
        'default_min_area_m2': 1.0,
        'expected_resolution_m': 0.3,
        'checkpoint_path': None,
    }]
    monkeypatch.setattr(api, 'get_building_model_providers', lambda: providers)
    monkeypatch.setattr(project_inference, 'get_building_model_providers', lambda: providers)
    if runner is not None:
        monkeypatch.setattr(api, 'run_project_building_inference', runner)


def _new_project(tmp_path, monkeypatch):
    registry = ProjectRegistry(str(tmp_path / 'projects'))
    monkeypatch.setattr(api, 'project_registry', registry)
    api.PROJECT_JOBS.clear()
    raster_path = tmp_path / 'test.tif'
    _create_rgb_raster(raster_path)
    client = TestClient(app)
    with raster_path.open('rb') as handle:
        response = client.post(
            '/api/projects',
            files={'file': ('test.tif', handle.read(), 'image/tiff')},
            data={'name': 'Inference test', 'locality': 'Test locality'},
        )
    assert response.status_code == 201, response.text
    return client, registry, response.json()['project_id']


def _wait_for_job(client, job_id, timeout_seconds=5):
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        response = client.get(f'/api/jobs/{job_id}')
        assert response.status_code == 200, response.text
        job = response.json()
        if job['status'] in {'completed', 'failed'}:
            return job
        time.sleep(0.01)
    raise AssertionError('Project building-model job did not finish within five seconds.')


def test_project_manifest_reference_flag_and_discrepancy_requires_reference(tmp_path, monkeypatch):
    client, _, project_id = _new_project(tmp_path, monkeypatch)

    manifest = client.get(f'/api/projects/{project_id}/layers/manifest')
    assert manifest.status_code == 200, manifest.text
    assert manifest.json()['has_reference_layer'] is False
    assert all(layer['id'] != 'buildings' for layer in manifest.json()['layers'])
    demo_manifest = client.get('/api/projects/SIH26012_INDIA_CANDIDATE_01_LALPUR/layers/manifest')
    assert demo_manifest.json()['has_reference_layer'] is True
    assert any(layer['id'] == 'buildings' for layer in demo_manifest.json()['layers'])

    discrepancy = client.get(f'/api/models/discrepancy?project_id={project_id}')
    assert discrepancy.status_code == 422
    assert discrepancy.json()['detail'] == 'No reference layer for this project.'


def test_project_process_publishes_runner_output_to_project_layer(tmp_path, monkeypatch):
    client, registry, project_id = _new_project(tmp_path, monkeypatch)
    checkpoint = tmp_path / 'model.pth'
    checkpoint.write_bytes(b'test model weights')
    _install_enabled_test_provider(monkeypatch)
    for provider in project_inference.get_building_model_providers():
        provider['checkpoint_path'] = str(checkpoint)

    def fake_predictor(provider_id, checkpoint_path):
        def predict(tile):
            probability = np.zeros((512, 512), dtype=np.float32)
            probability[:8, :8] = 0.95
            return probability
        return predict, 'cpu'

    monkeypatch.setattr(project_inference, '_load_predictor', fake_predictor)
    started = client.post(f'/api/projects/{project_id}/process/buildings', json={'provider': 'whu'})
    assert started.status_code == 202, started.text
    job = _wait_for_job(client, started.json()['job_id'])

    assert job['status'] == 'completed', job
    assert job['result_summary']['layer_id'] == 'ai_predictions'
    assert job['feature_count'] > 0
    layer = client.get(f'/api/projects/{project_id}/layers/ai_predictions')
    assert layer.status_code == 200
    assert layer.json()['features'][0]['properties']['source'] == 'ai'
    assert layer.json()['features'][0]['properties']['provider'] == 'whu'
    feature_id = layer.json()['features'][0]['id']
    details = client.get(f'/api/projects/{project_id}/features/{feature_id}')
    assert details.status_code == 200
    approved = client.put(
        f'/api/projects/{project_id}/features/{feature_id}',
        json={'review_status': 'approved', 'notes': 'Reviewed'},
    )
    assert approved.status_code == 200, approved.text
    geometry = details.json()['feature']['geometry']
    edited = client.post(
        f'/api/projects/{project_id}/features/{feature_id}/edit-geometry',
        json={'geometry': geometry, 'edit_reason': 'Test geometry edit'},
    )
    assert edited.status_code == 200, edited.text
    exported = client.get(f'/api/projects/{project_id}/export')
    assert exported.status_code == 200
    assert exported.json()['export_metadata']['project_id'] == project_id
    assert any(feature['id'] == feature_id for feature in exported.json()['features'])
    scores = client.get(f'/api/projects/{project_id}/scores')
    assert scores.status_code == 200, scores.text
    assert feature_id in scores.json()['scores']
    warnings = client.get(f'/api/projects/{project_id}/warnings')
    assert warnings.status_code == 200, warnings.text
    assert not any('parcel' in warning['warning_type'] for warning in warnings.json()['warnings'])
    assert registry.get_project(project_id)['model_run_ids']
    state_path = os.path.join(registry.project_storage_dir(project_id), 'store_state.json')
    assert os.path.isfile(state_path)


def test_failed_project_run_does_not_publish_prediction_state(tmp_path, monkeypatch):
    client, _, project_id = _new_project(tmp_path, monkeypatch)

    def fail_runner(**kwargs):
        raise RuntimeError('synthetic inference failure')

    _install_enabled_test_provider(monkeypatch, runner=fail_runner)
    started = client.post(f'/api/projects/{project_id}/process/buildings', json={'provider': 'whu'})
    job = _wait_for_job(client, started.json()['job_id'])

    assert job['status'] == 'failed'
    assert 'synthetic inference failure' in job['error_message']
    layer = client.get(f'/api/projects/{project_id}/layers/ai_predictions')
    assert layer.status_code == 200
    assert layer.json()['features'] == []


def test_disabled_provider_returns_exact_reason_without_fallback(tmp_path, monkeypatch):
    client, _, project_id = _new_project(tmp_path, monkeypatch)
    reason = 'WHU checkpoint is not present in the local Hugging Face cache.'
    monkeypatch.setattr(api, 'get_building_model_providers', lambda: [{
        'id': 'whu', 'enabled': False, 'reason': reason,
    }])

    response = client.post(f'/api/projects/{project_id}/process/buildings', json={'provider': 'whu'})

    assert response.status_code == 503
    assert response.json()['detail'] == reason
    layer = client.get(f'/api/projects/{project_id}/layers/ai_predictions')
    assert layer.status_code == 200
    assert layer.json()['features'] == []


def test_real_whu_provider_on_tiny_project_returns_layer_or_clear_disabled_reason(tmp_path, monkeypatch):
    client, _, project_id = _new_project(tmp_path, monkeypatch)
    registry_response = client.get('/api/models/providers')
    assert registry_response.status_code == 200, registry_response.text
    provider = next(item for item in registry_response.json()['providers'] if item['id'] == 'whu')
    started = client.post(
        f'/api/projects/{project_id}/process/buildings',
        json={'provider': 'whu', 'min_area_m2': 0.0},
    )
    if not provider['enabled']:
        assert started.status_code == 503
        assert started.json()['detail'] == provider['reason']
        pytest.skip(f"Local WHU provider unavailable: {provider['reason']}")

    assert started.status_code == 202, started.text
    job = _wait_for_job(client, started.json()['job_id'])
    assert job['status'] == 'completed', job.get('error_message')
    assert job['result_summary']['layer_id'] == 'ai_predictions'
    assert job['feature_count'] >= 0
    layer = client.get(f'/api/projects/{project_id}/layers/ai_predictions')
    assert layer.status_code == 200, layer.text
    assert len(layer.json()['features']) == job['feature_count']


@pytest.mark.parametrize('provider_id', ['deeplab_spacenet', 'deeplab_lalpur_finetuned'])
def test_real_deeplab_provider_on_tiny_project(provider_id, tmp_path, monkeypatch):
    client, _, project_id = _new_project(tmp_path, monkeypatch)
    response = client.get('/api/models/providers')
    assert response.status_code == 200, response.text
    provider = next(item for item in response.json()['providers'] if item['id'] == provider_id)
    started = client.post(
        f'/api/projects/{project_id}/process/buildings',
        json={'provider': provider_id, 'min_area_m2': 0.0},
    )
    if not provider['enabled']:
        assert started.status_code == 503
        assert started.json()['detail'] == provider['reason']
        pytest.skip(f"Local {provider_id} unavailable: {provider['reason']}")

    assert started.status_code == 202, started.text
    job = _wait_for_job(client, started.json()['job_id'], timeout_seconds=120)
    assert job['status'] == 'completed', job.get('error_message')
    assert job['result_summary']['provider_id'] == provider_id
    assert job['result_summary']['layer_id'] == 'ai_predictions'
    assert job['feature_count'] >= 0


def test_second_concurrent_project_run_is_rejected(tmp_path, monkeypatch):
    client, _, project_id = _new_project(tmp_path, monkeypatch)
    entered = threading.Event()
    release = threading.Event()

    def blocked_runner(**kwargs):
        entered.set()
        assert release.wait(5)
        return {
            'features': [],
            'output_path': os.path.join(kwargs['run_dir'], 'predictions.geojson'),
            'metadata': {
                'created_at': '2026-10-05T00:00:00+00:00',
                'resolution_m': 0.3,
                'source_ground_resolution_m': 0.3,
                'resampling_factor_source_pixels_per_output_pixel': 1.0,
                'tile_count': 1,
                'elapsed_seconds': 0.1,
                'warnings': [],
            },
        }

    _install_enabled_test_provider(monkeypatch, runner=blocked_runner)
    first = client.post(f'/api/projects/{project_id}/process/buildings', json={'provider': 'whu'})
    assert first.status_code == 202, first.text
    assert entered.wait(2)
    try:
        second = client.post(f'/api/projects/{project_id}/process/buildings', json={'provider': 'whu'})
        assert second.status_code == 409
        assert second.json()['detail'] == 'A job is already running.'
    finally:
        release.set()
    assert _wait_for_job(client, first.json()['job_id'])['status'] == 'completed'