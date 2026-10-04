import io
import json
from pathlib import Path

import numpy as np
import rasterio
from fastapi.testclient import TestClient
from PIL import Image
from rasterio.transform import from_origin

from backend import api
from backend.api import app
from backend.services.project_registry import ProjectRegistry


def _create_rgb_raster(path, *, left=0.0, bottom=0.0, right=10.0, top=10.0):
    with rasterio.open(
        path,
        'w',
        driver='GTiff',
        height=32,
        width=32,
        count=3,
        dtype='uint8',
        crs='EPSG:3857',
        transform=from_origin(left, top, (right - left) / 32.0, (top - bottom) / 32.0),
    ) as dataset:
        rgb = np.full((3, 32, 32), 120, dtype='uint8')
        dataset.write(rgb)


def test_project_payload_includes_bounds_wgs84_and_manifest_tile_metadata(tmp_path, monkeypatch):
    registry = ProjectRegistry(str(tmp_path / 'projects'))
    monkeypatch.setattr(api, 'project_registry', registry)
    raster_path = tmp_path / 'sample.tif'
    _create_rgb_raster(raster_path, left=1000.0, bottom=1000.0, right=1010.0, top=1010.0)

    client = TestClient(app)
    with raster_path.open('rb') as handle:
        created = client.post(
            '/api/projects',
            files={'file': ('sample.tif', handle.read(), 'image/tiff')},
            data={'name': 'Project One', 'locality': 'West field'},
        )
    assert created.status_code == 201, created.text
    project_id = created.json()['project_id']

    project = client.get(f'/api/projects/{project_id}')
    assert project.status_code == 200, project.text
    payload = project.json()
    assert 'bounds_wgs84' in payload['raster']
    bounds = payload['raster']['bounds_wgs84']
    assert len(bounds) == 2
    assert bounds[0][0] < bounds[1][0]
    assert bounds[0][1] < bounds[1][1]
    assert all(np.isfinite(value) for point in bounds for value in point)
    assert all(-90 <= point[0] <= 90 and -180 <= point[1] <= 180 for point in bounds)

    manifest = client.get(f'/api/projects/{project_id}/layers/manifest')
    assert manifest.status_code == 200, manifest.text
    manifest_payload = manifest.json()
    orthomosaic = next(layer for layer in manifest_payload['layers'] if layer['id'] == 'orthomosaic')
    assert orthomosaic['tile_url_template'].endswith('/{z}/{x}/{y}.png')
    assert orthomosaic['max_native_zoom'] >= 18
    assert orthomosaic['max_zoom'] >= orthomosaic['max_native_zoom']


def test_project_tile_endpoints_return_png_for_covered_tiles_and_transparent_for_outside_tiles(tmp_path, monkeypatch):
    registry = ProjectRegistry(str(tmp_path / 'projects'))
    monkeypatch.setattr(api, 'project_registry', registry)
    raster_path = tmp_path / 'sample.tif'
    _create_rgb_raster(raster_path, left=1000.0, bottom=1000.0, right=1010.0, top=1010.0)

    client = TestClient(app)
    with raster_path.open('rb') as handle:
        created = client.post(
            '/api/projects',
            files={'file': ('sample.tif', handle.read(), 'image/tiff')},
            data={'name': 'Project Two', 'locality': 'South field'},
        )
    assert created.status_code == 201, created.text
    project_id = created.json()['project_id']

    zoom = 18
    resolution = 2 * 20037508.342789244 / (256 * (2 ** zoom))
    x = int((1000.0 + 20037508.342789244) / (256 * resolution))
    y = int((20037508.342789244 - 1010.0) / (256 * resolution))
    hit = client.get(f'/api/projects/{project_id}/raster/tiles/{zoom}/{x}/{y}.png')
    assert hit.status_code == 200, hit.text
    assert hit.headers['content-type'].startswith('image/png')
    assert Image.open(io.BytesIO(hit.content)).convert('RGBA').getchannel('A').getbbox() is not None

    outside = client.get(f'/api/projects/{project_id}/raster/tiles/{zoom}/{x + 20}/{y}.png')
    assert outside.status_code == 200, outside.text
    assert outside.headers['content-type'].startswith('image/png')
    assert Image.open(io.BytesIO(outside.content)).convert('RGBA').getchannel('A').getbbox() is None


def test_unknown_project_returns_404_for_project_tile_and_manifest(tmp_path, monkeypatch):
    registry = ProjectRegistry(str(tmp_path / 'projects'))
    monkeypatch.setattr(api, 'project_registry', registry)
    client = TestClient(app)

    manifest = client.get('/api/projects/unknown-project/layers/manifest')
    assert manifest.status_code == 404, manifest.text

    tile = client.get('/api/projects/unknown-project/raster/tiles/0/0/0.png')
    assert tile.status_code == 404, tile.text


def test_legacy_project_payload_computes_validated_bounds_and_native_zoom(tmp_path, monkeypatch):
    registry = ProjectRegistry(str(tmp_path / 'projects'))
    registry.ensure_project_record(
        'project-legacy',
        name='Legacy project',
        locality='Test locality',
        raster_meta={
            'relative_path': 'rasters/legacy.tif',
            'crs': 'EPSG:3857',
            'bounds': {'minx': 1000.0, 'miny': 1000.0, 'maxx': 1010.0, 'maxy': 1010.0},
            'gsd_x': 0.25,
            'gsd_y': 0.25,
            'available': True,
        },
    )
    monkeypatch.setattr(api, 'project_registry', registry)
    client = TestClient(app)

    response = client.get('/api/projects/project-legacy')

    assert response.status_code == 200, response.text
    raster = response.json()['raster']
    assert len(raster['bounds_wgs84']) == 2
    assert all(np.isfinite(value) for point in raster['bounds_wgs84'] for value in point)
    assert 0 <= raster['max_native_zoom'] <= 24


def test_html_and_versioned_static_assets_are_no_cache(tmp_path):
    client = TestClient(app)

    for path in ('/', '/js/app.js?v=test-build', '/css/app.css?v=test-build'):
        response = client.get(path)
        assert response.status_code == 200, path
        assert 'no-cache' in response.headers.get('cache-control', '')


def test_project_tile_render_errors_are_not_returned_as_transparent_images(tmp_path, monkeypatch):
    registry = ProjectRegistry(str(tmp_path / 'projects'))
    monkeypatch.setattr(api, 'project_registry', registry)

    class BrokenRasterService:
        def get_tile_png(self, z, x, y, *, strict=False):
            raise RuntimeError('raster read failed')

    monkeypatch.setattr(registry, 'get_project_raster_service', lambda project_id: BrokenRasterService())
    client = TestClient(app)

    response = client.get('/api/projects/project-known/raster/tiles/14/8192/8192.png')

    assert response.status_code == 500
    assert 'raster read failed' in response.json()['detail']
