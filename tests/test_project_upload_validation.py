import io
import os

import numpy as np
import rasterio
import pytest
from rasterio.transform import from_origin
from PIL import Image
from fastapi.testclient import TestClient

from backend import api
from backend.api import app
from backend.services.project_registry import ProjectRegistry


def _write_test_geotiff(path, *, width=32, height=32, bands=3, dtype='uint8', crs='EPSG:3857'):
    arr = np.zeros((bands, height, width), dtype=dtype)
    for band_index in range(bands):
        arr[band_index] = (band_index + 1) * 10
    with rasterio.open(
        path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=bands,
        dtype=dtype,
        crs=crs,
        transform=from_origin(0, 1000, 1.0, 1.0)
    ) as dst:
        dst.write(arr)


def test_project_creation_persists_and_tiles_compatible_rgb_geotiff(tmp_path, monkeypatch):
    client = TestClient(app)
    registry = ProjectRegistry(str(tmp_path / 'projects'))
    monkeypatch.setattr(api, 'project_registry', registry)
    file_path = tmp_path / 'project_raster.tif'
    _write_test_geotiff(file_path)

    with file_path.open('rb') as fh:
        response = client.post(
            '/api/projects',
            files={'file': ('project_raster.tif', fh.read(), 'image/tiff')},
            data={'name': 'North block survey', 'locality': 'Lalpur, Gujarat'}
        )

    assert response.status_code == 201, response.text
    payload = response.json()
    assert payload['valid'] is True
    assert payload['project_id'].startswith('project-')
    assert payload['raster']['crs'] == 'EPSG:3857'
    assert payload['raster']['rgb_band_mapping'] == [1, 2, 3]
    assert payload['raster']['bands'] == 3
    assert payload['raster']['width'] == 32
    assert payload['raster']['height'] == 32
    assert payload['raster']['gsd_x'] == 1
    assert payload['record']['name'] == 'North block survey'
    assert payload['record']['locality'] == 'Lalpur, Gujarat'
    with rasterio.open(file_path) as source:
        assert payload['raster']['bounds'] == {
            'minx': source.bounds.left,
            'miny': source.bounds.bottom,
            'maxx': source.bounds.right,
            'maxy': source.bounds.top,
        }
        assert payload['raster']['gsd_y'] == abs(source.transform.e)

    restarted_registry = ProjectRegistry(str(tmp_path / 'projects'))
    persisted_project = restarted_registry.get_project(payload['project_id'])
    assert persisted_project['raster']['crs'] == 'EPSG:3857'
    assert os.path.isfile(restarted_registry.get_project_raster_path(payload['project_id']))

    tile = client.get(f"/api/projects/{payload['project_id']}/raster/tiles/14/8192/8191.png")
    assert tile.status_code == 200
    assert tile.headers['content-type'] == 'image/png'
    assert tile.content != b''
    assert np.asarray(Image.open(io.BytesIO(tile.content)).convert('RGBA'))[:, :, 3].max() > 0


def test_project_creation_rejects_non_geotiff(tmp_path, monkeypatch):
    client = TestClient(app)
    monkeypatch.setattr(api, 'project_registry', ProjectRegistry(str(tmp_path / 'projects')))

    response = client.post(
        '/api/projects',
        files={'file': ('bad.png', b'not-a-real-geotiff', 'image/png')},
        data={'name': 'Bad raster'}
    )

    assert response.status_code == 400
    payload = response.json()
    assert 'detail' in payload
    assert 'GeoTIFF' in str(payload['detail']) or 'georeference' in str(payload['detail']).lower()


def test_project_creation_rejects_missing_rgb_bands(tmp_path, monkeypatch):
    client = TestClient(app)
    monkeypatch.setattr(api, 'project_registry', ProjectRegistry(str(tmp_path / 'projects')))
    file_path = tmp_path / 'single_band.tif'
    _write_test_geotiff(file_path, bands=1)

    with file_path.open('rb') as fh:
        response = client.post(
            '/api/projects',
            files={'file': ('single_band.tif', fh.read(), 'image/tiff')},
            data={'name': 'Single band'}
        )

    assert response.status_code == 400
    assert 'at least three' in response.json()['detail']


@pytest.mark.parametrize(
    ('raster_options', 'message'),
    [
        ({'crs': None}, 'no CRS'),
        ({'crs': 'EPSG:4326'}, 'EPSG:3857'),
        ({'dtype': 'uint16'}, 'uint8 RGB'),
    ],
)
def test_project_creation_rejects_unsupported_geotiff_metadata(tmp_path, monkeypatch, raster_options, message):
    client = TestClient(app)
    monkeypatch.setattr(api, 'project_registry', ProjectRegistry(str(tmp_path / 'projects')))
    file_path = tmp_path / 'unsupported.tif'
    _write_test_geotiff(file_path, **raster_options)

    with file_path.open('rb') as fh:
        response = client.post(
            '/api/projects',
            files={'file': ('unsupported.tif', fh.read(), 'image/tiff')},
            data={'name': 'Unsupported raster'}
        )

    assert response.status_code == 400
    assert message in response.json()['detail']
