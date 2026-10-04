import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from tools.inria_ingest import build_manifest, city_from_name


def write_tile(path: Path, value: int = 0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=8,
        width=8,
        count=3,
        dtype="uint8",
        crs="EPSG:32610",
        transform=from_origin(500000, 4100000, 0.3, 0.3),
    ) as dst:
        dst.write(np.full((3, 8, 8), value, dtype=np.uint8))


def write_mask(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=8,
        width=8,
        count=1,
        dtype="uint8",
        crs="EPSG:32610",
        transform=from_origin(500000, 4100000, 0.3, 0.3),
    ) as dst:
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[2:4, 2:4] = 255
        dst.write(mask, 1)


def test_missing_inria_data_is_a_download_gate(tmp_path):
    manifest = build_manifest(tmp_path / "missing", tmp_path / "manifest.json")
    assert manifest["status"] == "WAITING_FOR_USER_DOWNLOAD"
    assert "download" in manifest["download_url"]


def test_city_level_split_and_pair_validation(tmp_path):
    root = tmp_path / "inria"
    for name in ("austin1", "austin2", "vienna1"):
        image = root / "images" / f"{name}.tif"
        mask = root / "gt" / f"{name}.tif"
        write_tile(image, value=80)
        write_mask(mask)
    manifest = build_manifest(root, tmp_path / "manifest.json", expected_size=(8, 8))
    assert manifest["status"] == "READY_FOR_TRAINING"
    assert manifest["validation_count"] == 1
    assert manifest["train_count"] == 2
    assert not ({tile["city"] for tile in manifest["tiles"] if tile["split"] == "train"} &
                {tile["city"] for tile in manifest["tiles"] if tile["split"] == "validation"})
    assert all(tile["building_pixels"] == 4 for tile in manifest["tiles"])


def test_city_parser_rejects_unknown_name():
    try:
        city_from_name(Path("unknown1.tif"))
    except ValueError as exc:
        assert "Cannot identify" in str(exc)
    else:
        raise AssertionError("unknown city should be rejected")