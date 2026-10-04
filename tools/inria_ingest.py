"""Validate and split the official Inria Aerial Image Labeling training data.

This tool never downloads the corpus. The official download form requires user
identity details, so the user places the archive/extracted files locally and
then runs this validator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import rasterio


OFFICIAL_DOWNLOAD_URL = "https://project.inria.fr/aerialimagelabeling/download/"
EXPECTED_CITIES = ("austin", "chicago", "kitsap", "tyrol", "vienna")
CITY_PATTERN = re.compile(r"^(austin|chicago|kitsap|tyrol|vienna)[_-]?")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def city_from_name(path: Path) -> str:
    match = CITY_PATTERN.match(path.stem.lower())
    if not match:
        raise ValueError(f"Cannot identify an Inria city from filename: {path.name}")
    return match.group(1)


def find_pairs(root: Path) -> list[tuple[Path, Path, str]]:
    image_files = sorted(root.rglob("*.tif")) + sorted(root.rglob("*.tiff"))
    image_files = [path for path in image_files if "gt" not in {part.lower() for part in path.parts}]
    pairs: list[tuple[Path, Path, str]] = []
    seen: set[str] = set()
    for image_path in image_files:
        city = city_from_name(image_path)
        key = image_path.stem.lower()
        if key in seen:
            raise ValueError(f"Duplicate image tile stem: {image_path.stem}")
        seen.add(key)
        candidates = [
            image_path.parent.parent / "gt" / image_path.name,
            image_path.parent.parent / "ground_truth" / image_path.name,
            image_path.with_name(f"{image_path.stem}_gt{image_path.suffix}"),
        ]
        mask_path = next((candidate for candidate in candidates if candidate.is_file()), None)
        if mask_path is None:
            raise ValueError(f"Missing building mask for image: {image_path}")
        pairs.append((image_path, mask_path, city))
    return pairs


def validate_pair(image_path: Path, mask_path: Path, expected_size: tuple[int, int] = (5000, 5000)) -> dict[str, Any]:
    with rasterio.open(image_path) as image, rasterio.open(mask_path) as mask:
        if image.count != 3:
            raise ValueError(f"Expected 3 RGB bands in {image_path}, found {image.count}")
        if (image.width, image.height) != expected_size or (mask.width, mask.height) != expected_size:
            raise ValueError(f"Expected {expected_size} pixels for {image_path.name}")
        if (mask.width, mask.height) != (image.width, image.height):
            raise ValueError(f"Image/mask dimensions differ for {image_path.name}")
        if image.crs is None:
            raise ValueError(f"Image CRS is missing: {image_path}")
        if image.res[0] <= 0 or image.res[1] <= 0:
            raise ValueError(f"Image resolution is invalid: {image_path}")
        mask_values = np.unique(mask.read(1))
        if not set(mask_values.tolist()).issubset({0, 1, 255}):
            raise ValueError(f"Unexpected mask values in {mask_path}: {mask_values.tolist()}")
        return {
            "width": image.width,
            "height": image.height,
            "bands": image.count,
            "crs": str(image.crs),
            "resolution": [float(image.res[0]), float(image.res[1])],
            "bounds": list(image.bounds),
            "building_pixels": int(np.count_nonzero(mask.read(1) > 0)),
            "image_sha256": sha256_file(image_path),
            "mask_sha256": sha256_file(mask_path),
        }


def build_manifest(root: Path, output_path: Path, expected_size: tuple[int, int] = (5000, 5000)) -> dict[str, Any]:
    pairs = find_pairs(root)
    if not pairs:
        return {
            "status": "WAITING_FOR_USER_DOWNLOAD",
            "download_url": OFFICIAL_DOWNLOAD_URL,
            "message": "Place the official labeled Inria training archive under the configured root and rerun this command.",
            "root": str(root),
        }

    records = []
    for image_path, mask_path, city in pairs:
        metadata = validate_pair(image_path, mask_path, expected_size=expected_size)
        records.append({
            "city": city,
            "image": str(image_path),
            "mask": str(mask_path),
            **metadata,
        })
    cities = sorted({record["city"] for record in records})
    unknown = sorted(set(cities) - set(EXPECTED_CITIES))
    if unknown:
        raise ValueError(f"Unexpected Inria city names: {unknown}")
    validation_city = cities[-1]
    for record in records:
        record["split"] = "validation" if record["city"] == validation_city else "train"
    manifest = {
        "status": "READY_FOR_TRAINING",
        "dataset": "Inria Aerial Image Labeling training subset",
        "source_url": "https://project.inria.fr/aerialimagelabeling/",
        "download_url": OFFICIAL_DOWNLOAD_URL,
        "root": str(root),
        "expected_tile_size": list(expected_size),
        "city_split_rule": f"Hold out complete city/region '{validation_city}' for validation; train on all other cities.",
        "cities": cities,
        "tiles": records,
        "tile_count": len(records),
        "train_count": sum(record["split"] == "train" for record in records),
        "validation_count": sum(record["split"] == "validation" for record in records),
        "official_test_used": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/external/inria_aerial"))
    parser.add_argument("--output", type=Path, default=Path("data/external/inria_aerial/manifest.json"))
    args = parser.parse_args()
    manifest = build_manifest(args.root, args.output)
    print(json.dumps({key: manifest[key] for key in ("status", "download_url", "tile_count", "train_count", "validation_count") if key in manifest}, indent=2))
    return 0 if manifest["status"] == "READY_FOR_TRAINING" else 2


if __name__ == "__main__":
    raise SystemExit(main())