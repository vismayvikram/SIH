# SIH26012 Feature Review Platform — Implementation Progress Report
*Updated: 2026-09-29 · Verified against the actual workspace, local model outputs, and running FastAPI server*

---

## 📋 Executive Summary

The platform is running locally at http://127.0.0.1:8000 and the live WHU model path is operational in this workspace. The real Lalpur orthomosaic is present at `data/acquisition/SIH26012_INDIA_CANDIDATE_01/working/lalpur_orthomosaic.tif` (1.57 GB, EPSG:3857, approx. 0.0338 m/pixel), and a real WHU U-Net++ inference run already produced a georeferenced prediction GeoJSON at `data/local_model_run/predicted_buildings_4326.geojson`.

## 2026-09-29 Wiring Correction

The discrepancy API now evaluates the explicitly selected prediction collection that the map displays. The active source is persisted with a source mode, model ID/revision, run ID, output path, creation time, feature count, and mock flag. Legacy `store_state.json` prediction arrays without this metadata are ignored, so stale mock features cannot silently become the WHU result.

The current active artifact is `WHU saved run`, run `run-whu-1790689369`, from `data/local_model_run/predicted_buildings_4326.geojson`. It contains 67 valid predictions and the reference layer contains 317 valid features. Deterministic one-to-one projected matching reproduces:

| IoU threshold | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| 0.35 | 8 | 59 | 309 | 11.94% | 2.52% | 4.16% |
| 0.50 | 4 | 63 | 313 | 5.97% | 1.26% | 2.08% |

These are agreement results against the user-confirmed, locally aligned reference set. They are not official or legal accuracy benchmarks. The old `8 TP, 166 FP, 309 FN` result was produced from the wrong persisted mock source and must not be presented as the WHU score.

The 57-versus-67 discrepancy is reconciled by restoring the canonical 67-feature artifact from the saved WHU probability raster at the documented 0.45 threshold. The project polygonizer reproduces the required 8/59/309 and 4/63/313 metrics; no geometries were fabricated.

The UI now labels `Mock predictions`, `WHU saved run`, and `Fresh WHU inference`, supports loading/clearing the saved run, and shows matched, model-only, and reference-only IDs. Real parcels remain empty and `/api/parcels/rag` remains `not_evaluated`.

### Exploratory threshold sweep

The saved `building_probability.tif` was polygonized with the existing 10 m² EPSG:32643 area filter and evaluated with the same global IoU matcher as the API. This is an exploratory full-AOI sweep, not a spatially held-out validation experiment, so no tuned setting is promoted as an unbiased accuracy result.

| Pixel threshold | Valid polygons | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.25 | 98 | 15 | 83 | 302 | 15.31% | 4.73% | 7.23% |
| 0.35 | 85 | 11 | 74 | 306 | 12.94% | 3.47% | 5.47% |
| 0.45 baseline | 67 | 8 | 59 | 309 | 11.94% | 2.52% | 4.17% |
| 0.50 | 57 | 5 | 52 | 312 | 8.77% | 1.58% | 2.67% |
| 0.65 | 47 | 4 | 43 | 313 | 8.51% | 1.26% | 2.20% |

The threshold sweep does not resolve the sparse/fragmented prediction problem. No local fine-tuning was run because the requested held-out train/validation/test blocks have not been created. A valid follow-up requires geographically separated blocks.

The current evidence shows a working WHU model demo evaluated against a user-confirmed local reference set, not an official or legal cadastral accuracy benchmark. DeepLab remains unavailable and is not part of the comparison.

---

## ✅ Verified Evidence

### 1. Raster and app state
- Raster file exists and is georeferenced: `lalpur_orthomosaic.tif`
- Raster metadata verified with rasterio:
  - CRS: `EPSG:3857`
  - Bounds: `(8098996.3782, 2636558.4073, 8099677.1552, 2637264.506)`
  - Shape: `20886 x 20137`
  - Bands: `4` (`uint8`), with red/green/blue/alpha channels
  - GSD: `~0.0338 m/pixel`
- FastAPI app responds successfully at `/api/health`, `/api/models/discrepancy`, and `/api/parcels/rag`.
- The local server is live and returns layer counts including `buildings: 317`, `parcels: 0`, `ai_predictions: 9`.

### 2. Real model run
- Model identifier: `giswqs/whu-building-unetplusplus-efficientnet-b4`
- Revision: `09df9efd323bbd3d56b98b4857129eb9b5baa2d3`
- Weight SHA-256: `922af7c96c0dc44256ab8b4d1a071f2151e0a921c997af80b55bb766bcc30dc6`
- Weight size: `84,027,346 bytes` (~80.13 MB)
- Run artifact: `data/local_model_run/MODEL_RUN.md`
- Live output: `data/local_model_run/predicted_buildings_4326.geojson`
- The canonical saved baseline contains 67 valid polygons at the documented 0.45 threshold. The earlier live run recorded 57 polygons at threshold 0.50 and is not the canonical saved baseline.

### 3. QGIS visual QA status
- The project contains screenshot and project files in `data/acquisition/SIH26012_INDIA_CANDIDATE_01/qgis/`.
- User confirmation: the 317 building-reference alignment was confirmed in QGIS.
- Result: `QGIS visual QA: confirmed; reference remains a local, non-legal benchmark set`.

### 4. Parcel status
- Real parcel layer is empty at `data/acquisition/SIH26012_INDIA_CANDIDATE_01/working/blank_parcel_template_4326.geojson`.
- `/api/parcels/rag` correctly returns `status: not_evaluated` with a clear reason.
- Synthetic parcel scoring remains demo-only and is clearly labeled as such.

### 5. Repository safety and history
- `.gitignore` is configured to ignore rasters, GeoJSON, QGIS files, local weights, and output directories.
- `gh`/GitHub visibility was not confirmed in this environment; no remote change or history rewrite was performed.
- The repo warning remains valid: untracking does not erase files from older Git history.

---

## 📊 Actual Sprint Status by Requirement

| Item | Status | Evidence |
|---|---|---|
| Real raster displayed in-browser | ✅ Yes | Local raster tiles are served by the app and the TIFF metadata is valid. |
| QGIS visual alignment | ✅ Confirmed | User confirmed the 317 building-reference alignment in QGIS; reference remains local and non-legal. |
| Model smoke tiles / full AOI | ✅ Full AOI run executed | WHU inference reports 72 tiles processed and 0 failed; canonical saved baseline is the separately restored 67-feature 0.45-threshold artifact. |
| Model output integrated into the map | ✅ Yes | The saved WHU GeoJSON is selectable without overwriting the canonical artifact; fresh runs use run-specific paths. |
| IoU / precision / recall / F1 | ✅ Local reference agreement | WHU baseline reproduces 8 TP / 59 FP / 309 FN at IoU 0.35 and 4 TP / 63 FP / 313 FN at IoU 0.50. |
| DeepLab RGB Footprint Extract | ❌ Not available | No real checkpoint exists locally; the adapter now fails explicitly and cannot return WHU, mock, empty-success, or synthetic polygons. |
| Benchmark status | ⚠️ Not claimed | Benchmark dataset adapters remain explicit `PENDING_ACQUISITION`; no fabricated benchmark pass. |
| Persistence across restart | ✅ Verified | The FeatureStore persistence test passes and the state survives restart when the same state file is used. |
| Zero real parcels | ✅ Confirmed | The real parcel layer is empty and `/api/parcels/rag` returns `not_evaluated`. |
| Repository visibility/history | ⚠️ Unknown / not changed | No `gh` verification and no remote history rewrite performed. |

---

## 🧪 Test Evidence

```text
$ python -m pytest -q
54 passed in 14.73s
```

```text
$ python -m pytest tests/test_platform.py -m model_integration -q
1 passed, 53 deselected in 4.71s
```

These are the exact fresh verification results from the current workspace.

---

## Verdict

Plain-English verdict: `working model demo`.

Reason: the project has a real WHU model checkpoint, a successful local GPU run over the Lalpur AOI, and georeferenced prediction output evaluated against a user-confirmed local reference set. It is not an official or legal cadastral accuracy benchmark, and DeepLab has not run because its checkpoint and executable loader are absent.

### What you personally need to do next
1. Obtain the real RGB Footprint Extract SpaceNet checkpoint and its verified inference loader, then run it into a separate `data/local_model_run/runs/deeplab/<run_id>/` directory.
2. Compare that genuine output to the preserved WHU baseline with the same evaluator.
3. Keep the local-reference and non-legal-benchmark caveat in any result summary.
4. No remote repository changes were made.

---

## Model Run Notes
- Model: WHU Building Detection — EfficientNet-B4 + U-Net++
- Revision: `09df9efd323bbd3d56b98b4857129eb9b5baa2d3`
- Weight SHA-256: `922af7c96c0dc44256ab8b4d1a071f2151e0a921c997af80b55bb766bcc30dc6`
- Local output directory: `data/local_model_run/`
- Tile count: `72`
- Failed tiles: `0`
- Device: `cuda`
- Elapsed time: `5.25 s`
- Alignment disposition: `confirmed by user in QGIS; local reference set, not an official/legal accuracy benchmark`

## 2026-09-30 Local Provider Path Audit

- Local branch/commit: `main` at `9c8b44c` (`v3`); the working tree is dirty and contains local, unpushed provider/UI changes.
- Root cause of identical WHU/DeepLab map output: the frontend DeepLab option submitted `mode=live`, so the API instantiated WHU. The frontend now sends explicit `mode=deeplab`.
- DeepLab asset status: no `.pth`, `.pt`, `.ckpt`, `.bin`, or `.safetensors` checkpoint was found in the workspace. The available `data/local_model_run/lalpur_rgb_0.30m.tif` is input imagery, not weights.
- DeepLab safety change: placeholder/LFS pointer files are rejected, and the adapter raises an explicit error rather than generating constant-probability polygons or returning WHU/mock output.
- Fresh WHU output change: default live runs now write below `data/local_model_run/runs/whu/<run_id>/`; the canonical saved baseline is not overwritten.
- Canonical WHU artifacts: GeoJSON SHA-256 `604F19D8B97B9CC957341F08D94EFC77C3B637CD540A0AD35D308485DD4F3FFD`; probability raster SHA-256 `34D49D402CEA624472856A9777107655F545D01F1B101457DAB2E90BA4FBFDDE`; input RGB raster SHA-256 `B55D274FE453FB461DC4E4A24697B5B1A78960C6D4E155C6862BB4FEFB1DF7B0`.
- Canonical WHU run ID: `run-whu-baseline-045`; valid feature count: 67; probability threshold: 0.45.
- Latest focused validation: `6 passed, 55 deselected` for DeepLab safety, source-selection, discrepancy, and WHU baseline checks.

