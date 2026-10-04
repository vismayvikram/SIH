# SIH26012 — AI-Assisted Cadastral Review Platform

**Smart India Hackathon 2026 · Team Plot Twist (Team ID 149942)**
**Problem Statement:** SIH26012 — AI-Based Automated Urban Parcel Mapping and Cadastral Feature Extraction System using Drone Imagery
**Theme:** Smart Automation · **Category:** Software · **Ministry:** Rural Development (DoLR)

 **Repo:** https://github.com/vismayvikram/SIH

---

## What is this?

A web-based AI cadastral review assistant. It takes a drone orthomosaic (ORI), extracts building footprints with a selected deep-learning model, validates the result with deterministic GIS rules, ranks review items, and lets a human reviewer edit, approve, or reject every feature before exporting a preliminary GeoJSON map.

> **We do not use AI where deterministic GIS is more reliable.**
> AI extracts uncertain visual features → the GIS engine validates their spatial relationships → the human reviewer approves the result.

The platform is a **review-and-triage tool**, not an automatic land-ownership system.

### The problem it addresses

| Problem | How the platform responds |
|---|---|
| Manual digitisation of drone ORI is slow and labour-heavy | AI drafts building footprints as clean, GIS-ready polygons |
| Dense settlements have irregular, overlapping geometries | Rule-based topology engine catches overlaps and malformed shapes |
| AI output is hard for officials to trust | Every warning is explained in plain language; human sign-off and an audit trail on every decision |
| Field teams don't know where to go first | Transparent 0–100 review-priority score with Red / Amber / Green queues |

---

## Key features (built and working)

**Data and raster pipeline**
- Real Lalpur Village (Gujarat) orthomosaic: 1.57 GB, 20,137 × 20,886 px, ~3.4 cm GSD, EPSG:3857
- On-the-fly XYZ tile server, so multi-GB rasters render smoothly in the browser
- Multi-project upload with validation (band count, dimensions, CRS, 2 GB size limit, free disk space)

**AI feature extraction**
- Project-scoped **WHU U-Net++ (EfficientNet-B4)**, **DeepLab SpaceNet**, and **Lalpur fine-tuned DeepLab** providers; unavailable providers are disabled with their local prerequisite listed
- Windowed 512 × 512 inference with 256-pixel overlap, model-resolution resampling, nodata/alpha masking, thresholding, polygonization, and minimum-area filtering
- Run metadata includes the target resolution, latitude-corrected source ground resolution, resampling ratio, input/checkpoint SHA-256, and provider
- Predictions are stored with the project, appear in its editable review layer, and remain isolated from Lalpur
- Model-vs-reference comparison is available only for projects with reference buildings; Lalpur comparison uses its local 317-footprint reference set

**GIS topology engine (Shapely / GEOS)** — geometry validity, polygon overlap, building-road spatial overlap, parcel crossing/outside coverage, and low-confidence detection checks. Each generated warning includes the measured evidence, configured tolerance, numeric exceedance, stable rule ID, and a suggested review action. Real parcel checks run only when a non-empty parcel layer is loaded; Lalpur currently has none.

Review thresholds are configured in `backend/services/review_config.py`. Area and distance measurements use projected metric geometry. The thresholds and severity bands are prototype triage settings, not cadastral or legal standards. Suggested actions are review guidance only; an outside-parcel warning explicitly recommends mapping initiation rather than enforcement.

**Review-priority scoring** — explainable parcel score (0–100) with reasons per parcel. Red ≥ 70 (urgent), Amber 40–69 (recommended), Green < 40 (low priority).

**Web-GIS workspace** — three-panel Leaflet interface: layers and warnings on the left, map in the centre, feature inspector on the right. Edit geometry, set status (Approve / Reject / Under Review), compare original vs. edited geometry, revert, and read the full audit log. Vertex editing remains active through repeated drags and ends on Save, Cancel, or Escape. Warning summaries are deterministic templates built only from fields in the warning payload; no LLM call is made.

**Persistence and export** — state survives restarts; import and export of RFC 7946 GeoJSON bundles with metadata, audit history, and a "preliminary, not legally final" disclaimer.

**Context layer** — RGB vegetation-index detection for green-cover context.

---

## Current status and honest limitations

We prefer to be clear about what is and isn't done.

| Area | Status |
|---|---|
| Orthomosaic serving, building extraction, topology, scoring, Web-GIS, export | Built and tested |
| Change detection (old vs. new buildings) | Matching logic runs inside the discrepancy API. A dedicated `/api/analytics/changes` endpoint that labels new / removed / modified buildings is **in progress** |
| Real parcel data | The Lalpur parcel template has **0 features**, because no official cadastral layer was available for this area. Parcel-based scoring and warnings are demonstrated on a **synthetic** parcel layer. The engine accepts any reference layer, so real records (e.g. Bhu-Naksha) can be plugged in once access is granted |
| Roads | OpenStreetMap vectors are used as reference. No road AI model yet |
| Land-use | Vegetation index only. No multi-class land-use model |
| Model accuracy | Pretrained and fine-tuned providers are experimental. The Lalpur fine-tuned model used Lalpur's 317 reference footprints for training; Lalpur metrics are in-sample and performance on other imagery is unvalidated. Human review stays mandatory |

### What the platform never claims
- It does **not** determine legal property ownership boundaries.
- It does **not** produce legally final cadastral records.
- It does **not** map individual apartment units.
- It does **not** claim to work on any city, drone, or image without local validation.

All AI outputs are labelled as suggestions, and every feature carries a `source` tag (AI / reference / human-edited).

---

## System architecture

```text
Drone orthomosaic (GeoTIFF)
   ↓
Validation and preprocessing (CRS, extent, tiling)
   ↓
AI feature extraction (U-Net++ / DeepLab)
   ↓
Raster-to-vector conversion (polygonization)
   ↓
Deterministic GIS topology engine (Shapely / GEOS)
   ↓
Review-priority scoring (Red / Amber / Green)
   ↓
Web-GIS review: edit · approve · reject · audit trail
   ↓
GeoJSON export (preliminary, traceable)
```

Every map feature belongs to one of three categories, kept visually separate in the UI:
1. **Detected features** — from AI models
2. **Reference features** — existing parcels and building references
3. **Suggestions and warnings** — conflicts, confidence scores, topology issues

---

## Tech stack

| Layer | Technologies |
|---|---|
| AI / ML | PyTorch, segmentation-models-pytorch, U-Net++ (EfficientNet-B4), Hugging Face Hub |
| Geospatial | Rasterio / GDAL, Shapely / GEOS, pyproj, QGIS (visual QA) |
| Backend | Python 3.10+, FastAPI, Pydantic v2, Uvicorn |
| Frontend | Leaflet 1.9, HTML5 / CSS3 / JavaScript, XYZ tile server |
| Quality and output | pytest (115 tests), GeoJSON (RFC 7946) |

---

## Project structure

```text
SIH/
├── backend/
│   ├── api.py                    # FastAPI REST endpoints
│   └── services/
│       ├── raster_service.py     # XYZ tile server
│       ├── project_registry.py   # Multi-project upload and validation
│       ├── data_loader.py        # Reference buildings, roads
│       ├── whu_model.py          # WHU U-Net++ inference
│       ├── model_adapter.py      # DeepLab SpaceNet adapter
│       ├── topology.py           # 5-rule topology engine
│       ├── scoring.py            # Review-priority scoring
│       ├── greenery_detection.py # Vegetation index
│       ├── warning_narrator.py   # Plain-English narration
│       └── feature_store.py      # Persistence and audit trail
├── frontend/
│   ├── index.html
│   └── js/                       # app, map, inspector, warnings, model
├── data/                         # Orthomosaic, model runs, store state
└── tests/                        # pytest suite
```

---

## Getting started

> Adjust the commands below to match your repository if they differ.

**Prerequisites:** Python 3.10+, GDAL-capable environment (rasterio), and optionally a CUDA GPU (CPU inference also works).

```bash
# 1. Clone
git clone https://github.com/vismayvikram/SIH.git
cd SIH

# 2. Create a virtual environment and install dependencies
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. Place the orthomosaic
# data/acquisition/SIH26012_INDIA_CANDIDATE_01/working/lalpur_orthomosaic.tif

# 4. Start the server
uvicorn backend.api:app --reload --port 8000

# 5. Open the app
# http://localhost:8000
```

**Run the tests**
```bash
pytest
```

## Using your own imagery

You can open any local RGB GeoTIFF as a project without replacing the Lalpur demo.

1. Open the dashboard and choose Create project.
2. Upload a georeferenced RGB GeoTIFF in EPSG:3857.
3. Give it a project name and optional locality label.
4. The server validates CRS, band count, dimensions, and file size before creating a project-scoped raster workspace.

The Lalpur case remains available as the built-in demo project, while uploaded imagery is stored under its own project directory and stays isolated from the demo data and demo model runs.

### Run a building model on a project

Open a project, then choose **AI Building Model** from the map toolbar (or the Actions menu). Choose an available provider, select the whole raster or draw a rectangle, adjust the confidence threshold if needed, and run. For large rasters, drawing an area reduces the number of tiles. A second concurrent model job is rejected with `A job is already running.`

The providers target approximately 0.30 m ground resolution. Project inference reads 512-pixel windows with 256-pixel overlap, skips empty/nodata pixels, and resamples through a windowed raster view; it does not load the full source raster into memory. For EPSG:3857 rasters, the source ground resolution is calculated at the raster-centre latitude:

```text
ground GSD = EPSG:3857 pixel size × cos(raster-centre latitude)
resampling ratio = source ground GSD / requested model GSD
```

Coarser-than-target imagery produces a warning rather than an inference rejection. The run summary records both source ground GSD and chosen inference resolution. Generated footprints are unverified suggestions in that project's `ai_predictions` layer; they can be inspected, edited, approved/rejected, and included in that project's GeoJSON export.

Model dependencies are optional for the rest of the application. Install `requirements-model.txt` to provide the PyTorch/Hugging Face WHU runtime. The SpaceNet DeepLab provider also requires the pinned source under `data/local_model_run/vendor/rgb-footprint-extract` and its SpaceNet checkpoint under that tree's `weights/spaceNet` directory. The Lalpur fine-tuned provider uses:

```text
data/local_model_run/lalpur_improvement/run-deeplab-20261004-141612/fine_tuned_checkpoint.pth
```

The fine-tuned model was trained using Lalpur's 317 reference footprints. Its Lalpur metrics are in-sample; performance on other imagery is unvalidated. A provider with missing weights, model source, or runtime dependencies is shown disabled with the exact local requirement. The application does not substitute mock or Lalpur predictions for a project run.

Project-model API workflow:

```text
GET  /api/models/providers
POST /api/projects/{project_id}/process/buildings
GET  /api/jobs/{job_id}                         # poll queued/running/completed/failed
GET  /api/projects/{project_id}/layers/manifest # refresh after completion
```

The POST body accepts `provider`, `threshold`, `aoi` (GeoJSON Polygon/Feature or `null` for the whole image), `resolution_m`, and `min_area_m2`. The POST returns HTTP 202; the job status includes the result layer ID, feature count, scale summary, and output path on success. Project manifests expose `has_reference_layer`; model-vs-reference comparison is unavailable for projects without reference buildings and returns HTTP 422. Lalpur retains its own reference comparison.

## Deploy with Docker

Docker is the recommended deployment path because Rasterio and the upload workflow
need a writable data directory. The image excludes local imagery and model artifacts;
upload project rasters after starting the service, or provision them separately.

```bash
docker compose up --build -d
docker compose ps
```

Open `http://localhost:8000`. Uploaded projects, review state, and generated runs
persist in the `sih-data` Docker volume across container replacements. Back up this
volume as application data. To use another port, set `PORT` before starting Compose.
For production, terminate TLS at a trusted reverse proxy and restrict network access
to the review interface; the API does not provide user authentication.

To deploy without Compose, build the image and mount a persistent Docker volume at
`/app/data`:

```bash
docker build -t sih-review .
docker run --rm -p 8000:8000 -v sih-data:/app/data sih-review
```

---

## Main API endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/raster/tiles/{z}/{x}/{y}.png` | Orthomosaic XYZ tiles |
| POST | `/api/projects` | Upload and validate a new GeoTIFF project |
| GET | `/api/models/discrepancy` | Agreement metrics: precision, recall, F1, IoU |
| POST | `/api/models/select-precomputed` | Switch to the saved WHU run |
| POST | `/api/models/select-deeplab` | Switch to DeepLab SpaceNet |
| POST | `/api/greenery/detect` | Vegetation-index detection |
| POST | `/api/warnings/narrate` | Plain-English warning narration |
| GET | `/api/parcels/rag` | Red / Amber / Green parcel queues |
| GET | `/api/export` | Export GeoJSON bundle |
| POST | `/api/import` | Import GeoJSON bundle |

---

## Feasibility and impact

- **Fits existing programmes:** designed for drone imagery already captured under SVAMITVA (3.30 lakh villages surveyed, 3.19 crore property cards) and the NAKSHA urban survey pilot (157 ULBs, 4,484 sq km).
- **Open-source stack:** no licensing cost; can run on existing government servers.
- **Local-first deployment:** sensitive land-record data stays inside the government network.
- **Scales through tiling:** tiled inference and on-demand map tiles handle multi-GB orthomosaics; GPU batch jobs can scale across districts.

**Who benefits:** survey officials review flagged areas instead of digitising every roof · field teams visit ranked hotspots first · urban local bodies get faster, cleaner base maps · property owners get more accurate records and fewer boundary disputes.

---

## Roadmap

**Near term**
- Dedicated vector change comparator (`/api/analytics/changes`): new, removed, modified and unchanged buildings, feeding the priority score
- Real parcel layer integration once partner data is available
- One-click demo scenario preset

**Future**
- Fine-tune on SVAMITVA / NAKSHA tiles with geographically held-out test blocks
- Road and pathway extraction model; multi-class land-use segmentation
- T-UNet bi-temporal change detection (needs co-registered two-date imagery)
- CA-Markov urban growth forecasting and heat-resilience layer (needs multi-year and thermal data)
- Trained parcel-boundary model (needs verified survey ground truth)

---

## Data and licensing

Model weights, datasets, and imagery are tracked with source URL, licence, attribution, download date, and purpose. The WHU U-Net++ EfficientNet-B4 weights come from Hugging Face (`giswqs`). The SpaceNet DeepLab checkpoint is loaded through a pinned `rgb-footprint-extract` loader. Check each source's licence before any use beyond this prototype.

## References

1. Zhou et al., *UNet++: A Nested U-Net Architecture for Medical Image Segmentation*, DLMIA 2018
2. Ji, Wei & Lu, *Multisource Building Extraction (WHU Building Dataset)*, IEEE TGRS 2019
3. Huang et al., *UNet 3+*, ICASSP 2020
4. Kirillov et al., *Segment Anything*, ICCV 2023
5. Zhong et al., *T-UNet: Triplet UNet for Change Detection*, Geo-spatial Information Science 2024
6. DoLR — NAKSHA urban land survey programme; PIB — SVAMITVA drone survey updates

---

## Team

**Team Plot Twist** · Smart India Hackathon 2026 · Team ID 149942

*Outputs are preliminary suggestions for surveyor review and are not legal cadastral records.*