/**
 * SIH26012 Platform - Main Application Coordinator
 */
import { ApiClient } from './api.js';
import { MapController } from './map.js';
import { FeatureInspector } from './inspector.js';
import { WarningCenter } from './warnings.js';
import { ModelModalController } from './model.js';

class Application {
  constructor() {
    this.map = null;
    this.inspector = null;
    this.warningCenter = null;
    this.modelController = null;
  }

  async init() {
    console.log('Initializing SIH26012 Feature Review Platform...');

    // Initialize Map
    this.map = new MapController('map', {
      onFeatureSelect: (fid, layerName, feature) => this.handleFeatureSelect(fid, layerName),
      onGeometryChange: (fid, newGeom) => this.handleGeometryChange(newGeom),
      onDraftCreated: (geom) => this.handleDraftCreated(geom)
    });
    this.map.init();

    // Initialize Inspector
    this.inspector = new FeatureInspector('feature-inspector-container', {
      onStatusChanged: (fid, newStatus) => this.handleStatusChanged(fid, newStatus),
      onGeometrySaved: (fid) => this.refreshAllData(),
      onGeometryReverted: (fid) => this.refreshAllData(),
      onStartVertexEdit: () => this.map.startVertexEditing(),
      onWarningClick: (warning) => this.map.focusOnWarning(warning)
    });

    // Initialize Warning Center
    this.warningCenter = new WarningCenter('bottom-warning-tray-container', {
      onWarningClick: (warning) => this.map.focusOnWarning(warning)
    });

    // Initialize Model Modals
    this.modelController = new ModelModalController({
      onInferenceSuccess: (res) => this.handleInferenceSuccess(res)
    });

    // Bind UI controls
    this.bindLayerToggles();
    this.bindBasemapButtons();
    this.bindHeaderActions();
    this.bindDrawingTools();

    // Initial Data Load
    await this.loadInitialData();
  }

  async loadInitialData() {
    try {
      // 1. Load Metadata
      const metadata = await ApiClient.getMetadata();
      this.updateMetadataNotice(metadata);

      // 2. Load Vector Layers
      const [buildings, roads, osmRoads, parcels, synthetic] = await Promise.all([
        ApiClient.getLayer('buildings'),
        ApiClient.getLayer('roads'),
        ApiClient.getLayer('osm_roads'),
        ApiClient.getLayer('parcels'),
        ApiClient.getLayer('synthetic')
      ]);

      const predictionSource = await ApiClient.getPredictionSource();
      this.updatePredictionSourceBadge(predictionSource);

      this.map.loadLayerData('buildings', buildings);
      this.map.loadLayerData('roads', roads);
      this.map.loadLayerData('osm_roads', osmRoads);
      this.map.loadLayerData('synthetic', synthetic);

      // Update layer counter badges in sidebar
      this.updateLayerBadges({
        buildings: buildings.total_features,
        roads: roads.total_features,
        osm_roads: osmRoads.total_features,
        parcels: parcels.total_features,
        synthetic: synthetic.total_features
      });

      // 3. Load Warnings
      const warningsData = await ApiClient.getWarnings();
      this.warningCenter.setWarnings(warningsData.warnings);

      this.showToast(`Loaded ${buildings.total_features} Lalpur buildings & ${synthetic.total_features} synthetic test fixtures.`);
    } catch (err) {
      console.error('Initialization error:', err);
      this.showToast('Error loading layers: ' + err.message, 'danger');
    }
  }

  async refreshAllData() {
    const [buildings, synthetic, warningsData] = await Promise.all([
      ApiClient.getLayer('buildings'),
      ApiClient.getLayer('synthetic'),
      ApiClient.getWarnings()
    ]);
    this.map.loadLayerData('buildings', buildings);
    this.map.loadLayerData('synthetic', synthetic);
    this.warningCenter.setWarnings(warningsData.warnings);
  }

  handleFeatureSelect(featureId, layerName) {
    this.inspector.loadFeature(featureId, layerName);
  }

  handleGeometryChange(newGeometry) {
    this.inspector.notifyGeometryEdited(newGeometry);
  }

  async handleDraftCreated(geometry) {
    try {
      const draft = await ApiClient.createDraft(geometry, 'draft_polygon', 'Human reviewer digitized draft footprint');
      const draftsLayer = await ApiClient.getLayer('drafts');
      this.map.loadLayerData('drafts', draftsLayer);
      this.showToast(`Created draft feature ${draft.feature.id}!`);
      this.map.selectFeature(draft.feature.id, 'drafts', draft.feature, null);
    } catch (err) {
      this.showToast('Failed to save draft: ' + err.message, 'danger');
    }
  }

  handleStatusChanged(featureId, newStatus) {
    this.map.refreshLayerStyles();
    this.showToast(`Updated ${featureId} status to "${newStatus}"`);
    this.refreshAllData();
  }

  async handleInferenceSuccess(result) {
    const aiLayer = await ApiClient.getLayer('ai_predictions');
    this.map.loadLayerData('ai_predictions', aiLayer);
    const warningsData = await ApiClient.getWarnings();
    this.warningCenter.setWarnings(warningsData.warnings);
    this.updatePredictionSourceBadge(result.metadata || { source_label: 'No active prediction set', selected: result.detected_count > 0 });
    this.showToast(`Added ${result.detected_count} AI footprints!`);
  }

  updatePredictionSourceBadge(source) {
    const badge = document.getElementById('prediction-source-badge');
    if (!badge) return;
    badge.textContent = source.selected === false ? 'No active prediction set' : `${source.source_label} · ${source.prediction_count} features`;
  }

  bindLayerToggles() {
    const toggles = [
      { id: 'toggle-buildings', layer: 'buildings' },
      { id: 'toggle-roads', layer: 'roads' },
      { id: 'toggle-osm', layer: 'osm_roads' },
      { id: 'toggle-synthetic', layer: 'synthetic' },
      { id: 'toggle-drafts', layer: 'drafts' },
      { id: 'toggle-ai', layer: 'ai_predictions' }
    ];

    toggles.forEach(t => {
      const el = document.getElementById(t.id);
      if (el) {
        el.onchange = (e) => {
          this.map.toggleLayer(t.layer, e.target.checked);
        };
      }
    });

    // Blank parcel toggle info
    const parcelToggle = document.getElementById('toggle-parcels');
    if (parcelToggle) {
      parcelToggle.onchange = (e) => {
        if (e.target.checked) {
          alert('Real Cadastral Parcel Template has 0 features for this AOI. No official vector parcels exist. Building footprints are not parcels.');
        }
      };
    }
  }

  bindBasemapButtons() {
    const btnGeotiff = document.getElementById('btn-basemap-geotiff');
    const btnOsm = document.getElementById('btn-basemap-osm');
    const btnNeutral = document.getElementById('btn-basemap-neutral');

    const updateActive = (activeBtn) => {
      [btnGeotiff, btnOsm, btnNeutral].forEach(b => {
        if (b) b.classList.remove('active');
      });
      if (activeBtn) activeBtn.classList.add('active');
    };

    if (btnGeotiff) {
      btnGeotiff.onclick = () => {
        this.map.setBasemap('geotiff');
        updateActive(btnGeotiff);
      };
    }

    if (btnOsm) {
      btnOsm.onclick = () => {
        this.map.setBasemap('osm');
        updateActive(btnOsm);
      };
    }

    if (btnNeutral) {
      btnNeutral.onclick = () => {
        this.map.setBasemap('neutral');
        updateActive(btnNeutral);
      };
    }
  }

  bindHeaderActions() {
    // Export GeoJSON
    const btnExport = document.getElementById('btn-export-geojson');
    if (btnExport) {
      btnExport.onclick = () => {
        window.location.href = '/api/export';
        this.showToast('Downloading reviewed GeoJSON bundle...');
      };
    }

    // Import GeoJSON trigger
    const btnImport = document.getElementById('btn-import-geojson');
    const importFileInput = document.getElementById('import-file-input');
    if (btnImport && importFileInput) {
      btnImport.onclick = () => importFileInput.click();
      importFileInput.onchange = async (e) => {
        const file = e.target.files[0];
        if (!file) return;
        try {
          const text = await file.text();
          const json = JSON.parse(text);
          const res = await ApiClient.importGeoJSON(json);
          this.showToast(`Imported ${res.imported_count} features successfully!`);
          await this.loadInitialData();
        } catch (err) {
          this.showToast('Import failed: ' + err.message, 'danger');
        }
        importFileInput.value = '';
      };
    }

    // Provenance modal trigger
    const pillProvenance = document.getElementById('pill-provenance-info');
    const modalProvenance = document.getElementById('modal-provenance');
    if (pillProvenance && modalProvenance) {
      pillProvenance.onclick = () => modalProvenance.classList.add('active');
    }
  }

  bindDrawingTools() {
    const btnDrawPolygon = document.getElementById('btn-tool-draw-polygon');
    const btnFinishDraw = document.getElementById('btn-tool-finish-draw');
    const btnCancelDraw = document.getElementById('btn-tool-cancel-draw');

    if (btnDrawPolygon) {
      btnDrawPolygon.onclick = () => {
        this.map.startDrawingDraft();
        btnDrawPolygon.style.display = 'none';
        if (btnFinishDraw) btnFinishDraw.style.display = 'inline-flex';
        if (btnCancelDraw) btnCancelDraw.style.display = 'inline-flex';
        this.showToast('Click on the map to add polygon vertices. Finish when done.');
      };
    }

    if (btnFinishDraw) {
      btnFinishDraw.onclick = () => {
        this.map.finishDrawingDraft();
        btnFinishDraw.style.display = 'none';
        if (btnCancelDraw) btnCancelDraw.style.display = 'none';
        if (btnDrawPolygon) btnDrawPolygon.style.display = 'inline-flex';
      };
    }

    if (btnCancelDraw) {
      btnCancelDraw.onclick = () => {
        this.map.cancelDrawingDraft();
        btnFinishDraw.style.display = 'none';
        btnCancelDraw.style.display = 'none';
        if (btnDrawPolygon) btnDrawPolygon.style.display = 'inline-flex';
        this.showToast('Cancelled draft drawing.');
      };
    }
  }

  updateLayerBadges(counts) {
    const bldBadge = document.getElementById('badge-count-buildings');
    const rdBadge = document.getElementById('badge-count-roads');
    const osmBadge = document.getElementById('badge-count-osm');
    const synthBadge = document.getElementById('badge-count-synthetic');

    if (bldBadge) bldBadge.innerText = counts.buildings;
    if (rdBadge) rdBadge.innerText = counts.roads;
    if (osmBadge) osmBadge.innerText = counts.osm_roads;
    if (synthBadge) synthBadge.innerText = counts.synthetic;
  }

  updateMetadataNotice(metadata) {
    const noticeEl = document.getElementById('raster-status-text');
    if (noticeEl && metadata.raster_orthomosaic_status) {
      const ro = metadata.raster_orthomosaic_status;
      if (ro.browser_service_status === 'AVAILABLE_LOCAL_XYZ_TILES') {
        const sizeMb = (ro.file_size_bytes / (1024 * 1024)).toFixed(1);
        noticeEl.innerHTML = `
          <strong>Drone Raster (GeoTIFF):</strong> ${ro.filename}<br>
          <span style="font-size: 10.5px; color: var(--accent-cyan);">Status: Active Local XYZ Tile Server (${sizeMb} MB, 4-band EPSG:3857)</span><br>
          <span style="font-size: 10px; color: var(--text-faint);">Tiled dynamically via rasterio. OpenStreetMap attribution preserved.</span>
        `;
      } else {
        noticeEl.innerHTML = `
          <strong>Drone Raster Status:</strong> Offline / Fallback<br>
          <span style="font-size: 10.5px; color: var(--accent-rose);">${ro.reason || 'GeoTIFF raster file unavailable.'}</span><br>
          <span style="font-size: 10px; color: var(--text-faint);">Falling back to OpenStreetMap / Neutral Dark canvas.</span>
        `;
      }
    }
  }

  showToast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    if (!container) return;
    const toast = document.createElement('div');
    toast.className = 'toast';
    if (type === 'danger') toast.style.borderLeftColor = 'var(--accent-rose)';
    toast.innerText = message;
    container.appendChild(toast);
    setTimeout(() => {
      toast.style.opacity = '0';
      setTimeout(() => toast.remove(), 300);
    }, 3500);
  }
}

// Start app on DOM ready
document.addEventListener('DOMContentLoaded', () => {
  const app = new Application();
  app.init();
});
