/**
 * SIH26012 Platform - Leaflet Map Controller
 * Handles vector rendering, layer toggles, vertex editing, drawing drafts, and warning focus.
 */

export class MapController {
  constructor(containerId, options = {}) {
    this.containerId = containerId;
    this.onFeatureSelect = options.onFeatureSelect || (() => {});
    this.onGeometryChange = options.onGeometryChange || (() => {});
    this.onDraftCreated = options.onDraftCreated || (() => {});
    this.onProjectTrace = options.onProjectTrace || (() => {});

    this.map = null;
    this.basemaps = {};
    this.currentBasemap = 'osm';

    this.layers = {
      buildings: L.geoJSON(null, { style: (f) => this.getBuildingStyle(f), onEachFeature: (f, l) => this.bindFeatureEvents(f, l, 'buildings') }),
      roads: L.geoJSON(null, { style: () => this.getRoadStyle(), onEachFeature: (f, l) => this.bindFeatureEvents(f, l, 'roads') }),
      osm_roads: L.geoJSON(null, { style: () => this.getOsmRoadStyle(), onEachFeature: (f, l) => this.bindFeatureEvents(f, l, 'osm_roads') }),
      synthetic: L.geoJSON(null, { style: (f) => this.getSyntheticStyle(f), onEachFeature: (f, l) => this.bindFeatureEvents(f, l, 'synthetic') }),
      drafts: L.geoJSON(null, { style: () => this.getDraftStyle(), onEachFeature: (f, l) => this.bindFeatureEvents(f, l, 'drafts') }),
      ai_predictions: L.geoJSON(null, { style: () => this.getAiStyle(), onEachFeature: (f, l) => this.bindFeatureEvents(f, l, 'ai_predictions') }),
      ai_whu_predictions: L.geoJSON(null, { style: () => this.getAiStyle('#38bdf8'), onEachFeature: (f, l) => this.bindFeatureEvents(f, l, 'ai_whu_predictions') }),
      ai_deeplab_predictions: L.geoJSON(null, { style: () => this.getAiStyle('#f59e0b'), onEachFeature: (f, l) => this.bindFeatureEvents(f, l, 'ai_deeplab_predictions') }),
      ai_deeplab_finetuned_predictions: L.geoJSON(null, {
        style: () => ({ color: '#e11d48', fillColor: '#e11d48', weight: 2, opacity: 0.95, fillOpacity: 0.18, dashArray: '5 4' }),
        interactive: false
      }),
      ai_mock_predictions: L.geoJSON(null, { style: () => this.getAiStyle('#a855f7'), onEachFeature: (f, l) => this.bindFeatureEvents(f, l, 'ai_mock_predictions') }),
      greenery: L.geoJSON(null, { style: () => this.getGreeneryStyle() }),
      warnings: L.layerGroup()
    };

    this.selectedFeatureId = null;
    this.selectedLayer = null;
    this.selectedFeature = null;

    // Vertex edit handles
    this.editHandlesGroup = L.layerGroup();
    this.isEditingVertices = false;
    this.editSession = null;

    // Draw draft state
    this.isDrawing = false;
    this.draftPoints = [];
    this.draftPreviewLine = null;
    this.draftMarkersGroup = L.layerGroup();
  }

  init(initialCenter = [23.040453, 72.757580], zoom = 18) {
    this.map = L.map(this.containerId, {
      center: initialCenter,
      zoom: zoom,
      zoomControl: false,
      maxZoom: 21,
      minZoom: 14
    });

    // Add zoom control top right
    L.control.zoom({ position: 'topright' }).addTo(this.map);
    L.control.scale({ imperial: false, position: 'bottomright' }).addTo(this.map);

    // Basemaps
    this.basemaps.geotiff = L.tileLayer('/api/raster/tiles/{z}/{x}/{y}.png', {
      maxZoom: 21,
      minZoom: 14,
      attribution: 'Drone Orthomosaic (3.38 cm GSD GeoTIFF) | &copy; <a href="https://www.openstreetmap.org/copyright" target="_blank">OpenStreetMap contributors</a>',
      errorTileUrl: 'data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7'
    });

    this.basemaps.osm = L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
      maxZoom: 19,
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright" target="_blank">OpenStreetMap contributors</a>'
    });

    this.basemaps.neutral = L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
      maxZoom: 20,
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> &copy; <a href="https://carto.com/attributions">CARTO</a>'
    });

    // Keep the map inert until a project is explicitly opened.
    this.currentBasemap = 'geotiff';

    // Map click handler (deselect feature or add vertex in draw mode)
    this.map.on('click', (e) => this.handleMapClick(e));
    document.addEventListener('keydown', (event) => {
      if (event.key === 'Escape' && this.modelAreaDrawCallback) {
        this.cancelModelAreaDrawing();
        return;
      }
      if (event.key === 'Escape' && this.editSession?.active) {
        document.getElementById('btn-cancel-geometry')?.click();
      }
    });
  }

  refreshSize() {
    if (!this.map) return Promise.resolve();
    return new Promise((resolve) => requestAnimationFrame(() => {
      this.map.invalidateSize({ pan: false });
      resolve();
    }));
  }

  setBasemap(type) {
    if (this.currentBasemap === type || !this.basemaps[type]) return;
    this.map.removeLayer(this.basemaps[this.currentBasemap]);
    this.basemaps[type].addTo(this.map);
    this.currentBasemap = type;
  }

  setImageryStatus(message, state = 'error') {
    const host = document.getElementById('map');
    if (!host) return;
    let errorBox = document.getElementById('project-imagery-error');
    if (!errorBox) {
      errorBox = document.createElement('div');
      errorBox.id = 'project-imagery-error';
      errorBox.className = 'project-imagery-error';
      host.appendChild(errorBox);
    }
    errorBox.textContent = message;
    errorBox.hidden = !message;
    errorBox.dataset.state = state;
  }

  clearProjectState() {
    for (const layer of Object.values(this.layers)) {
      layer.clearLayers?.();
      if (this.map.hasLayer(layer)) this.map.removeLayer(layer);
    }
    this.editHandlesGroup.clearLayers();
    this.draftMarkersGroup.clearLayers();
    if (this.draftPreviewLine) this.map.removeLayer(this.draftPreviewLine);
    if (this.map.hasLayer(this.editHandlesGroup)) this.map.removeLayer(this.editHandlesGroup);
    if (this.map.hasLayer(this.draftMarkersGroup)) this.map.removeLayer(this.draftMarkersGroup);
    for (const basemap of Object.values(this.basemaps)) {
      if (this.map.hasLayer(basemap)) this.map.removeLayer(basemap);
    }
    if (this.projectRasterLayer && this.map.hasLayer(this.projectRasterLayer)) {
      this.map.removeLayer(this.projectRasterLayer);
    }
    this.projectRasterLayer = null;
    this.currentBasemap = null;
    this.selectedFeatureId = null;
    this.selectedLayer = null;
    this.selectedFeature = null;
    this.isDrawing = false;
    this.draftPoints = [];
    this.draftPreviewLine = null;
    this.editSession = null;
    this.isEditingVertices = false;
  }

  openProjectRaster(project, orthomosaic) {
    const raster = project.raster || {};
    const bounds = raster.bounds_wgs84;
    if (!raster.bounds || raster.crs !== 'EPSG:3857') {
      throw new Error('This project does not have a displayable EPSG:3857 raster.');
    }
    if (!Array.isArray(bounds) || bounds.length !== 2 || !bounds.every((point) => Array.isArray(point) && point.length === 2 && point.every(Number.isFinite))) {
      throw new Error('This project raster is missing a valid WGS84 extent.');
    }
    if (bounds[0][0] < -90 || bounds[0][0] > 90 || bounds[1][0] < -90 || bounds[1][0] > 90 ||
      bounds[0][1] < -180 || bounds[0][1] > 180 || bounds[1][1] < -180 || bounds[1][1] > 180) {
      throw new Error('This project raster has WGS84 bounds outside valid coordinate ranges.');
    }
    if (!orthomosaic?.tile_url_template) throw new Error('The project manifest has no orthomosaic tile URL.');

    this.clearProjectState();
    this.setImageryStatus('Loading project imagery...', 'loading');
    const maxNativeZoom = Number(orthomosaic.max_native_zoom ?? raster.max_native_zoom);
    if (!Number.isInteger(maxNativeZoom) || maxNativeZoom < 0 || maxNativeZoom > 24) {
      throw new Error('The project manifest has an invalid native zoom limit.');
    }
    this.projectRasterLayer = L.tileLayer(
      orthomosaic.tile_url_template,
      {
        minZoom: 0,
        maxNativeZoom,
        maxZoom: Math.max(maxNativeZoom, 21),
        attribution: project.name || 'Project orthomosaic',
        errorTileUrl: 'data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7'
      }
    );
    let firstTileSettled = false;
    let firstTileTimeout;
    const firstTile = new Promise((resolve, reject) => {
      firstTileTimeout = setTimeout(() => {
        if (firstTileSettled) return;
        firstTileSettled = true;
        const error = new Error('Timed out waiting for the first project raster tile.');
        this.onProjectTrace('first tile errored', error.message);
        this.setImageryStatus(error.message, 'error');
        reject(error);
      }, 45000);
      this.projectRasterLayer.on('tileload', (event) => {
        if (firstTileSettled) return;
        const tileUrl = this.projectRasterLayer.getTileUrl(event.coords);
        this.onProjectTrace('first tile loaded', tileUrl);
        firstTileSettled = true;
        clearTimeout(firstTileTimeout);
        this.setImageryStatus('Project imagery loaded.', 'success');
        resolve();
      });
      this.projectRasterLayer.on('tileerror', (event) => {
        const tileUrl = event.coords ? this.projectRasterLayer.getTileUrl(event.coords) : 'project raster tile';
        this.setImageryStatus(`Could not load imagery for this project: ${tileUrl}`, 'error');
        if (firstTileSettled) {
          this.onProjectTrace('tile errored', tileUrl);
          return;
        }
        this.onProjectTrace('first tile errored', tileUrl);
        firstTileSettled = true;
        clearTimeout(firstTileTimeout);
        reject(new Error(`Could not load project raster tile: ${tileUrl}`));
      });
      try {
        this.projectRasterLayer.addTo(this.map);
        this.currentBasemap = 'project';
        this.map.setMinZoom(0);
        this.map.setMaxZoom(Math.max(maxNativeZoom, 21));
        this.map.fitBounds(bounds, { padding: [24, 24], maxZoom: Math.min(20, maxNativeZoom) });
        this.onProjectTrace('tile layer added', orthomosaic.tile_url_template);
      } catch (error) {
        firstTileSettled = true;
        clearTimeout(firstTileTimeout);
        reject(error);
      }
    });
    return firstTile;
  }

  activateDemoRaster() {
    this.clearProjectState();
    this.setImageryStatus('');
    this.basemaps.geotiff.addTo(this.map);
    this.currentBasemap = 'geotiff';
    this.map.setMinZoom(14);
    this.map.setMaxZoom(21);
    Object.values(this.layers).forEach((layer) => layer.addTo(this.map));
    this.editHandlesGroup.addTo(this.map);
    this.draftMarkersGroup.addTo(this.map);
    this.map.setView([23.040453, 72.757580], 18);
  }

  startModelAreaDrawing(onComplete) {
    this.cancelModelAreaDrawing(false);
    this.modelAreaDrawCallback = onComplete;
    if (this.modelAreaRectangle && this.map.hasLayer(this.modelAreaRectangle)) {
      this.map.removeLayer(this.modelAreaRectangle);
    }
    const container = this.map.getContainer();
    container.style.cursor = 'crosshair';
    this.map.dragging.disable();
    let start = null;
    let rectangle = null;
    const finish = (bounds) => {
      this.map.off('mousedown', handlers.onDown);
      this.map.off('mousemove', handlers.onMove);
      this.map.off('mouseup', handlers.onUp);
      this.modelAreaHandlers = null;
      this.map.dragging.enable();
      container.style.cursor = '';
      const callback = this.modelAreaDrawCallback;
      this.modelAreaDrawCallback = null;
      if (bounds && rectangle) {
        this.modelAreaRectangle = rectangle;
        const southWest = bounds.getSouthWest();
        const northEast = bounds.getNorthEast();
        callback({
          type: 'Polygon',
          coordinates: [[
            [southWest.lng, southWest.lat],
            [northEast.lng, southWest.lat],
            [northEast.lng, northEast.lat],
            [southWest.lng, northEast.lat],
            [southWest.lng, southWest.lat],
          ]],
        });
      } else {
        if (rectangle && this.map.hasLayer(rectangle)) this.map.removeLayer(rectangle);
        callback?.(null);
      }
    };
    const onDown = (event) => {
      start = event.latlng;
      rectangle = L.rectangle(L.latLngBounds(start, start), {
        color: '#00b074', weight: 2, fillOpacity: 0.08, dashArray: '5 4',
      }).addTo(this.map);
    };
    const onMove = (event) => {
      if (start && rectangle) rectangle.setBounds(L.latLngBounds(start, event.latlng));
    };
    const onUp = (event) => {
      if (!start || !rectangle) return finish(null);
      const bounds = L.latLngBounds(start, event.latlng);
      if (bounds.getNorth() === bounds.getSouth() || bounds.getEast() === bounds.getWest()) return finish(null);
      rectangle.setBounds(bounds);
      finish(bounds);
    };
    const handlers = { onDown, onMove, onUp };
    this.modelAreaHandlers = handlers;
    this.map.on('mousedown', onDown);
    this.map.on('mousemove', onMove);
    this.map.on('mouseup', onUp);
  }

  cancelModelAreaDrawing(complete = true) {
    if (!this.modelAreaDrawCallback) return;
    const handlers = this.modelAreaHandlers;
    if (handlers) {
      this.map.off('mousedown', handlers.onDown);
      this.map.off('mousemove', handlers.onMove);
      this.map.off('mouseup', handlers.onUp);
    }
    this.modelAreaHandlers = null;
    this.map.dragging.enable();
    this.map.getContainer().style.cursor = '';
    const callback = this.modelAreaDrawCallback;
    this.modelAreaDrawCallback = null;
    if (this.modelAreaRectangle && this.map.hasLayer(this.modelAreaRectangle)) {
      this.map.removeLayer(this.modelAreaRectangle);
      this.modelAreaRectangle = null;
    }
    if (complete) callback(null);
  }

  toggleLayer(layerName, visible) {
    const layer = this.layers[layerName];
    if (!layer) return;
    if (visible) {
      if (!this.map.hasLayer(layer)) this.map.addLayer(layer);
    } else {
      if (this.map.hasLayer(layer)) this.map.removeLayer(layer);
    }
  }

  loadLayerData(layerName, geojsonData) {
    const layer = this.layers[layerName];
    if (!layer) return;
    layer.clearLayers();
    if (geojsonData && geojsonData.features) {
      layer.addData(geojsonData);
    }
  }

  getBuildingStyle(feature) {
    const status = feature.properties?.review_status || 'unverified';
    const isSelected = feature.id === this.selectedFeatureId;

    let color = '#b48954';
    let fillOpacity = 0.35;

    if (status === 'approved') {
      color = '#7c7b5d';
      fillOpacity = 0.45;
    } else if (status === 'rejected') {
      color = '#9a6b55';
      fillOpacity = 0.45;
    } else if (status === 'under_review') {
      color = '#b9895b';
      fillOpacity = 0.45;
    }

    if (isSelected) {
      return {
        color: '#ffffff',
        weight: 3,
        fillColor: '#b48954',
        fillOpacity: 0.65,
        dashArray: null
      };
    }

    return {
      color: color,
      weight: 1.5,
      fillColor: color,
      fillOpacity: fillOpacity
    };
  }

  getRoadStyle() {
    return {
      color: '#f59e0b',
      weight: 2,
      fillColor: '#f59e0b',
      fillOpacity: 0.25
    };
  }

  getOsmRoadStyle() {
    return {
      color: '#7c7b5d',
      weight: 3,
      dashArray: '6, 6',
      opacity: 0.85
    };
  }

  getSyntheticStyle(feature) {
    const fType = feature.properties?.feature_type;
    const isSelected = feature.id === this.selectedFeatureId;

    let color = '#b9895b';
    let dash = '4, 4';
    if (fType === 'synthetic_parcel') {
      color = '#9a6b55';
      dash = '8, 4';
    } else if (fType === 'synthetic_road_corridor') {
      color = '#d7c8b3';
    }

    return {
      color: isSelected ? '#ffffff' : color,
      weight: isSelected ? 3 : 2,
      fillColor: color,
      fillOpacity: isSelected ? 0.6 : 0.3,
      dashArray: dash
    };
  }

  getGreeneryStyle() {
    return {
      color: '#166534',
      weight: 1.5,
      fillColor: '#22c55e',
      fillOpacity: 0.38,
      dashArray: '3, 2'
    };
  }

  getDraftStyle() {
    return {
      color: '#34d399',
      weight: 2.5,
      fillColor: '#10b981',
      fillOpacity: 0.35,
      dashArray: '4, 4'
    };
  }

  getAiStyle(color = '#818cf8') {
    return {
      color,
      weight: 2,
      fillColor: color,
      fillOpacity: 0.4
    };
  }

  bindFeatureEvents(feature, leafletLayer, layerName) {
    leafletLayer.on('click', (e) => {
      L.DomEvent.stopPropagation(e);
      this.selectFeature(feature.id, layerName, feature, leafletLayer);
    });

    const fid = feature.id || feature.properties?.feature_id;
    const ftype = feature.properties?.feature_type || 'feature';
    const status = feature.properties?.review_status || 'unverified';
    leafletLayer.bindTooltip(`<strong>${fid}</strong> (${ftype})<br><span style="text-transform:capitalize">${status}</span>`, {
      sticky: true,
      className: 'custom-map-tooltip'
    });
  }

  selectFeature(featureId, layerName, feature, leafletLayer) {
    if (this.editSession?.active && featureId === this.editSession.featureId) {
      return;
    }
    if (this.editSession?.active && featureId !== this.editSession.featureId) {
      this.cancelVertexEditing('Switched feature; unsaved vertex edits were discarded.');
    }
    this.clearVertexHandles();
    this.selectedFeatureId = featureId;
    this.selectedLayer = layerName;
    this.selectedFeature = feature;

    // Refresh styles across layers to reflect selection
    this.refreshLayerStyles();

    // Notify inspector
    this.onFeatureSelect(featureId, layerName, feature);
  }

  clearSelection() {
    this.selectedFeatureId = null;
    this.selectedLayer = null;
    this.selectedFeature = null;
    this.clearVertexHandles();
    this.refreshLayerStyles();
    this.onFeatureSelect(null, null, null);
  }

  refreshLayerStyles() {
    for (const [name, layer] of Object.entries(this.layers)) {
      if (layer.setStyle) {
        if (name === 'buildings') layer.setStyle((f) => this.getBuildingStyle(f));
        else if (name === 'synthetic') layer.setStyle((f) => this.getSyntheticStyle(f));
      }
    }
  }

  startVertexEditing() {
    if (!this.selectedFeature || !this.selectedFeature.geometry) return;
    if (this.editSession?.active) return;

    const geom = JSON.parse(JSON.stringify(this.selectedFeature.geometry));
    if (geom.type !== 'Polygon') return;

    this.clearVertexHandles();
    this.isEditingVertices = true;
    this.editSession = {
      active: true,
      featureId: this.selectedFeatureId,
      originalGeometry: JSON.parse(JSON.stringify(geom)),
      draftGeometry: geom,
      handleMarkers: [],
      activeHandleIndex: null,
      moveHandler: null,
      upHandler: null
    };

    this.editSession.moveHandler = (event) => {
      if (!this.editSession || this.editSession.activeHandleIndex === null) return;
      const activeIndex = this.editSession.activeHandleIndex;
      const draftCoords = this.editSession.draftGeometry.coordinates[0];
      const point = [event.latlng.lng, event.latlng.lat];
      draftCoords[activeIndex] = point;
      if (activeIndex === 0) {
        draftCoords[draftCoords.length - 1] = point;
      }
      this.editSession.handleMarkers[activeIndex]?.setLatLng([point[1], point[0]]);
      this.updateActiveFeatureGeometry(this.editSession.draftGeometry, { previewOnly: true });
      this.onGeometryChange(this.selectedFeatureId, this.editSession.draftGeometry);
    };

    this.editSession.upHandler = () => {
      if (!this.editSession) return;
      this.editSession.activeHandleIndex = null;
      this.map.dragging.enable();
    };

    this.map.on('mousemove', this.editSession.moveHandler);
    this.map.on('mouseup', this.editSession.upHandler);

    const coords = this.editSession.draftGeometry.coordinates[0];
    const statusText = document.getElementById('geometry-edit-status');
    if (statusText) {
      statusText.innerHTML = '<span style="color: var(--accent-amber)">● Editing vertices… Save or Cancel to finish.</span>';
    }

    coords.forEach((coord, idx) => {
      if (idx === coords.length - 1 && idx > 0) return;

      const marker = L.circleMarker([coord[1], coord[0]], {
        radius: 6,
        color: '#ffffff',
        fillColor: '#854628',
        fillOpacity: 1,
        weight: 2
      }).addTo(this.editHandlesGroup);

      marker.on('mousedown', (e) => {
        L.DomEvent.stopPropagation(e);
        if (!this.editSession?.active) return;
        this.editSession.activeHandleIndex = idx;
        this.map.dragging.disable();
        if (e && e.originalEvent) {
          e.originalEvent.preventDefault();
        }
      });
      this.editSession.handleMarkers[idx] = marker;
    });
  }

  cancelVertexEditing(message = 'Vertex edits cancelled and the original geometry was restored.') {
    if (!this.editSession?.active) return;
    const original = JSON.parse(JSON.stringify(this.editSession.originalGeometry));
    this.selectedFeature.geometry = original;
    this.updateActiveFeatureGeometry(original, { previewOnly: false });
    this.clearVertexHandles();
    const statusText = document.getElementById('geometry-edit-status');
    if (statusText) {
      statusText.innerHTML = `<span style="color: var(--accent-amber)">${message}</span>`;
    }
  }

  finishVertexEditing() {
    if (!this.editSession?.active) return;
    this.clearVertexHandles();
  }

  clearVertexHandles() {
    if (this.editSession?.activeHandleIndex !== null && this.editSession?.activeHandleIndex !== undefined) {
      this.map.dragging.enable();
    }
    this.isEditingVertices = false;
    if (this.editSession?.moveHandler) this.map.off('mousemove', this.editSession.moveHandler);
    if (this.editSession?.upHandler) this.map.off('mouseup', this.editSession.upHandler);
    this.editSession = null;
    this.editHandlesGroup.clearLayers();
  }

  updateActiveFeatureGeometry(newGeometry, options = { previewOnly: false }) {
    if (!this.selectedFeature) return;
    this.selectedFeature.geometry = newGeometry;
    const targetLayer = this.layers[this.selectedLayer];
    if (targetLayer) {
      targetLayer.eachLayer((l) => {
        if (l.feature && l.feature.id === this.selectedFeatureId) {
          if (l.setLatLngs) {
            const latlngs = newGeometry.coordinates[0].map(c => [c[1], c[0]]);
            l.setLatLngs([latlngs]);
          }
        }
      });
    }
    if (!options.previewOnly) {
      this.refreshLayerStyles();
    }
  }

  // Draw draft tool
  startDrawingDraft() {
    this.clearSelection();
    this.isDrawing = true;
    this.draftPoints = [];
    this.draftMarkersGroup.clearLayers();
    if (this.draftPreviewLine) {
      this.map.removeLayer(this.draftPreviewLine);
      this.draftPreviewLine = null;
    }
  }

  handleMapClick(e) {
    if (this.isEditingVertices || this.editSession?.active) {
      return;
    }

    if (this.isDrawing) {
      const latlng = e.latlng;
      this.draftPoints.push([latlng.lng, latlng.lat]);

      // Add vertex marker
      L.circleMarker(latlng, {
        radius: 5,
        color: '#10b981',
        fillColor: '#ffffff',
        fillOpacity: 1
      }).addTo(this.draftMarkersGroup);

      // Update preview line
      const latlngs = this.draftPoints.map(p => [p[1], p[0]]);
      if (!this.draftPreviewLine) {
        this.draftPreviewLine = L.polyline(latlngs, { color: '#10b981', dashArray: '4,4' }).addTo(this.map);
      } else {
        this.draftPreviewLine.setLatLngs(latlngs);
      }
      return;
    }

    // Normal click outside deselects
    this.clearSelection();
  }

  finishDrawingDraft() {
    if (!this.isDrawing || this.draftPoints.length < 3) {
      this.cancelDrawingDraft();
      return;
    }

    // Close polygon ring
    const closed = [...this.draftPoints, this.draftPoints[0]];
    const geometry = {
      type: 'Polygon',
      coordinates: [closed]
    };

    this.cancelDrawingDraft();
    this.onDraftCreated(geometry);
  }

  cancelDrawingDraft() {
    this.isDrawing = false;
    this.draftPoints = [];
    this.draftMarkersGroup.clearLayers();
    if (this.draftPreviewLine) {
      this.map.removeLayer(this.draftPreviewLine);
      this.draftPreviewLine = null;
    }
  }

  focusOnWarning(warning) {
    this.layers.warnings.clearLayers();

    if (warning.affected_coordinates && warning.affected_coordinates.length === 2) {
      const [lon, lat] = warning.affected_coordinates;
      this.map.flyTo([lat, lon], 19, { duration: 1 });

      const circle = L.circleMarker([lat, lon], {
        radius: 14,
        color: '#f43f5e',
        weight: 3,
        fillColor: '#f43f5e',
        fillOpacity: 0.4
      }).addTo(this.layers.warnings);

      circle.bindPopup(`<strong>Topology Warning (${warning.warning_type})</strong><br>${warning.explanation}`).openPopup();
    } else if (warning.feature_ids && warning.feature_ids.length > 0) {
      // Find feature on map
      const fid = warning.feature_ids[0];
      for (const [name, layer] of Object.entries(this.layers)) {
        if (layer.eachLayer) {
          layer.eachLayer((l) => {
            if (l.feature && (l.feature.id === fid || l.feature.properties?.feature_id === fid)) {
              this.map.flyToBounds(l.getBounds(), { maxZoom: 19, duration: 1 });
              this.selectFeature(fid, name, l.feature, l);
            }
          });
        }
      }
    }
  }
}
