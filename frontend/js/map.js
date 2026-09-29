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
      warnings: L.layerGroup()
    };

    this.selectedFeatureId = null;
    this.selectedLayer = null;
    this.selectedFeature = null;

    // Vertex edit handles
    this.editHandlesGroup = L.layerGroup();
    this.isEditingVertices = false;

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

    // Default to geotiff basemap if available, otherwise fallback to osm
    this.currentBasemap = 'geotiff';
    this.basemaps[this.currentBasemap].addTo(this.map);

    // Add all vector layers to map by default
    for (const [name, layer] of Object.entries(this.layers)) {
      layer.addTo(this.map);
    }
    this.editHandlesGroup.addTo(this.map);
    this.draftMarkersGroup.addTo(this.map);

    // Map click handler (deselect feature or add vertex in draw mode)
    this.map.on('click', (e) => this.handleMapClick(e));
  }

  setBasemap(type) {
    if (this.currentBasemap === type || !this.basemaps[type]) return;
    this.map.removeLayer(this.basemaps[this.currentBasemap]);
    this.basemaps[type].addTo(this.map);
    this.currentBasemap = type;
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

  getDraftStyle() {
    return {
      color: '#34d399',
      weight: 2.5,
      fillColor: '#10b981',
      fillOpacity: 0.35,
      dashArray: '4, 4'
    };
  }

  getAiStyle() {
    return {
      color: '#818cf8',
      weight: 2,
      fillColor: '#6366f1',
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
    this.clearVertexHandles();
    this.isEditingVertices = true;

    const geom = this.selectedFeature.geometry;
    if (geom.type !== 'Polygon') return;

    const coords = geom.coordinates[0]; // Exterior ring
    coords.forEach((coord, idx) => {
      // Don't duplicate closing vertex handle
      if (idx === coords.length - 1 && idx > 0) return;

      const marker = L.circleMarker([coord[1], coord[0]], {
        radius: 6,
        color: '#ffffff',
        fillColor: '#854628',
        fillOpacity: 1,
        weight: 2
      }).addTo(this.editHandlesGroup);

      // Make draggable
      let isDragging = false;
      marker.on('mousedown', (e) => {
        isDragging = true;
        this.map.dragging.disable();
      });

      this.map.on('mousemove', (e) => {
        if (!isDragging) return;
        marker.setLatLng(e.latlng);
        // Update coordinate in array
        coords[idx] = [e.latlng.lng, e.latlng.lat];
        if (idx === 0) coords[coords.length - 1] = [e.latlng.lng, e.latlng.lat];
        // Redraw
        this.updateActiveFeatureGeometry(geom);
      });

      this.map.on('mouseup', () => {
        if (isDragging) {
          isDragging = false;
          this.map.dragging.enable();
          this.onGeometryChange(this.selectedFeatureId, geom);
        }
      });
    });
  }

  clearVertexHandles() {
    this.isEditingVertices = false;
    this.editHandlesGroup.clearLayers();
  }

  updateActiveFeatureGeometry(newGeometry) {
    if (!this.selectedFeature) return;
    this.selectedFeature.geometry = newGeometry;
    // Find leaflet layer and update coordinates
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
