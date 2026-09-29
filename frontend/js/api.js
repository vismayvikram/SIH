/**
 * SIH26012 Platform - API Client Module
 */
const API_BASE = '/api';

export const ApiClient = {
  async getMetadata() {
    const res = await fetch(`${API_BASE}/metadata`);
    if (!res.ok) throw new Error('Failed to load system metadata');
    return res.json();
  },

  async getHealth() {
    const res = await fetch(`${API_BASE}/health`);
    if (!res.ok) throw new Error('Failed to check API health');
    return res.json();
  },

  async getLayer(layerName) {
    const res = await fetch(`${API_BASE}/layers/${layerName}`);
    if (!res.ok) throw new Error(`Failed to load layer: ${layerName}`);
    return res.json();
  },

  async getWarnings() {
    const res = await fetch(`${API_BASE}/warnings`);
    if (!res.ok) throw new Error('Failed to fetch topology warnings');
    return res.json();
  },

  async getScores() {
    const res = await fetch(`${API_BASE}/scores`);
    if (!res.ok) throw new Error('Failed to fetch review scores');
    return res.json();
  },

  async getFeatureDetails(featureId) {
    const res = await fetch(`${API_BASE}/features/${encodeURIComponent(featureId)}`);
    if (!res.ok) throw new Error(`Failed to fetch details for feature: ${featureId}`);
    return res.json();
  },

  async updateFeatureStatus(featureId, reviewStatus, notes, reviewerLabel = 'demo-reviewer') {
    const res = await fetch(`${API_BASE}/features/${encodeURIComponent(featureId)}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ review_status: reviewStatus, notes, reviewer_label: reviewerLabel })
    });
    if (!res.ok) throw new Error('Failed to update feature status');
    return res.json();
  },

  async saveGeometryEdit(featureId, newGeometry, editReason, reviewerLabel = 'demo-reviewer') {
    const res = await fetch(`${API_BASE}/features/${encodeURIComponent(featureId)}/edit-geometry`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ geometry: newGeometry, edit_reason: editReason, reviewer_label: reviewerLabel })
    });
    if (!res.ok) throw new Error('Failed to save geometry edit');
    return res.json();
  },

  async revertGeometry(featureId) {
    const res = await fetch(`${API_BASE}/features/${encodeURIComponent(featureId)}/revert`, {
      method: 'POST'
    });
    if (!res.ok) throw new Error('Failed to revert feature geometry');
    return res.json();
  },

  async createDraft(geometry, featureType = 'draft_polygon', notes = '', reviewerLabel = 'demo-reviewer') {
    const res = await fetch(`${API_BASE}/features/draft`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ geometry, feature_type: featureType, notes, reviewer_label: reviewerLabel })
    });
    if (!res.ok) throw new Error('Failed to save draft feature');
    return res.json();
  },

  async runModelInference(mode = 'live', modelName = 'giswqs/whu-building-unetplusplus-efficientnet-b4', confidenceThreshold = 0.5, simulateFailure = false) {
    const res = await fetch(`${API_BASE}/models/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        mode,
        model_name: modelName,
        model_version: '09df9efd323bbd3d56b98b4857129eb9b5baa2d3',
        confidence_threshold: confidenceThreshold,
        simulate_failure: simulateFailure
      })
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Model inference failed');
    }
    return res.json();
  },

  async runLiveModelJob(confidenceThreshold = 0.50) {
    const res = await fetch(`${API_BASE}/models/run-inference`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ confidence_threshold: confidenceThreshold })
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Failed to start live model job');
    }
    return res.json();
  },

  async getJobStatus() {
    const res = await fetch(`${API_BASE}/models/job-status`);
    if (!res.ok) throw new Error('Failed to fetch job status');
    return res.json();
  },

  async getBenchmarks() {
    const res = await fetch(`${API_BASE}/models/benchmarks`);
    if (!res.ok) throw new Error('Failed to fetch benchmark manifests');
    return res.json();
  },

  async importGeoJSON(geojsonData) {
    const res = await fetch(`${API_BASE}/import`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(geojsonData)
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Failed to import GeoJSON bundle');
    }
    return res.json();
  },

  async getDiscrepancies(iouThreshold = 0.35) {
    const res = await fetch(`${API_BASE}/models/discrepancy?iou_threshold=${iouThreshold}`);
    if (!res.ok) throw new Error('Failed to fetch model vs reference discrepancies');
    return res.json();
  },

  async getPredictionSource() {
    const res = await fetch(`${API_BASE}/models/prediction-source`);
    if (!res.ok) throw new Error('Failed to fetch active prediction source');
    return res.json();
  },

  async selectPrecomputedPrediction() {
    const res = await fetch(`${API_BASE}/models/select-precomputed`, { method: 'POST' });
    if (!res.ok) throw new Error('Failed to select saved WHU prediction');
    return res.json();
  },

  async clearPredictionSource() {
    const res = await fetch(`${API_BASE}/models/prediction-source`, { method: 'DELETE' });
    if (!res.ok) throw new Error('Failed to clear active prediction source');
    return res.json();
  },

  async getParcelRag() {
    const res = await fetch(`${API_BASE}/parcels/rag`);
    if (!res.ok) throw new Error('Failed to fetch parcel RAG status');
    return res.json();
  }
};
