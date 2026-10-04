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

  async getProjects() {
    const res = await fetch(`${API_BASE}/projects`);
    if (!res.ok) throw new Error('Failed to load project registry');
    return res.json();
  },

  async getProject(projectId) {
    const res = await fetch(`${API_BASE}/projects/${encodeURIComponent(projectId)}`);
    if (!res.ok) throw new Error(`Failed to load project: ${projectId}`);
    return res.json();
  },

  async getBuildingModelProviders() {
    const res = await fetch(`${API_BASE}/models/providers`);
    if (!res.ok) {
      const error = await res.json().catch(() => ({ detail: 'Failed to load building-model providers.' }));
      throw new Error(error.detail || 'Failed to load building-model providers.');
    }
    return res.json();
  },

  async runProjectBuildingModel(projectId, options) {
    const res = await fetch(`${API_BASE}/projects/${encodeURIComponent(projectId)}/process/buildings`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(options),
    });
    const result = await res.json().catch(() => ({ detail: 'Project model request returned invalid JSON.' }));
    if (!res.ok) throw new Error(result.detail || `Project model request failed (${res.status}).`);
    return { ...result, http_status: res.status };
  },

  async getProjectJob(jobId) {
    const res = await fetch(`${API_BASE}/jobs/${encodeURIComponent(jobId)}`);
    if (!res.ok) {
      const error = await res.json().catch(() => ({ detail: `Failed to read model job ${jobId}.` }));
      throw new Error(error.detail || `Failed to read model job ${jobId}.`);
    }
    return res.json();
  },

  async getProjectModelRuns(projectId, signal) {
    const res = await fetch(`${API_BASE}/projects/${encodeURIComponent(projectId)}/model-runs`, { signal });
    if (!res.ok) {
      const error = await res.json().catch(() => ({ detail: 'Failed to load saved model runs.' }));
      throw new Error(error.detail || 'Failed to load saved model runs.');
    }
    return res.json();
  },

  async uploadProjectRaster(file, name, locality) {
    const form = new FormData();
    form.append('file', file);
    form.append('name', name);
    form.append('locality', locality);
    const res = await fetch(`${API_BASE}/projects`, {
      method: 'POST',
      body: form
    });
    if (!res.ok) {
      const error = await res.json();
      throw new Error(error.detail || 'Project upload failed');
    }
    return res.json();
  },

  async getHealth() {
    const res = await fetch(`${API_BASE}/health`);
    if (!res.ok) throw new Error('Failed to check API health');
    return res.json();
  },

  async getLayer(layerName) {
    const res = await fetch(`${API_BASE}/layers/${layerName}`);
    if (!res.ok) {
      let detail = `Failed to load layer: ${layerName}`;
      try {
        const error = await res.json();
        detail = error.detail || detail;
      } catch (_) {
        // Keep the layer-specific error when the response is not JSON.
      }
      throw new Error(detail);
    }
    return res.json();
  },

  async getProjectLayers(projectId) {
    const res = await fetch(`${API_BASE}/projects/${encodeURIComponent(projectId)}/layers`);
    if (!res.ok) {
      const error = await res.json().catch(() => ({ detail: `Failed to load project layers for ${projectId}` }));
      throw new Error(error.detail || `Failed to load project layers for ${projectId}`);
    }
    return res.json();
  },

  async getProjectManifest(projectId) {
    const res = await fetch(`${API_BASE}/projects/${encodeURIComponent(projectId)}/layers/manifest`);
    if (!res.ok) {
      const error = await res.json().catch(() => ({ detail: `Failed to load project manifest for ${projectId}` }));
      throw new Error(error.detail || `Failed to load project manifest for ${projectId}`);
    }
    return res.json();
  },

  async getProjectWarnings(projectId) {
    const res = await fetch(`${API_BASE}/projects/${encodeURIComponent(projectId)}/warnings`);
    if (!res.ok) {
      const error = await res.json().catch(() => ({ detail: `Failed to load warnings for ${projectId}` }));
      throw new Error(error.detail || `Failed to load warnings for ${projectId}`);
    }
    return res.json();
  },

  async getWarnings() {
    const res = await fetch(`${API_BASE}/warnings`);
    if (!res.ok) throw new Error('Failed to fetch topology warnings');
    return res.json();
  },

  async narrateWarnings(warningIds) {
    const res = await fetch(`${API_BASE}/warnings/narrate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ warning_ids: warningIds })
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Warning phrasing request failed');
    }
    return res.json();
  },

  async getGreeneryResults() {
    const res = await fetch(`${API_BASE}/greenery/results`);
    if (!res.ok) throw new Error('Failed to load saved greenery candidates');
    return res.json();
  },

  async detectGreenery(options = {}) {
    const res = await fetch(`${API_BASE}/greenery/detect`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(options)
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'RGB greenery detection failed');
    }
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

  async getProjectFeatureDetails(projectId, featureId) {
    const res = await fetch(`${API_BASE}/projects/${encodeURIComponent(projectId)}/features/${encodeURIComponent(featureId)}`);
    if (!res.ok) {
      const error = await res.json().catch(() => ({ detail: `Failed to fetch project feature ${featureId}.` }));
      throw new Error(error.detail || `Failed to fetch project feature ${featureId}.`);
    }
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

  async updateProjectFeatureStatus(projectId, featureId, reviewStatus, notes, reviewerLabel = 'demo-reviewer') {
    const res = await fetch(`${API_BASE}/projects/${encodeURIComponent(projectId)}/features/${encodeURIComponent(featureId)}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ review_status: reviewStatus, notes, reviewer_label: reviewerLabel }),
    });
    if (!res.ok) {
      const error = await res.json().catch(() => ({ detail: 'Failed to update project feature status.' }));
      throw new Error(error.detail || 'Failed to update project feature status.');
    }
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

  async saveProjectGeometryEdit(projectId, featureId, newGeometry, editReason, reviewerLabel = 'demo-reviewer') {
    const res = await fetch(`${API_BASE}/projects/${encodeURIComponent(projectId)}/features/${encodeURIComponent(featureId)}/edit-geometry`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ geometry: newGeometry, edit_reason: editReason, reviewer_label: reviewerLabel }),
    });
    if (!res.ok) {
      const error = await res.json().catch(() => ({ detail: 'Failed to save project feature geometry.' }));
      throw new Error(error.detail || 'Failed to save project feature geometry.');
    }
    return res.json();
  },

  async revertGeometry(featureId) {
    const res = await fetch(`${API_BASE}/features/${encodeURIComponent(featureId)}/revert`, {
      method: 'POST'
    });
    if (!res.ok) throw new Error('Failed to revert feature geometry');
    return res.json();
  },

  async revertProjectGeometry(projectId, featureId) {
    const res = await fetch(`${API_BASE}/projects/${encodeURIComponent(projectId)}/features/${encodeURIComponent(featureId)}/revert`, { method: 'POST' });
    if (!res.ok) {
      const error = await res.json().catch(() => ({ detail: 'Failed to revert project feature geometry.' }));
      throw new Error(error.detail || 'Failed to revert project feature geometry.');
    }
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

  async runModelInference(mode = 'live', modelName = 'giswqs/whu-building-unetplusplus-efficientnet-b4', confidenceThreshold = 0.5, simulateFailure = false, morphologyOpeningPx = 0, morphologyClosingPx = 0) {
    const res = await fetch(`${API_BASE}/models/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        mode,
        model_name: modelName,
        model_version: '09df9efd323bbd3d56b98b4857129eb9b5baa2d3',
        confidence_threshold: confidenceThreshold,
        morphology_opening_px: morphologyOpeningPx,
        morphology_closing_px: morphologyClosingPx,
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

  async getDiscrepancies(iouThreshold = 0.35, projectId = null) {
    const url = new URL(`${API_BASE}/models/discrepancy`);
    url.searchParams.set('iou_threshold', String(iouThreshold));
    if (projectId) url.searchParams.set('project_id', projectId);
    const res = await fetch(url);
    if (!res.ok) throw new Error('Failed to fetch model vs reference discrepancies');
    return res.json();
  },

  async getPredictionSource(projectId = null) {
    const url = projectId ? `${API_BASE}/projects/${encodeURIComponent(projectId)}/models/prediction-source` : `${API_BASE}/models/prediction-source`;
    const res = await fetch(url);
    if (!res.ok) throw new Error('Failed to fetch active prediction source');
    return res.json();
  },

  async selectPrecomputedPrediction(projectId = null) {
    const url = projectId ? `${API_BASE}/projects/${encodeURIComponent(projectId)}/models/select-precomputed` : `${API_BASE}/models/select-precomputed`;
    const res = await fetch(url, { method: 'POST' });
    if (!res.ok) throw new Error('Failed to select saved WHU prediction');
    return res.json();
  },

  async clearPredictionSource(projectId = null) {
    const url = projectId ? `${API_BASE}/projects/${encodeURIComponent(projectId)}/models/prediction-source` : `${API_BASE}/models/prediction-source`;
    const res = await fetch(url, { method: 'DELETE' });
    if (!res.ok) throw new Error('Failed to clear active prediction source');
    return res.json();
  },

  async getParcelRag() {
    const res = await fetch(`${API_BASE}/parcels/rag`);
    if (!res.ok) throw new Error('Failed to fetch parcel RAG status');
    return res.json();
  }
};
