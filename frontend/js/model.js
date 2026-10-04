/**
 * SIH26012 Platform - AI Building Model & Benchmark Modals Controller
 */
import { ApiClient } from './api.js?v=sih-20261005-02';

export class ModelModalController {
  constructor(options = {}) {
    this.onInferenceSuccess = options.onInferenceSuccess || (() => {});
    this.getProjectContext = options.getProjectContext || (() => ({}));
    this.onProjectModelComplete = options.onProjectModelComplete || (async () => {});
    this.onTrace = options.onTrace || (() => {});
    this.onDrawArea = options.onDrawArea || (() => {});
    this.onToast = options.onToast || (() => {});
    this.modelModal = document.getElementById('modal-ai-model');
    this.benchmarkModal = document.getElementById('modal-benchmarks');
    this.discrepancyModal = document.getElementById('modal-discrepancy');
    this.providers = [];
    this.selectedProvider = null;
    this.areaMode = 'whole';
    this.area = null;
    this.lastSecondsPerTile = {};
    this.isRunning = false;
    this.providersLoaded = false;

    this.bindButtons();
  }

  bindButtons() {
    ['btn-open-model-modal', 'btn-open-model-modal-toolbar', 'btn-model-sidebar-cta'].forEach((id) => {
      const button = document.getElementById(id);
      if (button) button.onclick = async () => {
        this.onTrace('button clicked', id);
        await this.openModelModal();
      };
    });

    // Open benchmarks modal button
    const btnOpenBenchmarks = document.getElementById('btn-open-benchmark-modal');
    if (btnOpenBenchmarks) {
      btnOpenBenchmarks.onclick = () => this.openBenchmarkModal();
    }

    // Open discrepancies modal button
    const btnOpenDiscrepancy = document.getElementById('btn-open-discrepancy-modal');
    if (btnOpenDiscrepancy) {
      btnOpenDiscrepancy.onclick = () => this.openDiscrepancyModal();
    }

    // Close buttons
    document.querySelectorAll('.modal-close-trigger').forEach(btn => {
      btn.onclick = () => this.closeModals();
    });

    // Run inference button
    const btnRunInference = document.getElementById('btn-run-inference');
    if (btnRunInference) {
      btnRunInference.onclick = async () => { await this.executeInference(); };
    }

    document.getElementById('btn-close-ai-model')?.addEventListener('click', () => this.closeModelModal());
    document.getElementById('btn-cancel-ai-model')?.addEventListener('click', () => this.closeModelModal());
    document.getElementById('btn-model-draw-area')?.addEventListener('click', () => this.beginAreaDrawing());
    document.getElementById('model-provider-cards')?.addEventListener('click', (event) => {
      const card = event.target.closest('[data-provider-id]');
      if (card && !card.disabled) this.selectProvider(card.dataset.providerId);
    });
    document.getElementById('model-conf-slider')?.addEventListener('input', () => this.updateThresholdLabel());
    document.querySelectorAll('input[name="model-area-mode"]').forEach((radio) => {
      radio.addEventListener('change', () => {
        this.areaMode = radio.value;
        this.updateAreaEstimate();
      });
    });
    document.getElementById('model-resolution-m')?.addEventListener('input', () => this.updateAreaEstimate());

    const btnSaved = document.getElementById('btn-select-saved-whu');
    if (btnSaved) btnSaved.onclick = () => this.selectSavedPrediction();
    const btnClear = document.getElementById('btn-clear-prediction-source');
    if (btnClear) btnClear.onclick = () => this.clearPredictionSource();
  }

  openModelModal() {
    return this.showModelModal();
  }

  async showModelModal() {
    if (!this.modelModal) return;
    if (!this.modelModal.open) this.modelModal.showModal();
    this.onTrace('modal opened', this.getProjectContext().projectId || 'no active project');
    if (!this.providersLoaded) await this.loadProviders();
    else this.updateAreaEstimate();
  }

  closeModelModal() {
    if (this.modelModal?.open) this.modelModal.close();
  }

  async loadProviders() {
    const cards = document.getElementById('model-provider-cards');
    const status = document.getElementById('model-inference-status');
    const runButton = document.getElementById('btn-run-inference');
    if (cards) cards.textContent = 'Checking local model providers...';
    if (runButton) runButton.disabled = true;
    try {
      const result = await ApiClient.getBuildingModelProviders();
      this.providers = result.providers || [];
      this.providersLoaded = true;
      this.renderProviders();
      const selected = this.providers.find((provider) => provider.id === this.selectedProvider && provider.enabled);
      const firstEnabled = this.providers.find((provider) => provider.enabled);
      if (selected) this.renderProviders();
      else if (firstEnabled) this.selectProvider(firstEnabled.id);
      else {
        this.selectedProvider = null;
        this.renderProviders();
        if (status) status.textContent = 'No model provider is ready. Each disabled provider lists its exact local requirement.';
      }
    } catch (error) {
      this.providers = [];
      this.selectedProvider = null;
      if (cards) cards.textContent = `Model provider service unavailable: ${error.message}`;
      if (status) status.textContent = `Run is disabled: ${error.message}`;
      if (runButton) runButton.disabled = true;
    }
  }

  escapeHtml(value) {
    return String(value ?? '').replace(/[&<>"']/g, (character) => ({
      '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
    })[character]);
  }

  renderProviders() {
    const cards = document.getElementById('model-provider-cards');
    if (!cards) return;
    cards.innerHTML = this.providers.map((provider) => `
      <button class="model-provider-card ${provider.id === this.selectedProvider ? 'selected' : ''}"
        type="button" data-provider-id="${this.escapeHtml(provider.id)}" role="radio"
        aria-checked="${provider.id === this.selectedProvider}" ${provider.enabled ? '' : 'disabled'}>
        <span class="model-provider-card-heading">
          <strong>${this.escapeHtml(provider.label)}</strong>
          <span>${provider.enabled ? 'Ready' : 'Unavailable'}</span>
        </span>
        <span>${this.escapeHtml(provider.description)}</span>
        <span><strong>Imagery:</strong> ${this.escapeHtml(provider.resolution_label)}</span>
        <span class="model-provider-caveat">${this.escapeHtml(provider.caveat)}</span>
        ${provider.reason ? `<span class="model-provider-reason">${this.escapeHtml(provider.reason)}</span>` : ''}
      </button>
    `).join('');
  }

  selectProvider(providerId) {
    const provider = this.providers.find((entry) => entry.id === providerId && entry.enabled);
    if (!provider) return;
    this.selectedProvider = providerId;
    this.onTrace('provider chosen', providerId);
    const threshold = document.getElementById('model-conf-slider');
    const resolution = document.getElementById('model-resolution-m');
    const minimumArea = document.getElementById('model-min-area');
    if (threshold) threshold.value = String(provider.default_threshold);
    if (resolution) resolution.value = String(provider.expected_resolution_m);
    if (minimumArea) minimumArea.value = String(provider.default_min_area_m2);
    this.renderProviders();
    this.updateThresholdLabel();
    this.updateAreaEstimate();
    const runButton = document.getElementById('btn-run-inference');
    if (runButton) runButton.disabled = !this.getProjectContext().projectId || this.isRunning;
  }

  updateThresholdLabel() {
    const slider = document.getElementById('model-conf-slider');
    const label = document.getElementById('model-threshold-value');
    if (slider && label) label.value = Number(slider.value).toFixed(2);
    this.updateAreaEstimate();
  }

  updateAreaEstimate() {
    const estimate = document.getElementById('model-area-estimate');
    const provider = this.providers.find((entry) => entry.id === this.selectedProvider);
    const context = this.getProjectContext();
    if (!estimate || !provider || !context.raster) return;
    const raster = context.raster;
    const centerLatitude = (raster.bounds_wgs84?.[0]?.[0] + raster.bounds_wgs84?.[1]?.[0]) / 2;
    const sourceGroundGsd = Number(raster.gsd_x || raster.gsd_y) * Math.cos((centerLatitude * Math.PI) / 180);
    const targetGsd = Number(document.getElementById('model-resolution-m')?.value || provider.expected_resolution_m);
    let widthM = Number(raster.width) * sourceGroundGsd;
    let heightM = Number(raster.height) * sourceGroundGsd;
    if (this.areaMode === 'draw' && this.area) {
      const ring = this.area.coordinates?.[0] || [];
      const longitudes = ring.map((point) => point[0]);
      const latitudes = ring.map((point) => point[1]);
      if (longitudes.length && latitudes.length) {
        const lat = (Math.min(...latitudes) + Math.max(...latitudes)) / 2;
        widthM = (Math.max(...longitudes) - Math.min(...longitudes)) * 111320 * Math.cos((lat * Math.PI) / 180);
        heightM = (Math.max(...latitudes) - Math.min(...latitudes)) * 110574;
      }
    }
    if (!Number.isFinite(targetGsd) || targetGsd <= 0 || !Number.isFinite(sourceGroundGsd)) {
      estimate.textContent = 'Tile count and runtime estimate unavailable for this raster.';
      return;
    }
    const widthPixels = widthM / targetGsd;
    const heightPixels = heightM / targetGsd;
    const countTiles = (pixels) => Math.max(1, Math.ceil(Math.max(0, pixels - 512) / 256) + 1);
    const tileCount = countTiles(widthPixels) * countTiles(heightPixels);
    const measuredSecondsPerTile = this.lastSecondsPerTile[provider.id] ?? provider.runtime_sample?.seconds_per_tile;
    const runtimeBasis = this.lastSecondsPerTile[provider.id]
      ? "this provider's last local run"
      : provider.runtime_sample
        ? `a saved ${provider.runtime_sample.tile_count}-tile run${provider.runtime_sample.device ? ` on ${provider.runtime_sample.device}` : ''}`
        : '';
    const runtime = Number.isFinite(measuredSecondsPerTile)
      ? `Rough runtime: about ${(tileCount * measuredSecondsPerTile / 60).toFixed(1)} min, based on ${runtimeBasis}.`
      : 'Rough runtime: unavailable; no saved local timing for this provider yet.';
    estimate.textContent = `${this.areaMode === 'draw' && this.area ? 'Drawn area' : 'Whole image'} · about ${tileCount} overlapping tiles · ${runtime} Draw area is recommended for large images.`;
  }

  beginAreaDrawing() {
    const context = this.getProjectContext();
    if (!context.projectId) return;
    this.areaMode = 'draw';
    this.area = null;
    this.closeModelModal();
    this.onTrace('area drawing started', context.projectId);
    this.onDrawArea((aoi) => {
      this.area = aoi;
      if (!aoi) this.areaMode = 'whole';
      const wholeRadio = document.querySelector('input[name="model-area-mode"][value="whole"]');
      const drawRadio = document.querySelector('input[name="model-area-mode"][value="draw"]');
      if (wholeRadio) wholeRadio.checked = this.areaMode === 'whole';
      if (drawRadio) drawRadio.checked = this.areaMode === 'draw';
      this.onTrace('area selected', aoi ? 'rectangle selected' : 'drawing cancelled');
      this.updateAreaEstimate();
      this.showModelModal();
    });
  }

  async selectSavedPrediction() {
    try {
      const result = await ApiClient.selectPrecomputedPrediction();
      this.onInferenceSuccess(result);
      const statusBox = document.getElementById('model-inference-status');
      if (statusBox) statusBox.innerHTML = `<div style="color: var(--accent-emerald);">Loaded ${result.detected_count} features from ${result.metadata.source_label} (${result.metadata.run_id}).</div>`;
    } catch (err) {
      const statusBox = document.getElementById('model-inference-status');
      if (statusBox) statusBox.innerHTML = `<div style="color: var(--accent-rose);">${err.message}</div>`;
    }
  }

  async clearPredictionSource() {
    try {
      await ApiClient.clearPredictionSource();
      this.onInferenceSuccess({ detected_count: 0, metadata: { source_label: 'No active prediction set' } });
    } catch (err) {
      const statusBox = document.getElementById('model-inference-status');
      if (statusBox) statusBox.innerHTML = `<div style="color: var(--accent-rose);">${err.message}</div>`;
    }
  }

  async openDiscrepancyModal(iouThreshold = 0.35) {
    const context = this.getProjectContext();
    if (!context.hasReferenceLayer) return;
    if (!this.discrepancyModal) return;
    this.discrepancyModal.classList.add('active');
    const contentBox = document.getElementById('discrepancy-content-area');
    if (!contentBox) return;

    contentBox.innerHTML = '<p>Evaluating AI vs reference spatial discrepancies...</p>';
    try {
      const data = await ApiClient.getDiscrepancies(iouThreshold, context.projectId);
      if (data.status === 'no_predictions') {
        contentBox.innerHTML = `
          <div style="background: rgba(245, 158, 11, 0.12); border: 1px solid rgba(245, 158, 11, 0.3); border-radius: 6px; padding: 12px; color: #fbbf24; margin-bottom: 12px; font-size: 12px;">
            ℹ️ ${data.message}
          </div>
          <div style="font-size: 11px; color: var(--accent-amber); margin-bottom: 8px;">
            ⚠️ Status: <strong>${data.alignment_status}</strong>
          </div>
          <p style="font-size: 11.5px; color: var(--text-muted);">${data.disclaimer}</p>
        `;
        return;
      }

      const m = data.metrics || {};
      contentBox.innerHTML = `
        <div style="background: rgba(245, 158, 11, 0.12); border: 1px solid rgba(245, 158, 11, 0.3); border-radius: 6px; padding: 10px; font-size: 11.5px; color: #fbbf24; margin-bottom: 10px;">
          ⚠️ <strong>Visual QA Alignment Status:</strong> <code>${data.alignment_status}</code><br>
          <span style="font-size: 10.5px; color: var(--text-muted);">Metrics represent spatial agreement with the user-aligned local reference set, not an official or legal accuracy benchmark.</span>
        </div>

        <div style="background: rgba(180, 137, 84, 0.1); border: 1px solid rgba(133, 70, 40, 0.35); border-radius: 6px; padding: 10px; font-size: 11.5px; color: #854628; margin-bottom: 12px; display: flex; justify-content: space-between; align-items: center;">
          <div>⚖️ <strong>Spatial Discrepancy Comparator:</strong> ${data.comparator_nature}</div>
          <div style="display: flex; align-items: center; gap: 6px;">
            <label style="font-size: 11px;">IoU Threshold:</label>
            <select id="disc-iou-select" style="background: var(--bg-dark); color: #fff; border: 1px solid var(--border-subtle); border-radius: 4px; padding: 2px 6px; font-size: 11px;">
              <option value="0.35" ${iouThreshold === 0.35 ? 'selected' : ''}>0.35 (Default)</option>
              <option value="0.50" ${iouThreshold === 0.50 ? 'selected' : ''}>0.50 (Strict)</option>
            </select>
          </div>
        </div>

        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; margin-bottom: 12px;">
          <div style="background: var(--bg-card); padding: 8px; border-radius: 6px; border: 1px solid var(--border-subtle); text-align: center;">
            <div style="font-size: 10px; color: var(--text-faint);">MATCHED (TP)</div>
            <div style="font-size: 18px; font-weight: 700; color: var(--accent-emerald);">${m.matched_pairs_count}</div>
          </div>
          <div style="background: var(--bg-card); padding: 8px; border-radius: 6px; border: 1px solid var(--border-subtle); text-align: center;">
            <div style="font-size: 10px; color: var(--text-faint);">MODEL ONLY (FP)</div>
            <div style="font-size: 18px; font-weight: 700; color: var(--accent-amber);">${m.model_only_detections}</div>
          </div>
          <div style="background: var(--bg-card); padding: 8px; border-radius: 6px; border: 1px solid var(--border-subtle); text-align: center;">
            <div style="font-size: 10px; color: var(--text-faint);">REF ONLY (FN)</div>
            <div style="font-size: 18px; font-weight: 700; color: var(--accent-rose);">${m.reference_only_footprints}</div>
          </div>
          <div style="background: var(--bg-card); padding: 8px; border-radius: 6px; border: 1px solid var(--border-subtle); text-align: center;">
            <div style="font-size: 10px; color: var(--text-faint);">PRECISION / RECALL</div>
            <div style="font-size: 14px; font-weight: 700; color: var(--accent-cyan); margin-top: 4px;">P: ${(m.precision*100).toFixed(1)}% / R: ${(m.recall*100).toFixed(1)}%</div>
          </div>
        </div>

        <div style="font-size: 12px; margin-bottom: 8px; display: flex; justify-content: space-between;">
          <strong>Matched Pairs (Sample):</strong>
          <span style="font-size: 11px; color: var(--text-faint);">Total AI: ${data.total_ai_predictions} | Total Ref: ${data.total_reference_features}</span>
        </div>
        <div style="max-height: 160px; overflow-y: auto; background: rgba(0,0,0,0.3); border-radius: 6px; padding: 6px; font-family: var(--font-mono); font-size: 11px;">
          ${data.matched_pairs && data.matched_pairs.length > 0 ? data.matched_pairs.map(p => `
            <div style="padding: 4px 6px; border-bottom: 1px solid rgba(255,255,255,0.05); display: flex; justify-content: space-between;">
              <span style="color: var(--accent-indigo);">${p.ai_id}</span>
              <span style="color: var(--text-faint);">↔</span>
              <span style="color: var(--accent-cyan);">${p.ref_id}</span>
              <span style="color: var(--accent-emerald);">IoU: ${(p.iou * 100).toFixed(1)}%</span>
            </div>
          `).join('') : '<div style="color: var(--text-faint); padding: 8px;">No matched pairs at current IoU threshold.</div>'}
        </div>

        <div style="margin-top: 10px; display: grid; grid-template-columns: 1fr 1fr; gap: 8px; font-size: 10.5px; color: var(--text-faint);">
          <div><strong>Model-only IDs:</strong><br>${(data.model_only_features || []).join(', ') || 'None'}</div>
          <div><strong>Reference-only IDs:</strong><br>${(data.reference_only_features || []).join(', ') || 'None'}</div>
        </div>

        <div style="margin-top: 10px; font-size: 10.5px; color: var(--text-faint);">
          ${data.disclaimer}
        </div>
      `;

      const sel = document.getElementById('disc-iou-select');
      if (sel) {
        sel.onchange = (e) => this.openDiscrepancyModal(parseFloat(e.target.value));
      }
    } catch (err) {
      contentBox.innerHTML = `<p style="color: var(--accent-rose)">Failed to load discrepancies: ${err.message}</p>`;
    }
  }

  async openBenchmarkModal() {
    if (!this.benchmarkModal) return;
    this.benchmarkModal.classList.add('active');
    const contentBox = document.getElementById('benchmark-content-area');
    if (contentBox) {
      contentBox.innerHTML = '<p>Loading registered benchmark specifications...</p>';
      try {
        const manifest = await ApiClient.getBenchmarks();
        const datasets = manifest.registered_datasets;
        contentBox.innerHTML = `
          <div style="background: rgba(245, 158, 11, 0.12); border: 1px solid rgba(245, 158, 11, 0.3); border-radius: 6px; padding: 10px; font-size: 11.5px; color: #fbbf24; margin-bottom: 12px;">
            ℹ️ ${manifest.disclaimer}
          </div>
          ${Object.entries(datasets).map(([key, d]) => `
            <div style="background: var(--bg-card); border: 1px solid var(--border-subtle); border-radius: 8px; padding: 12px; margin-bottom: 10px;">
              <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                <strong style="color: var(--accent-cyan); font-size: 13px;">${d.name}</strong>
                <span class="status-badge" style="background: rgba(244, 63, 94, 0.2); color: #fb7185;">${d.status}</span>
              </div>
              <div style="font-size: 11.5px; color: var(--text-muted); display: flex; flex-direction: column; gap: 4px;">
                <div>Resolution: <strong>${d.resolution}</strong> | Coverage: <strong>${d.coverage}</strong></div>
                <div>License: <strong>${d.license}</strong></div>
                <div>URL / Reference: <a href="${d.url}" target="_blank" style="color: var(--accent-cyan);">${d.url}</a></div>
                <div style="margin-top: 4px; background: rgba(0,0,0,0.3); padding: 6px; border-radius: 4px; font-family: var(--font-mono); font-size: 10.5px; color: #e2e8f0;">
                  ${d.download_method}
                </div>
                <div style="font-size: 10.5px; color: var(--text-faint); margin-top: 2px;">
                  <em>${d.status_details}</em>
                </div>
              </div>
            </div>
          `).join('')}
        `;
      } catch (err) {
        contentBox.innerHTML = `<p style="color: var(--accent-rose)">Failed to load benchmarks: ${err.message}</p>`;
      }
    }
  }

  closeModals() {
    this.closeModelModal();
    if (this.benchmarkModal) this.benchmarkModal.classList.remove('active');
    if (this.discrepancyModal) this.discrepancyModal.classList.remove('active');
  }

  async executeInference() {
    const confSlider = document.getElementById('model-conf-slider');
    const statusBox = document.getElementById('model-inference-status');
    const btnRun = document.getElementById('btn-run-inference');
    const context = this.getProjectContext();
    const provider = this.providers.find((entry) => entry.id === this.selectedProvider && entry.enabled);
    if (!provider || !context.projectId) {
      if (statusBox) statusBox.textContent = 'Choose an available provider and open a project first.';
      return;
    }
    if (this.areaMode === 'draw' && !this.area) {
      if (statusBox) statusBox.textContent = 'Draw a rectangle on the map or choose Whole image.';
      return;
    }
    const resolutionValue = document.getElementById('model-resolution-m')?.value;
    const minAreaValue = document.getElementById('model-min-area')?.value;
    const request = {
      provider: provider.id,
      threshold: Number(confSlider?.value ?? provider.default_threshold),
      aoi: this.areaMode === 'draw' ? this.area : null,
      resolution_m: resolutionValue ? Number(resolutionValue) : null,
      min_area_m2: minAreaValue ? Number(minAreaValue) : provider.default_min_area_m2,
    };
    this.isRunning = true;
    if (btnRun) {
      btnRun.disabled = true;
      btnRun.textContent = 'Running...';
    }
    if (statusBox) statusBox.textContent = 'Preparing project imagery and model weights...';
    try {
      this.onTrace('request sent', `${provider.id} on ${context.projectId}`);
      const started = await ApiClient.runProjectBuildingModel(context.projectId, request);
      this.onTrace('response status', String(started.http_status));
      let job = await ApiClient.getProjectJob(started.job_id);
      while (job.status === 'queued' || job.status === 'running') {
        const progress = job.progress || {};
        if (statusBox) statusBox.textContent = `Running ${provider.label}: ${progress.done || 0} / ${progress.total || '?'} tiles (${progress.percent || 0}%).`;
        await new Promise((resolve) => setTimeout(resolve, 700));
        job = await ApiClient.getProjectJob(started.job_id);
      }
      if (job.status !== 'completed') throw new Error(job.error_message || `Model job ended with status ${job.status}.`);
      const summary = job.result_summary || {};
      await this.onProjectModelComplete(job, summary);
      this.onTrace('layer added', `${summary.layer_id || 'ai_predictions'} · ${summary.feature_count ?? job.feature_count} features`);
      const featureCount = summary.feature_count ?? job.feature_count ?? 0;
      const tileCount = summary.tile_count;
      const sourceGsd = summary.source_ground_resolution_m;
      const targetGsd = summary.inference_resolution_m;
      if (Number.isFinite(tileCount) && Number.isFinite(job.elapsed_seconds)) {
        this.lastSecondsPerTile[provider.id] = job.elapsed_seconds / Math.max(tileCount, 1);
      }
      if (statusBox) {
        statusBox.textContent = `${featureCount} buildings found. Inference resolution: ${Number.isFinite(targetGsd) ? `${targetGsd.toFixed(2)} m/pixel` : 'provider default'}.${Number.isFinite(sourceGsd) ? ` Source ground resolution: ${sourceGsd.toFixed(3)} m/pixel.` : ''}${summary.warnings?.length ? ` Warning: ${summary.warnings.join(' ')}` : ''}`;
      }
      this.onTrace('complete', `${featureCount} buildings`);
    } catch (err) {
      if (statusBox) statusBox.textContent = `Model run failed: ${err.message}`;
      this.onTrace('failed', err.message);
      this.onToast(`Model run failed: ${err.message}`, 'danger');
    } finally {
      this.isRunning = false;
      if (btnRun) {
        btnRun.disabled = !this.providers.some((entry) => entry.id === this.selectedProvider && entry.enabled);
        btnRun.textContent = 'Run model';
      }
    }
  }
}
