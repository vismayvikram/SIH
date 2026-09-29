/**
 * SIH26012 Platform - AI Building Model & Benchmark Modals Controller
 */
import { ApiClient } from './api.js';

export class ModelModalController {
  constructor(options = {}) {
    this.onInferenceSuccess = options.onInferenceSuccess || (() => {});
    this.modelModal = document.getElementById('modal-ai-model');
    this.benchmarkModal = document.getElementById('modal-benchmarks');
    this.discrepancyModal = document.getElementById('modal-discrepancy');

    this.bindButtons();
  }

  bindButtons() {
    // Open inference modal button
    const btnOpenModel = document.getElementById('btn-open-model-modal');
    if (btnOpenModel) {
      btnOpenModel.onclick = () => this.openModelModal();
    }

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
      btnRunInference.onclick = () => this.executeInference();
    }

    const btnSaved = document.getElementById('btn-select-saved-whu');
    if (btnSaved) btnSaved.onclick = () => this.selectSavedPrediction();
    const btnClear = document.getElementById('btn-clear-prediction-source');
    if (btnClear) btnClear.onclick = () => this.clearPredictionSource();
  }

  openModelModal() {
    if (this.modelModal) this.modelModal.classList.add('active');
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
    if (!this.discrepancyModal) return;
    this.discrepancyModal.classList.add('active');
    const contentBox = document.getElementById('discrepancy-content-area');
    if (!contentBox) return;

    contentBox.innerHTML = '<p>Evaluating AI vs reference spatial discrepancies...</p>';
    try {
      const data = await ApiClient.getDiscrepancies(iouThreshold);
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
    if (this.modelModal) this.modelModal.classList.remove('active');
    if (this.benchmarkModal) this.benchmarkModal.classList.remove('active');
    if (this.discrepancyModal) this.discrepancyModal.classList.remove('active');
  }

  async executeInference() {
    const modelSelect = document.getElementById('model-select');
    const confSlider = document.getElementById('model-conf-slider');
    const simulateFailureCheck = document.getElementById('model-simulate-failure');
    const statusBox = document.getElementById('model-inference-status');
    const btnRun = document.getElementById('btn-run-inference');

    const providerId = modelSelect ? modelSelect.value : 'whu';
    const mode = providerId === 'deeplab' ? 'deeplab' : 'live';
    const modelName = providerId === 'deeplab'
      ? 'aatifjiwani/rgb-footprint-extract'
      : 'giswqs/whu-building-unetplusplus-efficientnet-b4';
    const confThreshold = confSlider ? parseFloat(confSlider.value) : 0.5;
    const simulateFailure = simulateFailureCheck ? simulateFailureCheck.checked : false;

    btnRun.disabled = true;
    statusBox.innerHTML = `
      <div style="display: flex; align-items: center; gap: 8px; color: var(--accent-cyan);">
        <span>⚡ Executing inference (${mode.toUpperCase()} mode: ${modelName})...</span>
      </div>
    `;

    try {
      const result = await ApiClient.runModelInference(mode, modelName, confThreshold, simulateFailure);
      statusBox.innerHTML = `
        <div style="color: var(--accent-emerald); font-weight: 500;">
          ✓ Inference Success (${mode.toUpperCase()} mode)! Extracted ${result.detected_count} building footprints.
          <div style="font-size: 11px; color: var(--text-muted); margin-top: 4px;">Features loaded into active review layer with source="ai_building_model".</div>
        </div>
      `;
      btnRun.disabled = false;
      this.onInferenceSuccess(result);
    } catch (err) {
      statusBox.innerHTML = `
        <div style="color: var(--accent-rose); font-weight: 500;">
          ✕ Inference Failed (Handled Gracefully):
          <div style="font-size: 11px; margin-top: 4px;">${err.message}</div>
        </div>
      `;
      btnRun.disabled = false;
    }
  }
}
