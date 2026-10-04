/*
 * SIH26012 Platform - Topology Warning Center Controller
 * Manages bottom collapsible tray, severity badges, filtering, click-to-zoom,
 * and deterministic, payload-grounded warning summaries.
 */
import { ApiClient } from './api.js?v=sih-20261005-02';

export class WarningCenter {
  constructor(containerId, options = {}) {
    this.container = document.getElementById(containerId);
    this.onWarningClick = options.onWarningClick || (() => {});
    this.allWarnings = [];
    this.currentFilter = 'all';
    this.isExpanded = false;
    this.narratives = {};
    this.narrationNotice = '';
    this.isNarrating = false;
  }

  escapeHtml(value) {
    return String(value ?? '').replace(/[&<>"']/g, (character) => ({
      '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
    })[character]);
  }

  renderEvidence(warning) {
    const evidence = warning.evidence || {};
    const tolerance = warning.tolerance || {};
    const exceedance = warning.exceeds_tolerance_by || {};
    const evidenceRows = Object.entries(evidence).map(([name, value]) => `
      <div><span>${this.escapeHtml(name.replace(/_/g, ' '))}</span><strong>${this.escapeHtml(typeof value === 'number' ? value.toFixed(3) : value)}</strong></div>
    `).join('');
    const toleranceText = tolerance.name
      ? `${tolerance.name.replace(/_/g, ' ')}: ${tolerance.value}${tolerance.unit ? ` ${tolerance.unit}` : ''}`
      : '';
    const exceedanceText = Object.entries(exceedance)
      .filter(([name]) => name !== 'ratio')
      .map(([name, value]) => `${name.replace(/_/g, ' ')}: ${Number(value).toFixed(3)}`)
      .join('; ');
    const action = warning.suggested_action || {};
    if (!evidenceRows && !toleranceText && !action.text) return '';
    return `
      <div class="warning-evidence">
        ${evidenceRows}
        ${toleranceText ? `<div><span>Review tolerance</span><strong>${this.escapeHtml(toleranceText)}</strong></div>` : ''}
        ${exceedanceText ? `<div><span>Beyond tolerance</span><strong>${this.escapeHtml(exceedanceText)}</strong></div>` : ''}
        ${action.text ? `<div class="warning-action"><span class="status-badge">${this.escapeHtml(action.action_type || 'review')}</span><strong>${this.escapeHtml(action.text)}</strong></div>` : ''}
      </div>`;
  }

  setWarnings(warnings) {
    this.allWarnings = warnings || [];
    this.render();
  }

  toggleExpand() {
    this.isExpanded = !this.isExpanded;
    this.render();
  }

  getFilteredWarnings() {
    if (this.currentFilter === 'synthetic') {
      return this.allWarnings.filter(w => w.source && w.source.includes('synthetic'));
    }
    if (this.currentFilter === 'real') {
      return this.allWarnings.filter(w => !w.source || !w.source.includes('synthetic'));
    }
    return this.allWarnings;
  }

  async narrateVisibleWarnings() {
    const warnings = this.getFilteredWarnings();
    if (!warnings.length || this.isNarrating) return;
    this.isNarrating = true;
    this.narrationNotice = 'Preparing deterministic summaries from warning evidence…';
    this.render();
    try {
      let failedBatches = 0;
      for (let offset = 0; offset < warnings.length; offset += 20) {
        const batch = warnings.slice(offset, offset + 20);
        try {
          const result = await ApiClient.narrateWarnings(batch.map(w => w.warning_id));
          for (const warning of batch) {
            const item = result.narratives?.[warning.warning_id];
            if (item?.text) {
              this.narratives[warning.warning_id] = {
                text: item.text,
                source: item.source || 'template',
                template: warning.explanation || ''
              };
            }
          }
        } catch (error) {
          failedBatches += 1;
        }
      }
      this.narrationNotice = failedBatches
        ? `Some summaries were unavailable; measured warning fields remain visible. (${failedBatches} batch(es) unavailable.)`
        : 'Summaries are generated deterministically from warning evidence and suggested actions.';
    } finally {
      this.isNarrating = false;
      this.isExpanded = true;
      this.render();
    }
  }

  render() {
    if (!this.container) return;
    const total = this.allWarnings.length;
    const synthCount = this.allWarnings.filter(w => w.source && w.source.includes('synthetic')).length;
    const realCount = total - synthCount;
    const filtered = this.getFilteredWarnings();
    const hasSyntheticData = synthCount > 0;
    const chips = hasSyntheticData ? `
      <div style="display: flex; gap: 4px; margin-right: 8px;">
        <button class="btn btn-secondary ${this.currentFilter === 'all' ? 'active' : ''}" id="filter-warn-all" style="padding: 2px 8px; font-size: 10px;">All (${total})</button>
        <button class="btn btn-secondary ${this.currentFilter === 'synthetic' ? 'active' : ''}" id="filter-warn-synth" style="padding: 2px 8px; font-size: 10px;">Synthetic (${synthCount})</button>
        <button class="btn btn-secondary ${this.currentFilter === 'real' ? 'active' : ''}" id="filter-warn-real" style="padding: 2px 8px; font-size: 10px;">Real (${realCount})</button>
      </div>
    ` : '';

    this.container.innerHTML = `
      <div class="warning-tray-header" id="warning-header-toggle">
        <div class="warning-tray-title">
          <span>Topology Warning Center</span>
          <span class="warning-counter-badge">${total} Active</span>
        </div>
        <div style="display: flex; align-items: center; gap: 8px;">
          ${chips}
          <button class="btn btn-secondary" id="btn-narrate-warnings" type="button" ${filtered.length && !this.isNarrating ? '' : 'disabled'} title="Generate deterministic summaries from warning evidence">${this.isNarrating ? 'Working…' : 'Summarize'}</button>
          <span style="font-size: 13px; color: var(--text-muted);">${this.isExpanded ? '▼ Hide' : '▲ Expand'}</span>
        </div>
      </div>

      ${this.isExpanded ? `
        <div class="warning-tray-body">
          ${this.narrationNotice ? `<div class="warning-narration-notice" role="status" style="padding: 8px 10px; margin-bottom: 8px; border: 1px solid var(--border-subtle); border-radius: 8px; color: var(--text-muted); font-size: 11px;">${this.escapeHtml(this.narrationNotice)}</div>` : ''}
          ${total === 0 ? '<div style="padding: 12px; color: var(--text-faint); text-align: center;">No layers to check yet.</div>' : filtered.length > 0 ? filtered.map(w => {
            const stored = this.narratives[w.warning_id];
            const currentNarrative = stored && stored.template === (w.explanation || '') ? stored : null;
            const displayedText = currentNarrative?.text || w.explanation || '';
            return `
              <div class="warning-card severity-${this.escapeHtml(w.severity)}" data-wid="${this.escapeHtml(w.warning_id)}">
                <div class="warning-info">
                  <div class="warning-title-line">
                    <span style="text-transform: capitalize;">${this.escapeHtml((w.warning_type || '').replace(/_/g, ' '))}</span>
                    <span class="status-badge">${this.escapeHtml(w.severity)}</span>
                    ${w.source && w.source.includes('synthetic') ? '<span class="badge-synth">DEMO FIXTURE</span>' : ''}
                  </div>
                  <div class="warning-desc">${this.escapeHtml(displayedText)}</div>
                  ${this.renderEvidence(w)}
                  <div style="font-size: 10px; color: var(--text-faint); margin-top: 2px;">
                    Affected features: <strong>${this.escapeHtml(w.feature_ids ? w.feature_ids.join(', ') : 'None')}</strong> | Rule: <em>${this.escapeHtml(w.plain_language_rule || '')}</em>
                  </div>
                </div>
                <button class="btn btn-primary btn-icon btn-zoom-warning" data-wid="${this.escapeHtml(w.warning_id)}">Zoom to Conflict</button>
              </div>`;
          }).join('') : '<div style="padding: 12px; color: var(--text-faint); text-align: center;">No warnings match this filter.</div>'}
        </div>
      ` : ''}
    `;

    const toggleHeader = document.getElementById('warning-header-toggle');
    if (toggleHeader) {
      toggleHeader.onclick = (event) => {
        if (event.target.closest('button')) return;
        this.toggleExpand();
      };
    }

    const btnAll = document.getElementById('filter-warn-all');
    const btnSynth = document.getElementById('filter-warn-synth');
    const btnReal = document.getElementById('filter-warn-real');
    const btnNarrate = document.getElementById('btn-narrate-warnings');

    if (btnAll) btnAll.onclick = (event) => { event.stopPropagation(); this.currentFilter = 'all'; this.isExpanded = true; this.render(); };
    if (btnSynth) btnSynth.onclick = (event) => { event.stopPropagation(); this.currentFilter = 'synthetic'; this.isExpanded = true; this.render(); };
    if (btnReal) btnReal.onclick = (event) => { event.stopPropagation(); this.currentFilter = 'real'; this.isExpanded = true; this.render(); };
    if (btnNarrate) btnNarrate.onclick = (event) => { event.stopPropagation(); this.narrateVisibleWarnings(); };

    this.container.querySelectorAll('.btn-zoom-warning').forEach(button => {
      button.onclick = (event) => {
        event.stopPropagation();
        const warningId = button.getAttribute('data-wid');
        const warning = this.allWarnings.find(item => item.warning_id === warningId);
        if (warning) this.onWarningClick(warning);
      };
    });
  }
}
