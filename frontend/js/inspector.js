/**
 * SIH26012 Platform - Feature Inspector Controller
 * Renders feature metadata, review status buttons, score gauge, audit trail, and vertex edit controls.
 */
import { ApiClient } from './api.js?v=sih-20261005-02';

export class FeatureInspector {
  constructor(containerId, options = {}) {
    this.container = document.getElementById(containerId);
    this.onStatusChanged = options.onStatusChanged || (() => {});
    this.onGeometrySaved = options.onGeometrySaved || (() => {});
    this.onGeometryReverted = options.onGeometryReverted || (() => {});
    this.onStartVertexEdit = options.onStartVertexEdit || (() => {});
    this.onCancelVertexEdit = options.onCancelVertexEdit || (() => {});
    this.onWarningClick = options.onWarningClick || (() => {});
    this.getProjectId = options.getProjectId || (() => null);

    this.currentFeatureId = null;
    this.currentFeature = null;
    this.currentDetails = null;
    this.pendingGeometry = null;

    this.renderEmpty();
  }

  renderEmpty() {
    this.container.innerHTML = `
      <div class="inspector-empty">
        <div class="inspector-empty-icon">📍</div>
        <p><strong>No Feature Selected</strong></p>
        <p style="font-size: 11px; margin-top: 6px;">Click on any building footprint, road polygon, OSM centerline, or synthetic fixture to review details and edit boundaries.</p>
      </div>
    `;
  }

  renderWarningEvidence(warning) {
    const evidence = Object.entries(warning.evidence || {}).map(([name, value]) =>
      `<div><span>${this.escapeHtml(name.replace(/_/g, ' '))}</span><strong>${this.escapeHtml(typeof value === 'number' ? value.toFixed(3) : value)}</strong></div>`
    ).join('');
    const tolerance = warning.tolerance || {};
    const action = warning.suggested_action || {};
    if (!evidence && !tolerance.name && !action.text) return '';
    return `
      <div class="warning-evidence">
        ${evidence}
        ${tolerance.name ? `<div><span>Review tolerance</span><strong>${this.escapeHtml(tolerance.name.replace(/_/g, ' '))}: ${this.escapeHtml(tolerance.value)} ${this.escapeHtml(tolerance.unit || '')}</strong></div>` : ''}
        ${action.text ? `<div class="warning-action"><span class="status-badge">${this.escapeHtml(action.action_type || 'review')}</span><strong>${this.escapeHtml(action.text)}</strong></div>` : ''}
      </div>`;
  }

  escapeHtml(value) {
    return String(value ?? '').replace(/[&<>"']/g, (character) => ({
      '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
    })[character]);
  }

  async loadFeature(featureId, layerName) {
    if (!featureId) {
      this.currentFeatureId = null;
      this.currentFeature = null;
      this.currentDetails = null;
      this.pendingGeometry = null;
      this.renderEmpty();
      return;
    }

    this.currentFeatureId = featureId;
    this.pendingGeometry = null;
    this.container.innerHTML = `
      <div class="inspector-empty">
        <p>Loading feature details...</p>
      </div>
    `;

    try {
      const projectId = this.getProjectId();
      const details = projectId && projectId !== 'SIH26012_INDIA_CANDIDATE_01_LALPUR'
        ? await ApiClient.getProjectFeatureDetails(projectId, featureId)
        : await ApiClient.getFeatureDetails(featureId);
      this.currentDetails = details;
      this.currentFeature = details.feature;
      this.render();
    } catch (err) {
      this.container.innerHTML = `
        <div class="inspector-empty" style="color: var(--accent-rose);">
          <p>Failed to load feature details.</p>
          <p style="font-size: 11px;">${err.message}</p>
        </div>
      `;
    }
  }

  notifyGeometryEdited(newGeometry) {
    this.pendingGeometry = newGeometry;
    const saveBtn = document.getElementById('btn-save-geometry');
    const statusText = document.getElementById('geometry-edit-status');
    if (saveBtn) saveBtn.disabled = false;
    if (statusText) statusText.innerHTML = '<span style="color: var(--accent-amber)">● Unsaved vertex modifications detected</span>';
  }

  render() {
    if (!this.currentFeature) return;

    const feat = this.currentFeature;
    const props = feat.properties || {};
    const scoreBd = this.currentDetails.score_breakdown;
    const warnings = this.currentDetails.associated_warnings || [];
    const hasEdits = this.currentDetails.has_geometry_edits;

    const fid = feat.id || props.feature_id;
    const ftype = props.feature_type || 'building';
    const source = props.source || 'project_vaayu_sample';
    const status = props.review_status || 'unverified';
    const area = props.area_sqm ? `${props.area_sqm.toFixed(1)} m²` : '—';
    const roof = props.roof_type || 'Unknown';
    const floors = props.no_floors || 1;
    const editedBy = props.edited_by || 'Unedited';
    const editedAt = props.edited_at ? new Date(props.edited_at).toLocaleTimeString() : '—';

    // Priority color
    let priorityClass = 'priority-low';
    let barColor = '#10b981';
    if (scoreBd && scoreBd.priority === 'high') {
      priorityClass = 'priority-high';
      barColor = '#f43f5e';
    } else if (scoreBd && scoreBd.priority === 'medium') {
      priorityClass = 'priority-medium';
      barColor = '#f59e0b';
    }
    const scoreVal = scoreBd ? scoreBd.total_score : 10;

    this.container.innerHTML = `
      <div class="inspector-content">
        <!-- Header Info -->
        <div class="feature-header-card">
          <div class="feature-title-row">
            <span class="feature-id">${fid}</span>
            <span class="status-badge status-${status}">${status}</span>
          </div>
          <div class="feature-badge-row">
            <span class="status-badge" style="background: rgba(180, 137, 84, 0.15); color: #b48954;">${ftype}</span>
            <span class="status-badge" style="background: rgba(124, 123, 93, 0.15); color: #7c7b5d;">source: ${source}</span>
            ${props.confidence !== null && props.confidence !== undefined ? 
              `<span class="status-badge" style="background: rgba(180, 137, 84, 0.15); color: #854628;">conf: ${(props.confidence * 100).toFixed(0)}%</span>` : ''}
          </div>
        </div>

        <!-- Transparent Review-Priority Score Gauge -->
        <div class="score-box">
          <div class="score-header">
            <div>
              <div style="font-size: 11px; text-transform: uppercase; color: var(--text-faint); font-weight: 600;">Review Priority Score</div>
              <div class="score-num-wrap">
                <span class="score-number" style="color: ${barColor}">${scoreVal}</span>
                <span class="score-max">/ 100</span>
              </div>
            </div>
            <span class="score-priority-tag ${priorityClass}">${scoreBd ? scoreBd.priority : 'low'}</span>
          </div>

          <div class="score-bar-bg">
            <div class="score-bar-fill" style="width: ${scoreVal}%; background: ${barColor};"></div>
          </div>

          <!-- Contributing Rules Breakdown -->
          <details style="cursor: pointer; margin-top: 6px;">
            <summary style="font-size: 10.5px; color: var(--accent-cyan); font-weight: 500;">
              View Rule Contributions (${scoreBd ? scoreBd.rules_triggered.length : 0} active)
            </summary>
            <div class="score-rules-list" style="margin-top: 8px;">
              ${scoreBd && scoreBd.rules_triggered.length > 0 ? 
                scoreBd.rules_triggered.map(r => `
                  <div class="score-rule-item">
                    <span>${r.rule_name}</span>
                    <strong style="color: ${r.points > 0 ? 'var(--accent-rose)' : 'var(--accent-emerald)'}">
                      ${r.points > 0 ? '+' : ''}${r.points}
                    </strong>
                  </div>
                `).join('') : '<div style="font-size: 10.5px; color: var(--text-faint);">No priority penalties applied.</div>'
              }
            </div>
          </details>
          <div class="score-disclaimer-note">
            ⚠️ prototype heuristic—not a validated survey-priority model
          </div>
        </div>

        <!-- Physical Attributes -->
        <div class="attr-grid">
          <div class="attr-cell">
            <div class="attr-label">Calculated Area</div>
            <div class="attr-val">${area}</div>
          </div>
          <div class="attr-cell">
            <div class="attr-label">Roof Type</div>
            <div class="attr-val">${roof}</div>
          </div>
          <div class="attr-cell">
            <div class="attr-label">Floors</div>
            <div class="attr-val">${floors}</div>
          </div>
          <div class="attr-cell">
            <div class="attr-label">Locality / LGD</div>
            <div class="attr-val">Lalpur (511638)</div>
          </div>
        </div>

        <!-- Associated Topology Warnings -->
        ${warnings.length > 0 ? `
          <div class="form-group">
            <div class="section-title">
              <span>Active Topology Warnings (${warnings.length})</span>
            </div>
            ${warnings.map(w => `
              <div class="warning-card severity-${w.severity}" style="margin-bottom: 6px;" data-wid="${w.warning_id}">
                <div class="warning-info">
                  <div class="warning-title-line">
                    <span>${w.warning_type}</span>
                  </div>
                  <div class="warning-desc">${w.explanation}</div>
                  ${this.renderWarningEvidence(w)}
                </div>
                <button class="btn btn-secondary btn-icon btn-zoom-warning" title="Focus Conflict" data-wid="${w.warning_id}">🔍</button>
              </div>
            `).join('')}
          </div>
        ` : ''}

        <!-- Reviewer Actions -->
        <div class="inspector-actions">
          <div class="section-title">Review Disposition</div>
          <div class="status-btn-group">
            <button class="btn btn-warning" id="btn-status-under-review">Under Review</button>
            <button class="btn btn-success" id="btn-status-approve">Approve</button>
            <button class="btn btn-danger" id="btn-status-reject">Reject</button>
          </div>

          <!-- Notes -->
          <div class="form-group">
            <label class="form-label">Reviewer Notes</label>
            <textarea class="form-textarea" id="inspector-notes-input" placeholder="Add inspection notes or field survey notes...">${props.notes || ''}</textarea>
            <button class="btn btn-secondary" id="btn-save-notes" style="margin-top: 6px; align-self: flex-end;">Save Notes</button>
          </div>

          <!-- Polygon Vertex Editing -->
          <div class="geometry-edit-card">
            <div style="display: flex; justify-content: space-between; align-items: center;">
              <strong style="font-size: 11.5px; color: var(--accent-cyan);">Polygon Boundary Modification</strong>
              <button class="btn btn-secondary btn-icon" id="btn-start-edit" title="Edit vertices on map">✏️ Edit Vertices</button>
            </div>
            
            <div id="geometry-edit-status" style="font-size: 10.5px; color: var(--text-faint);">
              ${hasEdits ? '<span style="color: var(--accent-cyan)">● Feature geometry has human modifications</span>' : 'Geometry is currently unedited.'}
            </div>

            <div class="form-group" style="margin-top: 4px;">
              <label class="form-label">Edit Reason / Audit Comment</label>
              <input type="text" class="form-input" id="edit-reason-input" value="${props.edit_reason || 'Boundary refinement'}" placeholder="Reason for geometry alteration" />
            </div>

            <div style="display: flex; gap: 8px; margin-top: 4px;">
              <button class="btn btn-primary" id="btn-save-geometry" style="flex: 1;" disabled>Save Geometry</button>
              <button class="btn btn-secondary" id="btn-cancel-geometry" style="flex: 1;">Cancel</button>
              <button class="btn btn-secondary" id="btn-revert-geometry" style="flex: 1;" ${hasEdits ? '' : 'disabled'}>Revert</button>
            </div>
          </div>

          <!-- Audit Trail -->
          <div style="font-size: 10px; color: var(--text-faint); margin-top: 4px; border-top: 1px solid var(--border-subtle); padding-top: 8px;">
            <div>Last modified by: <strong>${editedBy}</strong></div>
            <div>Timestamp: <strong>${editedAt}</strong></div>
            <div style="margin-top: 4px; font-style: italic;">${props.provenance_citation || ''}</div>
          </div>
        </div>
      </div>
    `;

    this.bindEvents();
  }

  bindEvents() {
    // Status update buttons
    const btnApprove = document.getElementById('btn-status-approve');
    const btnReject = document.getElementById('btn-status-reject');
    const btnUnderReview = document.getElementById('btn-status-under-review');
    const btnSaveNotes = document.getElementById('btn-save-notes');
    const notesInput = document.getElementById('inspector-notes-input');

    if (btnApprove) btnApprove.onclick = () => this.handleStatusUpdate('approved');
    if (btnReject) btnReject.onclick = () => this.handleStatusUpdate('rejected');
    if (btnUnderReview) btnUnderReview.onclick = () => this.handleStatusUpdate('under_review');
    if (btnSaveNotes) btnSaveNotes.onclick = () => this.handleStatusUpdate(this.currentFeature.properties.review_status);

    // Geometry edit buttons
    const btnStartEdit = document.getElementById('btn-start-edit');
    const btnSaveGeom = document.getElementById('btn-save-geometry');
    const btnCancelGeom = document.getElementById('btn-cancel-geometry');
    const btnRevertGeom = document.getElementById('btn-revert-geometry');
    const reasonInput = document.getElementById('edit-reason-input');

    if (btnStartEdit) btnStartEdit.onclick = () => this.onStartVertexEdit();
    if (btnCancelGeom) {
      btnCancelGeom.onclick = () => {
        this.pendingGeometry = null;
        const statusText = document.getElementById('geometry-edit-status');
        if (statusText) statusText.innerHTML = '<span style="color: var(--accent-amber)">● Vertex edits cancelled.</span>';
        this.onCancelVertexEdit();
      };
    }
    if (btnSaveGeom) {
      btnSaveGeom.onclick = async () => {
        if (!this.pendingGeometry) return;
        const reason = reasonInput ? reasonInput.value : 'Manual vertex edit';
        try {
          const projectId = this.getProjectId();
          if (projectId && projectId !== 'SIH26012_INDIA_CANDIDATE_01_LALPUR') {
            await ApiClient.saveProjectGeometryEdit(projectId, this.currentFeatureId, this.pendingGeometry, reason, 'demo-reviewer');
          } else {
            await ApiClient.saveGeometryEdit(this.currentFeatureId, this.pendingGeometry, reason, 'demo-reviewer');
          }
          this.pendingGeometry = null;
          await this.loadFeature(this.currentFeatureId, this.currentDetails.layer);
          this.onGeometrySaved(this.currentFeatureId);
        } catch (err) {
          alert('Failed to save geometry: ' + err.message);
        }
      };
    }

    if (btnRevertGeom) {
      btnRevertGeom.onclick = async () => {
        try {
          const projectId = this.getProjectId();
          if (projectId && projectId !== 'SIH26012_INDIA_CANDIDATE_01_LALPUR') {
            await ApiClient.revertProjectGeometry(projectId, this.currentFeatureId);
          } else {
            await ApiClient.revertGeometry(this.currentFeatureId);
          }
          await this.loadFeature(this.currentFeatureId, this.currentDetails.layer);
          this.onGeometryReverted(this.currentFeatureId);
        } catch (err) {
          alert('Failed to revert geometry: ' + err.message);
        }
      };
    }

    // Warning focus buttons
    this.container.querySelectorAll('.btn-zoom-warning').forEach(btn => {
      btn.onclick = (e) => {
        e.stopPropagation();
        const wid = btn.getAttribute('data-wid');
        const warn = this.currentDetails.associated_warnings.find(w => w.warning_id === wid);
        if (warn) this.onWarningClick(warn);
      };
    });
  }

  async handleStatusUpdate(newStatus) {
    const notesInput = document.getElementById('inspector-notes-input');
    const notes = notesInput ? notesInput.value : '';
    try {
      const projectId = this.getProjectId();
      if (projectId && projectId !== 'SIH26012_INDIA_CANDIDATE_01_LALPUR') {
        await ApiClient.updateProjectFeatureStatus(projectId, this.currentFeatureId, newStatus, notes, 'demo-reviewer');
      } else {
        await ApiClient.updateFeatureStatus(this.currentFeatureId, newStatus, notes, 'demo-reviewer');
      }
      await this.loadFeature(this.currentFeatureId, this.currentDetails.layer);
      this.onStatusChanged(this.currentFeatureId, newStatus);
    } catch (err) {
      alert('Failed to update status: ' + err.message);
    }
  }
}
