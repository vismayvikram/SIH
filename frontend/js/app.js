/**
 * SIH26012 Platform - Main Application Coordinator
 */
import { ApiClient } from './api.js?v=sih-20261005-02';
import { MapController } from './map.js?v=sih-20261005-02';
import { FeatureInspector } from './inspector.js?v=sih-20261005-02';
import { WarningCenter } from './warnings.js?v=sih-20261005-02';
import { ModelModalController } from './model.js?v=sih-20261005-02';

class Application {
  constructor() {
    this.map = null;
    this.inspector = null;
    this.warningCenter = null;
    this.modelController = null;
    this.currentView = null;
    this.projectModeControlStates = null;
    this.activeProjectId = null;
    this.activeProjectRaster = null;
    this.hasReferenceLayer = false;
    this.buildId = window.APP_BUILD_ID || 'sih-20261005-02';
    this.apiRequestLog = [];
    this.consoleErrors = [];
    this.openProjectTrace = [];
    this.currentProjectManifest = [];
  }

  getBuildId() {
    return this.buildId;
  }

  trace(step, detail = '') {
    this.openProjectTrace.push({ timestamp: new Date().toISOString(), step, detail: String(detail || '') });
    if (this.openProjectTrace.length > 100) this.openProjectTrace.shift();
  }

  recordApiRequest(method, url, status) {
    this.apiRequestLog.push({ timestamp: new Date().toISOString(), method, url, status });
    if (this.apiRequestLog.length > 30) this.apiRequestLog.shift();
  }

  recordConsoleError(message, file = '', line = 0) {
    this.consoleErrors.push({ timestamp: new Date().toISOString(), message, file, line });
    if (this.consoleErrors.length > 20) this.consoleErrors.shift();
  }

  registerGlobalDiagnostics() {
    const originalFetch = window.fetch.bind(window);
    window.fetch = async (...args) => {
      const [resource, init] = args;
      const method = (init && init.method) ? init.method.toUpperCase() : 'GET';
      const url = typeof resource === 'string' ? resource : resource?.url || 'request';
      try {
        const response = await originalFetch(...args);
        this.recordApiRequest(method, url, response.status);
        return response;
      } catch (error) {
        this.recordApiRequest(method, url, 'NETWORK_ERROR');
        throw error;
      }
    };

    window.addEventListener('error', (event) => {
      const message = event.message || event.error?.message || 'Unknown error';
      this.recordConsoleError(message, event.filename || '', event.lineno || 0);
      this.showToast(`Something went wrong: ${message}`, 'danger');
    });

    window.addEventListener('unhandledrejection', (event) => {
      const reason = event.reason;
      const message = reason?.message || String(reason || 'Unknown promise rejection');
      const stackLocation = reason?.stack?.match(/(.+):(\d+):(\d+)/);
      this.recordConsoleError(message, stackLocation?.[1] || '', Number(stackLocation?.[2] || 0));
      this.showToast(`Something went wrong: ${message}`, 'danger');
    });
  }

  safeInit(label, callback) {
    try {
      return callback();
    } catch (error) {
      const message = error?.message || String(error);
      console.error(`${label} failed:`, error);
      this.showToast(`${label} failed: ${message}`, 'danger');
      return null;
    }
  }

  async clearProjectState() {
    this.currentProjectManifest = [];
    this.hasReferenceLayer = false;
    this.activeProjectRaster = null;
    if (this.warningCenter) this.warningCenter.setWarnings([]);
    if (this.inspector) await this.inspector.loadFeature(null);
    this.map?.clearProjectState();
    this.updatePredictionSourceBadge({ selected: false, source_label: 'No active prediction set' });
    this.updateGreeneryStatus(0, true);
    const emptyState = document.getElementById('project-empty-state');
    if (emptyState) emptyState.hidden = false;
    const manifestContainer = document.getElementById('layer-manifest-container');
    if (manifestContainer) manifestContainer.innerHTML = '';
  }

  renderLayerManifest(manifest = []) {
    this.currentProjectManifest = manifest;
    const container = document.getElementById('layer-manifest-container');
    if (!container) return;
    const quietMode = !manifest || manifest.length === 0;
    container.innerHTML = quietMode
      ? '<div class="layer-card"><div class="layer-info"><div class="layer-name-wrap"><span class="layer-name">No layers yet</span><span class="layer-meta">The project raster is available but no review layers have been imported.</span></div></div></div>'
      : manifest.map((layer) => {
          const style = layer.style || {};
          const countLabel = typeof layer.feature_count === 'number' ? `${layer.feature_count} features` : '0 features';
          const visibleText = layer.visible_default ? 'visible by default' : 'off by default';
          const readOnlyText = layer.read_only ? 'read only' : 'editable';
          const color = style.color || '#60a5fa';
          return `
            <div class="layer-card">
              <div class="layer-info">
                <div class="layer-color-dot" style="background: ${color};"></div>
                <div class="layer-name-wrap">
                  <span class="layer-name">${this.escapeHtml(layer.label || layer.id)}</span>
                  <span class="layer-meta">${this.escapeHtml(countLabel)} · ${this.escapeHtml(layer.kind || 'review')} · ${this.escapeHtml(readOnlyText)} · ${this.escapeHtml(visibleText)}</span>
                </div>
              </div>
              <label class="toggle-switch">
                <input type="checkbox" data-layer-id="${this.escapeHtml(layer.id)}" ${layer.visible_default ? 'checked' : ''} ${layer.read_only ? 'disabled' : ''}>
                <span class="toggle-slider"></span>
              </label>
            </div>
          `;
        }).join('');

    container.querySelectorAll('input[data-layer-id]').forEach((toggle) => {
      toggle.addEventListener('change', (event) => {
        const layerId = event.target.dataset.layerId;
        if (!layerId || !this.map) return;
        this.map.toggleLayer(layerId, event.target.checked);
      });
    });
    const hasAiFeatures = manifest.some((layer) => layer.id === 'ai_predictions' && Number(layer.feature_count) > 0);
    const projectCta = document.getElementById('project-empty-state');
    if (projectCta) projectCta.hidden = hasAiFeatures;
  }

  setReferenceUi(hasReference) {
    this.hasReferenceLayer = Boolean(hasReference);
    document.querySelectorAll('[data-reference-only]').forEach((element) => {
      element.hidden = !this.hasReferenceLayer;
      if (!this.hasReferenceLayer) element.classList.remove('active');
    });
  }

  bindDashboardActions() {
    const openDemoButtons = [
      document.getElementById('btn-open-demo-primary'),
      document.getElementById('btn-open-demo-secondary'),
      document.getElementById('btn-open-demo-card')
    ].filter(Boolean);

    openDemoButtons.forEach((button) => {
      button.onclick = async () => { await this.openDemoProject(); };
    });

    const backButton = document.getElementById('btn-back-dashboard');
    if (backButton) {
      backButton.onclick = () => this.showDashboardView();
    }

    ['btn-create-project', 'btn-create-project-list'].forEach((id) => {
      document.getElementById(id)?.addEventListener('click', () => this.openProjectDialog());
    });

    document.getElementById('nav-projects')?.addEventListener('click', () => {
      document.getElementById('projects-section')?.scrollIntoView({ behavior: 'smooth' });
    });

    document.getElementById('btn-close-project-dialog')?.addEventListener('click', () => this.closeProjectDialog());
    document.getElementById('btn-cancel-project')?.addEventListener('click', () => this.closeProjectDialog());
    document.getElementById('create-project-form')?.addEventListener('submit', (event) => this.handleProjectCreate(event));
    document.getElementById('dashboard-project-list')?.addEventListener('click', async (event) => {
      const button = event.target.closest('[data-open-project]');
      if (button) await this.handleOpenProject(button.dataset.openProject);
    });

    window.addEventListener('hashchange', () => this.syncView());
  }

  syncView() {
    const view = window.location.hash === '#demo' ? 'demo' : 'dashboard';
    this.currentView = view;
    document.body.classList.toggle('view-dashboard', view === 'dashboard');
    if (view === 'demo') {
      window.location.hash = '#demo';
      return this.map?.refreshSize() || Promise.resolve();
    } else if (window.location.hash === '#demo') {
      window.history.replaceState(null, '', window.location.pathname);
    }
    return Promise.resolve();
  }

  showDashboardView() {
    window.location.hash = '#dashboard';
    this.syncView();
  }

  async showDemoView(project = null) {
    this.activeProjectId = 'SIH26012_INDIA_CANDIDATE_01_LALPUR';
    await this.clearProjectState();
    const demoProject = project || await ApiClient.getProject(this.activeProjectId);
    this.activeProjectRaster = demoProject.raster;
    this.setReferenceUi(true);
    this.setProjectModeControls(false);
    this.setWorkspaceProject('Lalpur Village', 'LGD 511638', {
      crs: 'EPSG:3857', width: 20137, height: 20886, gsd_x: 0.0338, gsd_y: 0.0338
    }, 'Lalpur AOI');
    window.location.hash = '#demo';
    await this.syncView();
    this.map.activateDemoRaster();
    const runCardPromise = this.loadDemoRunCard();
    await this.loadInitialData();
    await runCardPromise;
  }

  async loadDemoRunCard() {
    const card = document.getElementById('demo-about-run');
    const body = document.getElementById('demo-about-run-body');
    if (!card || !body) return;
    card.hidden = false;
    body.replaceChildren();
    const message = document.createElement('div');
    message.textContent = 'Loading saved run...';
    body.appendChild(message);
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 8000);
    try {
      const payload = await ApiClient.getProjectModelRuns(this.activeProjectId, controller.signal);
      const run = payload.runs?.[0];
      body.replaceChildren();
      if (!run) {
        this.renderUnavailableDemoRun(body);
        return;
      }
      const metadata = run.model_metadata || {};
      const summary = document.createElement('div');
      summary.textContent = `${metadata.provider_label || metadata.model_name || 'Saved model run'} · ${run.feature_count ?? metadata.valid_feature_count ?? run.features?.length ?? 0} buildings · ${metadata.inference_resolution_m ? `${metadata.inference_resolution_m} m/pixel` : 'resolution unavailable'} · ${run.created_at || 'timestamp unavailable'}`;
      body.appendChild(summary);
    } catch (error) {
      body.replaceChildren();
      this.renderUnavailableDemoRun(body);
    } finally {
      clearTimeout(timeout);
    }
  }

  renderUnavailableDemoRun(body) {
    const message = document.createElement('div');
    message.textContent = 'Not available for this run';
    const retry = document.createElement('button');
    retry.className = 'btn btn-secondary';
    retry.type = 'button';
    retry.textContent = 'Retry';
    retry.addEventListener('click', () => this.loadDemoRunCard());
    body.append(message, retry);
  }

  async openDemoProject() {
    this.trace('clicked', 'Lalpur demo');
    this.hideOpenProjectFailure();
    let step = 1;
    try {
      step = 1;
      const project = await ApiClient.getProject('SIH26012_INDIA_CANDIDATE_01_LALPUR');
      if (!project?.project_id || !project.raster) throw new Error('Demo project response is missing project metadata.');
      this.trace('project fetched', project.project_id);
      step = 2;
      this.validateProjectBounds(project);
      this.trace('bounds computed', project.raster.bounds_wgs84);
      await this.showDemoView(project);
      this.trace('complete', 'Lalpur demo opened');
    } catch (error) {
      this.showOpenProjectFailure(step, error);
    }
  }

  async init() {
    console.log('Initializing SIH26012 Feature Review Platform...');

    // Initialize Map
    this.map = new MapController('map', {
      onFeatureSelect: (fid, layerName, feature) => this.handleFeatureSelect(fid, layerName),
      onGeometryChange: (fid, newGeom) => this.handleGeometryChange(newGeom),
      onDraftCreated: (geom) => this.handleDraftCreated(geom),
      onProjectTrace: (step, detail) => {
        this.trace(step, detail);
        if (step === 'first tile loaded') this.recordApiRequest('GET', detail, 200);
        if (step === 'first tile errored' || step === 'tile errored') this.recordApiRequest('GET', detail, 'TILE_ERROR');
      }
    });
    this.map.init();

    // Initialize Inspector
    this.inspector = new FeatureInspector('feature-inspector-container', {
      getProjectId: () => this.activeProjectId,
      onStatusChanged: (fid, newStatus) => this.handleStatusChanged(fid, newStatus),
      onGeometrySaved: (fid) => {
        this.map.finishVertexEditing();
        this.refreshAllData();
      },
      onGeometryReverted: (fid) => this.refreshAllData(),
      onStartVertexEdit: () => this.map.startVertexEditing(),
      onCancelVertexEdit: () => this.map.cancelVertexEditing(),
      onWarningClick: (warning) => this.map.focusOnWarning(warning)
    });

    // Initialize Warning Center
    this.warningCenter = new WarningCenter('bottom-warning-tray-container', {
      onWarningClick: (warning) => this.map.focusOnWarning(warning)
    });

    // Initialize Model Modals
    this.modelController = new ModelModalController({
      onInferenceSuccess: (res) => this.handleInferenceSuccess(res),
      getProjectContext: () => ({
        projectId: this.activeProjectId,
        raster: this.activeProjectRaster,
        hasReferenceLayer: this.hasReferenceLayer,
      }),
      onProjectModelComplete: (job, summary) => this.refreshProjectModelLayer(job, summary),
      onTrace: (step, detail) => this.trace(`model ${step}`, detail),
      onDrawArea: (callback) => this.map.startModelAreaDrawing(callback),
      onToast: (message, type) => this.showToast(message, type),
    });

    // Bind UI controls
    this.registerGlobalDiagnostics();
    this.bindDashboardActions();
    this.bindLayerToggles();
    this.bindGreeneryControls();
    this.bindBasemapButtons();
    this.bindHeaderActions();
    this.bindDrawingTools();
    this.setBuildIdDisplay();
    this.setReferenceUi(false);
    await this.syncView();
    await this.loadProjectSummary();
  }

  setBuildIdDisplay() {
    const buildIdDisplay = document.getElementById('app-build-id');
    if (buildIdDisplay) buildIdDisplay.textContent = `build: ${this.getBuildId()}`;
  }

  async loadProjectSummary() {
    try {
      const payload = await ApiClient.getProjects();
      const project = payload.projects?.[0];
      this.renderProjectList(payload.projects || []);
      if (!project) return;

      const projectTitle = document.querySelector('.featured-project h3');
      const previewTitle = document.querySelector('.dashboard-preview-card .preview-map-header span');
      const locationText = document.querySelector('.project-kicker');
      const statusBadges = document.querySelectorAll('.featured-project .status-badge, .dashboard-preview-card .status-badge');

      if (projectTitle) projectTitle.textContent = project.name;
      if (previewTitle) previewTitle.textContent = project.name;
      if (locationText) locationText.textContent = project.locality;
      statusBadges.forEach((badge) => {
        badge.textContent = project.status === 'demo_ready' ? 'Demo ready' : 'Raster unavailable';
      });

      const projectLabel = document.querySelector('.featured-project .project-meta-grid');
      if (projectLabel) {
        projectLabel.innerHTML = `
          <div><strong>Orthomosaic</strong><span>${project.raster?.available ? 'Available' : 'Unavailable'}</span></div>
          <div><strong>Reference footprints</strong><span>${project.reference_layer_ids?.length ? '317' : '0'}</span></div>
          <div><strong>Parcel layer</strong><span>Not available</span></div>
          <div><strong>Model status</strong><span>${project.model_run_ids?.length ? 'Experimental' : 'Planned'}</span></div>
        `;
      }
    } catch (err) {
      console.warn('Project summary unavailable:', err);
    }
  }

  renderProjectList(projects) {
    const list = document.getElementById('dashboard-project-list');
    if (!list) return;
    list.innerHTML = projects.map((project) => {
      const hasRaster = Boolean(project.raster?.available && project.raster?.relative_path);
      const rasterDetails = hasRaster ? this.formatRasterDetails(project.raster) : 'No raster uploaded';
      const statusLabel = project.project_id === 'SIH26012_INDIA_CANDIDATE_01_LALPUR'
        ? (project.status === 'demo_ready' ? 'Demo ready' : 'Raster unavailable')
        : (hasRaster ? 'Raster ready' : 'Needs raster');
      return `
        <article class="dashboard-project-item">
          <div class="dashboard-project-item-copy">
            <div class="project-kicker">${this.escapeHtml(project.locality || 'Unknown locality')}</div>
            <h3>${this.escapeHtml(project.name || project.project_id)}</h3>
            <p>${hasRaster ? `${this.escapeHtml(project.raster.original_filename || 'GeoTIFF available')} · ` : ''}${this.escapeHtml(rasterDetails)}</p>
          </div>
          <div class="dashboard-project-item-actions">
            <span class="status-badge ${hasRaster ? 'status-approved' : ''}">${this.escapeHtml(statusLabel)}</span>
            <button class="btn btn-primary" type="button" data-open-project="${this.escapeHtml(project.project_id)}" ${hasRaster ? '' : 'disabled'}>Open project</button>
          </div>
        </article>`;
    }).join('');
  }

  escapeHtml(value) {
    return String(value).replace(/[&<>"']/g, (character) => ({
      '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
    })[character]);
  }

  openProjectDialog() {
    const dialog = document.getElementById('create-project-dialog');
    const status = document.getElementById('project-upload-status');
    if (status) status.textContent = '';
    if (dialog && !dialog.open) dialog.showModal();
  }

  closeProjectDialog() {
    const dialog = document.getElementById('create-project-dialog');
    if (dialog?.open) dialog.close();
  }

  async handleProjectCreate(event) {
    event.preventDefault();
    const form = event.currentTarget;
    const values = new FormData(form);
    const name = String(values.get('name') || '').trim();
    const locality = String(values.get('locality') || '').trim();
    const file = values.get('file');
    const status = document.getElementById('project-upload-status');
    const submit = document.getElementById('btn-submit-project');
    if (!(file instanceof File) || !file.size) {
      if (status) status.textContent = 'Choose a GeoTIFF file to continue.';
      return;
    }

    if (submit) {
      submit.disabled = true;
      submit.textContent = 'Validating raster...';
    }
    if (status) status.textContent = 'Uploading and validating the GeoTIFF. Large files may take a moment.';
    try {
      const result = await ApiClient.uploadProjectRaster(file, name, locality);
      form.reset();
      this.closeProjectDialog();
      await this.loadProjectSummary();
      await this.handleOpenProject(result.project_id);
      this.showToast(`Created project: ${name}`);
    } catch (error) {
      if (status) status.textContent = error.message;
    } finally {
      if (submit) {
        submit.disabled = false;
        submit.textContent = 'Validate & Create';
      }
    }
  }

  async handleOpenProject(projectId) {
    this.trace('clicked', projectId);
    this.hideOpenProjectFailure();
    this.activeProjectId = projectId;
    let step = 1;
    let firstTilePromise = null;
    try {
      step = 1;
      const project = await ApiClient.getProject(projectId);
      if (!project || !project.project_id || !project.raster) {
        throw new Error('Project response is missing project_id or raster metadata.');
      }
      this.trace('project fetched', project.project_id);
      if (project.project_id === 'SIH26012_INDIA_CANDIDATE_01_LALPUR') {
        await this.showDemoView(project);
        this.trace('complete', 'Lalpur demo opened');
        return;
      }

      step = 2;
      const bounds = this.validateProjectBounds(project);
      project.raster.bounds_wgs84 = bounds;
      this.trace('bounds computed', JSON.stringify(bounds));

      step = 3;
      const manifestPayload = await ApiClient.getProjectManifest(projectId);
      if (!Array.isArray(manifestPayload?.layers)) {
        throw new Error('Project manifest response is missing its layers array.');
      }
      const orthomosaic = manifestPayload.layers.find((layer) => layer?.id === 'orthomosaic');
      if (!orthomosaic) throw new Error('Project manifest has no orthomosaic entry.');
      this.trace('manifest fetched', `${manifestPayload.layers.length} layers`);
      this.setReferenceUi(manifestPayload.has_reference_layer);

      step = 4;
      this.activeProjectId = project.project_id;
      await this.clearProjectState();
      this.activeProjectId = project.project_id;
      this.activeProjectRaster = project.raster;
      this.setReferenceUi(manifestPayload.has_reference_layer);
      const demoRunCard = document.getElementById('demo-about-run');
      if (demoRunCard) demoRunCard.hidden = true;
      this.trace('project state cleared', project.project_id);
      this.setProjectModeControls(true);
      this.setWorkspaceProject(project.name, `${project.locality} · Raster preview only`, project.raster, project.locality);
      await this.showDemoViewWithoutReset();
      this.trace('workspace shown', project.project_id);

      step = 5;
      firstTilePromise = this.map.openProjectRaster(project, orthomosaic);
      step = 6;
      this.renderLayerManifest(manifestPayload.layers);
      const emptyState = document.getElementById('project-empty-state');
      if (emptyState) emptyState.hidden = manifestPayload.layers.length > 0;
      this.trace('sidebar rendered', `${manifestPayload.layers.length} layers`);
      step = 7;
      await firstTilePromise;
      step = 8;
      this.trace('complete', project.project_id);
    } catch (error) {
      firstTilePromise?.catch(() => {});
      this.trace('failed', `step ${step}: ${error?.message || error}`);
      this.showOpenProjectFailure(step, error);
    }
  }

  validateProjectBounds(project) {
    const bounds = project?.raster?.bounds_wgs84;
    if (!Array.isArray(bounds) || bounds.length !== 2 || !bounds.every(
      (point) => Array.isArray(point) && point.length === 2 && point.every(Number.isFinite)
    )) {
      throw new Error('Project has no finite WGS84 bounds in [[south, west], [north, east]] order.');
    }
    const [[south, west], [north, east]] = bounds;
    if (south < -90 || north > 90 || south >= north || west < -180 || east > 180 || west >= east) {
      throw new Error('Project WGS84 bounds are outside valid latitude/longitude ranges.');
    }
    return [[south, west], [north, east]];
  }

  showOpenProjectFailure(step, error) {
    const banner = document.getElementById('project-open-error');
    const message = `Opening project failed at step ${step}: ${error?.message || String(error)}`;
    if (banner) {
      banner.textContent = message;
      banner.hidden = false;
    }
    this.showToast(message, 'danger');
  }

  hideOpenProjectFailure() {
    const banner = document.getElementById('project-open-error');
    if (banner) {
      banner.textContent = '';
      banner.hidden = true;
    }
  }

  async showDemoViewWithoutReset() {
    window.location.hash = '#demo';
    await this.syncView();
  }

  async refreshProjectModelLayer(job, summary) {
    const projectId = job.project_id || this.activeProjectId;
    if (!projectId) throw new Error('No project ID returned for the completed model job.');
    if (projectId !== this.activeProjectId) return;
    const [layersPayload, manifestPayload, warningsPayload, sourcePayload] = await Promise.all([
      ApiClient.getProjectLayers(projectId),
      ApiClient.getProjectManifest(projectId),
      ApiClient.getProjectWarnings(projectId),
      ApiClient.getPredictionSource(projectId),
    ]);
    this.currentProjectManifest = manifestPayload.layers || [];
    this.setReferenceUi(manifestPayload.has_reference_layer);
    this.renderLayerManifest(this.currentProjectManifest);
    const predictions = layersPayload.layers?.ai_predictions || { type: 'FeatureCollection', features: [] };
    this.map.loadLayerData('ai_predictions', predictions);
    this.map.toggleLayer('ai_predictions', predictions.features?.length > 0);
    this.warningCenter.setWarnings(warningsPayload.warnings || []);
    this.updatePredictionSourceBadge(sourcePayload || { selected: false, source_label: 'No active prediction set' });
    const emptyState = document.getElementById('project-empty-state');
    if (emptyState) emptyState.hidden = predictions.features?.length > 0;
    this.trace('project manifest refreshed', `${this.currentProjectManifest.length} layers`);
    this.trace('layer added', `${summary.layer_id || job.layer_id} · ${summary.feature_count ?? job.feature_count} features`);
  }

  formatRasterDetails(raster) {
    const gsd = Number(raster.gsd_x);
    const gsdLabel = Number.isFinite(gsd)
      ? `${gsd < 0.1 ? `${(gsd * 100).toFixed(2)} cm` : `${gsd.toFixed(2)} m`}/px`
      : 'GSD unavailable';
    return `${raster.crs || 'CRS unavailable'} · ${raster.width} × ${raster.height} · ${gsdLabel}`;
  }

  setWorkspaceProject(name, locality, raster, locationLabel = locality) {
    const nameElement = document.getElementById('workspace-project-name');
    const localityElement = document.getElementById('workspace-project-locality');
    const mapInfo = document.getElementById('workspace-map-info');
    const statusGsd = document.getElementById('status-gsd');
    const statusLocation = document.getElementById('status-location');
    if (nameElement) nameElement.textContent = name;
    if (localityElement) localityElement.textContent = locationLabel;
    if (mapInfo) mapInfo.textContent = raster ? this.formatRasterDetails(raster) : 'Raster metadata unavailable';
    if (statusLocation) statusLocation.textContent = locationLabel || 'No project open';
    if (statusGsd) {
      const gsd = raster && Number(raster.gsd_x || raster.gsd_y);
      statusGsd.textContent = Number.isFinite(gsd) ? `${gsd < 0.1 ? `${(gsd * 100).toFixed(2)} cm` : `${gsd.toFixed(2)} m`} GSD` : 'GSD unavailable';
    }
    const buildIdDisplay = document.getElementById('app-build-id');
    if (buildIdDisplay) buildIdDisplay.textContent = `build: ${this.getBuildId()}`;
  }

  setProjectModeControls(isProjectRaster) {
    const controlIds = [
      'toggle-buildings', 'toggle-roads', 'toggle-osm', 'toggle-parcels',
      'toggle-synthetic', 'toggle-drafts', 'toggle-ai-whu', 'toggle-ai-deeplab', 'toggle-ai-deeplab-finetuned', 'toggle-ai-mock',
      'toggle-greenery', 'greenery-threshold', 'greenery-min-area', 'btn-run-greenery', 'btn-import-geojson'
    ];
    const controls = controlIds.map((id) => document.getElementById(id)).filter(Boolean);
    if (isProjectRaster) {
      if (!this.projectModeControlStates) {
        this.projectModeControlStates = controls.map((element) => ({ element, disabled: element.disabled }));
      }
      controls.forEach((element) => { element.disabled = true; });
      return;
    }
    this.projectModeControlStates?.forEach(({ element, disabled }) => { element.disabled = disabled; });
    this.projectModeControlStates = null;
  }

  async loadInitialData() {
    const isDemoProject = this.activeProjectId === 'SIH26012_INDIA_CANDIDATE_01_LALPUR';
    try {
      const metadata = await ApiClient.getMetadata();
      this.updateMetadataNotice(metadata);

      if (!isDemoProject) {
        const [layersPayload, warningsData, predictionSource, manifestData] = await Promise.all([
          ApiClient.getProjectLayers(this.activeProjectId),
          ApiClient.getProjectWarnings(this.activeProjectId),
          ApiClient.getPredictionSource(this.activeProjectId),
          ApiClient.getProjectManifest(this.activeProjectId)
        ]);

        this.clearProjectState();
        this.renderLayerManifest(manifestData.layers || []);
        const emptyState = document.getElementById('project-empty-state');
        if (emptyState) emptyState.hidden = manifestData.layers?.length > 0;

        const layerMap = layersPayload.layers || {};
        const featureCollectionNames = [
          'buildings', 'roads', 'osm_roads', 'parcels', 'synthetic', 'drafts',
          'ai_predictions', 'ai_whu_predictions', 'ai_deeplab_predictions',
          'ai_deeplab_finetuned_predictions', 'ai_mock_predictions'
        ];
        featureCollectionNames.forEach((layerName) => {
          const collection = layerMap[layerName] || { type: 'FeatureCollection', features: [] };
          this.map?.loadLayerData(layerName, collection);
          const layerToggle = document.querySelector(`input[data-layer-id="${layerName}"]`);
          if (layerToggle) layerToggle.checked = !!(manifestData.layers || []).find((entry) => entry.id === layerName)?.visible_default;
        });
        this.warningCenter.setWarnings(warningsData.warnings || []);
        this.updatePredictionSourceBadge(predictionSource || { selected: false, source_label: 'No active prediction set' });
        this.updateLayerBadges({ buildings: 0, roads: 0, osm_roads: 0, parcels: 0, synthetic: 0 });
        this.showToast('Project loaded. No layers yet for this project.');
        return;
      }

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
      for (const layerName of ['ai_whu_predictions', 'ai_deeplab_predictions', 'ai_mock_predictions']) {
        const providerLayer = await ApiClient.getLayer(layerName);
        this.map.loadLayerData(layerName, providerLayer);
      }
      try {
        const finetuned = await ApiClient.getLayer('ai_deeplab_finetuned_predictions');
        this.map.loadLayerData('ai_deeplab_finetuned_predictions', finetuned);
        this.map.toggleLayer(
          'ai_deeplab_finetuned_predictions',
          Boolean(document.getElementById('toggle-ai-deeplab-finetuned')?.checked)
        );
        this.renderFineTunedProviderDetails(finetuned.metadata?.provider);
        const countBadge = document.getElementById('badge-count-finetuned');
        if (countBadge) countBadge.textContent = `${finetuned.total_features} features`;
      } catch (candidateError) {
        const providerStatus = document.getElementById('finetuned-provider-status');
        if (providerStatus) providerStatus.textContent = `Saved candidate unavailable: ${candidateError.message}`;
        const candidateToggle = document.getElementById('toggle-ai-deeplab-finetuned');
        if (candidateToggle) candidateToggle.disabled = true;
      }
      try {
        const greenery = await ApiClient.getGreeneryResults();
        this.map.loadLayerData('greenery', greenery);
        const greeneryToggle = document.getElementById('toggle-greenery');
        this.map.toggleLayer('greenery', Boolean(greeneryToggle?.checked));
        this.updateGreeneryStatus(greenery.features?.length || 0, greenery.metadata?.status === 'not_run');
      } catch (greeneryError) {
        console.warn('Saved greenery candidates unavailable:', greeneryError);
      }

      this.updateLayerBadges({
        buildings: buildings.total_features,
        roads: roads.total_features,
        osm_roads: osmRoads.total_features,
        parcels: parcels.total_features,
        synthetic: synthetic.total_features
      });

      const warningsData = await ApiClient.getWarnings();
      this.warningCenter.setWarnings(warningsData.warnings);

      this.showToast(`Loaded ${buildings.total_features} Lalpur buildings & ${synthetic.total_features} synthetic test fixtures.`);
    } catch (err) {
      console.error('Initialization error:', err);
      this.showToast('Error loading layers: ' + err.message, 'danger');
    }
  }

  async refreshAllData() {
    if (this.activeProjectId && this.activeProjectId !== 'SIH26012_INDIA_CANDIDATE_01_LALPUR') {
      const [layersPayload, warningsPayload, manifestPayload, sourcePayload] = await Promise.all([
        ApiClient.getProjectLayers(this.activeProjectId),
        ApiClient.getProjectWarnings(this.activeProjectId),
        ApiClient.getProjectManifest(this.activeProjectId),
        ApiClient.getPredictionSource(this.activeProjectId),
      ]);
      this.currentProjectManifest = manifestPayload.layers || [];
      this.setReferenceUi(manifestPayload.has_reference_layer);
      this.renderLayerManifest(this.currentProjectManifest);
      for (const [name, collection] of Object.entries(layersPayload.layers || {})) {
        this.map.loadLayerData(name, collection);
      }
      this.warningCenter.setWarnings(warningsPayload.warnings || []);
      this.updatePredictionSourceBadge(sourcePayload);
      return;
    }
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

  getDiagnosticsText() {
    const manifestCount = Array.isArray(this.currentProjectManifest) ? this.currentProjectManifest.length : 0;
    const requestSummary = this.apiRequestLog.slice(-30).map((entry) => `${entry.method} ${entry.url} ${entry.status}`).join('\n');
    const errorSummary = this.consoleErrors.slice(-20).map((entry) => `${entry.timestamp} ${entry.message} | ${entry.file || 'unknown file'}:${entry.line || 0}`).join('\n');
    const traceSummary = this.openProjectTrace.map((entry) => `${entry.timestamp} ${entry.step}${entry.detail ? ` | ${entry.detail}` : ''}`).join('\n');
    return [
      `build_id: ${this.getBuildId()}`,
      `active_project_id: ${this.activeProjectId || 'none'}`,
      `manifest_layers: ${manifestCount}`,
      '',
      'api_requests:',
      requestSummary || 'none',
      '',
      'console_errors:',
      errorSummary || 'none',
      '',
      'open_project_trace:',
      traceSummary || 'none',
    ].join('\n');
  }

  async handleInferenceSuccess(result) {
    const aiLayer = await ApiClient.getLayer('ai_predictions');
    this.map.loadLayerData('ai_predictions', aiLayer);
    const providerLayer = {
      whu: 'ai_whu_predictions',
      deeplab: 'ai_deeplab_predictions',
      mock: 'ai_mock_predictions'
    }[result.metadata?.provider_id];
    if (providerLayer && result.features) {
      this.map.loadLayerData(providerLayer, {
        type: 'FeatureCollection',
        features: result.features
      });
    }
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

  renderFineTunedProviderDetails(provider) {
    if (!provider) return;
    const metrics = provider.metrics;
    const candidate = metrics.candidate;
    const baseline = metrics.baseline;
    const setText = (id, value) => {
      const element = document.getElementById(id);
      if (element) element.textContent = value;
    };
    const fixed = (value) => (Number(value) * 100).toFixed(1);

    setText('finetuned-provider-status', provider.provider_label);
    setText(
      'finetuned-provider-run',
      `Provider ${provider.provider_id} · run ${provider.run_id} · threshold ${provider.confidence_threshold.toFixed(2)} · ${provider.target_gsd_m.toFixed(2)} m GSD · ${provider.training_patch_count} patches · ${provider.epochs} epochs · stride ${provider.inference_stride}.`
    );
    setText(
      'finetuned-provider-pixel-metrics',
      `Pixel-mask F1 at threshold 0.50: ${fixed(candidate.pixel.f1)}% (fine-tuned) vs ${fixed(baseline.pixel.f1)}% (baseline). Pixel F1 measures thresholded mask pixels, not building-detection accuracy.`
    );
    setText(
      'finetuned-provider-object-metrics',
      `Object matching at footprint IoU >= 0.35: ${candidate.polygon.tp} TP / ${candidate.polygon.fp} FP / ${candidate.polygon.fn} FN; object F1 ${fixed(candidate.polygon.f1)}%. At IoU >= 0.50: object F1 ${fixed(candidate.polygon.f1_iou_0_50)}%. Object F1 is the harmonic mean of one-to-one object-match precision and recall.`
    );
    setText(
      'finetuned-provider-baseline-metrics',
      `Baseline at IoU >= 0.35: ${baseline.polygon.tp} TP / ${baseline.polygon.fp} FP / ${baseline.polygon.fn} FN; object F1 ${fixed(baseline.polygon.f1)}%. At IoU >= 0.50: object F1 ${fixed(baseline.polygon.f1_iou_0_50)}%.`
    );
    setText(
      'finetuned-provider-hashes',
      `Checkpoint SHA-256 ${provider.checkpoint_sha256} · input raster SHA-256 ${provider.input_raster_sha256} · reference GeoJSON SHA-256 ${provider.reference_geojson_sha256}`
    );
  }

  bindLayerToggles() {
    const toggles = [
      { id: 'toggle-buildings', layer: 'buildings' },
      { id: 'toggle-roads', layer: 'roads' },
      { id: 'toggle-osm', layer: 'osm_roads' },
      { id: 'toggle-synthetic', layer: 'synthetic' },
      { id: 'toggle-drafts', layer: 'drafts' },
      { id: 'toggle-ai-whu', layer: 'ai_whu_predictions' },
      { id: 'toggle-ai-deeplab', layer: 'ai_deeplab_predictions' },
      { id: 'toggle-ai-deeplab-finetuned', layer: 'ai_deeplab_finetuned_predictions' },
      { id: 'toggle-ai-mock', layer: 'ai_mock_predictions' },
      { id: 'toggle-greenery', layer: 'greenery' }
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

  bindGreeneryControls() {
    const button = document.getElementById('btn-run-greenery');
    const toggle = document.getElementById('toggle-greenery');
    const threshold = document.getElementById('greenery-threshold');
    const thresholdValue = document.getElementById('greenery-threshold-value');
    const minArea = document.getElementById('greenery-min-area');
    if (toggle) toggle.onchange = (event) => this.map.toggleLayer('greenery', event.target.checked);
    if (threshold && thresholdValue) threshold.oninput = () => { thresholdValue.textContent = Number(threshold.value).toFixed(2); };
    if (!button) return;
    button.onclick = async () => {
      const status = document.getElementById('greenery-status');
      button.disabled = true;
      button.textContent = 'Calculating…';
      if (status) status.textContent = 'Computing RGB Excess Green candidates from the Lalpur orthomosaic…';
      try {
        const indexThreshold = threshold ? Number(threshold.value) : 0.15;
        const minAreaSqm = minArea ? Number(minArea.value) : 5.0;
        const result = await ApiClient.detectGreenery({ index_threshold: indexThreshold, min_area_sqm: minAreaSqm });
        this.map.loadLayerData('greenery', result);
        if (toggle) toggle.checked = true;
        this.map.toggleLayer('greenery', true);
        this.updateGreeneryStatus(result.features?.length || 0, false);
        this.showToast(`Added ${result.features?.length || 0} RGB greenery candidates. These are heuristic results, not a trained model.`);
      } catch (error) {
        if (status) status.textContent = `Detection failed: ${error.message}`;
        this.showToast(`RGB greenery detection failed: ${error.message}`, 'danger');
      } finally {
        button.disabled = false;
        button.textContent = 'Run RGB greenery index';
      }
    };
  }

  updateGreeneryStatus(count, neverRun = false) {
    const status = document.getElementById('greenery-status');
    if (!status) return;
    status.textContent = neverRun ? 'Not run yet · RGB heuristic, no training' : `${count} candidate patches · review required`;
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
    const btnOpenDiagnostics = document.getElementById('btn-open-diagnostics');
    const diagnosticsDialog = document.getElementById('diagnostics-dialog');
    if (btnOpenDiagnostics && diagnosticsDialog) {
      btnOpenDiagnostics.onclick = () => {
        const output = document.getElementById('diagnostics-output');
        if (output) {
          output.value = this.getDiagnosticsText();
        }
        diagnosticsDialog.showModal ? diagnosticsDialog.showModal() : diagnosticsDialog.setAttribute('open', 'open');
      };
    }
    const btnCopyDiagnostics = document.getElementById('btn-copy-diagnostics');
    if (btnCopyDiagnostics) {
      btnCopyDiagnostics.onclick = async () => {
        const output = document.getElementById('diagnostics-output');
        if (!output) return;
        await navigator.clipboard.writeText(output.value);
        this.showToast('Diagnostics copied to clipboard.');
      };
    }
    const btnCloseDiagnostics = document.getElementById('btn-close-diagnostics');
    if (btnCloseDiagnostics && diagnosticsDialog) {
      btnCloseDiagnostics.onclick = () => diagnosticsDialog.close ? diagnosticsDialog.close() : diagnosticsDialog.removeAttribute('open');
    }

    // Export GeoJSON
    const btnExport = document.getElementById('btn-export-geojson');
    if (btnExport) {
      btnExport.onclick = () => {
        const projectId = this.activeProjectId;
        const exportUrl = projectId && projectId !== 'SIH26012_INDIA_CANDIDATE_01_LALPUR'
          ? `/api/projects/${encodeURIComponent(projectId)}/export`
          : '/api/export';
        window.location.href = exportUrl;
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
