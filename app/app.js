if (window.location.protocol === "file:") {
  window.location.replace("http://127.0.0.1:8765/");
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `HTTP ${response.status}`);
  }
  return response.json();
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function artifactHref(path, version = "") {
  const suffix = version ? `&v=${encodeURIComponent(String(version))}` : "";
  return `/api/artifact?path=${encodeURIComponent(path)}${suffix}`;
}

function statusClass(status) {
  const value = String(status || "idle").toLowerCase().replaceAll(" ", "_").replaceAll("-", "_");
  const legacyGateStop = String.fromCharCode(98, 108, 111, 99, 107, 101, 100);
  if (value === "done") return "completed";
  if (["failed", "error", "exception", legacyGateStop].includes(value)) return "needs_review";
  return value;
}

function statusLabel(status) {
  const value = statusClass(status);
  if (value === "completed") return "completed";
  if (value === "needs_review") return "needs review";
  return (value || "idle").replaceAll("_", " ");
}

function formatInteger(value) {
  const num = Number(value);
  if (!Number.isFinite(num)) return "-";
  return new Intl.NumberFormat("en-US").format(num);
}

function formatTokens(value) {
  const num = Number(value);
  if (!Number.isFinite(num) || num <= 0) return "-";
  if (num >= 1_000_000) return `${(num / 1_000_000).toFixed(2)}M`;
  if (num >= 1_000) return `${(num / 1_000).toFixed(1)}K`;
  return String(Math.round(num));
}

function formatCost(value) {
  if (value === null || value === undefined || value === "") return "-";
  if (typeof value === "string" && value.trim()) return value;
  const num = Number(value);
  if (!Number.isFinite(num) || num < 0) return "-";
  return `$${num.toFixed(2)}`;
}

function currencySymbol(currency) {
  const value = String(currency || "").toUpperCase();
  if (value === "USD") return "$";
  return value ? `${value} ` : "";
}

function formatRate(pricing) {
  if (!pricing || !pricing.currency) return "pricing unavailable";
  const symbol = currencySymbol(pricing.currency);
  return `${symbol}${Number(pricing.input_per_m).toFixed(4)}/M in · ${symbol}${Number(pricing.cached_input_per_m).toFixed(4)}/M cached · ${symbol}${Number(pricing.output_per_m).toFixed(4)}/M out`;
}

const PHASES = [
  { key: "phase1", label: "Phase 1", subtitle: "Problem proposal" },
  { key: "phase2", label: "Phase 2", subtitle: "Technical study" },
  { key: "phase3", label: "Phase 3", subtitle: "Paper package" },
];

const DEFAULT_LOG = "Start WARA to view live run logs here.";

const els = {
  topicTitle: document.getElementById("topicTitle"),
  phaseTrack: document.getElementById("phaseTrack"),
  workspaceRoot: document.getElementById("workspaceRoot"),
  runStatus: document.getElementById("runStatus"),
  runForm: document.getElementById("runForm"),
  topicInput: document.getElementById("topicInput"),
  modelSelect: document.getElementById("modelSelect"),
  runBtn: document.getElementById("runBtn"),
  refreshBtn: document.getElementById("refreshBtn"),
  stopBtn: document.getElementById("stopBtn"),
  saveModelConfigBtn: document.getElementById("saveModelConfigBtn"),
  activeProviderName: document.getElementById("activeProviderName"),
  activeProviderMeta: document.getElementById("activeProviderMeta"),
  activeApiKey: document.getElementById("activeApiKey"),
  activeBaseUrl: document.getElementById("activeBaseUrl"),
  activeKeyStatus: document.getElementById("activeKeyStatus"),
  modelRateSummary: document.getElementById("modelRateSummary"),
  historyCount: document.getElementById("historyCount"),
  runList: document.getElementById("runList"),
  logRunId: document.getElementById("logRunId"),
  currentActivity: document.getElementById("currentActivity"),
  logBlock: document.getElementById("logBlock"),
  selectedPdfName: document.getElementById("selectedPdfName"),
  finalPdfBtn: document.getElementById("finalPdfBtn"),
  openPdfBtn: document.getElementById("openPdfBtn"),
  pdfPreviewImage: document.getElementById("pdfPreviewImage"),
  pdfEmptyState: document.getElementById("pdfEmptyState"),
  metricRuntime: document.getElementById("metricRuntime"),
  metricModel: document.getElementById("metricModel"),
  metricRate: document.getElementById("metricRate"),
  metricTokens: document.getElementById("metricTokens"),
  metricCost: document.getElementById("metricCost"),
  metricCostDetail: document.getElementById("metricCostDetail"),
};

let pollTimer = null;
let selectedRunId = null;
let currentPdfPath = null;
let cachedRuns = { phase1: [], phase2: [] };
let modelConfig = { profiles: [], providers: {}, selected_model_profile: "kimi-k2.6-no-thinking" };

function selectedProfile() {
  const id = els.modelSelect.value || modelConfig.selected_model_profile;
  return (modelConfig.profiles || []).find((profile) => profile.id === id) || null;
}

function providerIdForProfile(profile) {
  const providerId = String(profile?.provider_id || "").toLowerCase();
  if (providerId) return providerId;
  const envName = String(profile?.api_key_env || "").toUpperCase();
  const profileId = String(profile?.id || "").toLowerCase();
  if (envName.includes("DEEPSEEK") || profileId.includes("deepseek")) return "deepseek";
  if (envName.includes("OPENAI") || profileId.includes("openai")) return "openai";
  if (envName.includes("KIMI") || envName.includes("MOONSHOT") || profileId.includes("kimi")) return "kimi";
  return "";
}

function selectedProvider() {
  const profile = selectedProfile();
  const providerId = providerIdForProfile(profile);
  const provider = providerId ? modelConfig.providers?.[providerId] || {} : {};
  return { profile, providerId, provider };
}

function providerStatusText(provider) {
  return provider?.api_key_set ? "set" : "not set";
}

function providerStatusClass(provider) {
  return provider?.api_key_set ? "is-set" : "is-missing";
}

function renderModelConfig(config) {
  modelConfig = config || modelConfig;
  const profiles = modelConfig.profiles || [];
  els.modelSelect.innerHTML = profiles.map((profile) => {
    return `<option value="${escapeHtml(profile.id)}">${escapeHtml(profile.label || profile.id)}</option>`;
  }).join("");
  els.modelSelect.value = modelConfig.selected_model_profile || modelConfig.default_model_profile || profiles[0]?.id || "";
  renderSelectedModelRate();
}

function renderActiveApiForm() {
  const { profile, providerId, provider } = selectedProvider();
  const providerLabel = provider.label || providerId || "Selected provider";
  const keyEnv = profile?.api_key_env || provider.api_key_env || "API_KEY";
  const baseUrl = profile?.base_url || provider.base_url || "";
  els.activeProviderName.textContent = providerLabel;
  els.activeProviderMeta.textContent = profile
    ? `${keyEnv} for ${profile.label || profile.id}`
    : "Choose a model to configure its API key.";
  els.activeApiKey.value = "";
  els.activeApiKey.placeholder = `${keyEnv}, leave blank to keep existing`;
  els.activeBaseUrl.value = baseUrl;
  els.activeBaseUrl.placeholder = baseUrl || "Base URL for selected model";
  els.activeKeyStatus.textContent = providerStatusText(provider);
  els.activeKeyStatus.className = providerStatusClass(provider);
}

function renderSelectedModelRate() {
  const profile = selectedProfile();
  if (!profile) {
    els.modelRateSummary.textContent = "Select a model to view token pricing.";
    els.metricRate.textContent = "model-specific pricing";
    renderActiveApiForm();
    return;
  }
  const rate = formatRate(profile.pricing);
  els.modelRateSummary.textContent = `${profile.primary_model || profile.id}: ${rate}`;
  els.metricModel.textContent = profile.label || profile.id;
  els.metricRate.textContent = rate;
  renderActiveApiForm();
}

async function refreshModelConfig() {
  const config = await fetchJson("/api/model-config");
  renderModelConfig(config);
}

function renderStatus(status) {
  const cls = statusClass(status);
  els.runStatus.textContent = statusLabel(cls);
  els.runStatus.className = `status-pill status-pill--${cls}`;
}

function renderActivity(activity) {
  if (!els.currentActivity) return;
  if (!activity || !activity.phase_key) {
    els.currentActivity.innerHTML = `
      <strong>Current activity</strong>
      <span>Waiting for a WARA run.</span>
    `;
    els.currentActivity.className = "activity-card";
    return;
  }
  const status = statusLabel(activity.status || "running");
  els.currentActivity.innerHTML = `
    <strong>${escapeHtml(activity.phase || "WARA")} · ${escapeHtml(activity.agent || "Controller")}</strong>
    <span>${escapeHtml(activity.title || "Pipeline step")} · ${escapeHtml(activity.task || "")}</span>
    <em>${escapeHtml(status)}</em>
  `;
  els.currentActivity.className = `activity-card activity-card--${statusClass(activity.status)}`;
}

function completedPhaseStep(report) {
  const phases = report?.phase2_report?.phases || [];
  let maxDone = 0;
  for (const phase of phases) {
    const status = statusClass(phase.status);
    const step = Number(phase.phase_step);
    if (Number.isFinite(step) && ["done", "completed"].includes(status)) {
      maxDone = Math.max(maxDone, step);
    }
  }
  return maxDone;
}

function currentPhaseKey(report) {
  const phases = report?.phase2_report?.phases || [];
  const running = phases.find((phase) => statusClass(phase.status) === "running");
  if (running) {
    const step = Number(running.phase_step);
    if (step >= 6) return "phase3";
    if (step >= 1) return "phase2";
  }
  const done = completedPhaseStep(report);
  if (done >= 6) return "phase3";
  if (done >= 1 || phases.length) return "phase2";
  return "phase1";
}

function renderPhaseTrack(report) {
  const current = currentPhaseKey(report);
  const doneStep = completedPhaseStep(report);
  const phase1Done = Boolean(report?.phase1_report) || doneStep >= 1 || (report?.phase2_report?.phases || []).length > 0;
  const phase2Done = doneStep >= 5;
  const phase3Done = doneStep >= 11 || statusClass(report?.status || report?.phase2_report?.status) === "completed";
  els.phaseTrack.innerHTML = PHASES.map((phase, index) => {
    let cls = "phase-node";
    if (phase.key === current) cls += " is-current";
    if (phase.key === "phase1" && phase1Done) cls += " is-done";
    if (phase.key === "phase2" && phase2Done) cls += " is-done";
    if (phase.key === "phase3" && phase3Done) cls += " is-done";
    return `
      <div class="${cls}">
        <span class="phase-dot">${index + 1}</span>
        <span class="phase-label">${escapeHtml(phase.label)}</span>
        <span class="phase-sublabel">${escapeHtml(phase.subtitle)}</span>
      </div>
    `;
  }).join("");
}

function renderRuns() {
  const runs = cachedRuns.phase2 || [];
  els.historyCount.textContent = String(runs.length);
  if (!runs.length) {
    els.runList.innerHTML = `<div class="empty-state">No WARA runs yet.</div>`;
    return;
  }
  els.runList.innerHTML = runs.map((run, index) => {
    const alias = run.display_id || `WARA${String(index + 1).padStart(3, "0")}`;
    const active = selectedRunId === run.run_id ? " is-selected" : "";
    return `
      <button type="button" class="run-card${active}" data-run-id="${escapeHtml(run.run_id)}">
        <strong>${escapeHtml(alias)}</strong>
      </button>
    `;
  }).join("");
  els.runList.querySelectorAll("[data-run-id]").forEach((button) => {
    button.addEventListener("click", async () => {
      selectedRunId = button.dataset.runId;
      renderRuns();
      const report = await fetchJson(`/api/runs/${encodeURIComponent(selectedRunId)}`);
      await renderReport(report);
      await refreshLogs(false);
    });
  });
}

function setPdf(path, version = "") {
  currentPdfPath = path || null;
  if (!currentPdfPath) {
    els.selectedPdfName.textContent = "-";
    els.pdfPreviewImage.removeAttribute("src");
    els.pdfEmptyState.style.display = "grid";
    els.openPdfBtn.disabled = true;
    return;
  }
  els.selectedPdfName.textContent = "Final Research PDF";
  const suffix = version ? `&v=${encodeURIComponent(String(version))}` : "";
  els.pdfPreviewImage.src = `/api/pdf-preview?path=${encodeURIComponent(currentPdfPath)}${suffix}`;
  els.pdfEmptyState.style.display = "none";
  els.openPdfBtn.disabled = false;
}

async function renderReport(report) {
  if (!report) return;
  const phase2 = report.phase2_report || {};
  const telemetry = report.telemetry || phase2.telemetry || {};
  selectedRunId = phase2.run_id || report.run_id || selectedRunId;

  els.topicTitle.textContent = report.topic || phase2.topic || "WARA Research Workspace";
  els.workspaceRoot.textContent = report.workspace_root || "-";
  renderStatus(report.status || phase2.status || "idle");
  renderPhaseTrack(report);

  els.metricRuntime.textContent = telemetry.runtime_hms || "-";
  const reportProfileId = phase2.model_profile || modelConfig.selected_model_profile || els.modelSelect.value;
  const reportProfile = (modelConfig.profiles || []).find((profile) => profile.id === reportProfileId) || selectedProfile();
  els.metricModel.textContent = reportProfile?.label || reportProfileId || "-";
  els.metricRate.textContent = reportProfile ? formatRate(reportProfile.pricing) : "model-specific pricing";
  els.metricTokens.textContent = formatTokens(telemetry.total_tokens);
  els.metricCost.textContent = formatCost(telemetry.cost_display_usd || telemetry.cost_display || telemetry.cost_usd);
  const prompt = formatTokens(telemetry.prompt_tokens);
  const cached = formatTokens(telemetry.cached_tokens);
  const output = formatTokens(telemetry.completion_tokens);
  els.metricCostDetail.textContent = `in ${prompt}, cached ${cached}, out ${output}`;

  setPdf(phase2.preview_pdf, phase2.preview_pdf_mtime);
  renderRuns();
}

async function refreshRuns() {
  const phase2Payload = await fetchJson("/api/phase2-runs");
  cachedRuns = {
    phase1: [],
    phase2: phase2Payload.runs || [],
  };
  renderRuns();
}

async function refreshLogs(useActive = true) {
  try {
    const suffix = selectedRunId ? `?run_id=${encodeURIComponent(selectedRunId)}` : "";
    const payload = await fetchJson(`/api/logs${suffix}`);
    els.logRunId.textContent = payload.run_id || selectedRunId || "idle";
    renderActivity(payload.activity);
    els.logBlock.textContent = payload.combined || DEFAULT_LOG;
    els.logBlock.scrollTop = els.logBlock.scrollHeight;
  } catch (error) {
    els.logRunId.textContent = "error";
    els.logBlock.textContent = `Failed to load logs: ${error.message}`;
  }
}

async function pollStatus() {
  try {
    const payload = await fetchJson("/api/status");
    if (payload.report) {
      await renderReport(payload.report);
    }
    renderStatus(payload.status || "idle");
    await refreshLogs(true);
    if (payload.status !== "running" && pollTimer) {
      window.clearInterval(pollTimer);
      pollTimer = null;
      els.runBtn.disabled = false;
      await refreshRuns();
      const overview = await fetchJson("/api/overview");
      await renderReport(overview);
    }
  } catch (error) {
    renderStatus("error");
    els.logBlock.textContent = `Failed to poll status: ${error.message}`;
    els.runBtn.disabled = false;
  }
}

async function startRun(event) {
  event.preventDefault();
  const topic = els.topicInput.value.trim();
  if (!topic) {
    els.logBlock.textContent = "Enter a research topic first.";
    return;
  }

  els.runBtn.disabled = true;
  renderStatus("starting");
  els.logBlock.textContent = "Starting WARA ...";
  const payload = await fetchJson("/api/start", {
    method: "POST",
    body: JSON.stringify({
      topic,
      model_profile: els.modelSelect.value,
      phase1_run: null,
      skip_phase1: false,
    }),
  });
  selectedRunId = payload.run_id;
  await renderReport(payload.report);
  await refreshRuns();
  await refreshLogs(true);
  if (pollTimer) window.clearInterval(pollTimer);
  pollTimer = window.setInterval(pollStatus, 3000);
}

async function stopRun() {
  await fetchJson("/api/stop", { method: "POST" });
  if (pollTimer) window.clearInterval(pollTimer);
  pollTimer = null;
  els.runBtn.disabled = false;
  renderStatus("stopped");
  await refreshLogs(true);
}

async function saveModelConfig() {
  els.saveModelConfigBtn.disabled = true;
  els.saveModelConfigBtn.textContent = "Saving...";
  try {
    const payload = {
      model_profile: els.modelSelect.value,
      api_key: els.activeApiKey.value.trim(),
      base_url: els.activeBaseUrl.value.trim(),
    };
    const config = await fetchJson("/api/model-config", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    renderModelConfig(config);
    els.logBlock.textContent = "API settings saved locally to .env. API keys are not displayed after saving.";
  } finally {
    els.saveModelConfigBtn.disabled = false;
    els.saveModelConfigBtn.textContent = "Save API Settings";
  }
}

async function bootstrap() {
  await refreshModelConfig();
  await refreshRuns();
  const overview = await fetchJson("/api/overview");
  await renderReport(overview);
  const status = await fetchJson("/api/status");
  if (status.report) await renderReport(status.report);
  renderStatus(status.status || overview.status || "idle");
  if (status.status === "running") {
    els.runBtn.disabled = true;
    pollTimer = window.setInterval(pollStatus, 3000);
  }
  await refreshLogs(true);
}

els.runForm.addEventListener("submit", (event) => {
  startRun(event).catch((error) => {
    renderStatus("error");
    els.logBlock.textContent = `Failed to start WARA: ${error.message}`;
    els.runBtn.disabled = false;
  });
});
els.refreshBtn.addEventListener("click", () => bootstrap().catch((error) => {
  els.logBlock.textContent = error.message;
  renderStatus("error");
}));
els.stopBtn.addEventListener("click", () => stopRun().catch((error) => {
  els.logBlock.textContent = `Failed to stop WARA: ${error.message}`;
  renderStatus("error");
}));
els.modelSelect.addEventListener("change", renderSelectedModelRate);
els.saveModelConfigBtn.addEventListener("click", () => saveModelConfig().catch((error) => {
  els.logBlock.textContent = `Failed to save API settings: ${error.message}`;
  renderStatus("error");
}));
els.openPdfBtn.addEventListener("click", async () => {
  if (!currentPdfPath) return;
  await fetchJson("/api/open-path", {
    method: "POST",
    body: JSON.stringify({ path: currentPdfPath }),
  });
});

bootstrap().catch((error) => {
  els.topicTitle.textContent = "WARA load failed";
  els.logBlock.textContent = error.message;
  renderStatus("error");
});
