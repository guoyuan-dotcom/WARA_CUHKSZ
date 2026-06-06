if (window.location.protocol === "file:") {
  window.location.replace("http://127.0.0.1:8765/review.html");
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

async function postForm(url, formData) {
  const response = await fetch(url, {
    method: "POST",
    body: formData,
  });
  if (!response.ok) {
    const text = await response.text();
    let parsedError = "";
    try {
      const payload = JSON.parse(text);
      parsedError = payload.error || "";
    } catch {
      parsedError = "";
    }
    throw new Error(parsedError || text || `HTTP ${response.status}`);
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

function statusClass(status) {
  const value = String(status || "idle").toLowerCase().replaceAll(" ", "_").replaceAll("-", "_");
  if (["failed", "error", "exception"].includes(value)) return "needs_review";
  if (["done", "completed"].includes(value)) return "completed";
  return value;
}

function setStatus(status) {
  const cls = statusClass(status);
  els.reviewStatus.textContent = cls.replaceAll("_", " ");
  els.reviewStatus.className = `status-pill status-pill--${cls}`;
}

const els = {
  reviewStatus: document.getElementById("reviewStatus"),
  reviewForm: document.getElementById("reviewForm"),
  reviewPdf: document.getElementById("reviewPdf"),
  reviewPdfName: document.getElementById("reviewPdfName"),
  reviewPaperId: document.getElementById("reviewPaperId"),
  reviewModelSelect: document.getElementById("reviewModelSelect"),
  reviewMaxTokens: document.getElementById("reviewMaxTokens"),
  reviewBtn: document.getElementById("reviewBtn"),
  clearReviewBtn: document.getElementById("clearReviewBtn"),
  reviewSaveModelConfigBtn: document.getElementById("reviewSaveModelConfigBtn"),
  reviewModelRateSummary: document.getElementById("reviewModelRateSummary"),
  reviewActiveProviderName: document.getElementById("reviewActiveProviderName"),
  reviewActiveProviderMeta: document.getElementById("reviewActiveProviderMeta"),
  reviewActiveApiKey: document.getElementById("reviewActiveApiKey"),
  reviewActiveBaseUrl: document.getElementById("reviewActiveBaseUrl"),
  reviewActiveKeyStatus: document.getElementById("reviewActiveKeyStatus"),
  reviewIdBadge: document.getElementById("reviewIdBadge"),
  reviewSummary: document.getElementById("reviewSummary"),
  reviewScoreGrid: document.getElementById("reviewScoreGrid"),
  reviewIssueGrid: document.getElementById("reviewIssueGrid"),
  reviewRawJson: document.getElementById("reviewRawJson"),
};

let modelConfig = { profiles: [], providers: {}, selected_model_profile: "kimi-k2.6-no-thinking" };

function selectedProfile() {
  const id = els.reviewModelSelect.value || modelConfig.selected_model_profile;
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

function renderActiveApiForm() {
  const profile = selectedProfile();
  const providerId = providerIdForProfile(profile);
  const provider = providerId ? modelConfig.providers?.[providerId] || {} : {};
  const providerLabel = provider.label || providerId || "Selected provider";
  const keyEnv = profile?.api_key_env || provider.api_key_env || "API_KEY";
  const baseUrl = profile?.base_url || provider.base_url || "";
  els.reviewActiveProviderName.textContent = providerLabel;
  els.reviewActiveProviderMeta.textContent = profile
    ? `${keyEnv} for ${profile.label || profile.id}`
    : "Choose a model to configure its API key.";
  els.reviewActiveApiKey.value = "";
  els.reviewActiveApiKey.placeholder = `${keyEnv}, leave blank to keep existing`;
  els.reviewActiveBaseUrl.value = baseUrl;
  els.reviewActiveBaseUrl.placeholder = baseUrl || "Base URL for selected model";
  els.reviewActiveKeyStatus.textContent = provider?.api_key_set ? "set" : "not set";
  els.reviewActiveKeyStatus.className = provider?.api_key_set ? "is-set" : "is-missing";
}

function renderModelConfig(config) {
  modelConfig = config || modelConfig;
  const profiles = modelConfig.profiles || [];
  els.reviewModelSelect.innerHTML = profiles.map((profile) => {
    return `<option value="${escapeHtml(profile.id)}">${escapeHtml(profile.label || profile.id)}</option>`;
  }).join("");
  els.reviewModelSelect.value = modelConfig.selected_model_profile || modelConfig.default_model_profile || profiles[0]?.id || "";
  renderSelectedModelRate();
}

function renderSelectedModelRate() {
  const profile = selectedProfile();
  els.reviewModelRateSummary.textContent = profile
    ? `${profile.primary_model || profile.id}: ${formatRate(profile.pricing)}`
    : "Select a model to view token pricing.";
  renderActiveApiForm();
}

async function refreshModelConfig() {
  renderModelConfig(await fetchJson("/api/model-config"));
}

async function saveModelConfig() {
  els.reviewSaveModelConfigBtn.disabled = true;
  els.reviewSaveModelConfigBtn.textContent = "Saving...";
  try {
    const config = await fetchJson("/api/model-config", {
      method: "POST",
      body: JSON.stringify({
        model_profile: els.reviewModelSelect.value,
        api_key: els.reviewActiveApiKey.value.trim(),
        base_url: els.reviewActiveBaseUrl.value.trim(),
      }),
    });
    renderModelConfig(config);
    setStatus("idle");
  } finally {
    els.reviewSaveModelConfigBtn.disabled = false;
    els.reviewSaveModelConfigBtn.textContent = "Save API Settings";
  }
}

function scoreClass(score) {
  const value = Number(score);
  if (!Number.isFinite(value)) return "";
  if (value >= 80) return "is-strong";
  if (value >= 60) return "is-medium";
  return "is-weak";
}

function profileTitle(profileKey, review = {}) {
  if (review.profile_label) return review.profile_label;
  if (profileKey === "research_validity") return "Manuscript-Level Research Validity";
  if (profileKey === "optimization_maturity") return "Optimization Research Maturity";
  return profileKey.replaceAll("_", " ");
}

function profileEntriesFromPayload(payload) {
  const reviews = payload?.reviews && typeof payload.reviews === "object"
    ? payload.reviews
    : (payload?.review ? { research_validity: payload.review } : {});
  return ["research_validity", "optimization_maturity"]
    .filter((key) => reviews[key])
    .map((key) => [key, reviews[key]]);
}

function renderList(title, items, className = "") {
  const values = Array.isArray(items) ? items.filter(Boolean) : [];
  if (!values.length) return "";
  return `
    <div class="review-list ${className}">
      <h3>${escapeHtml(title)}</h3>
      <ul>${values.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>
    </div>
  `;
}

function renderProfileSummary(profileKey, review, payload) {
  const score = Number(review.overall_score);
  const maxScore = Number(review.max_score || 100);
  const scoreText = Number.isFinite(score) ? `${score}/${Number.isFinite(maxScore) ? maxScore : 100}` : "-";
  return `
    <article class="profile-summary-card">
      <div class="review-score-orb ${scoreClass(score)}">${escapeHtml(scoreText)}</div>
      <div>
        <strong>${escapeHtml(profileTitle(profileKey, review))}</strong>
        <span>${escapeHtml(review.summary || "No summary was returned.")}</span>
        <em>Confidence: ${escapeHtml(review.confidence ?? "-")} · Model: ${escapeHtml(review.model_profile || payload.model_profile || "-")}</em>
      </div>
    </article>
  `;
}

function renderDimensionSection(profileKey, review) {
  const dims = review.dimension_scores || {};
  const cards = Object.entries(dims).map(([key, value]) => {
    const item = value || {};
    const label = item.label || key.replaceAll("_", " ");
    return `
      <article class="dimension-card">
        <span>${escapeHtml(label)}</span>
        <strong>${escapeHtml(item.score ?? "-")}/${escapeHtml(item.max_score ?? "-")}</strong>
        <p>${escapeHtml(item.justification || "")}</p>
      </article>
    `;
  }).join("");
  return `
    <section class="dimension-section">
      <h3>${escapeHtml(profileTitle(profileKey, review))}</h3>
      <div class="dimension-section-grid">${cards}</div>
    </section>
  `;
}

function renderReview(payload) {
  const entries = profileEntriesFromPayload(payload);
  els.reviewSummary.classList.add("has-score", "is-dual-score");
  els.reviewIdBadge.textContent = payload?.review_id || entries[0]?.[1]?.review_id || "completed";
  els.reviewSummary.innerHTML = entries.map(([profileKey, review]) => {
    return renderProfileSummary(profileKey, review, payload);
  }).join("") || `
    <strong>Review complete.</strong>
    <span>No scoring profiles were returned.</span>
  `;

  els.reviewScoreGrid.innerHTML = entries.map(([profileKey, review]) => {
    return renderDimensionSection(profileKey, review);
  }).join("");

  els.reviewIssueGrid.innerHTML = entries.map(([profileKey, review]) => {
    const title = profileTitle(profileKey, review);
    return [
      renderList(`${title} Strengths`, review.strengths, "is-positive"),
      renderList(`${title} Weaknesses`, review.weaknesses),
      renderList(`${title} Critical Issues`, review.critical_issues, "is-critical"),
      renderList(`${title} Major Issues`, review.major_issues, "is-warning"),
      renderList(`${title} Minor Issues`, review.minor_issues),
    ].filter(Boolean).join("");
  }).filter(Boolean).join("") || `<div class="empty-state">No strengths or weaknesses were returned.</div>`;

  els.reviewRawJson.textContent = JSON.stringify(payload, null, 2);
}

function clearReview() {
  els.reviewForm.reset();
  els.reviewPdfName.textContent = "No file selected.";
  els.reviewIdBadge.textContent = "no report";
  els.reviewSummary.classList.remove("has-score", "is-dual-score");
  els.reviewSummary.innerHTML = `
    <strong>Waiting for a manuscript.</strong>
    <span>The scoring agent reviews only the uploaded manuscript text and returns research-validity and optimization-maturity scores.</span>
  `;
  els.reviewScoreGrid.innerHTML = "";
  els.reviewIssueGrid.innerHTML = "";
  els.reviewRawJson.textContent = "{}";
  renderSelectedModelRate();
  setStatus("idle");
}

async function submitReview(event) {
  event.preventDefault();
  const file = els.reviewPdf.files?.[0];
  if (!file) {
    setStatus("error");
    els.reviewSummary.innerHTML = `<strong>No manuscript selected.</strong><span>Please choose a manuscript file first.</span>`;
    return;
  }
  const form = new FormData();
  form.append("pdf", file);
  form.append("paper_id", els.reviewPaperId.value.trim() || file.name.replace(/\.pdf$/i, ""));
  form.append("model_profile", els.reviewModelSelect.value);
  form.append("max_tokens", els.reviewMaxTokens.value || "12000");

  els.reviewBtn.disabled = true;
  els.reviewBtn.textContent = "Reviewing...";
  setStatus("running");
  els.reviewSummary.classList.remove("has-score", "is-dual-score");
  els.reviewSummary.innerHTML = `<strong>Review in progress.</strong><span>The manuscript text is being extracted and scored by the selected model using both paper rubrics.</span>`;
  try {
    const payload = await postForm("/api/review-pdf", form);
    renderReview(payload);
    setStatus("completed");
  } catch (error) {
    setStatus("error");
    els.reviewSummary.classList.remove("has-score", "is-dual-score");
    els.reviewSummary.innerHTML = `<strong>Review failed.</strong><span>${escapeHtml(error.message)}</span>`;
  } finally {
    els.reviewBtn.disabled = false;
    els.reviewBtn.textContent = "Review Manuscript";
  }
}

els.reviewPdf.addEventListener("change", () => {
  const file = els.reviewPdf.files?.[0];
  els.reviewPdfName.textContent = file ? `${file.name} · ${(file.size / 1024 / 1024).toFixed(2)} MB` : "No file selected.";
  if (file && !els.reviewPaperId.value.trim()) {
    els.reviewPaperId.value = file.name.replace(/\.pdf$/i, "");
  }
});
els.reviewModelSelect.addEventListener("change", renderSelectedModelRate);
els.reviewSaveModelConfigBtn.addEventListener("click", () => saveModelConfig().catch((error) => {
  setStatus("error");
  els.reviewSummary.innerHTML = `<strong>Could not save API settings.</strong><span>${escapeHtml(error.message)}</span>`;
}));
els.reviewForm.addEventListener("submit", (event) => {
  submitReview(event).catch((error) => {
    setStatus("error");
    els.reviewSummary.innerHTML = `<strong>Review failed.</strong><span>${escapeHtml(error.message)}</span>`;
  });
});
els.clearReviewBtn.addEventListener("click", clearReview);

refreshModelConfig().catch((error) => {
  setStatus("error");
  els.reviewSummary.innerHTML = `<strong>Review page failed to load.</strong><span>${escapeHtml(error.message)}</span>`;
});
