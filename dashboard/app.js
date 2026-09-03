const API_BASE = window.location.origin;

const els = {
  apiStatus: document.getElementById("api-status"),
  dropzone: document.getElementById("dropzone"),
  fileInput: document.getElementById("file-input"),
  analyzeUpload: document.getElementById("analyze-upload"),
  confThreshold: document.getElementById("conf-threshold"),
  confValue: document.getElementById("conf-value"),
  hazardAlert: document.getElementById("hazard-alert"),
  hazardSummary: document.getElementById("hazard-summary"),
  safeAlert: document.getElementById("safe-alert"),
  imageStage: document.getElementById("image-stage"),
  metricModel: document.getElementById("metric-model"),
  metricInference: document.getElementById("metric-inference"),
  metricObjects: document.getElementById("metric-objects"),
  detectionsBody: document.getElementById("detections-body"),
  loading: document.getElementById("loading"),
  loadingText: document.getElementById("loading-text"),
  webcamVideo: document.getElementById("webcam-video"),
  webcamCanvas: document.getElementById("webcam-canvas"),
  webcamHint: document.getElementById("webcam-hint"),
  startWebcam: document.getElementById("start-webcam"),
  captureWebcam: document.getElementById("capture-webcam"),
  liveScan: document.getElementById("live-scan"),
  modelSelect: document.getElementById("model-select"),
  modelCheckboxes: document.getElementById("model-checkboxes"),
  singleModelSettings: document.getElementById("single-model-settings"),
  compareModelSettings: document.getElementById("compare-model-settings"),
  singleResults: document.getElementById("single-results"),
  compareResults: document.getElementById("compare-results"),
  compareGrid: document.getElementById("compare-grid"),
  compareStripWrap: document.getElementById("compare-strip-wrap"),
  compareOriginalStage: document.getElementById("compare-original-stage"),
  compareThreeWrap: document.getElementById("compare-three-wrap"),
  compareThreeGrid: document.getElementById("compare-three-grid"),
  compareMetricsWrap: document.getElementById("compare-metrics-wrap"),
  compareMetricsBody: document.getElementById("compare-metrics-body"),
  compareDownloadBtn: document.getElementById("compare-download-btn"),
  benchmarkPanel: document.getElementById("benchmark-panel"),
  imageLightbox: document.getElementById("image-lightbox"),
  lightboxImage: document.getElementById("lightbox-image"),
  lightboxCaption: document.getElementById("lightbox-caption"),
  lightboxClose: document.getElementById("lightbox-close"),
  benchmarkBody: document.getElementById("benchmark-body"),
  benchmarkRecommendation: document.getElementById("benchmark-recommendation"),
  selectionBanner: document.getElementById("selection-banner"),
  resultsTitle: document.getElementById("results-title"),
};

const MODEL_DISPLAY_ORDER = ["yolov5", "yolo11s", "yolov8s"];

const MODEL_META = {
  yolov5: { run: "Run 4", label: "Production" },
  yolo11s: { run: "Run 2", label: "Tested" },
  yolov8s: { run: "Run 3", label: "Tested" },
};

const RUN3_FALLBACK_BENCHMARK = {
  cylinder_recall: 0.72,
  map50: 0.656,
  map50_95: 0.45,
  recall: 0.663,
  fps: 68.7,
};

let selectedFile = null;
let webcamStream = null;
let liveScanTimer = null;
let isAnalyzing = false;
let analysisMode = "single";
let modelCatalog = [];
let compareInputUrl = null;
let lastCompareModels = [];
let modelReady = false;
let apiOnline = false;
let defaultModelId = "yolov5";
const IS_RENDER_HOST = window.location.hostname.includes("onrender.com");

function setLoading(active, message = "Running inference…") {
  els.loading.classList.toggle("hidden", !active);
  els.loadingText.textContent = message;
}

function formatClassName(name) {
  return name.replace(/_/g, " ");
}

function hazardBadge(type) {
  const cls = type === "explosive" ? "explosive" : "toxic";
  return `<span class="badge ${cls}">${type}</span>`;
}

function formatPct(value) {
  if (value === null || value === undefined) return "—";
  return `${(value * 100).toFixed(1)}%`;
}

function formatFps(value) {
  if (value === null || value === undefined) return "—";
  return value.toFixed(1);
}

function roleClass(role) {
  if (role === "production") return "production";
  if (role === "experimental") return "experimental";
  return "candidate";
}

function cleanModelName(model) {
  if (!model) return "Model";
  const fromCatalog = modelCatalog.find((item) => modelId(item) === modelId(model));
  const name = model.name || fromCatalog?.name || model.short_name || fromCatalog?.short_name || modelId(model);
  return String(name).replace(/\s*[—-]\s*candidate.*$/i, "").trim();
}

function modelId(model) {
  return model.id || model.model_id;
}

function modelRunLabel(model) {
  return MODEL_META[modelId(model)]?.run || "";
}

function modelTagsHtml(model, speedster = null) {
  const tags = [];
  const role = model.role === "candidate" && hasBenchmark(model) ? "experimental" : model.role;
  tags.push(`<span class="model-tag ${roleClass(role)}">${statusLabel({ ...model, role })}</span>`);
  if (speedster && speedster.id === modelId(model)) {
    tags.push('<span class="model-tag speed">Fastest FPS</span>');
  }
  return tags.join(" ");
}

function statusLabel(model) {
  const id = modelId(model);
  if (id === "yolov5" || model.role === "production") return "Production";
  if (hasBenchmark(model)) {
    if (id === "yolov8s") return "Tested · Run 3";
    if (id === "yolo11s") return "Tested · Run 2";
    return "Tested";
  }
  if (model.badge && !/not yet trained/i.test(model.badge)) return model.badge;
  return "Unavailable";
}

function hasBenchmark(model) {
  const bench = effectiveBenchmark(model);
  return bench.map50 !== null && bench.map50 !== undefined;
}

function effectiveBenchmark(model) {
  const bench = { ...(model.benchmark || {}) };
  if (modelId(model) === "yolov8s" && (bench.map50 === null || bench.map50 === undefined)) {
    return { ...RUN3_FALLBACK_BENCHMARK };
  }
  return bench;
}

function normalizeModel(model) {
  if (modelId(model) !== "yolov8s") return model;
  if (hasBenchmark(model)) {
    return {
      ...model,
      role: model.role === "candidate" ? "experimental" : model.role,
      badge: "Tested",
      name: "YOLOv8s-Seg",
      inference_available: model.inference_available !== false,
    };
  }
  return {
    ...model,
    role: "experimental",
    badge: "Tested",
    name: "YOLOv8s-Seg",
    inference_available: true,
    benchmark: { ...RUN3_FALLBACK_BENCHMARK },
  };
}

function displayModelName(model) {
  const name = cleanModelName(model);
  const run = modelRunLabel(model);
  return run ? `${name} <span class="model-run-label">${run}</span>` : name;
}

function modelSortIndex(model) {
  const idx = MODEL_DISPLAY_ORDER.indexOf(modelId(model));
  return idx === -1 ? 99 : idx;
}

function sortModelsForDisplay(models) {
  return [...models].sort((a, b) => {
    const orderDiff = modelSortIndex(a) - modelSortIndex(b);
    if (orderDiff !== 0) return orderDiff;
    return a.short_name.localeCompare(b.short_name);
  });
}

function fastestModel(models) {
  return models.reduce((best, model) => {
    const fps = effectiveBenchmark(model).fps;
    if (fps === null || fps === undefined) return best;
    if (!best || fps > (effectiveBenchmark(best).fps || 0)) return model;
    return best;
  }, null);
}

function renderSelectionBanner() {
  const tested = modelCatalog.filter(hasBenchmark);
  if (!tested.length) {
    els.selectionBanner.classList.add("hidden");
    return;
  }

  const winner = tested.find((model) => modelId(model) === "yolov5") || tested[0];
  const speedster = fastestModel(tested);
  const bench = effectiveBenchmark(winner);

  els.selectionBanner.classList.remove("hidden");
  els.selectionBanner.innerHTML = `
    <div class="selection-banner-main">
      <span class="selection-badge production">Selected for production</span>
      <strong>${cleanModelName(winner)}</strong>
      <span>— best mAP@50 (${formatPct(bench.map50)}) and Cylinder recall (${formatPct(bench.cylinder_recall)})</span>
    </div>
    ${
      speedster && modelId(speedster) !== modelId(winner)
        ? `<p class="selection-banner-note">${cleanModelName(speedster)} (Run 3) reaches ${formatFps(effectiveBenchmark(speedster).fps)} FPS for latency-sensitive deployments.</p>`
        : ""
    }
  `;
}

function selectedCompareIds() {
  return [...els.modelCheckboxes.querySelectorAll("input[type=checkbox]:checked")].map(
    (input) => input.value
  );
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function orderedCompareIds(modelIds) {
  const unique = [...new Set(modelIds)];
  return MODEL_DISPLAY_ORDER.filter((id) => unique.includes(id));
}

function catalogEntry(modelIdValue) {
  return modelCatalog.find((model) => modelId(model) === modelIdValue);
}

function predictResponseToCompareItem(data, error = null) {
  const catalog = catalogEntry(data?.model_id);
  if (error || !data) {
    return {
      model_id: catalog?.id || data?.model_id || "unknown",
      model_name: catalog?.name || data?.model_name || "Model",
      role: catalog?.role || "experimental",
      badge: catalog?.badge,
      inference_available: catalog?.inference_available ?? false,
      error,
      hazard_detected: null,
      hazard_summary: [],
      class_counts: [],
      detections: [],
      inference_ms: null,
      benchmark: effectiveBenchmark(catalog || {}),
    };
  }
  return {
    model_id: data.model_id,
    model_name: data.model_name,
    role: catalog?.role || "experimental",
    badge: catalog?.badge,
    inference_available: true,
    error: null,
    hazard_detected: data.hazard_detected,
    hazard_summary: data.hazard_summary,
    class_counts: data.class_counts,
    detections: data.detections,
    inference_ms: data.inference_ms,
    image_width: data.image_width,
    image_height: data.image_height,
    annotated_image_base64: data.annotated_image_base64,
    benchmark: effectiveBenchmark(catalog || {}),
  };
}

function inferenceErrorMessage(err, status) {
  const message = err?.message || String(err);
  if (status === 502 || status === 503 || /502|503|gateway|timed out/i.test(message)) {
    return (
      "Server timed out or is still waking up on Render. " +
      "Wait until the status pill turns green, try Single Model (YOLOv5s) first, then Compare."
    );
  }
  return message;
}

async function postPredict(file, modelIdValue, conf) {
  const formData = new FormData();
  formData.append("file", file, file.name || "capture.jpg");
  const res = await fetch(
    `${API_BASE}/predict?include_annotated_image=true&conf_threshold=${conf}&model_id=${modelIdValue}`,
    { method: "POST", body: formData }
  );
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    const detail = err.detail || `Request failed (${res.status})`;
    throw Object.assign(new Error(detail), { status: res.status });
  }
  return res.json();
}

async function compareModelsSequential(file, modelIds, conf) {
  const ordered = orderedCompareIds(modelIds);
  const models = [];
  let imageWidth = null;
  let imageHeight = null;

  for (let index = 0; index < ordered.length; index += 1) {
    const id = ordered[index];
    const label = cleanModelName(catalogEntry(id) || { id });
    setLoading(
      true,
      IS_RENDER_HOST
        ? `Cloud compare ${index + 1}/${ordered.length}: ${label} (may take 1–2 min each)…`
        : `Comparing ${index + 1}/${ordered.length}: ${label}…`
    );
    try {
      const data = await postPredict(file, id, conf);
      imageWidth = data.image_width;
      imageHeight = data.image_height;
      models.push(predictResponseToCompareItem(data));
    } catch (err) {
      models.push(predictResponseToCompareItem(null, inferenceErrorMessage(err, err.status)));
      if (IS_RENDER_HOST && (err.status === 502 || err.status === 503)) {
        break;
      }
    }
  }

  return { models, image_width: imageWidth, image_height: imageHeight };
}

async function bootstrapHealth() {
  if (!IS_RENDER_HOST) {
    await checkHealth();
    return;
  }
  for (let attempt = 0; attempt < 40; attempt += 1) {
    await checkHealth();
    if (apiOnline && modelReady) return;
    await sleep(3000);
  }
}

function availableModels() {
  return modelCatalog.filter((model) => model.inference_available);
}

function renderModelControls() {
  els.modelSelect.innerHTML = "";
  els.modelCheckboxes.innerHTML = "";

  sortModelsForDisplay(modelCatalog).forEach((model) => {
    const option = document.createElement("option");
    option.value = model.id;
    const suffix = model.inference_available ? "" : " (unavailable)";
    option.textContent = `${cleanModelName(model)}${suffix}`;
    option.disabled = !model.inference_available;
    if (model.id === defaultModelId) option.selected = true;
    els.modelSelect.appendChild(option);

    const label = document.createElement("label");
    label.className = "model-checkbox";
    const input = document.createElement("input");
    input.type = "checkbox";
    input.value = model.id;
    input.checked = model.inference_available;
    input.disabled = !model.inference_available;
    const text = document.createElement("span");
    text.innerHTML = `<strong>${displayModelName(model)}</strong> ${modelTagsHtml(model)}`;
    label.appendChild(input);
    label.appendChild(text);
    if (model.id === "yolov8s" && hasBenchmark(model)) {
      const note = document.createElement("small");
      note.textContent = "Fastest inference (~69 FPS) · compare with YOLOv5s / YOLO11s";
      label.appendChild(note);
    }
    els.modelCheckboxes.appendChild(label);
  });
}

function renderBenchmarkTable() {
  els.benchmarkBody.innerHTML = "";
  const tested = sortModelsForDisplay(modelCatalog).filter(hasBenchmark);
  const speedster = fastestModel(tested);

  sortModelsForDisplay(modelCatalog).forEach((model) => {
    const bench = effectiveBenchmark(model);
    const testedModel = hasBenchmark(model);
    const row = document.createElement("tr");
    if (model.id === "yolov5") row.className = "winner-row";
    else if (speedster && modelId(speedster) === modelId(model)) row.className = "speed-row";
    const fpsCell =
      testedModel && speedster && modelId(speedster) === modelId(model)
        ? `<strong class="speed-highlight">${formatFps(bench.fps)}</strong> <span class="speed-badge">Fastest</span>`
        : testedModel
          ? formatFps(bench.fps)
          : "—";
    row.innerHTML = `
      <td>
        <strong>${cleanModelName(model)}</strong>
        ${modelRunLabel(model) ? `<div class="model-run-label">${modelRunLabel(model)}</div>` : ""}
      </td>
      <td>${statusLabel(model)}${speedster && modelId(speedster) === modelId(model) ? ' <span class="speed-badge">Fastest</span>' : ""}</td>
      <td>${testedModel ? formatPct(bench.map50) : "—"}</td>
      <td>${testedModel ? formatPct(bench.cylinder_recall) : "—"}</td>
      <td>${fpsCell}</td>
    `;
    els.benchmarkBody.appendChild(row);
  });
}

async function loadModelCatalog() {
  try {
    const res = await fetch(`${API_BASE}/models`);
    const data = await res.json();
    if (!res.ok) throw new Error("Failed to load model catalog");
    defaultModelId = data.default_model_id || "yolov5";
    modelCatalog = (data.models || []).map(normalizeModel);
    els.benchmarkRecommendation.textContent = data.recommendation || "";
    renderModelControls();
    renderBenchmarkTable();
    renderSelectionBanner();
  } catch (err) {
    els.benchmarkBody.innerHTML =
      `<tr class="empty-row"><td colspan="5">Could not load model catalog: ${err.message}</td></tr>`;
  }
}

async function checkHealth() {
  try {
    const res = await fetch(`${API_BASE}/health`);
    const data = await res.json();
    apiOnline = res.ok;
    modelReady = Boolean(res.ok && data.model_loaded);
    const warming = Boolean(data.warming);
    els.apiStatus.classList.toggle("online", apiOnline);
    els.apiStatus.classList.toggle("offline", !apiOnline);
    const statusLabel = els.apiStatus.querySelector("span:last-child");
    if (!res.ok) {
      statusLabel.textContent = "API unreachable";
    } else if (data.warmup_error) {
      statusLabel.textContent = `Warmup issue · ${data.warmup_error}`;
    } else if (warming) {
      statusLabel.textContent = "Warming default model…";
    } else if (modelReady) {
      statusLabel.textContent = `API online · ${data.device} · ${data.models_available} models`;
    } else if (IS_RENDER_HOST) {
      statusLabel.textContent = "API online · models load on first run";
    } else {
      statusLabel.textContent = "API online · models load on demand";
    }
    els.analyzeUpload.disabled = !apiOnline || (IS_RENDER_HOST && analysisMode === "compare" && !modelReady);
    els.analyzeUpload.title = apiOnline
      ? ""
      : IS_RENDER_HOST
        ? "Server may take up to 60s to wake up on Render."
        : "Wait for the API to come online.";
  } catch {
    apiOnline = false;
    modelReady = false;
    els.apiStatus.classList.add("offline");
    els.apiStatus.classList.remove("online");
    const statusLabel = els.apiStatus.querySelector("span:last-child");
    statusLabel.textContent = IS_RENDER_HOST
      ? "Waking up server (up to 60s)…"
      : "API unreachable";
    els.analyzeUpload.disabled = true;
  }
}

function setAnalysisMode(mode) {
  analysisMode = mode;
  document.querySelectorAll(".mode-btn").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.mode === mode);
  });
  els.singleModelSettings.classList.toggle("hidden", mode !== "single");
  els.compareModelSettings.classList.toggle("hidden", mode !== "compare");
  els.singleResults.classList.toggle("hidden", mode !== "single");
  els.compareResults.classList.toggle("hidden", mode !== "compare");
  els.benchmarkPanel.classList.toggle("hidden", mode === "compare");
  els.resultsTitle.textContent = mode === "compare" ? "Model Comparison" : "Detection Results";
  els.analyzeUpload.textContent = mode === "compare" ? "Compare Models" : "Run Analysis";
  if (mode === "compare" && selectedFile) {
    setCompareInputPreview(selectedFile);
  }
  if (mode === "compare" && !lastCompareModels.some((model) => model.annotated_image_base64)) {
    els.compareThreeWrap.classList.remove("hidden");
    els.compareThreeGrid.innerHTML =
      `<div class="placeholder compare-placeholder"><p>Upload an image and click <strong>Compare Models</strong> to see YOLOv5s, YOLO11s, and YOLOv8s side by side.</p></div>`;
  }
  els.analyzeUpload.disabled = !apiOnline || (IS_RENDER_HOST && mode === "compare" && !modelReady);
}

function renderSingleResults(data) {
  const hasHazard = data.hazard_detected;

  els.hazardAlert.classList.toggle("hidden", !hasHazard);
  els.safeAlert.classList.toggle("hidden", hasHazard);

  els.hazardSummary.innerHTML = "";
  if (hasHazard) {
    data.hazard_summary.forEach((item) => {
      const li = document.createElement("li");
      li.textContent = item;
      els.hazardSummary.appendChild(li);
    });
  }

  els.metricModel.textContent = data.model_name || "—";
  els.metricInference.textContent = `${data.inference_ms.toFixed(1)} ms`;
  els.metricObjects.textContent = String(data.detections.length);

  if (data.annotated_image_base64) {
    els.imageStage.innerHTML = `<img src="${data.annotated_image_base64}" alt="Annotated detection result" />`;
  }

  els.detectionsBody.innerHTML = "";
  if (!data.detections.length) {
    els.detectionsBody.innerHTML =
      '<tr class="empty-row"><td colspan="4">No detections above threshold.</td></tr>';
    return;
  }

  data.detections.forEach((det) => {
    const row = document.createElement("tr");
    row.innerHTML = `
      <td>${formatClassName(det.class_name)}</td>
      <td>${(det.confidence * 100).toFixed(1)}%</td>
      <td>${hazardBadge(det.hazard_type)}</td>
      <td>${det.risk_description}</td>
    `;
    els.detectionsBody.appendChild(row);
  });
}

function revokeCompareInputUrl() {
  if (compareInputUrl) {
    URL.revokeObjectURL(compareInputUrl);
    compareInputUrl = null;
  }
}

function setCompareInputPreview(file) {
  revokeCompareInputUrl();
  if (file) {
    compareInputUrl = URL.createObjectURL(file);
  }
}

function openLightbox(src, caption) {
  if (!src) return;
  els.lightboxImage.src = src;
  els.lightboxCaption.textContent = caption || "";
  els.imageLightbox.classList.remove("hidden");
}

function closeLightbox() {
  els.imageLightbox.classList.add("hidden");
  els.lightboxImage.removeAttribute("src");
  els.lightboxCaption.textContent = "";
}

function compareImageHtml(src, alt) {
  if (!src) {
    return `<div class="compare-image-empty"><p>No annotated output returned.</p></div>`;
  }
  return `
    <button class="compare-image-button" type="button" aria-label="Expand ${alt}">
      <img src="${src}" alt="${alt}" loading="lazy" />
      <span class="compare-image-zoom">Click to enlarge</span>
    </button>
  `;
}

function bindCompareImageButtons(container) {
  container.querySelectorAll(".compare-image-button").forEach((button) => {
    const img = button.querySelector("img");
    button.addEventListener("click", () => {
      openLightbox(img?.src, img?.alt || "");
    });
  });
}

function renderCompareOriginal(originalUrl) {
  if (!originalUrl) {
    els.compareStripWrap.classList.add("hidden");
    els.compareOriginalStage.innerHTML = "";
    return;
  }

  els.compareStripWrap.classList.remove("hidden");
  els.compareOriginalStage.innerHTML = compareImageHtml(originalUrl, "Original uploaded image");
  bindCompareImageButtons(els.compareOriginalStage);
}

function renderThreeModelComparison(models) {
  const ordered = MODEL_DISPLAY_ORDER.map((id) =>
    models.find((model) => modelId(model) === id) || { model_id: id, inference_available: false }
  );

  lastCompareModels = ordered;
  const hasAnyImage = ordered.some((model) => model.annotated_image_base64);
  if (!hasAnyImage) {
    els.compareThreeWrap.classList.add("hidden");
    els.compareDownloadBtn.classList.add("hidden");
    els.compareThreeGrid.innerHTML = `<div class="placeholder"><p>Run comparison to see all three model outputs.</p></div>`;
    return;
  }

  els.compareThreeWrap.classList.remove("hidden");
  els.compareDownloadBtn.classList.remove("hidden");

  els.compareThreeGrid.innerHTML = ordered
    .map((model) => {
      const id = modelId(model);
      const panelClass =
        id === "yolov5" ? "winner" : id === "yolov8s" ? "speed" : id === "yolo11s" ? "challenger" : "";
      const run = modelRunLabel(model);
      const status = model.error
        ? "Inference error"
        : !model.inference_available
          ? "Unavailable"
          : model.hazard_detected
            ? "Hazard detected"
            : "No hazard above threshold";

      return `
        <article class="compare-three-panel ${panelClass}" data-model-id="${id}">
          <div class="compare-three-panel-head">
            <div>
              <strong>${cleanModelName(model)}</strong>
              ${run ? `<span class="model-run-label">${run}</span>` : ""}
            </div>
            <span class="compare-three-status">${status}</span>
          </div>
          <div class="compare-three-image">
            ${
              model.annotated_image_base64
                ? compareImageHtml(model.annotated_image_base64, `${cleanModelName(model)} output`)
                : `<div class="compare-image-empty"><p>${model.error || "Model output unavailable"}</p></div>`
            }
          </div>
          <div class="compare-three-foot">
            <span>${model.inference_ms ? `${model.inference_ms.toFixed(1)} ms` : "—"}</span>
            <span>${model.detections?.length ?? 0} object(s)</span>
          </div>
        </article>
      `;
    })
    .join("");

  bindCompareImageButtons(els.compareThreeGrid);
}

function renderCompareMetricsTable(models) {
  const ordered = MODEL_DISPLAY_ORDER.map((id) =>
    models.find((model) => modelId(model) === id)
  ).filter(Boolean);

  if (!ordered.length) {
    els.compareMetricsWrap.classList.add("hidden");
    els.compareMetricsBody.innerHTML = "";
    return;
  }

  els.compareMetricsWrap.classList.remove("hidden");
  els.compareMetricsBody.innerHTML = ordered
    .map((model) => {
      const detections = (model.detections || [])
        .map((det) => `${formatClassName(det.class_name)} ${(det.confidence * 100).toFixed(0)}%`)
        .join(", ");
      return `
        <tr>
          <td><strong>${cleanModelName(model)}</strong></td>
          <td>${model.hazard_detected ? "Yes" : "No"}</td>
          <td>${model.detections?.length ?? 0}</td>
          <td>${model.inference_ms ? `${model.inference_ms.toFixed(1)} ms` : "—"}</td>
          <td>${detections || "—"}</td>
        </tr>
      `;
    })
    .join("");
}

async function downloadComparisonImage() {
  const panels = MODEL_DISPLAY_ORDER.map((id) => {
    const model = lastCompareModels.find((item) => modelId(item) === id);
    return { id, src: model?.annotated_image_base64 || null, label: cleanModelName(model || { name: id }) };
  }).filter((panel) => panel.src);

  if (panels.length < 2) {
    alert("Need at least two model images to build a comparison export.");
    return;
  }

  const images = await Promise.all(
    panels.map(
      (panel) =>
        new Promise((resolve, reject) => {
          const img = new Image();
          img.onload = () => resolve({ ...panel, img });
          img.onerror = () => reject(new Error(`Failed to load ${panel.label}`));
          img.src = panel.src;
        })
    )
  );

  const targetHeight = 720;
  const headerHeight = 42;
  const scaled = images.map(({ img, label }) => {
    const scale = targetHeight / img.naturalHeight;
    const width = Math.round(img.naturalWidth * scale);
    return { img, label, width, height: targetHeight };
  });

  const gap = 16;
  const padding = 20;
  const canvas = document.createElement("canvas");
  canvas.width = padding * 2 + scaled.reduce((sum, item) => sum + item.width, 0) + gap * (scaled.length - 1);
  canvas.height = padding * 2 + headerHeight + targetHeight;
  const ctx = canvas.getContext("2d");

  ctx.fillStyle = "#0f1419";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  let x = padding;
  scaled.forEach((item) => {
    ctx.fillStyle = "#94a3b8";
    ctx.font = "bold 18px Segoe UI, sans-serif";
    ctx.fillText(item.label, x, padding + 24);
    ctx.drawImage(item.img, x, padding + headerHeight, item.width, item.height);
    x += item.width + gap;
  });

  const link = document.createElement("a");
  link.download = "hazard_waste_three_model_comparison.png";
  link.href = canvas.toDataURL("image/png");
  link.click();
}

function renderCompareResults(data) {
  const models = [...(data.models || [])].sort(
    (a, b) => modelSortIndex(a) - modelSortIndex(b)
  );

  renderCompareOriginal(compareInputUrl);
  renderThreeModelComparison(models);
  renderCompareMetricsTable(models);
  els.compareGrid.innerHTML = "";
  els.compareGrid.classList.add("hidden");
}

async function predictImage(file) {
  if (isAnalyzing) return;
  if (!apiOnline) {
    alert(
      IS_RENDER_HOST
        ? "Server is waking up. Wait until the status turns green, then try again."
        : "API is offline. Wait until the status shows online, then try again."
    );
    return;
  }
  if (IS_RENDER_HOST && analysisMode === "compare" && !modelReady) {
    alert("Default model is still loading on Render. Wait until the status shows the API is online with models loaded.");
    return;
  }
  if (IS_RENDER_HOST && analysisMode === "compare") {
    const proceed = window.confirm(
      "On Render, each model runs as a separate request (~1–2 min per model on cloud CPU). Continue?"
    );
    if (!proceed) return;
  }
  isAnalyzing = true;

  const conf = parseFloat(els.confThreshold.value);
  const formData = new FormData();
  formData.append("file", file, file.name || "capture.jpg");

  try {
    if (analysisMode === "compare") {
      let modelIds = selectedCompareIds();
      if (!modelIds.length) {
        alert("Select at least one model for comparison.");
        return;
      }
      const compareIds =
        modelIds.length === MODEL_DISPLAY_ORDER.length
          ? MODEL_DISPLAY_ORDER.join(",")
          : modelIds.join(",");
      setCompareInputPreview(file);

      if (IS_RENDER_HOST) {
        const data = await compareModelsSequential(file, modelIds, conf);
        lastCompareModels = data.models;
        renderCompareResults(data);
      } else {
        setLoading(true, "Comparing all three models on the same image…");
        const res = await fetch(
          `${API_BASE}/predict/compare?include_annotated_image=true&conf_threshold=${conf}&model_ids=${compareIds}`,
          { method: "POST", body: formData }
        );
        if (!res.ok) {
          const err = await res.json().catch(() => ({}));
          throw Object.assign(
            new Error(err.detail || `Request failed (${res.status})`),
            { status: res.status }
          );
        }
        const data = await res.json();
        lastCompareModels = data.models || [];
        renderCompareResults(data);
      }
    } else {
      const selectedModelId = els.modelSelect.value;
      setLoading(
        true,
        IS_RENDER_HOST && !modelReady
          ? "Loading model on cloud CPU (first run ~30–60s)…"
          : "Running inference…"
      );
      const res = await fetch(
        `${API_BASE}/predict?include_annotated_image=true&conf_threshold=${conf}&model_id=${selectedModelId}`,
        { method: "POST", body: formData }
      );
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw Object.assign(
          new Error(err.detail || `Request failed (${res.status})`),
          { status: res.status }
        );
      }
      const data = await res.json();
      renderSingleResults(data);
    }
  } catch (err) {
    alert(`Inference failed: ${inferenceErrorMessage(err, err.status)}`);
  } finally {
    isAnalyzing = false;
    setLoading(false);
  }
}

function setupModeToggle() {
  document.querySelectorAll(".mode-btn").forEach((btn) => {
    btn.addEventListener("click", () => setAnalysisMode(btn.dataset.mode));
  });
}

function setupTabs() {
  document.querySelectorAll(".tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach((t) => t.classList.remove("active"));
      document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
      tab.classList.add("active");
      document.getElementById(`panel-${tab.dataset.tab}`).classList.add("active");
      if (tab.dataset.tab !== "webcam") {
        stopLiveScan();
      }
    });
  });
}

function setupUpload() {
  els.dropzone.addEventListener("click", () => els.fileInput.click());

  els.dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    els.dropzone.classList.add("dragover");
  });

  els.dropzone.addEventListener("dragleave", () => {
    els.dropzone.classList.remove("dragover");
  });

  els.dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    els.dropzone.classList.remove("dragover");
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith("image/")) {
      selectedFile = file;
      els.analyzeUpload.disabled = false;
      els.dropzone.querySelector(".dropzone-title").textContent = file.name;
      if (analysisMode === "compare") {
        setCompareInputPreview(file);
      }
    }
  });

  els.fileInput.addEventListener("change", () => {
    const file = els.fileInput.files[0];
    if (file) {
      selectedFile = file;
      els.analyzeUpload.disabled = false;
      els.dropzone.querySelector(".dropzone-title").textContent = file.name;
      if (analysisMode === "compare") {
        setCompareInputPreview(file);
      }
    }
  });

  els.analyzeUpload.addEventListener("click", () => {
    if (selectedFile) predictImage(selectedFile);
  });
}

async function startWebcam() {
  try {
    webcamStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "environment", width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false,
    });
    els.webcamVideo.srcObject = webcamStream;
    els.webcamVideo.parentElement.classList.add("active");
    els.captureWebcam.disabled = false;
    els.startWebcam.textContent = "Restart Camera";
  } catch (err) {
    alert(`Camera access failed: ${err.message}`);
  }
}

function captureWebcamFrame() {
  const video = els.webcamVideo;
  const canvas = els.webcamCanvas;
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(video, 0, 0);
  return new Promise((resolve) => {
    canvas.toBlob((blob) => {
      resolve(new File([blob], "webcam-capture.jpg", { type: "image/jpeg" }));
    }, "image/jpeg", 0.92);
  });
}

function stopLiveScan() {
  if (liveScanTimer) {
    clearInterval(liveScanTimer);
    liveScanTimer = null;
  }
  els.liveScan.checked = false;
}

function setupWebcam() {
  els.startWebcam.addEventListener("click", startWebcam);

  els.captureWebcam.addEventListener("click", async () => {
    const file = await captureWebcamFrame();
    await predictImage(file);
  });

  els.liveScan.addEventListener("change", async () => {
    if (els.liveScan.checked) {
      if (!webcamStream) {
        await startWebcam();
      }
      liveScanTimer = setInterval(async () => {
        if (!webcamStream || isAnalyzing) return;
        const file = await captureWebcamFrame();
        await predictImage(file);
      }, 3000);
    } else {
      stopLiveScan();
    }
  });
}

els.confThreshold.addEventListener("input", () => {
  els.confValue.textContent = parseFloat(els.confThreshold.value).toFixed(2);
});

setupModeToggle();
setupTabs();
setupUpload();
setupWebcam();
setupLightbox();
loadModelCatalog();
bootstrapHealth();
setInterval(checkHealth, 5000);

function setupLightbox() {
  els.lightboxClose.addEventListener("click", closeLightbox);
  els.imageLightbox.addEventListener("click", (event) => {
    if (event.target === els.imageLightbox) closeLightbox();
  });
  els.compareDownloadBtn.addEventListener("click", () => {
    downloadComparisonImage().catch((err) => alert(`Could not export comparison: ${err.message}`));
  });
  window.addEventListener("keydown", (event) => {
    if (event.key === "Escape") closeLightbox();
  });
}

window.addEventListener("beforeunload", () => {
  stopLiveScan();
  revokeCompareInputUrl();
  if (webcamStream) {
    webcamStream.getTracks().forEach((t) => t.stop());
  }
});
