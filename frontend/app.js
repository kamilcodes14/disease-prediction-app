(() => {
  const API_BASE = window.API_BASE; // still used for the photo-screening endpoint only

  /* ----------------------------------------------------------------
     Field metadata for the two hand-curated datasets (labels, units,
     input widgets). Breast Cancer has 30 numeric features and is
     rendered generically further down.
  ---------------------------------------------------------------- */
  const HEART_FIELDS = [
    { key: "age", label: "Age", unit: "years", type: "number", min: 1, max: 120, step: 1, default: 63 },
    { key: "sex", label: "Sex", type: "select", options: [[1, "Male"], [0, "Female"]], default: 1 },
    { key: "cp", label: "Chest pain type", type: "select", options: [[0, "Typical angina"], [1, "Atypical angina"], [2, "Non-anginal pain"], [3, "Asymptomatic"]], default: 3 },
    { key: "trestbps", label: "Resting blood pressure", unit: "mm Hg", type: "number", min: 80, max: 220, step: 1, default: 145 },
    { key: "chol", label: "Serum cholesterol", unit: "mg/dl", type: "number", min: 100, max: 600, step: 1, default: 233 },
    { key: "fbs", label: "Fasting blood sugar > 120 mg/dl", type: "select", options: [[1, "Yes"], [0, "No"]], default: 1 },
    { key: "restecg", label: "Resting ECG result", type: "select", options: [[0, "Normal"], [1, "ST-T abnormality"], [2, "LV hypertrophy"]], default: 0 },
    { key: "thalach", label: "Max heart rate achieved", unit: "bpm", type: "number", min: 60, max: 220, step: 1, default: 150 },
    { key: "exang", label: "Exercise-induced angina", type: "select", options: [[1, "Yes"], [0, "No"]], default: 0 },
    { key: "oldpeak", label: "ST depression (exercise)", type: "number", min: 0, max: 10, step: 0.1, default: 2.3 },
    { key: "slope", label: "ST segment slope", type: "select", options: [[0, "Upsloping"], [1, "Flat"], [2, "Downsloping"]], default: 0 },
    { key: "ca", label: "Major vessels colored (fluoroscopy)", type: "select", options: [[0, "0"], [1, "1"], [2, "2"], [3, "3"], [4, "4"]], default: 0 },
    { key: "thal", label: "Thalassemia test result", type: "select", options: [[0, "Category 0"], [1, "Category 1"], [2, "Category 2"], [3, "Category 3"]], default: 1 },
  ];

  const DIABETES_FIELDS = [
    { key: "Pregnancies", label: "Pregnancies", unit: "count", type: "number", min: 0, max: 20, step: 1, default: 6 },
    { key: "Glucose", label: "Plasma glucose", unit: "mg/dl", type: "number", min: 0, max: 300, step: 1, default: 148 },
    { key: "BloodPressure", label: "Diastolic blood pressure", unit: "mm Hg", type: "number", min: 0, max: 200, step: 1, default: 72 },
    { key: "SkinThickness", label: "Triceps skinfold thickness", unit: "mm", type: "number", min: 0, max: 100, step: 1, default: 35 },
    { key: "Insulin", label: "2-hr serum insulin", unit: "mu U/ml", type: "number", min: 0, max: 900, step: 1, default: 0 },
    { key: "BMI", label: "Body mass index", unit: "kg/m²", type: "number", min: 0, max: 70, step: 0.1, default: 33.6 },
    { key: "DiabetesPedigreeFunction", label: "Diabetes pedigree function", type: "number", min: 0, max: 3, step: 0.001, default: 0.627 },
    { key: "Age", label: "Age", unit: "years", type: "number", min: 1, max: 120, step: 1, default: 50 },
  ];

  const BREAST_CANCER_DEFAULTS = {
    "mean radius": 17.99, "mean texture": 10.38, "mean perimeter": 122.8, "mean area": 1001.0,
    "mean smoothness": 0.118, "mean compactness": 0.278, "mean concavity": 0.3, "mean concave points": 0.147,
    "mean symmetry": 0.242, "mean fractal dimension": 0.079, "radius error": 1.095, "texture error": 0.905,
    "perimeter error": 8.589, "area error": 153.4, "smoothness error": 0.006, "compactness error": 0.049,
    "concavity error": 0.054, "concave points error": 0.016, "symmetry error": 0.03, "fractal dimension error": 0.006,
    "worst radius": 25.38, "worst texture": 17.33, "worst perimeter": 184.6, "worst area": 2019.0,
    "worst smoothness": 0.162, "worst compactness": 0.666, "worst concavity": 0.712, "worst concave points": 0.265,
    "worst symmetry": 0.46, "worst fractal dimension": 0.119,
  };

  const DATASET_CONFIG = {
    heart_disease: { label: "Heart Disease", fields: HEART_FIELDS },
    diabetes: { label: "Diabetes", fields: DIABETES_FIELDS },
    breast_cancer: {
      label: "Breast Cancer",
      fields: Object.entries(BREAST_CANCER_DEFAULTS).map(([key, def]) => ({
        key,
        label: key.replace(/\b\w/g, (c) => c.toUpperCase()),
        type: "number",
        step: 0.001,
        default: def,
      })),
    },
  };

  // Every model now runs fully client-side via ONNX Runtime Web.
  // No backend needed for structured predictions — just static files.
  const MODEL_FILES = {
    heart_disease: "models/heart_disease.onnx",
    diabetes: "models/diabetes.onnx",
    breast_cancer: "models/breast_cancer.onnx",
  };
  const METADATA_FILES = {
    heart_disease: "models/heart_disease_metadata.json",
    diabetes: "models/diabetes_metadata.json",
    breast_cancer: "models/breast_cancer_metadata.json",
  };

  let schema = null; // populated from the local metadata JSON files
  const sessionCache = {}; // dataset key -> ort.InferenceSession (loaded once, reused)

  /* ---------------------------------------------------------------- */
  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

  function riskClass(level) {
    return { low: "risk-low", moderate: "risk-moderate", high: "risk-high" }[level] || "risk-low";
  }

  function errorReadout(container, message) {
    container.innerHTML = `
      <div class="readout-empty">
        <p style="color:#F0897A;">${message}</p>
      </div>`;
  }

  /* ---------------------------------------------------------------- */
  async function loadSchema() {
    const entries = await Promise.all(
      Object.entries(METADATA_FILES).map(async ([key, path]) => {
        const res = await fetch(path);
        if (!res.ok) throw new Error(`couldn't load ${path}`);
        return [key, await res.json()];
      })
    );
    schema = Object.fromEntries(entries);

    const select = $("#dataset-select");
    select.innerHTML = Object.keys(DATASET_CONFIG)
      .map((key) => `<option value="${key}">${DATASET_CONFIG[key].label}</option>`)
      .join("");
    renderStructuredForm(select.value);

    const cardsHost = $("#dataset-cards");
    cardsHost.innerHTML = Object.entries(schema)
      .map(([key, meta]) => {
        const auc = meta.leaderboard[meta.best_model].roc_auc;
        return `<div class="dataset-card">
          <span class="name">${meta.display_name}</span>
          <span class="metric">${meta.best_model.replace(/_/g, " ")} · AUC ${auc}</span>
        </div>`;
      })
      .join("");
  }

  // Loads (and caches) the ONNX session for a dataset. Each model is
  // ~1KB–950KB — trivial to fetch, no backend round-trip involved.
  async function getSession(datasetKey) {
    if (sessionCache[datasetKey]) return sessionCache[datasetKey];
    const session = await ort.InferenceSession.create(MODEL_FILES[datasetKey]);
    sessionCache[datasetKey] = session;
    return session;
  }

  // Runs prediction fully in the browser — this replaces the old
  // POST /api/predict/structured call. Same response shape as before,
  // so renderStructuredResult() below needs zero changes.
  async function predictStructured(datasetKey, features) {
    const meta = schema[datasetKey];
    const order = meta.feature_names;
    const missing = order.filter((f) => !(f in features));
    if (missing.length) throw new Error(`missing features: ${missing.join(", ")}`);

    const input = Float32Array.from(order.map((f) => features[f]));
    const session = await getSession(datasetKey);
    const tensor = new ort.Tensor("float32", input, [1, order.length]);
    const results = await session.run({ input: tensor });

    // Pipeline was exported with zipmap=False, so "probabilities" is a
    // flat [1, 2] float tensor: [P(class 0), P(class 1)].
    const probs = results.probabilities.data;
    const proba = probs[1];
    const prediction = proba >= 0.5 ? 1 : 0;

    return {
      dataset: datasetKey,
      display_name: meta.display_name,
      positive_label: meta.positive_label,
      prediction,
      probability: Math.round(proba * 10000) / 10000,
      risk_level: proba >= 0.66 ? "high" : proba >= 0.33 ? "moderate" : "low",
      model_used: meta.best_model,
    };
  }

  function renderStructuredForm(datasetKey) {
    const cfg = DATASET_CONFIG[datasetKey];
    const form = $("#structured-form");
    form.innerHTML = cfg.fields
      .map((f) => {
        if (f.type === "select") {
          const opts = f.options
            .map(([v, l]) => `<option value="${v}" ${v === f.default ? "selected" : ""}>${l}</option>`)
            .join("");
          return `<div class="form-field">
            <label for="f-${f.key}">${f.label}</label>
            <select id="f-${f.key}" class="select" data-key="${f.key}" style="margin-bottom:0;">${opts}</select>
          </div>`;
        }
        return `<div class="form-field">
          <label for="f-${f.key}">${f.label}${f.unit ? ` <span class="unit">(${f.unit})</span>` : ""}</label>
          <input id="f-${f.key}" class="input" style="margin-bottom:0;" type="number"
            data-key="${f.key}" value="${f.default}"
            ${f.min !== undefined ? `min="${f.min}"` : ""} ${f.max !== undefined ? `max="${f.max}"` : ""} ${f.step !== undefined ? `step="${f.step}"` : ""} />
        </div>`;
      })
      .join("");
  }

  function renderStructuredResult(data) {
    const container = $("#structured-readout");
    const pct = Math.round(data.probability * 100);
    container.innerHTML = `
      <div class="readout-result">
        <div class="readout-top"><span>${data.display_name}</span><span>${data.model_used.replace(/_/g, " ")}</span></div>
        <div class="readout-value">${pct}%</div>
        <div class="readout-label">estimated probability — ${data.positive_label}</div>
        <div class="risk-tag ${riskClass(data.risk_level)}"><span class="dot"></span>${data.risk_level} risk</div>
        <div class="readout-bar-track"><div class="readout-bar-fill" style="width:${pct}%"></div></div>
        <div class="readout-meta">
          <div>Prediction <b>${data.prediction ? data.positive_label : "Not " + data.positive_label.toLowerCase()}</b></div>
          <div>Model <b>${data.model_used.replace(/_/g, " ")}</b></div>
        </div>
      </div>`;
  }

  function renderImageResult(data) {
    const container = $("#image-readout");
    const pct = Math.round(data.probability * 100);
    const metaRows = Object.entries(data.feature_report)
      .slice(0, 6)
      .map(([k, v]) => `<div>${k.replace(/_/g, " ")} <b>${v}</b></div>`)
      .join("");
    container.innerHTML = `
      <div class="readout-result">
        <div class="readout-top"><span>Photo screening</span><span>random forest</span></div>
        <div class="readout-value">${pct}%</div>
        <div class="readout-label">${data.label}</div>
        <div class="risk-tag ${riskClass(data.risk_level)}"><span class="dot"></span>${data.risk_level} risk</div>
        <div class="readout-bar-track"><div class="readout-bar-fill" style="width:${pct}%"></div></div>
        <div class="readout-meta">${metaRows}</div>
      </div>`;
  }

  /* ---------------------------------------------------------------- */
  function wireModeToggle() {
    $$(".mode-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        $$(".mode-btn").forEach((b) => { b.classList.remove("is-active"); b.setAttribute("aria-selected", "false"); });
        btn.classList.add("is-active");
        btn.setAttribute("aria-selected", "true");
        const mode = btn.dataset.mode;
        $$(".mode-pane").forEach((p) => p.classList.toggle("is-active", p.dataset.pane === mode));
      });
    });
  }

  function wireStructuredMode() {
    $("#dataset-select").addEventListener("change", (e) => renderStructuredForm(e.target.value));

    $("#structured-submit").addEventListener("click", async () => {
      const datasetKey = $("#dataset-select").value;
      const features = {};
      $$("#structured-form [data-key]").forEach((el) => {
        features[el.dataset.key] = parseFloat(el.value);
      });

      const readout = $("#structured-readout");
      readout.innerHTML = `<div class="readout-empty"><p>Running prediction…</p></div>`;
      try {
        const data = await predictStructured(datasetKey, features);
        renderStructuredResult(data);
      } catch (err) {
        errorReadout(readout, `Prediction failed: ${err.message}`);
        console.error(err);
      }
    });
  }

  function wireImageMode() {
    const dropzone = $("#dropzone");
    const input = $("#image-input");
    const preview = $("#image-preview");
    const submitBtn = $("#image-submit");
    let currentFile = null;

    function setFile(file) {
      if (!file || !file.type.startsWith("image/")) return;
      currentFile = file;
      preview.src = URL.createObjectURL(file);
      preview.hidden = false;
      submitBtn.disabled = false;
    }

    dropzone.addEventListener("click", () => input.click());
    dropzone.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") input.click(); });
    dropzone.setAttribute("tabindex", "0");
    input.addEventListener("change", (e) => setFile(e.target.files[0]));

    ["dragover", "dragenter"].forEach((evt) =>
      dropzone.addEventListener(evt, (e) => { e.preventDefault(); dropzone.classList.add("is-dragover"); })
    );
    ["dragleave", "drop"].forEach((evt) =>
      dropzone.addEventListener(evt, (e) => { e.preventDefault(); dropzone.classList.remove("is-dragover"); })
    );
    dropzone.addEventListener("drop", (e) => setFile(e.dataTransfer.files[0]));

    submitBtn.addEventListener("click", async () => {
      if (!currentFile) return;
      const readout = $("#image-readout");

      // Photo screening still needs a backend (scikit-image feature
      // extraction isn't ported to JS yet). If no API_BASE is set,
      // tell the user plainly instead of failing silently.
           readout.innerHTML = `<div class="readout-empty"><p>Analyzing photo…</p></div>`;
      const formData = new FormData();
      formData.append("file", currentFile);
      try {
        const res = await fetch(`${API_BASE}/api/predict/image`, { method: "POST", body: formData });
        if (!res.ok) throw new Error((await res.json()).detail || "analysis failed");
        renderImageResult(await res.json());
      } catch (err) {
        errorReadout(readout, `Couldn't reach the API — is the backend running at ${API_BASE}? (${err.message})`);
      }
    });
  }

  /* ---------------------------------------------------------------- */
  wireModeToggle();
  wireStructuredMode();
  wireImageMode();
  loadSchema().catch((err) => {
    errorReadout($("#structured-readout"), `Couldn't load model files. Make sure the models/ folder was uploaded alongside app.js.`);
    console.error(err);
  });
})();
