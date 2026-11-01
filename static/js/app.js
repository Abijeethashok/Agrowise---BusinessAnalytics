/* AgroWise front-end logic (Step 6)
   - builds the recommendation form from /api/meta
   - validates input, calls /api/recommend, renders the results
   - fills overview numbers on the analytics and about pages
   All text coming from the server is escaped before it is inserted. */
(function () {
  "use strict";

  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
  const esc = (v) =>
    String(v).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  function num(v, decimals) {
    const s = Number(v).toFixed(decimals);
    return decimals === 1 ? s.replace(/\.0$/, "") : s;
  }
  const DECIMALS = { ph: 2 };
  const fmt = (feature, v) => num(v, DECIMALS[feature] !== undefined ? DECIMALS[feature] : 1);

  async function fetchJSON(url, options) {
    const res = await fetch(url, options);
    let data = null;
    try { data = await res.json(); } catch (_) { /* not JSON */ }
    if (!res.ok) {
      const err = new Error((data && data.message) || "Request failed (" + res.status + ")");
      err.status = res.status;
      err.payload = data;
      throw err;
    }
    return data;
  }

  /* ------------------------------------------------------------------ results */
  function meter(frac, cls, label) {
    const pct = Math.max(0, Math.min(1, frac)) * 100;
    return '<div class="meter" role="img" aria-label="' + esc(label) + '"><span class="' + cls +
      '" style="width:' + Math.max(pct, 1.5).toFixed(1) + '%"></span></div>';
  }

  function renderAlerts(r) {
    const out = [];
    (r.cautions || []).forEach((t) => out.push('<div class="alert alert-warning" role="alert"><strong>Caution.</strong> ' + esc(t) + "</div>"));
    (r.warnings || []).forEach((t) => out.push('<div class="alert alert-info" role="alert">' + esc(t) + "</div>"));
    (r.notes || []).forEach((t) => out.push('<div class="alert alert-secondary" role="alert">' + esc(t) + "</div>"));
    return out.join("");
  }

  function renderTop3(r) {
    const cards = r.top3.map((c, i) =>
      '<div class="col-md-4"><article class="rec-card' + (i === 0 ? " rec-card-top" : "") + '">' +
      '<span class="rank">#' + c.rank + "</span>" +
      "<h3>" + esc(c.display) + "</h3>" +
      '<div class="metric"><span>Model support score</span><strong>' + esc(c.model_score_display) + "</strong></div>" +
      meter(c.model_score, "m-model", "Model support score " + c.model_score_display) +
      '<div class="metric"><span>Profile fit</span><strong>' + esc(c.profile_fit_display) + "</strong></div>" +
      meter(c.profile_fit, "m-fit", "Profile fit " + c.profile_fit_display) +
      '<p class="note mt-2 mb-0">' + c.features_inside + " of " + c.features_total +
      " inputs are inside this crop's typical range.</p></article></div>").join("");
    return '<section class="panel"><h2 class="panel-title" id="results-heading" tabindex="-1">Top three crops by model ranking</h2>' +
      '<div class="row g-3">' + cards + "</div>" +
      '<p class="note mt-3 mb-0">' + esc(r.disclaimer) + "</p></section>";
  }

  function renderWhy(r) {
    const a = r.top3[0].display, b = r.top3[1] ? r.top3[1].display : "";
    const rows = (r.distinguishing || []).map((d) =>
      "<tr><td>" + esc(d.label) + "</td><td>" + fmt(d.feature, d.mean_a) + " (" + fmt(d.feature, d.range_a[0]) + " to " + fmt(d.feature, d.range_a[1]) + ")</td>" +
      "<td>" + fmt(d.feature, d.mean_b) + " (" + fmt(d.feature, d.range_b[0]) + " to " + fmt(d.feature, d.range_b[1]) + ")</td></tr>").join("");
    const closest = (r.closest_by_profile || []).map((c) =>
      "<li><strong>" + esc(c.display) + "</strong>: profile fit " + esc(c.profile_fit_display) + ", " + c.features_inside + " of " + c.features_total +
      " inputs in range (model rank #" + c.rank + ")</li>").join("");
    return '<section class="panel"><h2 class="panel-title">Why these results?</h2>' +
      "<p>" + esc(r.explanation) + "</p>" +
      (rows ? '<h3 class="h6 mt-3">Conditions that most separate ' + esc(a) + " from " + esc(b) + " in the dataset</h3>" +
        '<div class="table-responsive"><table class="table table-sm table-agro"><thead><tr><th>Condition</th><th>' + esc(a) +
        " mean (typical range)</th><th>" + esc(b) + " mean (typical range)</th></tr></thead><tbody>" + rows + "</tbody></table></div>" : "") +
      '<h3 class="h6 mt-3">Closest profile matches (independent of the model ranking)</h3><ul class="mb-0">' + closest + "</ul></section>";
  }

  function rangeBar(c) {
    const lo = Math.min(c.min, c.input, c.p05), hi = Math.max(c.max, c.input, c.p95);
    const pad = (hi - lo) * 0.08 || 1;
    const a = lo - pad, span = hi + pad - a;
    const pos = (v) => (((v - a) / span) * 100);
    const w = (x, y) => Math.max(pos(y) - pos(x), 0.8).toFixed(2);
    const inside = c.status === "within typical range";
    return '<div class="rbar" role="img" aria-label="' + esc(c.label + ": your value " + fmt(c.feature, c.input) + " is " + c.status) + '">' +
      '<span class="rb-obs" style="left:' + pos(c.min).toFixed(2) + "%;width:" + w(c.min, c.max) + '%"></span>' +
      '<span class="rb-typ" style="left:' + pos(c.p05).toFixed(2) + "%;width:" + w(c.p05, c.p95) + '%"></span>' +
      '<span class="rb-mean" style="left:' + pos(c.mean).toFixed(2) + '%"></span>' +
      '<span class="rb-in ' + (inside ? "ok" : "out") + '" style="left:' + pos(c.input).toFixed(2) + '%"></span></div>';
  }

  function renderSelected(r) {
    const s = r.selected;
    if (!s) {
      return '<section class="panel"><h2 class="panel-title">Your selected crop</h2><p class="mb-0 text-muted-agro">' +
        "You did not choose a crop, so only recommendations are shown. Pick a crop in the form to see how well your conditions match it.</p></section>";
    }
    const gaps = (s.largest_gaps || []).map((g) =>
      "<li><strong>" + esc(g.label) + "</strong>: your value " + fmt(g.feature, g.input) + " is " + esc(g.status) +
      " (" + fmt(g.feature, g.p05) + " to " + fmt(g.feature, g.p95) + ").</li>").join("");
    const alts = (s.alternatives || []).map((a) => {
      const why = [];
      if (a.better_on_model_rank) why.push("ranked higher by the model (#" + a.rank + ")");
      if (a.better_on_profile_fit) why.push("closer profile fit (" + esc(a.profile_fit_display) + " vs " + esc(s.profile_fit_display) + ")");
      return "<li><strong>" + esc(a.display) + "</strong>: " + why.join("; ") + "</li>";
    }).join("");
    const rows = s.feature_comparison.map((c) =>
      '<div class="cmp-row"><div class="cmp-label">' + esc(c.label) + "</div>" +
      "<div>" + fmt(c.feature, c.input) + "</div>" +
      "<div>" + fmt(c.feature, c.p05) + " to " + fmt(c.feature, c.p95) + "</div>" +
      '<div class="rbar-cell">' + rangeBar(c) + "</div>" +
      '<div><span class="chip ' + (c.status === "within typical range" ? "ok" : "out") + '">' + esc(c.status) + "</span></div></div>").join("");
    return '<section class="panel"><h2 class="panel-title">Your selected crop: ' + esc(s.display) + "</h2>" +
      '<span class="badge-support ' + esc(s.label_key) + '">' + esc(s.label) + "</span>" +
      '<div class="stat-tiles">' +
      '<div class="stat-tile"><strong>#' + s.rank + "</strong><span>Model rank (of " + r.all_crops.length + ")</span></div>" +
      '<div class="stat-tile"><strong>' + esc(s.model_score_display) + "</strong><span>Model support score</span></div>" +
      '<div class="stat-tile"><strong>' + esc(s.profile_fit_display) + "</strong><span>Profile fit (" + s.features_inside + " of " + s.features_total + " in range)</span></div></div>" +
      '<p class="note">This label combines the model ranking and the profile fit. It describes similarity to the dataset records for ' + esc(s.display) +
      ' and is not a guarantee of growth or yield. <a href="/about#assessment">How the label is decided</a>.</p>' +
      (gaps ? '<h3 class="h6 mt-3">Largest differences from the typical range</h3><ul>' + gaps + "</ul>" : '<p class="mt-3 mb-0">All seven values are inside the typical range for ' + esc(s.display) + ".</p>") +
      (alts ? '<h3 class="h6 mt-3">Alternatives that match your inputs better</h3><ul class="alt-list">' + alts + "</ul>" : "") +
      '<h3 class="h6 mt-4">How your inputs compare with ' + esc(s.display) + " records</h3>" +
      '<div class="cmp-head"><div>Condition</div><div>Your input</div><div>Typical range</div><div>Position</div><div>Status</div></div>' + rows +
      '<div class="legend"><span><i class="lg-obs"></i>All records of this crop</span><span><i class="lg-typ"></i>Typical range (5th to 95th percentile)</span>' +
      '<span><i class="lg-mean"></i>Crop mean</span><span><i class="lg-in"></i>Your input</span></div>' +
      '<p class="note mt-2 mb-0">' + esc(s.reference_note) + "</p></section>";
  }

  function renderCompat(r) {
    return '<section class="panel"><h2 class="panel-title">Compatibility across crops</h2><div class="row g-4">' +
      '<div class="col-lg-5"><h3 class="h6">Feature-by-feature fit</h3>' +
      '<div class="chart-box chart-box-sm"><canvas id="chart-fit-radar" role="img" aria-label="Radar chart of how well each input fits the crop\'s typical range"></canvas></div></div>' +
      '<div class="col-lg-7"><h3 class="h6">Profile fit across all crops</h3>' +
      '<div class="chart-box chart-box-tall"><canvas id="chart-fit-all" role="img" aria-label="Bar chart of profile fit for every crop"></canvas></div></div></div>' +
      '<p class="note mt-3 mb-0">Fit is 100% when an input is inside the crop\'s typical range and falls toward 0% the further outside it is. Gold marks your selected crop and dark green the model\'s top-ranked crop. ' +
      "Profile fit compares your inputs with dataset records; it does not predict growth or yield.</p></section>";
  }

  function buildResultHTML(r) {
    return renderAlerts(r) + renderTop3(r) + renderSelected(r) + renderCompat(r) + renderWhy(r);
  }

  /* --------------------------------------------------------- recommendation page */
  let META = null;
  let sampleIndex = 0;

  function clearErrors() {
    $$(".is-invalid").forEach((el) => el.classList.remove("is-invalid"));
    $$(".invalid-feedback").forEach((el) => { el.textContent = ""; });
    const box = $("#form-alert");
    if (box) { box.hidden = true; box.textContent = ""; }
  }

  function showFieldErrors(errors) {
    clearErrors();
    let first = null;
    Object.keys(errors).forEach((key) => {
      const el = key === "selected_crop" ? $("#f-selected_crop") : $("#f-" + key);
      const msg = $("#err-" + key);
      if (el) { el.classList.add("is-invalid"); if (!first) first = el; }
      if (msg) msg.textContent = errors[key];
    });
    if (first) first.focus();
    const box = $("#form-alert");
    if (box && Object.keys(errors).length) {
      box.hidden = false;
      box.textContent = "Please correct the highlighted fields.";
    }
  }

  function validateForm() {
    const errors = {}, values = {};
    META.features.forEach((f) => {
      const raw = $("#f-" + f.key).value.trim();
      if (raw === "") { errors[f.key] = "Enter a number for " + f.label + "."; return; }
      const x = Number(raw);
      if (!Number.isFinite(x)) { errors[f.key] = f.label + " must be a number."; return; }
      if (x < f.hard_min || x > f.hard_max) { errors[f.key] = f.label + " must be between " + f.hard_min + " and " + f.hard_max + "."; return; }
      values[f.key] = x;
    });
    return { errors, values };
  }

  function buildForm() {
    $("#field-grid").innerHTML = META.features.map((f) =>
      '<div class="col-sm-6"><label class="form-label" for="f-' + f.key + '">' + esc(f.label) + "</label>" +
      '<input class="form-control" type="number" inputmode="decimal" id="f-' + f.key + '" name="' + f.key +
      '" step="' + f.step + '" min="' + f.hard_min + '" max="' + f.hard_max + '" required aria-describedby="help-' + f.key + " hint-" + f.key + " err-" + f.key + '">' +
      '<div class="form-text" id="help-' + f.key + '">' + esc(f.help) + "</div>" +
      '<div class="range-note">Observed in dataset: ' + fmt(f.key, f.dataset_min) + " to " + fmt(f.key, f.dataset_max) + "</div>" +
      '<div class="field-hint" id="hint-' + f.key + '" aria-live="polite"></div>' +
      '<div class="invalid-feedback" id="err-' + f.key + '"></div></div>').join("");

    $("#f-selected_crop").innerHTML = '<option value="">No crop selected (recommendations only)</option>' +
      META.crops.map((c) => '<option value="' + esc(c.key) + '">' + esc(c.display) + "</option>").join("");

    META.features.forEach((f) => {
      $("#f-" + f.key).addEventListener("input", (ev) => {
        const hint = $("#hint-" + f.key);
        const x = Number(ev.target.value);
        if (ev.target.value !== "" && Number.isFinite(x) && (x < f.dataset_min || x > f.dataset_max)) {
          hint.textContent = "Outside the dataset's observed range; the model will be extrapolating.";
        } else { hint.textContent = ""; }
      });
    });
  }

  function setLoading(on) {
    const btn = $("#btn-recommend"), res = $("#results");
    btn.disabled = on;
    btn.innerHTML = on ? '<span class="spinner-dot" aria-hidden="true"></span>Analysing…' : "Get recommendation";
    res.setAttribute("aria-busy", on ? "true" : "false");
  }

  function showResultError(message) {
    $("#results").innerHTML = '<div class="alert alert-danger" role="alert"><strong>Could not get a recommendation.</strong> ' + esc(message) + "</div>";
  }

  async function onSubmit(ev) {
    ev.preventDefault();
    const { errors, values } = validateForm();
    if (Object.keys(errors).length) { showFieldErrors(errors); return; }
    clearErrors();
    setLoading(true);
    try {
      const body = { inputs: values, selected_crop: $("#f-selected_crop").value || null };
      const result = await fetchJSON("/api/recommend", {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
      });
      $("#results").innerHTML = buildResultHTML(result);
      if (window.AgroCharts) { window.AgroCharts.drawCompat(result); }
      const heading = $("#results-heading");
      const reduce = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
      $("#results").scrollIntoView({ behavior: reduce ? "auto" : "smooth", block: "start" });
      if (heading) heading.focus({ preventScroll: true });
    } catch (err) {
      if (err.status === 400 && err.payload && err.payload.errors) {
        showFieldErrors(err.payload.errors);
      } else {
        showResultError(err.message);
      }
    } finally {
      setLoading(false);
    }
  }

  async function onSample() {
    const note = $("#sample-note");
    try {
      const crop = META.crops[sampleIndex % META.crops.length];
      sampleIndex += 1;
      const s = await fetchJSON("/api/sample?crop=" + encodeURIComponent(crop.key));
      META.features.forEach((f) => {
        const el = $("#f-" + f.key);
        el.value = s.inputs[f.key];
        el.dispatchEvent(new Event("input"));
      });
      clearErrors();
      note.textContent = s.note + " Click again for another crop.";
    } catch (err) {
      note.textContent = "Could not load a sample: " + err.message;
    }
  }

  function onReset() {
    $("#recommend-form").reset();
    clearErrors();
    $$(".field-hint").forEach((el) => { el.textContent = ""; });
    $("#sample-note").textContent = "";
    $("#results").innerHTML = '<div class="results-placeholder">Your results will appear here after you submit the form.</div>';
  }

  async function initRecommendPage() {
    const form = $("#recommend-form");
    if (!form) return;
    try {
      META = await fetchJSON("/api/meta");
    } catch (err) {
      const box = $("#form-alert");
      box.hidden = false;
      box.className = "alert alert-danger";
      box.textContent = "The prediction service is unavailable: " + err.message;
      $("#btn-recommend").disabled = true;
      return;
    }
    buildForm();
    form.addEventListener("submit", onSubmit);
    $("#btn-sample").addEventListener("click", onSample);
    $("#btn-reset").addEventListener("click", onReset);
  }

  /* ------------------------------------------------------- overview number binding */
  function formatOverview(v, kind) {
    if (v === null || v === undefined) return "—";
    if (kind === "pct") return (Number(v) * 100).toFixed(2) + "%";
    if (kind === "int") return Number(v).toLocaleString("en-US");
    return String(v);
  }

  async function bindOverview() {
    const els = $$("[data-ov]");
    if (!els.length) return;
    try {
      const ov = await fetchJSON("/api/analytics/overview");
      els.forEach((el) => { el.textContent = formatOverview(ov[el.dataset.ov], el.dataset.fmt); });
    } catch (err) {
      els.forEach((el) => { el.textContent = "—"; });
      const box = $("#ov-error");
      if (box) { box.hidden = false; box.textContent = "Could not load the summary figures: " + err.message; }
    }
  }

  document.addEventListener("DOMContentLoaded", function () {
    initRecommendPage();
    bindOverview();
  });

  window.AgroWise = { buildResultHTML: buildResultHTML, fmt: fmt, esc: esc };
})();