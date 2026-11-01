/* AgroWise analytics dashboard (Step 7)
   Every table and chart is built from /api/analytics/* (saved results).
   Each section loads independently, so one failure never blocks the others. */
(function () {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const esc = (v) =>
    String(v).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const AC = window.AgroCharts;
  const PAL = AC.PAL;

  const FEATURES = ["N", "P", "K", "ph", "temperature", "humidity", "rainfall"];
  const FALLBACK_LABEL = {
    N: "Nitrogen (N, dataset ratio)", P: "Phosphorus (P, dataset ratio)", K: "Potassium (K, dataset ratio)",
    temperature: "Temperature (°C)", humidity: "Relative humidity (%)", ph: "Soil pH", rainfall: "Rainfall (mm)",
  };
  const BUTTON_LABEL = { N: "Nitrogen", P: "Phosphorus", K: "Potassium", ph: "Soil pH", temperature: "Temperature", humidity: "Humidity", rainfall: "Rainfall" };
  const SPECIAL_CROPS = { kidneybeans: "Kidney Beans", mothbeans: "Moth Beans", mungbean: "Mung Bean", blackgram: "Black Gram", pigeonpeas: "Pigeon Peas" };

  const state = { meta: null, descriptive: null, anova: null, profiles: null, corr: {}, corrKind: "pearson", compareFeature: null };

  /* ------------------------------------------------------------------ helpers */
  const label = (f) => (state.meta && state.meta.features.find((x) => x.key === f) || {}).label || FALLBACK_LABEL[f] || f;
  const cropName = (key) => {
    if (state.meta) {
      const c = state.meta.crops.find((x) => x.key === key);
      if (c) return c.display;
    }
    return SPECIAL_CROPS[key] || (String(key).charAt(0).toUpperCase() + String(key).slice(1));
  };
  const isNum = (v) => v !== null && v !== undefined && v !== "" && Number.isFinite(Number(v));
  const f2 = (v) => (isNum(v) ? Number(v).toFixed(2) : "—");
  const f3 = (v) => (isNum(v) ? Number(v).toFixed(3) : "—");
  const f4 = (v) => (isNum(v) ? Number(v).toFixed(4) : "—");
  const int = (v) => (isNum(v) ? Number(v).toLocaleString("en-US") : "—");
  const pct = (v) => (isNum(v) ? (Number(v) * 100).toFixed(2) + "%" : "—");
  const pfmt = (p) => (!isNum(p) ? "—" : Number(p) < 0.001 ? "< 0.001" : Number(p).toFixed(3));

  async function getJSON(url) {
    const res = await fetch(url);
    let data = null;
    try { data = await res.json(); } catch (_) { /* not JSON */ }
    if (!res.ok) throw new Error((data && data.message) || "Request failed (" + res.status + ")");
    return data;
  }
  const analytics = (name) => getJSON("/api/analytics/" + name);

  function table(headers, rows) {
    return '<div class="table-responsive"><table class="table table-sm table-agro table-stat"><thead><tr>' +
      headers.map((h) => '<th scope="col">' + esc(h) + "</th>").join("") + "</tr></thead><tbody>" +
      rows.map((r) => "<tr>" + r.map((c) => "<td>" + c + "</td>").join("") + "</tr>").join("") + "</tbody></table></div>";
  }

  async function section(containerId, fn) {
    try {
      await fn();
    } catch (err) {
      const el = $(containerId);
      if (el) el.innerHTML = '<div class="alert alert-warning" role="alert">Could not load this section: ' + esc(err.message) + "</div>";
    }
  }

  function effectOf(row) {
    if (isNum(row.eta_squared_primary)) return Number(row.eta_squared_primary);
    return Number(row.primary_test === "Kruskal-Wallis" ? row.eta_squared_H : row.eta_squared);
  }

  /* --------------------------------------------------------------- data section */
  async function cropDistribution() {
    const d = await analytics("crop_distribution");
    const rows = d.rows;
    AC.make("chart-crop-dist", {
      type: "bar",
      data: { labels: rows.map((r) => r.display), datasets: [{ label: "Records", data: rows.map((r) => r.count), backgroundColor: PAL.leaf, borderRadius: 4 }] },
      options: {
        plugins: { legend: { display: false } },
        scales: {
          y: { beginAtZero: true, title: { display: true, text: "Number of records" } },
          x: { ticks: { autoSkip: false, maxRotation: 60, minRotation: 45 } },
        },
      },
    });
    const counts = rows.map((r) => r.count);
    const mn = Math.min.apply(null, counts), mx = Math.max.apply(null, counts);
    $("crop-dist-note").textContent = mn === mx
      ? "Every crop has exactly " + mn + " records, so the " + rows.length + " classes are perfectly balanced and accuracy and macro-F1 are directly comparable."
      : "Records per crop range from " + mn + " to " + mx + ". Macro-averaged metrics stop small classes from being overlooked.";
  }

  async function descriptive() {
    const d = await analytics("descriptive");
    state.descriptive = d.rows;
    const cols = [["mean", "Mean"], ["median", "Median"], ["std", "Std dev"], ["min", "Min"], ["q1", "Q1"], ["q3", "Q3"], ["max", "Max"], ["iqr", "IQR"], ["skewness", "Skewness"], ["cv_percent", "CV (%)"]];
    $("desc-table").innerHTML = table(["Variable"].concat(cols.map((c) => c[1])),
      d.rows.map((r) => [esc(label(r.feature))].concat(cols.map((c) => f2(r[c[0]])))));
  }

  /* ------------------------------------------------------------- distributions */
  async function histograms() {
    const d = await analytics("histograms");
    const order = FEATURES.filter((f) => d.features[f]);
    $("hist-grid").innerHTML = order.map((f) =>
      '<div class="col-md-6"><div class="card-soft"><h3 class="h6">' + esc(label(f)) + '</h3>' +
      '<div class="chart-box chart-box-sm"><canvas id="hist-' + f + '" role="img" aria-label="Histogram of ' + esc(label(f)) + '"></canvas></div>' +
      '<p class="note mt-2 mb-0" id="hist-note-' + f + '"></p></div></div>').join("");

    order.forEach((f) => {
      const h = d.features[f];
      const dec = f === "ph" ? 2 : 1;
      AC.make("hist-" + f, {
        type: "bar",
        data: {
          labels: h.counts.map((_, i) => ((h.edges[i] + h.edges[i + 1]) / 2).toFixed(dec)),
          datasets: [{ label: "Records", data: h.counts, backgroundColor: PAL.leaf, borderWidth: 0, barPercentage: 1, categoryPercentage: 1 }],
        },
        options: {
          plugins: {
            legend: { display: false },
            tooltip: { callbacks: { title: (items) => { const i = items[0].dataIndex; return label(f) + ": " + h.edges[i].toFixed(2) + " to " + h.edges[i + 1].toFixed(2); } } },
          },
          scales: {
            x: { title: { display: true, text: label(f) }, ticks: { maxTicksLimit: 8 } },
            y: { beginAtZero: true, title: { display: true, text: "Records" } },
          },
        },
      });
      const stat = (state.descriptive || []).find((r) => r.feature === f);
      if (stat) {
        const s = Number(stat.skewness);
        const shape = Math.abs(s) > 1 ? "strongly skewed" : Math.abs(s) > 0.5 ? "moderately skewed" : "roughly symmetric";
        $("hist-note-" + f).textContent = "Mean " + f2(stat.mean) + " · median " + f2(stat.median) + " · skewness " + f2(stat.skewness) + " (" + shape + ", rule of thumb).";
      }
    });
  }

  async function outliers() {
    const d = await analytics("outliers");
    $("outlier-table").innerHTML = table(["Variable", "Outliers across all records", "Outliers within each crop", "Records"],
      d.rows.map((r) => [esc(label(r.feature)), int(r.outliers_pooled_iqr), int(r.outliers_within_crop_iqr), int(r.n)]));
  }

  /* --------------------------------------------------------------- correlation */
  function heatColor(r) {
    const a = Math.min(1, Math.abs(r));
    const alpha = (0.12 + 0.78 * a).toFixed(2);
    return r >= 0 ? "rgba(47, 107, 63, " + alpha + ")" : "rgba(180, 87, 44, " + alpha + ")";
  }
  const shortName = (f) => AC.SHORT[f] || f;

  function renderCorrToggle() {
    const kinds = [["pearson", "Pearson r"], ["spearman", "Spearman ρ"]];
    $("corr-toggle").innerHTML = kinds.map((k) =>
      '<button type="button" class="toggle-btn' + (state.corrKind === k[0] ? " active" : "") + '" data-kind="' + k[0] +
      '" aria-pressed="' + (state.corrKind === k[0]) + '">' + k[1] + "</button>").join("");
  }

  function renderHeat() {
    const m = state.corr[state.corrKind];
    if (!m) return;
    const head = "<tr><th></th>" + m.col_labels.map((c) => '<th scope="col" title="' + esc(label(c)) + '">' + esc(shortName(c)) + "</th>").join("") + "</tr>";
    const body = m.matrix.map((row, i) =>
      '<tr><th scope="row" title="' + esc(label(m.row_labels[i])) + '">' + esc(shortName(m.row_labels[i])) + "</th>" +
      row.map((v, j) => {
        const a = Math.abs(Number(v));
        return '<td style="background:' + heatColor(Number(v)) + ";color:" + (a > 0.55 ? "#fff" : "#1d2a22") + '" title="' +
          esc(label(m.row_labels[i]) + " vs " + label(m.col_labels[j]) + ": " + f3(v)) + '">' + f2(v) + "</td>";
      }).join("") + "</tr>").join("");
    $("corr-heatmap").innerHTML = '<table class="heat" aria-label="' + (state.corrKind === "pearson" ? "Pearson" : "Spearman") + ' correlation matrix"><thead>' + head + "</thead><tbody>" + body + "</tbody></table>";
  }

  async function correlation() {
    const [p, s, pairs] = await Promise.all([analytics("correlation_pearson"), analytics("correlation_spearman"), analytics("correlation_pairs")]);
    state.corr = { pearson: p, spearman: s };
    renderCorrToggle();
    renderHeat();
    $("corr-toggle").addEventListener("click", (ev) => {
      const b = ev.target.closest("[data-kind]");
      if (!b) return;
      state.corrKind = b.dataset.kind;
      renderCorrToggle();
      renderHeat();
    });
    const rows = pairs.rows.slice(0, 10).map((r) => {
      const differs = Math.abs(Number(r.pearson_r) - Number(r.spearman_rho)) >= 0.3;
      return [esc(shortName(r.feature_1) + " & " + shortName(r.feature_2)), f3(r.pearson_r), f3(r.spearman_rho), esc(r.strength),
        differs ? '<span class="chip out">Pearson and Spearman differ by 0.30 or more</span>' : "—"];
    });
    $("corr-pairs").innerHTML = table(["Pair", "Pearson r", "Spearman ρ", "Strength (Pearson)", "Check"], rows);
  }

  /* ----------------------------------------------------------- crop comparison */
  function renderCompareControls(feats) {
    $("compare-controls").innerHTML = feats.map((f) =>
      '<button type="button" class="toggle-btn' + (state.compareFeature === f ? " active" : "") + '" data-feature="' + f +
      '" aria-pressed="' + (state.compareFeature === f) + '">' + esc(BUTTON_LABEL[f] || f) + "</button>").join("");
  }

  function drawCompare(f) {
    const rows = state.profiles.slice().sort((a, b) => a[f + "_mean"] - b[f + "_mean"]);
    AC.make("chart-compare", {
      data: {
        labels: rows.map((r) => cropName(r.crop)),
        datasets: [
          { type: "bar", label: "Typical range (5th to 95th percentile)", data: rows.map((r) => [r[f + "_p05"], r[f + "_p95"]]), backgroundColor: PAL.leafSoft, borderColor: PAL.leaf, borderWidth: 1, borderSkipped: false, borderRadius: 4 },
          { type: "line", label: "Crop mean", data: rows.map((r) => r[f + "_mean"]), showLine: false, pointRadius: 5, pointStyle: "rectRot", pointBackgroundColor: PAL.forest, pointBorderColor: "#fff" },
        ],
      },
      options: {
        plugins: { legend: { position: "bottom" } },
        scales: {
          y: { title: { display: true, text: label(f) } },
          x: { ticks: { autoSkip: false, maxRotation: 60, minRotation: 45 } },
        },
      },
    });
    let note = "Crops are sorted by their mean. Bars show where the central 90% of each crop's records lie, so overlapping bars mean crops that are hard to tell apart on this variable alone.";
    const a = (state.anova || []).find((r) => r.feature === f);
    if (a) note = "Difference between crops for this variable: eta-squared = " + f3(effectOf(a)) + " (" + esc(a.effect_size) + "). " + note;
    $("compare-note").textContent = note;
  }

  async function cropCompare() {
    const d = await analytics("profiles");
    state.profiles = d.rows;
    const feats = FEATURES.filter((f) => d.rows[0] && (f + "_mean") in d.rows[0]);
    state.compareFeature = feats[0];
    renderCompareControls(feats);
    $("compare-controls").addEventListener("click", (ev) => {
      const b = ev.target.closest("[data-feature]");
      if (!b) return;
      state.compareFeature = b.dataset.feature;
      renderCompareControls(feats);
      drawCompare(state.compareFeature);
    });
    drawCompare(state.compareFeature);
  }

  /* ---------------------------------------------------------- statistical tests */
  async function loadAnova() {
    const d = await analytics("anova");
    state.anova = d.rows;
  }

  function primaryStat(r) {
    if (r.primary_test === "Kruskal-Wallis") return "H = " + f2(r.h_kruskal);
    if (r.primary_test === "Welch's ANOVA") return "F = " + f2(r.f_welch);
    return "F = " + f2(r.f_classical);
  }

  async function tests() {
    if (!state.anova) await loadAnova();
    const rows = state.anova;
    const bonf = rows[0] ? rows[0].bonferroni_alpha : null;
    $("anova-hypotheses").textContent =
      "For each variable, H0: the mean is the same for all crops. H1: at least one crop has a different mean. Significance level 0.05, with a Bonferroni-adjusted level of " + f4(bonf) + " for the seven tests.";
    $("anova-table").innerHTML = table(
      ["Variable", "Primary test", "Statistic", "p-value", "Effect size (η²)", "Reject H0 (Bonferroni)", "Tukey: significant pairs"],
      rows.map((r) => [esc(label(r.feature)), esc(r.primary_test), esc(primaryStat(r)), pfmt(r.primary_p),
        f3(effectOf(r)) + ' <span class="chip ok">' + esc(r.effect_size) + "</span>",
        r.significant_bonferroni ? "Yes" : "No", int(r.tukey_significant_pairs) + " of " + int(r.tukey_total_pairs)]));
    $("anova-text").innerHTML = rows.map((r) =>
      "<li><strong>" + esc(label(r.feature)) + ":</strong> " + esc(r.primary_test) + ", p " + (isNum(r.primary_p) && Number(r.primary_p) < 0.001 ? "< 0.001" : "= " + pfmt(r.primary_p)) +
      ", η² = " + f3(effectOf(r)) + " (" + esc(r.effect_size) + "). " +
      (r.significant_bonferroni ? "Reject H0: crops differ on this variable." : "Fail to reject H0 at the Bonferroni level.") + "</li>").join("");

    const a = await analytics("anova_assumptions");
    $("assume-table").innerHTML = table(
      ["Variable", "Crops passing Shapiro-Wilk", "Levene p-value", "Largest / smallest variance", "Normality rule met", "Equal variances"],
      a.rows.map((r) => [esc(label(r.feature)), int(r.groups_passing_shapiro) + " of " + int(r.groups_total), pfmt(r.levene_p), f2(r.max_to_min_variance_ratio),
        r.normality_ok ? "Yes" : "No", r.equal_variance_ok ? "Yes" : "No"]));
  }

  /* ------------------------------------------------------------------- models */
  async function modelComparison() {
    const d = await analytics("model_comparison");
    const rows = d.rows;
    $("model-table").innerHTML = table(
      ["Model", "CV macro-F1 (mean ± sd)", "CV log-loss", "Test accuracy", "Test macro-F1", "Test top-3", "Test log-loss", ""],
      rows.map((r) => [esc(r.model), f4(r.cv_f1_macro_mean) + " ± " + f4(r.cv_f1_macro_std), f4(r.cv_log_loss_mean), pct(r.test_accuracy),
        pct(r.test_f1_macro), pct(r.test_top3_accuracy), f4(r.test_log_loss), r.selected ? '<span class="chip ok">selected</span>' : ""]));

    const names = rows.map((r) => r.model);
    const f1vals = rows.map((r) => Number(r.cv_f1_macro_mean)).concat(rows.map((r) => Number(r.test_f1_macro)));
    const truncated = Math.min.apply(null, f1vals) > 0.9;
    AC.make("chart-model-f1", {
      type: "bar",
      data: { labels: names, datasets: [
        { label: "Cross-validation macro-F1 (mean)", data: rows.map((r) => Number(r.cv_f1_macro_mean)), backgroundColor: PAL.leaf, borderRadius: 4 },
        { label: "Test macro-F1", data: rows.map((r) => Number(r.test_f1_macro)), backgroundColor: PAL.gold, borderRadius: 4 },
      ] },
      options: {
        plugins: { legend: { position: "bottom" } },
        scales: { y: { min: truncated ? 0.9 : 0, max: 1, title: { display: true, text: "Macro-F1" } } },
      },
    });
    $("f1-axis-note").textContent = truncated
      ? "The vertical axis starts at 0.90 so that small differences are visible. Differences this small are within the cross-validation spread."
      : "The vertical axis starts at zero.";

    AC.make("chart-model-ll", {
      type: "bar",
      data: { labels: names, datasets: [
        { label: "Cross-validation log-loss (mean)", data: rows.map((r) => Number(r.cv_log_loss_mean)), backgroundColor: PAL.leaf, borderRadius: 4 },
        { label: "Test log-loss", data: rows.map((r) => Number(r.test_log_loss)), backgroundColor: PAL.gold, borderRadius: 4 },
      ] },
      options: {
        plugins: { legend: { position: "bottom" } },
        scales: { y: { beginAtZero: true, title: { display: true, text: "Log-loss (lower is better)" } } },
      },
    });
  }

  async function robustness() {
    const d = await analytics("robustness");
    const levels = Array.from(new Set(d.rows.map((r) => Number(r.noise_level_x_feature_std)))).sort((a, b) => a - b);
    const names = Array.from(new Set(d.rows.map((r) => r.model)));
    const colors = [PAL.clay, PAL.gold, PAL.green, PAL.forest];
    AC.make("chart-robust", {
      type: "line",
      data: {
        labels: levels.map((l) => (l === 0 ? "No noise" : l.toFixed(2) + " × SD")),
        datasets: names.map((n, i) => ({
          label: n,
          data: levels.map((l) => {
            const r = d.rows.find((x) => x.model === n && Number(x.noise_level_x_feature_std) === l);
            return r ? Math.round(Number(r.accuracy) * 1000) / 10 : null;
          }),
          borderColor: colors[i % colors.length], backgroundColor: colors[i % colors.length], tension: 0.2, pointRadius: 4,
        })),
      },
      options: {
        plugins: { legend: { position: "bottom" } },
        scales: {
          y: { min: 0, max: 100, title: { display: true, text: "Test accuracy (%)" } },
          x: { title: { display: true, text: "Noise added to each input (multiple of its standard deviation)" } },
        },
      },
    });
  }

  async function importance() {
    const d = await analytics("feature_importance");
    const rows = d.rows.slice().sort((a, b) => b.perm_importance_mean - a.perm_importance_mean);
    AC.make("chart-importance", {
      type: "bar",
      data: { labels: rows.map((r) => label(r.feature)), datasets: [{ label: "Drop in accuracy when shuffled", data: rows.map((r) => Number(r.perm_importance_mean)), backgroundColor: PAL.green, borderRadius: 4 }] },
      options: {
        indexAxis: "y",
        plugins: {
          legend: { display: false },
          tooltip: { callbacks: { afterLabel: (c) => "± " + f4(rows[c.dataIndex].perm_importance_std) + " (std over repeats)" } },
        },
        scales: { x: { beginAtZero: true, title: { display: true, text: "Mean drop in test accuracy" } } },
      },
    });
  }

  async function confusion() {
    const [cm, pairs] = await Promise.all([analytics("confusion_matrix"), analytics("confused_pairs")]);
    const head = '<tr><th class="cm-corner"></th>' + cm.col_labels.map((c) => '<th class="cm-col" scope="col"><span>' + esc(cropName(c)) + "</span></th>").join("") + "</tr>";
    const body = cm.matrix.map((row, i) => {
      const total = row.reduce((a, b) => a + Number(b), 0) || 1;
      return '<tr><th scope="row" class="cm-row">' + esc(cropName(cm.row_labels[i])) + "</th>" + row.map((v, j) => {
        const n = Number(v);
        const tip = esc("True: " + cropName(cm.row_labels[i]) + " · Predicted: " + cropName(cm.col_labels[j]) + " · Count: " + n);
        if (n === 0) return '<td class="cm-zero" title="' + tip + '">·</td>';
        if (i === j) return '<td class="cm-diag" style="background:rgba(47, 107, 63, ' + (0.25 + 0.7 * (n / total)).toFixed(2) + ')" title="' + tip + '">' + n + "</td>";
        return '<td class="cm-err" title="' + tip + '">' + n + "</td>";
      }).join("") + "</tr>";
    }).join("");
    $("cm-wrap").innerHTML = '<table class="cm" aria-label="Confusion matrix on the held-out test set"><thead>' + head + "</thead><tbody>" + body + "</tbody></table>";

    const total = pairs.rows.reduce((a, r) => a + Number(r.count), 0);
    $("cm-errors").innerHTML = pairs.rows.length
      ? "<p class=\"mb-2\"><strong>" + total + " misclassified test record" + (total === 1 ? "" : "s") + ":</strong></p><ul class=\"mb-0\">" +
        pairs.rows.map((r) => "<li>True <strong>" + esc(cropName(r.true_crop)) + "</strong> predicted as <strong>" + esc(cropName(r.predicted_crop)) + "</strong>: " + int(r.count) + "</li>").join("") + "</ul>"
      : "<p class=\"mb-0\">No misclassifications on the held-out test set.</p>";
  }

  async function classReport() {
    const d = await analytics("classification_report");
    const rows = d.rows.filter((r) => r.name !== "accuracy");
    const nice = (n) => (n === "macro avg" ? "Macro average" : n === "weighted avg" ? "Weighted average" : cropName(n));
    $("class-report").innerHTML = table(["Crop", "Precision", "Recall", "F1-score", "Test records"],
      rows.map((r) => [esc(nice(r.name)), f3(r.precision), f3(r.recall), f3(r["f1-score"]), int(r.support)]));
  }

  /* --------------------------------------------------------------------- init */
  async function init() {
    if (!$("analytics-root")) return;
    if (!AC.ready()) {
      const w = $("chart-warning");
      if (w) { w.hidden = false; w.textContent = "The charting library could not be loaded, so charts are hidden. Tables are still shown. Check your internet connection and refresh."; }
    }
    try { state.meta = await getJSON("/api/meta"); } catch (_) { state.meta = null; }

    await section("desc-table", descriptive);
    await section("anova-table", loadAnova);
    await section("crop-dist-note", cropDistribution);
    await section("hist-grid", histograms);
    await section("outlier-table", outliers);
    await section("corr-heatmap", correlation);
    await section("compare-note", cropCompare);
    await section("anova-table", tests);
    await section("model-table", modelComparison);
    await section("robust-note", robustness);
    await section("importance-note", importance);
    await section("cm-wrap", confusion);
    await section("class-report", classReport);
  }

  document.addEventListener("DOMContentLoaded", function () { init(); });
  window.AgroAnalytics = { init: init, state: state };
})();