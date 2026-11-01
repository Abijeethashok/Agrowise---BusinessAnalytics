/* AgroWise chart helpers (Step 7)
   - shared palette and a safe chart factory around Chart.js
   - compatibility charts shown under a recommendation result */
(function () {
  "use strict";

  const PAL = {
    forest: "#1c3d29", green: "#2f6b3f", leaf: "#4c9459", leafSoft: "rgba(76, 148, 89, 0.38)",
    gold: "#b08a2e", clay: "#b4572c", sand: "#efe7d4", ink: "#1d2a22", muted: "#5a685d", grid: "#e1dac8",
  };
  const SHORT = { N: "N", P: "P", K: "K", temperature: "Temp", humidity: "Humidity", ph: "pH", rainfall: "Rainfall" };
  const registry = {};
  let styled = false;

  const ready = () => typeof Chart !== "undefined";

  function applyDefaults() {
    if (styled || !ready()) return;
    const family = (typeof getComputedStyle === "function" && document.body)
      ? getComputedStyle(document.body).fontFamily : "sans-serif";
    Chart.defaults.font.family = family;
    Chart.defaults.color = PAL.ink;
    Chart.defaults.borderColor = PAL.grid;
    Chart.defaults.maintainAspectRatio = false;
    const reduce = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    Chart.defaults.animation = reduce ? false : { duration: 350 };
    styled = true;
  }

  function make(id, config) {
    if (!ready()) return null;
    const el = document.getElementById(id);
    if (!el) return null;
    applyDefaults();
    if (registry[id]) registry[id].destroy();
    registry[id] = new Chart(el, config);
    return registry[id];
  }

  function chartsUnavailable(ids) {
    ids.forEach((id) => {
      const el = document.getElementById(id);
      if (el && el.parentNode) {
        el.parentNode.innerHTML = '<p class="note mb-0">Charts could not be loaded. Check your internet connection and refresh the page.</p>';
      }
    });
  }

  /* ---------------------------------------------- recommendation-page charts */
  function drawCompat(result) {
    const ids = ["chart-fit-radar", "chart-fit-all"];
    if (!ready()) { chartsUnavailable(ids); return; }

    // 1) feature-by-feature fit (radar)
    const feats = Object.keys(result.inputs);
    const sel = result.selected, top = result.top3;
    const picks = [];
    if (sel) picks.push({ row: sel, tag: "selected", color: PAL.gold });
    if (!sel || sel.crop !== top[0].crop) picks.push({ row: top[0], tag: "model rank #1", color: PAL.forest });
    if (picks.length < 2 && top[1]) picks.push({ row: top[1], tag: "model rank #2", color: PAL.clay });

    make("chart-fit-radar", {
      type: "radar",
      data: {
        labels: feats.map((f) => SHORT[f] || f),
        datasets: picks.map((p) => ({
          label: p.row.display + " (" + p.tag + ")",
          data: feats.map((f) => Math.round(((p.row.feature_fits || {})[f] || 0) * 100)),
          borderColor: p.color, backgroundColor: p.color + "22", pointBackgroundColor: p.color, borderWidth: 2,
        })),
      },
      options: {
        scales: { r: { min: 0, max: 100, ticks: { stepSize: 25, backdropColor: "transparent" }, pointLabels: { font: { size: 12 } } } },
        plugins: {
          legend: { position: "bottom" },
          tooltip: { callbacks: { label: (c) => c.dataset.label + ": " + c.parsed.r + "% fit" } },
        },
      },
    });

    // 2) profile fit across all crops (horizontal bars)
    const rows = result.all_crops.slice().sort((a, b) => b.profile_fit - a.profile_fit || a.rank - b.rank);
    const selKey = sel ? sel.crop : null;
    make("chart-fit-all", {
      type: "bar",
      data: {
        labels: rows.map((r) => r.display),
        datasets: [{
          label: "Profile fit (%)",
          data: rows.map((r) => Math.round(r.profile_fit * 1000) / 10),
          backgroundColor: rows.map((r) => (r.crop === selKey ? PAL.gold : r.rank === 1 ? PAL.green : PAL.leafSoft)),
          borderRadius: 4,
        }],
      },
      options: {
        indexAxis: "y",
        scales: {
          x: { min: 0, max: 100, title: { display: true, text: "Profile fit (%)" } },
          y: { ticks: { autoSkip: false, font: { size: 11 } } },
        },
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              afterLabel: (c) => {
                const r = rows[c.dataIndex];
                return "Model rank #" + r.rank + " · " + r.features_inside + " of " + r.features_total + " inputs in typical range";
              },
            },
          },
        },
      },
    });
  }

  window.AgroCharts = { PAL: PAL, SHORT: SHORT, ready: ready, make: make, drawCompat: drawCompat, chartsUnavailable: chartsUnavailable };
})();