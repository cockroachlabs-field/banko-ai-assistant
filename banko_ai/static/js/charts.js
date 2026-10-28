/* Thin Chart.js helpers that read colors from the theme tokens and redraw
   when the theme changes. */
(function () {
  "use strict";
  const live = new Set();

  // Some tokens are color-mix() expressions (the accent shades derive from
  // BRAND_ACCENT), which Chart.js cannot read. The browser resolves the
  // color on a swatch, and a 1x1 canvas turns it back into hex.
  const swatch = document.createElement("span");
  const probe = document.createElement("canvas").getContext("2d", { willReadFrequently: true });
  function resolve(value) {
    if (!value || /^#[0-9a-f]{6}$/i.test(value)) return value;
    swatch.style.color = "";
    swatch.style.color = value;
    if (!swatch.style.color) return value;
    document.documentElement.appendChild(swatch);
    const computed = getComputedStyle(swatch).color;
    swatch.remove();
    probe.clearRect(0, 0, 1, 1);
    probe.fillStyle = "#000000";
    probe.fillStyle = computed;
    probe.fillRect(0, 0, 1, 1);
    const [r, g, b] = probe.getImageData(0, 0, 1, 1).data;
    return "#" + [r, g, b].map((x) => x.toString(16).padStart(2, "0")).join("");
  }

  function css(name) { return resolve(getComputedStyle(document.documentElement).getPropertyValue(name).trim()); }

  function alpha(color, a) {
    const m = /^#([0-9a-f]{6})$/i.exec(color);
    if (!m) return "rgba(105, 51, 255, " + a + ")";
    const n = parseInt(m[1], 16);
    return "rgba(" + (n >> 16) + ", " + ((n >> 8) & 255) + ", " + (n & 255) + ", " + a + ")";
  }

  function axes(money) {
    return {
      x: { grid: { display: false }, ticks: { color: css("--mu"), maxRotation: 0, autoSkip: true, maxTicksLimit: 8 } },
      y: { grid: { color: css("--chart-grid") }, border: { display: false },
           ticks: { color: css("--mu"), maxTicksLimit: 5, callback: (v) => (money ? Banko.fmt.money0(v) : v) } },
    };
  }

  function register(canvas, build) {
    const entry = { canvas, build, chart: build() };
    live.add(entry);
    return { destroy() { entry.chart.destroy(); live.delete(entry); }, entry };
  }

  function line(canvas, { labels, series, money = true }) {
    return register(canvas, () => new Chart(canvas, {
      type: "line",
      data: {
        labels,
        datasets: series.map((s, i) => {
          const color = css(i === 0 ? "--chart-1" : "--chart-2");
          return { label: s.label, data: s.data, borderColor: color, borderDash: s.dashed ? [4, 4] : [],
                   borderWidth: i === 0 ? 2.4 : 1.6, pointRadius: 0, tension: 0.3, spanGaps: true,
                   fill: !!s.fill, backgroundColor: s.fill ? alpha(color, 0.16) : "transparent" };
        }),
      },
      options: { responsive: true, maintainAspectRatio: false, animation: false, interaction: { mode: "index", intersect: false },
                 plugins: { legend: { display: false },
                            tooltip: { callbacks: { label: (c) => (c.dataset.label ? c.dataset.label + ": " : "") + (money ? Banko.fmt.money(c.parsed.y) : c.parsed.y) } } },
                 scales: axes(money) },
    }));
  }

  function bar(canvas, { labels, series, money = true }) {
    return register(canvas, () => new Chart(canvas, {
      type: "bar",
      data: { labels, datasets: series.map((s, i) => ({ label: s.label, data: s.data, borderRadius: 6,
                                                       backgroundColor: css(i === 0 ? "--pos" : "--chart-1") })) },
      options: { responsive: true, maintainAspectRatio: false, animation: false,
                 plugins: { legend: { display: true, labels: { color: css("--mu"), boxWidth: 10 } },
                            tooltip: { callbacks: { label: (c) => c.dataset.label + ": " + (money ? Banko.fmt.money(c.parsed.y) : c.parsed.y) } } },
                 scales: axes(money) },
    }));
  }

  function doughnut(canvas, { labels, data, colors }) {
    return register(canvas, () => new Chart(canvas, {
      type: "doughnut",
      data: { labels, datasets: [{ data, backgroundColor: colors, borderColor: css("--card"), borderWidth: 2 }] },
      options: { responsive: true, maintainAspectRatio: false, animation: false, cutout: "68%",
                 plugins: { legend: { display: false }, tooltip: { callbacks: { label: (c) => c.label + ": " + Banko.fmt.money(c.parsed) } } } },
    }));
  }

  function clock(sec) {
    return sec == null ? "" : new Date(sec * 1000).toLocaleTimeString("en-US", { hour12: false });
  }

  // A time series updated in place once a second (one axis, 2px lines).
  // Hovering draws a crosshair and one tooltip with every series, value
  // first; thin markers stand at the seconds events happened, and the
  // tooltip for that second names them.
  function timeline(canvas, { series, format = (v) => String(v) }) {
    const state = { labels: [], data: series.map(() => []), events: [] };
    const tones = { bad: "--neg", good: "--pos", warn: "--warn", info: "--mu" };
    const marks = {
      id: "bankoTimeline",
      afterDatasetsDraw(chart) {
        const { ctx, chartArea: area, scales: { x } } = chart;
        const rule = (px, color) => {
          ctx.strokeStyle = color;
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(px, area.top);
          ctx.lineTo(px, area.bottom);
          ctx.stroke();
        };
        ctx.save();
        for (const ev of state.events) {
          const i = state.labels.indexOf(ev.second);
          if (i >= 0) rule(Math.round(x.getPixelForValue(i)) + 0.5, alpha(css(tones[ev.level] || "--mu"), 0.75));
        }
        const active = chart.tooltip ? chart.tooltip.getActiveElements() : [];
        if (active.length) rule(Math.round(active[0].element.x) + 0.5, css("--mu"));
        ctx.restore();
      },
    };
    const reg = register(canvas, () => new Chart(canvas, {
      type: "line",
      data: {
        labels: state.labels,
        datasets: series.map((s, i) => {
          const color = css(s.color);
          return { label: s.label, data: state.data[i], borderColor: color, backgroundColor: color,
                   borderDash: s.dashed ? [5, 4] : [], borderWidth: 2, borderCapStyle: "round", borderJoinStyle: "round",
                   pointRadius: 0, pointHoverRadius: 4, pointHoverBorderWidth: 2, pointHoverBorderColor: css("--card"),
                   tension: 0.25, spanGaps: true, fill: false };
        }),
      },
      options: {
        responsive: true, maintainAspectRatio: false, animation: false, interaction: { mode: "index", intersect: false },
        plugins: {
          legend: { display: false },
          tooltip: { callbacks: {
            title: (items) => clock(state.labels[items[0].dataIndex]),
            label: (c) => (c.parsed.y == null ? "no data" : format(c.parsed.y)) + "  " + c.dataset.label,
            afterBody: (items) => state.events.filter((e) => e.second === state.labels[items[0].dataIndex]).map((e) => e.text),
          } },
        },
        scales: {
          x: { grid: { display: false }, ticks: { color: css("--mu"), maxRotation: 0, autoSkip: true, maxTicksLimit: 6,
                                                 callback: (v, i) => clock(state.labels[i]) } },
          y: { beginAtZero: true, grid: { color: css("--chart-grid") }, border: { display: false },
               ticks: { color: css("--mu"), maxTicksLimit: 5, callback: (v) => format(v) } },
        },
      },
      plugins: [marks],
    }));
    return {
      destroy: reg.destroy,
      update(labels, data, events) {
        state.labels.splice(0, state.labels.length, ...labels);
        data.forEach((d, i) => state.data[i].splice(0, state.data[i].length, ...d));
        state.events = events || [];
        reg.entry.chart.update("none");
      },
    };
  }

  window.addEventListener("banko:theme", () => {
    for (const e of live) { e.chart.destroy(); e.chart = e.build(); }
  });

  Banko.charts = { line, bar, doughnut, timeline, css };
})();
