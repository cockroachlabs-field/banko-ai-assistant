/* Thin Chart.js helpers that read colors from the theme tokens and redraw
   when the theme changes. */
(function () {
  "use strict";
  const live = new Set();

  function css(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }

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

  window.addEventListener("banko:theme", () => {
    for (const e of live) { e.chart.destroy(); e.chart = e.build(); }
  });

  Banko.charts = { line, bar, doughnut, css };
})();
