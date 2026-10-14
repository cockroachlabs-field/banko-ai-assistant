document.addEventListener("alpine:init", () => {
  Alpine.data("dashboardPage", () => ({
    data: null, error: "", seeding: null, noData: false, charts: [],

    init() {
      window.addEventListener("banko:seeding", (ev) => {
        const p = ev.detail;
        if (p.user_id !== Banko.shell.user.user_id) return;
        this.seeding = p.state === "done" ? null : p;
        if (this.seeding) this.noData = false;  // the page loaded before seeding reported in
        if (p.state === "done") this.load();
      });
    },

    async load() {
      this.error = "";
      try {
        const [data, me] = await Promise.all([Banko.api("/api/v1/dashboard"), Banko.api("/api/v1/me")]);
        const running = !!(me.seeding && me.seeding.state === "running");
        const empty = !data.recent.length && Number(data.net_worth) === 0;
        this.seeding = running ? me.seeding : null;
        this.noData = empty && !running;
        this.data = data;
        if (!this.seeding && !this.noData) this.$nextTick(() => this.draw());
      } catch (e) {
        this.error = Banko.explain(e);
      }
    },

    draw() {
      for (const c of this.charts) c.destroy();
      const h = this.data.net_worth_history;
      const s = this.data.spending;
      this.charts = [
        Banko.charts.line(this.$refs.net, { labels: h.map((p) => Banko.fmt.month(p.month)),
          series: [{ label: "Net worth", data: h.map((p) => Number(p.net_worth)), fill: true }] }),
        Banko.charts.line(this.$refs.spend, { labels: s.last_month.map((p) => p.day),
          series: [{ label: "This month", data: s.this_month.map((p) => Number(p.total)) },
                   { label: "Last month", data: s.last_month.map((p) => Number(p.total)), dashed: true }] }),
      ];
    },

    netDelta() {
      if (!this.data) return 0;
      const h = this.data.net_worth_history;
      return h.length > 1 ? Number(h[h.length - 1].net_worth) - Number(h[h.length - 2].net_worth) : 0;
    },
    spendDelta() { return this.data ? Number(this.data.spending.this_total) - Number(this.data.spending.last_same_day) : 0; },
    monthName() { return new Date().toLocaleDateString("en-US", { month: "long" }); },
    dateBadge(day) { return Banko.fmt.date(day, { month: "short" }) + "<br>" + Banko.fmt.date(day, { day: "numeric" }); },
    seedingPct() { const p = this.seeding || {}; return p.total ? Math.round((p.done / p.total) * 100) : 5; },
    seedingText() { const p = this.seeding || {}; return p.total ? "Writing " + p.done + " of " + p.total + " ledger entries" : "Starting"; },
  }));
});
