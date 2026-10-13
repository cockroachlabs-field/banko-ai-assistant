document.addEventListener("alpine:init", () => {
  Alpine.data("cashflowPage", () => ({
    data: null, months: 6, error: "", charts: [],
    async load() {
      this.error = "";
      try { this.data = await Banko.api("/api/v1/cashflow?months=" + this.months); this.$nextTick(() => this.draw()); }
      catch (e) { this.error = Banko.explain(e); }
    },
    current() { return this.data && this.data.months[this.data.months.length - 1]; },
    draw() {
      for (const c of this.charts) c.destroy();
      const m = this.data.months;
      const cats = this.data.categories.slice(0, 8);
      this.charts = [
        Banko.charts.bar(this.$refs.flow, { labels: m.map((x) => Banko.fmt.month(x.month)),
          series: [{ label: "Income", data: m.map((x) => Number(x.income)) }, { label: "Spending", data: m.map((x) => Number(x.spending)) }] }),
        Banko.charts.doughnut(this.$refs.cats, { labels: cats.map((c) => c.category), data: cats.map((c) => Number(c.total)), colors: cats.map((c) => c.color) }),
      ];
    },
  }));
});
