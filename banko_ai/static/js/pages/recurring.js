document.addEventListener("alpine:init", () => {
  Alpine.data("recurringPage", () => ({
    items: null, error: "",
    async load() {
      try { const d = await Banko.api("/api/v1/recurring"); this.items = d.items.sort((a, b) => String(a.next_due).localeCompare(String(b.next_due))); }
      catch (e) { this.error = Banko.explain(e); }
    },
    badge(day) { return Banko.fmt.date(day, { month: "short" }) + "<br>" + Banko.fmt.date(day, { day: "numeric" }); },
    cadence(r) { return { weekly: "Weekly", monthly: "Monthly", yearly: "Yearly" }[r.cadence] || r.cadence; },
  }));
});
