document.addEventListener("alpine:init", () => {
  Alpine.data("coachPage", () => ({
    nudges: [], selected: null, loaded: false, busy: false, simText: "", simError: false,
    kinds: [{ type: "budget_threshold", label: "A budget is near its limit" },
            { type: "anomaly", label: "An unusual purchase" },
            { type: "recurring_drift", label: "A subscription price went up" }],
    init() { window.addEventListener("banko:nudge", () => this.load()); },
    async load() {
      try { this.nudges = (await Banko.api("/api/v1/coach/nudges")).nudges; }
      catch (e) { Alpine.store("toasts").push({ title: "Coach", text: Banko.explain(e), kind: "error" }); }
      finally { this.loaded = true; }
    },
    async open(n) {
      try { this.selected = await Banko.api("/api/v1/coach/nudges/" + n.nudge_id); }
      catch (e) { Alpine.store("toasts").push({ title: "Coach", text: Banko.explain(e), kind: "error" }); }
    },
    label(type) { return { budget_threshold: "budget", anomaly: "unusual purchase", recurring_drift: "price change" }[type] || "signal"; },
    async simulate(type) {
      this.busy = true; this.simText = ""; this.simError = false;
      try {
        await Banko.api("/api/v1/coach/simulate", { method: "POST", body: { signal_type: type } });
        this.simText = "Signal sent. The coach is thinking; its nudge lands here and as a toast.";
      } catch (e) { this.simText = Banko.explain(e); this.simError = true; }
      finally { this.busy = false; }
    },
  }));
});
