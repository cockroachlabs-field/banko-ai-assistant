// "How this was answered" in a window of its own. The Assistant sends each
// answer's trace over a channel that only this browser's Banko windows hear
// (assistant.js); this window shows the last one it got, and asks for it
// when it opens or is reloaded.
document.addEventListener("alpine:init", () => {
  Alpine.data("traceWindow", () => ({
    shown: null, copied: false,
    // The size of the steps and the plan, kept for the next time the window opens.
    size: Math.min(26, Math.max(11, Number(localStorage.getItem("banko.trace.size")) || 13)),
    init() {
      // The theme is the Assistant's: set there, it changes here too.
      window.addEventListener("storage", (e) => { if (e.key === "banko.theme") window.bankoTheme(); });
      if (!("BroadcastChannel" in window)) return;
      const channel = new BroadcastChannel("banko-trace");
      channel.onmessage = (e) => {
        const said = e.data || {};
        if (said.type === "trace") this.show(said.trace);
        // A plan names the user it ran for: it does not outlive that user's turn at the keyboard.
        else if (said.type === "user" && this.shown && this.shown.user !== said.user) this.show(null);
      };
      channel.postMessage({ type: "hello" });
    },
    show(trace) { this.shown = trace; this.copied = false; },
    resize(by) {
      this.size = Math.min(26, Math.max(11, this.size + by));
      localStorage.setItem("banko.trace.size", String(this.size));
    },
    // One row's detail and timing, with a separator only when both exist.
    line(s) { return [s.detail, s.ms ? Math.round(s.ms) + " ms" : ""].filter(Boolean).join(" · "); },
    async copyPlan() {
      const plan = this.shown.explain;
      try { await navigator.clipboard.writeText(plan); }
      catch (e) {
        const area = document.createElement("textarea");
        area.value = plan; document.body.appendChild(area); area.select(); document.execCommand("copy"); area.remove();
      }
      this.copied = true;
    },
  }));
});
