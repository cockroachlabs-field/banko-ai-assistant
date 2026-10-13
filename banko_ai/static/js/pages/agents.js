document.addEventListener("alpine:init", () => {
  Alpine.data("agentsPage", () => ({
    items: [], stats: {}, filter: "all", loaded: false, live: false, timer: null,
    agentNames: ["receipt", "fraud", "budget", "coach", "assistant"],
    init() {
      const socket = window.BankoSocket;
      if (socket) { this.live = socket.connected; socket.on("connect", () => { this.live = true; }); socket.on("disconnect", () => { this.live = false; }); }
      window.addEventListener("banko:agent", (e) => this.onEvent(e.detail));
      window.addEventListener("banko:nudge", () => this.refreshSoon());
    },
    async load() {
      try {
        const [act, stats] = await Promise.all([Banko.api("/api/v1/agents/activity"), Banko.api("/api/v1/agents/stats")]);
        this.merge(act.items); this.stats = stats;
      } catch (e) { Alpine.store("toasts").push({ title: "Agents", text: Banko.explain(e), kind: "error" }); }
      finally { this.loaded = true; }
    },
    merge(list) {
      const seen = new Map(this.items.map((i) => [i.id, i]));
      for (const i of list) seen.set(i.id, i);
      this.items = Array.from(seen.values()).sort((a, b) => String(b.at).localeCompare(String(a.at))).slice(0, 200);
    },
    onEvent(ev) {
      // Live receipt and assistant steps for this user show at once; anything
      // else (shared decision broadcasts) triggers a scoped refresh instead.
      if (ev.source === "activity" && ev.user_id === Banko.shell.user.user_id) {
        const id = "live:" + ev.agent_type + ":" + ev.timestamp + ":" + (ev.message || "");
        this.merge([{ id, agent: ev.agent_type, kind: ev.status, title: ev.agent_type + " agent", detail: ev.message || "",
                      confidence: null, region: ev.region || null, at: ev.timestamp }]);
      }
      this.refreshSoon();
    },
    refreshSoon() { clearTimeout(this.timer); this.timer = setTimeout(() => this.load(), 1500); },
    shown() { return this.filter === "all" ? this.items : this.items.filter((i) => i.agent === this.filter); },
  }));
});
