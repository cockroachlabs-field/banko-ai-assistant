document.addEventListener("alpine:init", () => {
  const personas = JSON.parse(document.getElementById("settings-personas").textContent);

  Alpine.data("settingsPage", () => ({
    personas,
    personaIds: new Set(personas.map((p) => p.user_id)),
    languages: Banko.shell.languages,
    themes: [["light", "Daylight"], ["dark", "Midnight"], ["system", "Match my system"]],
    me: null, ai: null, model: "", aiText: "", aiError: false, memories: [], health: null, cache: null,
    language: localStorage.getItem("banko.language") || (Banko.shell.languages[0] || { code: "en-US" }).code,
    autoRead: localStorage.getItem("banko.autoread") === "1",
    theme: localStorage.getItem("banko.theme") || Banko.shell.theme || "system",
    genUsers: 5, genMonths: 3, genBusy: false, genDone: 0, genTotal: 0, genSeen: new Set(), dataText: "", dataError: false,

    init() {
      // Count new users as their seeding finishes; persona resets report too, so skip those.
      window.addEventListener("banko:seeding", (e) => {
        const p = e.detail;
        if (!this.genTotal || p.state !== "done" || this.personaIds.has(p.user_id)) return;
        this.genSeen.add(p.user_id);
        this.genDone = Math.min(this.genSeen.size, this.genTotal);
      });
      // The header toggle cycles the theme; keep the radio in step.
      window.addEventListener("banko:theme", () => { this.theme = localStorage.getItem("banko.theme") || this.theme; });
    },
    async load() {
      const get = async (path) => { try { return await Banko.api(path); } catch (e) { return null; } };
      const [me, ai, mem, health, cache] = await Promise.all(
        [get("/api/v1/me"), get("/api/v1/ai"), get("/api/v1/memories"), get("/api/health"), get("/cache-stats")]);
      this.me = me; this.ai = ai; this.model = ai ? ai.model : "";
      this.memories = mem ? mem.memories : [];
      this.health = health; this.cache = cache;
    },
    components() {
      const c = (this.health && this.health.components) || {};
      return Object.keys(c).sort().map((name) => ({ name, state: c[name].state }));
    },
    schemaText() {
      const c = this.health && this.health.components && this.health.components.schema;
      return c && c.detail ? "Schema " + c.detail : "";
    },
    async switchModel() {
      this.aiText = ""; this.aiError = false;
      try {
        const r = await Banko.api("/api/v1/ai/model", { method: "POST", body: { model: this.model } });
        this.ai.model = this.model; this.aiText = r.message || "Model updated.";
      } catch (e) { this.aiText = Banko.explain(e); this.aiError = true; }
    },
    async generate() {
      this.genBusy = true; this.dataText = ""; this.dataError = false; this.genSeen = new Set(); this.genDone = 0; this.genTotal = 0;
      try {
        await Banko.api("/api/v1/admin/seed", { method: "POST", body: { users: this.genUsers, months: this.genMonths } });
        this.genTotal = this.genUsers;
        this.dataText = "Creating users from the seed profiles. They seed in the background.";
      } catch (e) { this.dataText = Banko.explain(e); this.dataError = true; }
      finally { this.genBusy = false; }
    },
    async reset(p) {
      if (!window.confirm("Reset " + p.display_name + "? Their money history is deleted and seeded again.")) return;
      this.dataText = ""; this.dataError = false;
      try {
        await Banko.api("/api/v1/admin/reset", { method: "POST", body: { persona: p.username } });
        this.dataText = "Resetting " + p.display_name + ". The history reseeds in the background.";
      } catch (e) { this.dataText = Banko.explain(e); this.dataError = true; }
    },
    async forget(m) {
      try {
        await Banko.api("/api/v1/memories/" + m.memory_id, { method: "DELETE" });
        this.memories = this.memories.filter((x) => x.memory_id !== m.memory_id);
      } catch (e) { Alpine.store("toasts").push({ title: "Memories", text: Banko.explain(e), kind: "error" }); }
    },
  }));
});
