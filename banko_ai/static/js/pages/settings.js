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
    region: "", moving: false, homeText: "", homeError: false,
    // The person's own assistant settings: what the server lists, and what the controls hold until Save.
    prefs: [], prefGroups: [], prefValues: {}, prefText: "", prefError: false, prefSaving: false,

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
      const [me, ai, mem, health, cache, mine] = await Promise.all(
        [get("/api/v1/me"), get("/api/v1/ai"), get("/api/v1/memories"), get("/api/health"), get("/cache-stats"),
         get("/api/v1/me/settings")]);
      if (mine) this.showPrefs(mine);
      this.me = me; this.ai = ai; this.model = ai ? ai.model : "";
      this.region = me ? (me.home_region || (me.regions || [])[0] || "") : "";
      // What the move said, kept across the reload that follows it.
      const moved = sessionStorage.getItem("banko.moved");
      if (moved) { sessionStorage.removeItem("banko.moved"); this.homeText = moved; }
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
    // A database with regions, and one's own account: a persona keeps the region its file gives it.
    canMove() { return !!(this.me && (this.me.regions || []).length && !Banko.shell.user.persona); },
    async moveHome() {
      this.moving = true; this.homeText = ""; this.homeError = false;
      try {
        const out = await Banko.api("/api/v1/me/home-region", { method: "POST", body: { region: this.region } });
        // The header's region and every page's reads come from the server: load again.
        sessionStorage.setItem("banko.moved", "You live in " + out.home_region + " now. " +
          Number(out.rows).toLocaleString("en-US") + (out.rows === 1 ? " row" : " rows") + " moved.");
        window.location.reload();
      } catch (e) { this.homeText = Banko.explain(e); this.homeError = true; this.moving = false; }
    },
    showPrefs(mine) {
      this.prefs = mine.settings; this.prefGroups = mine.groups;
      this.prefValues = Object.fromEntries(mine.settings.map((s) => [s.key, s.value]));
    },
    prefsIn(group) { return this.prefs.filter((s) => s.group === group); },
    // A setting that matters only beside another one's value (how similar, for a similar question).
    prefOff(s) { return !!(s.only_with && this.prefValues[s.only_with.key] !== s.only_with.value); },
    prefsEdited() { return this.prefs.some((s) => this.prefValues[s.key] !== s.value); },
    prefsOwn() { return this.prefs.some((s) => s.changed); },
    // What the choice on screen means, in the words the server gave for it.
    prefMeans(s) {
      if (s.kind !== "choice") return "";
      return ((s.choices.find((c) => c.value === this.prefValues[s.key]) || {}).help) || "";
    },
    prefShown(s, value) {
      if (s.kind === "flag") return value ? "on" : "off";
      if (s.kind === "choice") return ((s.choices.find((c) => c.value === value) || {}).label) || String(value);
      if (s.percent) return Math.round(Number(value) * 100) + "%";
      return String(value);
    },
    async savePrefs() { await this.sendPrefs({ ...this.prefValues }, "Saved. Your next question uses these."); },
    async resetPrefs() { await this.sendPrefs({}, "Back to the defaults."); },
    async sendPrefs(values, said) {
      this.prefSaving = true; this.prefText = ""; this.prefError = false;
      try {
        this.showPrefs(await Banko.api("/api/v1/me/settings", { method: "POST", body: { values } }));
        this.prefText = said;
      } catch (e) { this.prefText = Banko.explain(e); this.prefError = true; }
      finally { this.prefSaving = false; }
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
