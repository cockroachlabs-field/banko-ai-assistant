document.addEventListener("alpine:init", () => {
  Alpine.data("threadList", () => ({
    threads: [], current: null, editing: null, name: "",
    async init() {
      window.addEventListener("banko:thread", (e) => { this.current = e.detail.id; this.load(); });
      window.addEventListener("banko:answered", () => this.load());
      await this.load();
    },
    async load() { try { this.threads = (await Banko.api("/api/v1/assistant/threads")).threads; } catch (e) { /* the list is optional */ } },
    grouped() {
      const today = new Date(); today.setHours(0, 0, 0, 0);
      const groups = { Today: [], Yesterday: [], Earlier: [] };
      for (const t of this.threads) {
        const d = new Date(t.last_message_at); d.setHours(0, 0, 0, 0);
        const diff = Math.round((today - d) / 86400000);
        groups[diff <= 0 ? "Today" : diff === 1 ? "Yesterday" : "Earlier"].push(t);
      }
      return Object.entries(groups).filter(([, items]) => items.length).map(([title, items]) => ({ title, items }));
    },
    startRename(t) { this.editing = t.thread_id; this.name = t.title; },
    async saveName(t) {
      try { await Banko.api("/api/v1/assistant/threads/" + t.thread_id, { method: "PATCH", body: { title: this.name.trim() } }); }
      catch (e) { Alpine.store("toasts").push({ title: "Assistant", text: Banko.explain(e), kind: "error" }); }
      this.editing = null; await this.load();
    },
    async remove(t) {
      if (!window.confirm("Delete this conversation? Its checkpoints are removed too.")) return;
      try { await Banko.api("/api/v1/assistant/threads/" + t.thread_id, { method: "DELETE" }); }
      catch (e) { Alpine.store("toasts").push({ title: "Assistant", text: Banko.explain(e), kind: "error" }); }
      if (t.thread_id === this.current) window.location.assign("/assistant");
      else await this.load();
    },
  }));

  Alpine.data("tracePanel", () => ({
    message: null, memories: [],
    async init() {
      window.addEventListener("banko:answered", (e) => { this.message = e.detail.bot; this.loadMemories(); });
      window.addEventListener("banko:trace", (e) => { this.message = e.detail.message; });
      await this.loadMemories();
    },
    async loadMemories() { try { this.memories = (await Banko.api("/api/v1/memories")).memories; } catch (e) { this.memories = []; } },
    steps() {
      const m = this.message;
      return m.meta && m.meta.trace && m.meta.trace.length ? m.meta.trace : m.steps;
    },
  }));
});
