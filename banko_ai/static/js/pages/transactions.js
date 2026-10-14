document.addEventListener("alpine:init", () => {
  // One fetch per page, shared by the list and the drawer. A failure is
  // tried once more at once, and never kept, so the next caller fetches again.
  let categoriesPromise = null;
  const fetchCategories = () => Banko.api("/api/v1/categories").then((d) => d.items);
  const categories = () => (categoriesPromise = categoriesPromise || fetchCategories().catch(fetchCategories).then(
    (items) => items, (e) => { categoriesPromise = null; throw e; }));

  Alpine.data("transactionsPage", () => ({
    items: [], next: null, q: "", semantic: false, searchMs: 0, range: "30", kind: "", category: "",
    categories: [], error: "", busy: false, loaded: false, selected: null,
    ranges: [{ key: "30", label: "Last 30 days" }, { key: "90", label: "Last 90 days" }, { key: "all", label: "All time" }],
    init() {
      // Listeners first: nothing on this page waits for the category list.
      window.addEventListener("banko:open-transaction", (e) => { this.selected = e.detail.id; });
      window.addEventListener("banko:transaction-changed", () => this.load());
      const open = new URLSearchParams(location.search).get("open");
      if (open) this.$nextTick(() => Banko.emit("banko:open-transaction", { id: open }));
      this.loadCategories();
    },
    loadCategories() {
      if (!this.categories.length) categories().then((c) => { this.categories = c; }, () => { /* tried again on the next load */ });
    },
    params() {
      const p = new URLSearchParams({ limit: "40" });
      if (this.range !== "all") {
        const d = new Date(); d.setDate(d.getDate() - Number(this.range));
        p.set("date_from", d.toISOString().slice(0, 10));
      }
      if (this.kind) p.set("kind", this.kind);
      if (this.category) p.set("category", this.category);
      return p;
    },
    async load() {
      this.error = ""; this.semantic = false; this.busy = true;
      this.loadCategories();
      try {
        const data = await Banko.api("/api/v1/transactions?" + this.params());
        this.items = data.items; this.next = data.next_cursor;
      } catch (e) { this.error = Banko.explain(e); }
      finally { this.busy = false; this.loaded = true; }
    },
    async more() {
      this.busy = true;
      try {
        const p = this.params(); p.set("cursor", this.next);
        const data = await Banko.api("/api/v1/transactions?" + p);
        this.items = this.items.concat(data.items); this.next = data.next_cursor;
      } catch (e) { this.error = Banko.explain(e); }
      finally { this.busy = false; }
    },
    async search() {
      const q = this.q.trim();
      if (!q) return this.load();
      this.error = ""; this.busy = true;
      const t0 = performance.now();
      try {
        const data = await Banko.api("/api/v1/transactions?limit=25&q=" + encodeURIComponent(q));
        this.items = data.items; this.next = null; this.semantic = true;
        this.searchMs = Math.round(performance.now() - t0);
      } catch (e) { this.error = Banko.explain(e); }
      finally { this.busy = false; this.loaded = true; }
    },
    grouped() {
      if (this.semantic) return [{ label: "Closest matches", items: this.items }];
      const out = [];
      for (const t of this.items) {
        const label = Banko.fmt.dayLabel(t.posted_at);
        if (!out.length || out[out.length - 1].label !== label) out.push({ label, items: [] });
        out[out.length - 1].items.push(t);
      }
      return out;
    },
  }));

  Alpine.data("txDrawer", () => ({
    shown: false, t: null, categories: [], category: "", saving: false, saveError: "", readMs: 0,
    async open(id) {
      this.shown = true; this.t = null; this.saveError = "";
      // Before the detail, so the select has its options when the category is set.
      try { this.categories = await categories(); } catch (e) { this.categories = []; }
      const t0 = performance.now();
      try {
        this.t = await Banko.api("/api/v1/transactions/" + encodeURIComponent(id));
        this.readMs = Math.round(performance.now() - t0);
        this.category = this.t.category_label;
      } catch (e) { this.shown = false; Alpine.store("toasts").push({ title: "Transactions", text: Banko.explain(e), kind: "error" }); }
    },
    close() { this.shown = false; },
    decisionText(d) {
      const flagged = d.action && d.action.fraud_detected;
      const score = typeof d.confidence === "number" ? " · " + Banko.scoreText(d.decision_type, d.confidence) : "";
      return (flagged ? "Flagged by the fraud agent" : "✓ Fraud check passed") + score;
    },
    async save() {
      this.saving = true; this.saveError = "";
      try {
        this.t = await Banko.api("/api/v1/transactions/" + encodeURIComponent(this.t.expense_id), { method: "PATCH", body: { category: this.category } });
        Banko.emit("banko:transaction-changed", { id: this.t.expense_id });
      } catch (e) { this.saveError = Banko.explain(e); }
      finally { this.saving = false; }
    },
  }));
});
