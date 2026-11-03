document.addEventListener("alpine:init", () => {
  Alpine.data("budgetPage", () => ({
    data: null, categories: [], error: "", editing: null, draft: "", saveError: "", adding: false, newCategory: "", newAmount: "",
    async load() {
      this.error = "";
      try {
        const [data, cats] = await Promise.all([Banko.api("/api/v1/budgets"), Banko.api("/api/v1/categories")]);
        this.data = data; this.categories = cats.items;
      } catch (e) { this.error = Banko.explain(e); }
    },
    addable() {
      const have = new Set((this.data ? this.data.items : []).map((b) => b.key));
      return this.categories.filter((c) => (c.group === "needs" || c.group === "wants") && !have.has(c.key));
    },
    toggleAdd() {
      this.adding = !this.adding;
      // A select with no matching option shows its first one; start the
      // model there, so the category sent is the one on screen.
      if (this.adding) { this.newCategory = (this.addable()[0] || {}).label || ""; this.newAmount = ""; this.saveError = ""; }
    },
    edit(b) { this.editing = b.key; this.draft = String(b.budget); this.saveError = ""; },
    async put(category, amount) {
      this.saveError = "";
      try { await Banko.api("/api/v1/budgets", { method: "PUT", body: { category, amount: String(amount).trim() } }); await this.load(); return true; }
      catch (e) { this.saveError = Banko.explain(e); return false; }
    },
    async save(b) { if (await this.put(b.category, this.draft)) this.editing = null; },
    async add() { if (await this.put(this.newCategory, this.newAmount)) { this.adding = false; this.newAmount = ""; } },
  }));
});
