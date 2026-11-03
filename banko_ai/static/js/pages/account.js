document.addEventListener("alpine:init", () => {
  Alpine.data("accountPage", (id) => ({
    id, account: null, months: [], month: "", statement: null, error: "",
    async load() {
      try {
        const [account, list] = await Promise.all([Banko.api("/api/v1/accounts/" + id), Banko.api("/api/v1/accounts/" + id + "/statements")]);
        this.account = account; this.months = list.months || [];
        this.month = this.months[0] || "";
        if (this.month) await this.loadStatement();
      } catch (e) { this.error = Banko.explain(e); }
    },
    init() { window.addEventListener("banko:money-moved", () => this.load()); },
    async loadStatement() {
      this.statement = null;
      try { this.statement = await Banko.api("/api/v1/accounts/" + id + "/statements?month=" + this.month); }
      catch (e) { this.error = Banko.explain(e); }
    },
  }));
});
