document.addEventListener("alpine:init", () => {
  Alpine.data("moveMoney", () => ({
    shown: false, accounts: [], from: "", to: "", amount: "", key: "", busy: false, error: "",
    async open({ from = "", to = "" }) {
      this.error = ""; this.amount = ""; this.key = Banko.newKey();
      const data = await Banko.api("/api/v1/accounts");
      this.accounts = data.accounts;
      this.from = from || (this.accounts.find((a) => a.kind === "checking") || this.accounts[0] || {}).account_id || "";
      this.to = to || (this.accounts.find((a) => a.account_id !== this.from) || {}).account_id || "";
      this.shown = true;
    },
    close() { this.shown = false; },
    isCardPayment() { const t = this.accounts.find((a) => a.account_id === this.to); return !!t && t.kind === "credit"; },
    async submit() {
      this.busy = true; this.error = "";
      try {
        await Banko.api("/api/v1/transfers", { method: "POST", body: {
          from_account: this.from, to_account: this.to, amount: this.amount.trim(),
          kind: this.isCardPayment() ? "card_payment" : "transfer", idempotency_key: this.key } });
        this.shown = false;
        Alpine.store("toasts").push({ title: "Done", text: "Moved " + Banko.fmt.money(this.amount) });
        Banko.emit("banko:money-moved", {});
      } catch (e) { this.error = Banko.explain(e); }
      finally { this.busy = false; }
    },
  }));

  Alpine.data("accountsPage", () => ({
    data: null, history: [], error: "", chart: null,
    async load() {
      this.error = "";
      try {
        const [data, history] = await Promise.all([Banko.api("/api/v1/accounts"), Banko.api("/api/v1/accounts/history?months=12")]);
        this.data = data; this.history = history.items;
        this.$nextTick(() => this.draw());
      } catch (e) { this.error = Banko.explain(e); }
    },
    init() { window.addEventListener("banko:money-moved", () => this.load()); },
    draw() {
      if (this.chart) this.chart.destroy();
      this.chart = Banko.charts.line(this.$refs.net, { labels: this.history.map((p) => Banko.fmt.month(p.month)),
        series: [{ label: "Net worth", data: this.history.map((p) => Number(p.net_worth)), fill: true }] });
    },
    used(a) { return a.credit_limit ? Math.min(1, -Number(a.balance) / Number(a.credit_limit)) : 0; },
  }));
});
