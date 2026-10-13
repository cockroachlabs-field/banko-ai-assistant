document.addEventListener("alpine:init", () => {
  Alpine.data("goalsPage", () => ({
    goals: null, accounts: [], error: "", target: null, from: "", amount: "", key: "", busy: false, formError: "",
    async load() {
      try {
        const [goals, accounts] = await Promise.all([Banko.api("/api/v1/goals"), Banko.api("/api/v1/accounts")]);
        this.goals = goals.items; this.accounts = accounts.accounts;
      } catch (e) { this.error = Banko.explain(e); }
    },
    paceText(g) {
      if (Number(g.remaining) === 0) return "Done. Nice work.";
      if (!g.projected_completion) return "No contributions in the last three months.";
      return Banko.fmt.money0(g.monthly_pace) + " a month · done around " + Banko.fmt.date(g.projected_completion, { month: "short", year: "numeric" });
    },
    sources() { return this.accounts.filter((a) => a.kind !== "credit" && (!this.target || a.account_id !== this.target.account_id)); },
    startContribute(g) { this.target = g; this.amount = ""; this.formError = ""; this.key = Banko.newKey(); this.from = (this.sources()[0] || {}).account_id || ""; },
    async contribute() {
      this.busy = true; this.formError = "";
      try {
        await Banko.api("/api/v1/goals/" + this.target.goal_id + "/contribute", { method: "POST",
          body: { from_account: this.from, amount: this.amount.trim(), idempotency_key: this.key } });
        this.target = null; await this.load();
      } catch (e) { this.formError = Banko.explain(e); }
      finally { this.busy = false; }
    },
  }));
});
