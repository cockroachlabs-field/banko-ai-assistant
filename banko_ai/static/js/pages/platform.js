document.addEventListener("alpine:init", () => {
  const OPS = ["purchase", "transfer", "card_payment", "deposit", "page_load", "search", "report"];
  const OP_LABELS = { purchase: "Purchases", transfer: "Transfers", card_payment: "Card payments", deposit: "Deposits",
                      page_load: "Page loads", search: "Searches", report: "Reports" };
  const DISRUPTIVE = ["crash", "outage", "stop", "partition", "isolate", "brownout"];
  const EVERY = { live: 1000, cluster: 3000, data: 15000 };
  const ERROR_WORDS = { "40001": "transaction conflicts", "40003": "ambiguous commits", connection: "lost connections",
                        timeout: "timeouts", ambiguous: "commits lost in flight" };
  const SOURCES = { chaos: "From the chaos control plane", status_api: "From CockroachDB's HTTP API",
                    sql: "From SQL: replicas only, liveness unknown" };

  Alpine.data("platformPage", () => ({
    live: null, cluster: null, data: null, events: [], lastEvent: null, offline: false,
    controls: { load: true, chaos: false, defaults: { users: 100, rate: 50, workers: 16 },
                limits: { users: 2000, rate: 2000, workers: 64 } },
    form: { profile: "balanced", users: 100, rate: 50, workers: 16, duration_s: 0, ramp_s: 10,
            weights: Object.fromEntries(OPS.map((op) => [op, 0])) },
    formReady: false, busy: false, checking: false, message: "", charts: null, timers: {}, gone: false,
    ops: OPS,
    profiles: [["balanced", "Balanced"], ["payments", "Payments (writes)"], ["browsing", "Browsing (reads)"],
               ["search", "Search (vectors)"], ["custom", "Custom"]],
    durations: [[0, "Until stopped"], [60, "1 minute"], [300, "5 minutes"], [900, "15 minutes"], [3600, "1 hour"]],

    init() {
      for (const name of Object.keys(EVERY)) this.loop(name);
      this._wake = () => { if (!document.hidden) for (const name of Object.keys(EVERY)) this.fetch(name); };
      document.addEventListener("visibilitychange", this._wake);
    },
    destroy() {
      this.gone = true;
      for (const t of Object.values(this.timers)) clearTimeout(t);
      document.removeEventListener("visibilitychange", this._wake);
      if (this.charts) { this.charts.throughput.destroy(); this.charts.latency.destroy(); }
    },

    loop(name) {
      const tick = async () => {
        if (this.gone) return;
        if (!document.hidden) await this.fetch(name);
        this.timers[name] = setTimeout(tick, EVERY[name]);
      };
      tick();
    },
    async fetch(name) {
      const path = name === "live" ? "/api/v1/platform/live?since=" + (this.lastEvent ?? "") : "/api/v1/platform/" + name;
      try {
        const body = await Banko.api(path, { signal: AbortSignal.timeout(8000) });
        if (name === "live") this.receiveLive(body);
        else this[name] = body;
        if (name === "live") this.offline = false;
      } catch (e) {
        if (name === "live") this.offline = true;
      }
    },
    receiveLive(body) {
      this.live = body;
      this.controls = body.controls;
      const fresh = body.events.filter((e) => this.lastEvent == null || e.id > this.lastEvent);
      if (fresh.length) {
        this.lastEvent = fresh[fresh.length - 1].id;
        this.events = [...fresh.reverse(), ...this.events].slice(0, 100);
      }
      if (!this.formReady) {
        const c = body.load.config;
        const d = body.controls.defaults;
        Object.assign(this.form, c ? { profile: c.profile, users: c.users, rate: c.rate, workers: c.workers,
                                        duration_s: c.duration_s || 0, ramp_s: c.ramp_s }
                                   : { users: d.users, rate: d.rate, workers: d.workers });
        if (c && c.weights) Object.assign(this.form.weights, c.weights);
        this.formReady = true;
      }
      this.$nextTick(() => this.draw());
    },
    draw() {
      if (!this.$refs.throughput) return;
      if (!this.charts) {
        this.charts = {
          throughput: Banko.charts.timeline(this.$refs.throughput, { format: (v) => this.rate(v),
            series: [{ label: "completed", color: "--chart-1" }, { label: "errors", color: "--neg" }] }),
          latency: Banko.charts.timeline(this.$refs.latency, { format: (v) => this.ms(v),
            series: [{ label: "p50", color: "--chart-2" }, { label: "p95", color: "--chart-1" },
                     { label: "p99", color: "--chart-2", dashed: true }] }),
        };
      }
      const s = this.live.series;
      const marks = this.events.map((e) => ({ second: Math.floor(e.at), level: e.level, text: e.text }));
      this.charts.throughput.update(s.t, [s.ops, s.errors], marks);
      this.charts.latency.update(s.t, [s.p50, s.p95, s.p99], marks);
    },

    // -- formatting -----------------------------------------------------------------------------
    rate(v) {
      if (v == null || Number.isNaN(Number(v))) return "–";
      const n = Number(v);
      const s = n < 100 ? n.toFixed(1) : Math.round(n).toLocaleString("en-US");
      return s.endsWith(".0") ? s.slice(0, -2) : s;
    },
    ms(v) {
      if (v == null) return "–";
      const n = Number(v);
      if (n >= 1000) return (n / 1000).toFixed(1).replace(/\.0$/, "") + " s";
      return (n < 10 ? n.toFixed(1).replace(/\.0$/, "") : Math.round(n)) + " ms";
    },
    plural(n, word) { return this.count(n) + " " + word + (Number(n) === 1 ? "" : "s"); },
    count(v) { return v == null ? "–" : Number(v).toLocaleString("en-US"); },
    when(at) { return new Date(at * 1000).toLocaleTimeString("en-US", { hour12: false }); },
    opLabel(op) { return OP_LABELS[op] || op; },

    // -- state ----------------------------------------------------------------------------------
    running() { return !!this.live && ["preparing", "running", "stopping"].includes(this.live.load.state); },
    sum(key) { return this.live ? this.live.summary[key] : null; },
    stateText() {
      if (!this.live) return "Connecting";
      const l = this.live.load;
      if (l.state === "preparing") return "Preparing customers";
      if (l.state === "stopping") return "Stopping";
      if (l.state !== "running") return "Idle";
      const t = Math.round(l.elapsed_s);
      return "Running " + Math.floor(t / 60) + ":" + String(t % 60).padStart(2, "0") + " · " + l.users_ready + " customers";
    },
    stateTone() { return this.live && this.live.load.state === "running" ? "ok" : this.running() ? "warn" : ""; },
    errorClasses() {
      const c = this.live ? this.live.summary.error_classes : {};
      const parts = Object.entries(c || {}).map(([k, n]) => n + " " + (ERROR_WORDS[k] ? ERROR_WORDS[k] + " (" + k + ")" : k));
      return parts.length ? "Errors: " + parts.join(", ") : "";
    },
    opStat(op, key) { return this.live && this.live.ops[op] ? this.live.ops[op][key] : null; },
    opRows() {
      if (!this.live) return [];
      return OPS.filter((op) => this.live.ops[op]).map((op) => ({ op, ...this.live.ops[op] }));
    },
    regionRows() {
      if (!this.live) return [];
      const homed = {};
      for (const u of (this.data && this.data.users) || []) if (u.region) homed[u.region] = u.load;
      const serving = this.live.load.serving || {};
      return Object.entries(this.live.regions).map(([region, s]) => ({
        region, customers: region === "all" ? this.live.load.users_ready : (homed[region] ?? null),
        ops_s: s.ops_s, p95: s.p95, errors: s.errors, served_by: (serving[region] || {}).served_by,
      })).sort((a, b) => a.region.localeCompare(b.region));
    },
    integrityText() {
      const i = this.data && this.data.integrity;
      if (!i) return "";
      if (i.skipped) return i.skipped;
      return this.count(i.accounts) + " accounts checked " + Banko.fmt.relative(i.checked_at) +
             (i.follower_reads ? ", with follower reads" : "");
    },
    sourceText() { return this.cluster && this.cluster.topology ? SOURCES[this.cluster.topology.source] || "" : ""; },
    regionGroups() {
      const t = this.cluster && this.cluster.topology;
      if (!t) return [];
      const groups = {};
      for (const n of t.nodes) (groups[n.region || "this cluster"] ||= []).push(n);
      return Object.entries(groups).sort().map(([region, nodes]) => {
        const known = nodes.filter((n) => n.live !== null && n.live !== undefined);
        const up = known.filter((n) => n.live).length;
        return { region, nodes, summary: known.length ? up + " of " + nodes.length + " live" : this.plural(nodes.length, "node") };
      });
    },
    nodeState(n) {
      if (n.live === true) return n.leases == null ? "live" : "live · " + this.plural(n.leases, "lease");
      if (n.live === false) return n.status || "down";
      return n.replicas == null ? "" : this.plural(n.replicas, "replica");
    },
    leaseRows() {
      const by = (this.cluster && this.cluster.leases && this.cluster.leases.by_region) || {};
      const top = Math.max(1, ...Object.values(by));
      return Object.entries(by).sort().map(([region, leases]) => ({ region, leases, pct: Math.round((leases / top) * 100) }));
    },
    chaosName() { return this.cluster && this.cluster.chaos ? this.cluster.chaos.cluster.display_name || "cluster" : ""; },
    chaosLatency() { return this.cluster && this.cluster.chaos ? this.cluster.chaos.cluster.latency : ""; },
    eventTone(e) { return { bad: "bad", warn: "warn", good: "ok" }[e.level] || ""; },
    eventWord(e) { return { bad: "Problem", warn: "Warning", good: "Recovered" }[e.level] || e.source; },

    // -- actions --------------------------------------------------------------------------------
    payload(action) {
      const f = this.form;
      const body = { action, profile: f.profile, users: f.users, rate: f.rate, workers: f.workers,
                     duration_s: f.duration_s, ramp_s: f.ramp_s };
      if (f.profile === "custom") body.weights = { ...f.weights };
      return body;
    },
    async post(path, body) {
      return Banko.api("/api/v1/platform" + path, { method: "POST", body });
    },
    async act(fn) {
      this.busy = true;
      this.message = "";
      try { await fn(); } catch (e) { this.message = Banko.explain(e); } finally { this.busy = false; }
      this.fetch("live");
    },
    start() { return this.act(() => this.post("/load", this.payload("start"))); },
    apply() { return this.act(() => this.post("/load", this.payload("apply"))); },
    stop() { return this.act(() => this.post("/load", { action: "stop" })); },
    clearUsers() {
      if (!window.confirm("Remove every load customer and everything they own?")) return;
      return this.act(() => this.post("/load/users", { action: "clear" }));
    },
    checkIntegrity() {
      this.checking = true;
      this.post("/integrity/check", {})
        .then((report) => { if (this.data) this.data.integrity = report; })
        .catch((e) => { this.message = Banko.explain(e); })
        .finally(() => { this.checking = false; });
    },
    chaos(action, kind, value, label) {
      if (DISRUPTIVE.includes(action) && !window.confirm(label + "? This breaks the cluster for real until you recover it.")) return;
      return this.act(async () => {
        try {
          await this.post("/chaos", { action, kind, value, confirm: false });
        } catch (e) {
          if (e.status !== 409) throw e;
          if (!window.confirm("The control plane warns: " + Banko.explain(e) + ". Do it anyway?")) return;
          await this.post("/chaos", { action, kind, value, confirm: true });
        }
        this.fetch("cluster");
      });
    },
    latency(setting) { return this.act(() => this.post("/chaos/latency", { setting })); },
  }));
});
