/* The shell: theme, greeting, command palette, toasts, Socket.IO and
   drop-a-receipt-anywhere. */
document.addEventListener("alpine:init", () => {
  "use strict";
  let nextToast = 1;
  Alpine.store("toasts", {
    items: [],
    push({ title, text, kind = "info", action = null, sticky = false }) {
      const id = nextToast++;
      this.items.push({ id, title, text, kind, action });
      if (!sticky) setTimeout(() => this.drop(id), 9000);
      return id;
    },
    drop(id) { this.items = this.items.filter((t) => t.id !== id); },
  });

  const darkQuery = window.matchMedia("(prefers-color-scheme: dark)");
  const sprite = (name) => document.querySelector('link[rel="stylesheet"]').href.replace(/css\/app\.css.*$/, "vendor/lucide-sprite.svg") + "#i-" + name;

  Alpine.data("shell", () => ({
    theme: localStorage.getItem("banko.theme") || Banko.shell.theme || "system",
    sideOpen: false, menuOpen: false, paletteOpen: false,
    query: "", found: [], active: 0, dropping: false, dragDepth: 0,

    init() {
      this.applyTheme();
      darkQuery.addEventListener("change", () => this.applyTheme());
      this.connect();
      window.addEventListener("banko:open-transaction", (e) => {
        if (Banko.shell.page !== "transactions") window.location.assign("/transactions?open=" + encodeURIComponent(e.detail.id));
      });
      window.addEventListener("dragenter", (e) => this.dragEnter(e));
      window.addEventListener("dragleave", () => this.dragLeave());
      window.addEventListener("dragover", (e) => { if (this.hasFiles(e)) e.preventDefault(); });
      window.addEventListener("drop", (e) => this.dropFile(e));
      window.addEventListener("paste", (e) => this.pasteFile(e));
      window.addEventListener("banko:set-theme", (e) => {
        this.theme = e.detail.theme;
        localStorage.setItem("banko.theme", this.theme);
        this.applyTheme();
      });
    },

    applyTheme() {
      const dark = this.theme === "dark" || (this.theme === "system" && darkQuery.matches);
      document.documentElement.dataset.theme = dark ? "dark" : "light";
      Banko.emit("banko:theme", { dark });
    },
    cycleTheme() {
      this.theme = { light: "dark", dark: "system", system: "light" }[this.theme] || "system";
      localStorage.setItem("banko.theme", this.theme);
      this.applyTheme();
    },
    themeIcon() { return sprite({ light: "sun", dark: "moon", system: "monitor" }[this.theme] || "monitor"); },

    greeting() {
      const h = new Date().getHours();
      const part = h < 12 ? "Good morning" : h < 18 ? "Good afternoon" : "Good evening";
      return part + ", " + Banko.shell.user.display_name;
    },
    today() { return new Date().toLocaleDateString("en-US", { weekday: "long", month: "long", day: "numeric" }); },

    onKey(e) {
      const mod = e.metaKey || e.ctrlKey;
      if (mod && e.key.toLowerCase() === "k") { e.preventDefault(); this.paletteOpen ? (this.paletteOpen = false) : this.openPalette(); }
      else if (e.key === "Escape") { this.paletteOpen = false; this.menuOpen = false; this.sideOpen = false; }
    },

    openPalette() {
      this.paletteOpen = true; this.query = ""; this.found = []; this.active = 0;
      this.$nextTick(() => this.$refs.palette.focus());
    },
    async searchTransactions() {
      const q = this.query.trim();
      this.active = 0;
      if (q.length < 3) { this.found = []; return; }
      try {
        const data = await Banko.api("/api/v1/transactions?q=" + encodeURIComponent(q) + "&limit=6");
        this.found = data.items || [];
      } catch (e) { this.found = []; }
    },
    paletteGroups() {
      const q = this.query.trim().toLowerCase();
      let index = 0;
      const groups = [];
      const ask = q ? [{ id: "ask", icon: "sparkles", label: "Ask " + Banko.shell.brand.name + ": " + this.query.trim(), kind: "ask" }] : [];
      const pages = [];
      for (const g of Banko.shell.nav) for (const item of g.items) {
        if (!q || item.label.toLowerCase().includes(q)) pages.push({ id: "page-" + item.key, icon: item.icon, label: item.label, kind: "page", href: item.href });
      }
      const actions = [
        { id: "receipt", icon: "receipt", label: "Add a receipt", kind: "receipt" },
        { id: "new-chat", icon: "message-square", label: "New conversation", kind: "chat" },
      ].filter((a) => !q || a.label.toLowerCase().includes(q));
      const txns = this.found.map((t) => ({ id: "t-" + t.id, icon: "arrow-left-right", kind: "txn", txn: t.expense_id || t.id,
        label: (t.merchant || t.description) + " · " + Banko.fmt.money(t.amount), hint: Banko.fmt.date(t.posted_at) }));
      for (const [title, items] of [["Ask", ask], ["Transactions", txns], ["Pages", pages], ["Actions", actions]]) {
        if (items.length) groups.push({ title, items: items.map((i) => ({ ...i, index: index++ })) });
      }
      return groups;
    },
    paletteKey(e) {
      const all = this.paletteGroups().flatMap((g) => g.items);
      if (e.key === "ArrowDown") { e.preventDefault(); this.active = Math.min(this.active + 1, all.length - 1); }
      else if (e.key === "ArrowUp") { e.preventDefault(); this.active = Math.max(this.active - 1, 0); }
      else if (e.key === "Enter") { e.preventDefault(); const item = all[this.active]; if (item) this.run(item); }
    },
    run(item) {
      this.paletteOpen = false;
      if (item.kind === "page") window.location.assign(item.href);
      else if (item.kind === "ask") Banko.emit("banko:ask", { text: this.query.trim() });
      else if (item.kind === "chat") Banko.emit("banko:ask", { fresh: true });
      else if (item.kind === "receipt") Banko.emit("banko:pick-receipt", {});
      else if (item.kind === "txn") Banko.emit("banko:open-transaction", { id: item.txn });
    },

    connect() {
      if (!window.io) return;
      const socket = window.io({ transports: ["websocket", "polling"] });
      socket.on("connect", () => socket.emit("coach.join", { user_id: Banko.shell.user.user_id }));
      socket.on("coach.nudge", (n) => {
        Alpine.store("toasts").push({ title: "Coach", text: n.message, sticky: true,
          action: { label: "Ask " + Banko.shell.brand.name + " about this", run: () => Banko.emit("banko:ask", { context: { kind: "nudge", id: n.nudge_id } }) } });
        Banko.emit("banko:nudge", n);
      });
      socket.on("agent_activity", (a) => Banko.emit("banko:agent", { source: "activity", ...a }));
      socket.on("agent_decision", (d) => Banko.emit("banko:agent", { source: "decision", ...d }));
      socket.on("seeding", (p) => Banko.emit("banko:seeding", p));
      window.BankoSocket = socket;
    },

    hasFiles(e) { return !!(e.dataTransfer && Array.from(e.dataTransfer.types || []).includes("Files")); },
    dragEnter(e) { if (!this.hasFiles(e)) return; this.dragDepth += 1; this.dropping = true; },
    dragLeave() { this.dragDepth = Math.max(0, this.dragDepth - 1); if (!this.dragDepth) this.dropping = false; },
    dropFile(e) {
      if (!this.hasFiles(e)) return;
      e.preventDefault();
      this.dragDepth = 0; this.dropping = false;
      const file = e.dataTransfer.files[0];
      if (file) Banko.emit("banko:receipt", { file });
    },
    pasteFile(e) {
      const item = Array.from((e.clipboardData && e.clipboardData.items) || []).find((i) => i.kind === "file" && i.type.startsWith("image/"));
      if (!item) return;
      e.preventDefault();
      Banko.emit("banko:receipt", { file: item.getAsFile() });
    },
  }));
});
