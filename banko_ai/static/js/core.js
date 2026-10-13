/* Banko core: the shell data the server rendered, API and SSE clients,
   formatting. Loaded on every page before Alpine. */
(function () {
  "use strict";
  const node = document.getElementById("banko-shell");
  const shell = node ? JSON.parse(node.textContent) : {};
  const FALLBACK = "Something went wrong. Please try again.";

  class ApiError extends Error {
    constructor(status, message) { super(message); this.status = status; }
  }

  // An expired session falls back to the default persona on the server.
  // The page was rendered for one user; if the API answered for another,
  // go sign in again instead of showing someone else's money.
  function guard(res) {
    const served = res.headers.get("X-Banko-User");
    if (served && shell.user && served !== shell.user.user_id) {
      window.location.assign("/login");
      throw new ApiError(401, "Your session ended. Sign in again.");
    }
  }

  async function api(path, { method = "GET", body, form, signal, raw = false } = {}) {
    const opts = { method, signal, credentials: "same-origin", headers: { Accept: "application/json" } };
    if (form) opts.body = form;
    else if (body !== undefined) { opts.headers["Content-Type"] = "application/json"; opts.body = JSON.stringify(body); }
    const res = await fetch(path, opts);
    guard(res);
    if (raw) return res;
    const text = await res.text();
    let data = null;
    if (text) { try { data = JSON.parse(text); } catch (e) { data = null; } }
    if (!res.ok) throw new ApiError(res.status, (data && (data.error || data.message)) || FALLBACK);
    return data;
  }

  // Splits a Server-Sent Events buffer into complete events plus the rest.
  function parseSSE(buffer) {
    const events = [];
    const blocks = buffer.split("\n\n");
    const rest = blocks.pop();
    for (const block of blocks) {
      let event = "message";
      const data = [];
      for (const line of block.split("\n")) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        else if (line.startsWith("data:")) data.push(line.slice(5).replace(/^ /, ""));
      }
      if (!data.length) continue;
      let payload = data.join("\n");
      try { payload = JSON.parse(payload); } catch (e) { /* plain text data */ }
      events.push({ event, data: payload });
    }
    return { events, rest };
  }

  async function stream(path, { body, form, signal, onEvent }) {
    const res = await api(path, { method: "POST", body, form, signal, raw: true });
    if (!res.ok) {
      let message = FALLBACK;
      try { message = (await res.json()).error || FALLBACK; } catch (e) { /* not JSON */ }
      throw new ApiError(res.status, message);
    }
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const { events, rest } = parseSSE(buffer);
      buffer = rest;
      for (const ev of events) onEvent(ev.event, ev.data);
    }
  }

  const usd = new Intl.NumberFormat("en-US", { style: "currency", currency: "USD" });
  const usd0 = new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 0 });

  function day(value) {
    if (!value) return null;
    if (/^\d{4}-\d{2}-\d{2}$/.test(value)) {
      const [y, m, d] = value.split("-").map(Number);
      return new Date(y, m - 1, d);
    }
    return new Date(value);
  }

  const fmt = {
    money(v, { sign = false } = {}) {
      const n = Number(v || 0);
      const s = usd.format(Math.abs(n));
      if (n < 0) return "−" + s;
      return sign && n > 0 ? "+" + s : s;
    },
    money0(v) { const n = Number(v || 0); return (n < 0 ? "−" : "") + usd0.format(Math.abs(n)); },
    pct(x, digits = 0) { return (Number(x || 0) * 100).toFixed(digits) + "%"; },
    date(value, opts = { month: "short", day: "numeric" }) { const d = day(value); return d ? d.toLocaleDateString("en-US", opts) : ""; },
    time(value) { const d = day(value); return d ? d.toLocaleTimeString("en-US", { hour: "numeric", minute: "2-digit" }) : ""; },
    month(label) { const [y, m] = String(label).split("-").map(Number); return new Date(y, m - 1, 1).toLocaleDateString("en-US", { month: "short" }); },
    dayLabel(value) {
      const d = day(value);
      if (!d) return "";
      const today = new Date(); today.setHours(0, 0, 0, 0);
      const that = new Date(d); that.setHours(0, 0, 0, 0);
      const diff = Math.round((today - that) / 86400000);
      if (diff === 0) return "Today";
      if (diff === 1) return "Yesterday";
      return d.toLocaleDateString("en-US", { weekday: "long", month: "short", day: "numeric" });
    },
    relative(value) {
      const d = day(value);
      if (!d) return "";
      const s = Math.max(0, (Date.now() - d.getTime()) / 1000);
      if (s < 60) return "just now";
      if (s < 3600) return Math.floor(s / 60) + " min ago";
      if (s < 86400) return Math.floor(s / 3600) + " h ago";
      if (s < 172800) return "yesterday";
      return d.toLocaleDateString("en-US", { month: "short", day: "numeric" });
    },
  };

  // Deterministic colors per merchant name, so a merchant keeps its color.
  const PALETTE = ["#0F7B5F", "#2B59C3", "#6B4226", "#C9302C", "#6933FF", "#B45309", "#0E7490", "#7C3AED"];
  function monogram(name) {
    const words = String(name || "?").replace(/[^A-Za-z0-9 ]/g, " ").trim().split(/\s+/);
    const letters = ((words[0] || "?")[0] + ((words[1] || "")[0] || "")).toUpperCase();
    let h = 0;
    for (const c of String(name || "")) h = (h * 31 + c.charCodeAt(0)) >>> 0;
    return { letters, color: PALETTE[h % PALETTE.length] };
  }

  function emit(name, detail) { window.dispatchEvent(new CustomEvent(name, { detail })); }

  // Idempotency keys. crypto.getRandomValues works on plain http too, where
  // crypto.randomUUID is unavailable.
  function newKey() {
    const b = crypto.getRandomValues(new Uint8Array(16));
    return Array.from(b, (x) => x.toString(16).padStart(2, "0")).join("");
  }

  // API error codes the UI turns into plain copy. Anything else is already
  // neutral text from the server.
  const COPY = {
    insufficient_funds: "There isn't enough available in that account.",
    key_conflict: "That request was already used with a different amount.",
    not_found: "That item no longer exists.",
  };
  function explain(err) { const m = (err && err.message) || ""; return COPY[m] || m || FALLBACK; }

  window.Banko = { shell, api, stream, parseSSE, fmt, monogram, emit, newKey, explain, ApiError, FALLBACK };
})();
