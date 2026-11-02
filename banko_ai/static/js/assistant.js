/* The assistant, shared by the slide-over panel (every page) and the full
   Assistant page. Streams answers over SSE, shows tool steps, citations and
   proposals, uploads receipts, and keeps the "How this was answered" trace. */
document.addEventListener("alpine:init", () => {
  "use strict";
  let seq = 0;
  // The same shape answers.strip_citations removes: any [txn:...] marker, well formed or not.
  const MARKER = /[ \t]*\[\s*(?:cite\s*:\s*)?txn\s*:[^\]]*\]/gi;
  Banko.stripCitations = (text) => String(text || "").replace(MARKER, "");
  // Model answers are Markdown, so only the tags Markdown makes survive: no
  // style (one answer could blank the app), no forms and no images (an
  // image would fetch an outside URL). Links open in a new tab, without the
  // opener.
  const PURIFY = {
    ALLOWED_TAGS: ["p", "br", "hr", "strong", "b", "em", "i", "del", "s", "code", "pre", "blockquote",
                   "ul", "ol", "li", "a", "h1", "h2", "h3", "h4", "table", "thead", "tbody", "tr", "th", "td"],
    ALLOWED_ATTR: ["href", "title", "align", "start"],
    ALLOW_DATA_ATTR: false,
  };
  window.DOMPurify.addHook("afterSanitizeAttributes", (node) => {
    if (node.tagName === "A" && node.hasAttribute("href")) {
      node.setAttribute("target", "_blank");
      node.setAttribute("rel", "noopener noreferrer");
    }
  });
  Banko.renderAnswer = (text) =>
    window.DOMPurify.sanitize(window.marked.parse(Banko.stripCitations(text), { breaks: true }), PURIFY);
  // One trace row's detail and timing, with a separator only when both exist.
  Banko.traceLine = (s) => [s.detail, s.ms ? Math.round(s.ms) + " ms" : ""].filter(Boolean).join(" · ");
  const duration = (ms) => (!ms ? "" : ms >= 1000 ? (ms / 1000).toFixed(1) + " s" : Math.round(ms) + " ms");
  const count = (n) => Number(n || 0).toLocaleString("en-US");
  const RECEIPT_TYPES = /^(image\/|application\/pdf$)/;
  const STEP_LABELS = { ocr: "OCR", extract: "Extract", category: "Category", duplicate: "Dup check", post: "Posted", fraud: "Fraud", budget: "Budget" };
  // The panel remembers its thread per user, so after "Switch user" another
  // persona's thread never loads.
  const threadKey = () => "banko.thread:" + Banko.shell.user.user_id;
  const newBot = () => Alpine.reactive({ id: ++seq, role: "bot", steps: [], answer: "", citations: [], proposals: [], meta: {},
                                         route: "", cached: false, error: "", stopped: false, ms: 0, showTrace: false,
                                         answered: false, done: false, unsent: false, more: [], next: null, paging: false });
  // An answer that searched can page on: meta.search names the search and
  // the cursor of its next page, and a typed "show more" brings that page
  // along as items.
  const paged = (bot) => {
    const s = (bot.meta && bot.meta.search) || {};
    bot.more = (s.items || []).slice(); bot.next = s.next || null;
    return bot;
  };
  // A saved message as the chat shows it. An answer this server sent keeps
  // its meta (trace, plan, tokens) until the server restarts.
  const fromServer = (m) => m.role === "user"
    ? { id: ++seq, role: "user", text: m.content }
    : paged(Object.assign(newBot(), { answer: m.content, meta: m.meta || {}, cached: !!(m.meta && m.meta.cached),
                                      answered: true, done: true }));
  // A history read after the new turn landed already ends with it: keep the
  // exchange on screen and drop its copy from the history.
  function before(history, added) {
    const asked = added.find((m) => m.role === "user");
    const n = history.length;
    const last = n && history[n - 1].role === "user" ? n - 1 : n > 1 && history[n - 2].role === "user" ? n - 2 : -1;
    return asked && last >= 0 && history[last].text === asked.text ? history.slice(0, last) : history;
  }

  Alpine.data("assistant", (mode) => {
  // A thread switch in flight: a history load or a new thread. Anything that
  // adds a message or changes the thread waits for it, so a history that
  // arrives late never replaces what the user just did.
  let pending = null;
  const track = (promise) => {
    const p = promise.finally(() => { if (pending === p) pending = null; });
    pending = p;
    return p;
  };
  // A new conversation on its way: the thread the next question goes to.
  let creating = null;
  // Bumps whenever the conversation on screen is replaced, so a history that
  // was asked for before never lands in the new one.
  let epoch = 0;
  // Threads whose title was read back after an answer in this page.
  const titled = new Set();
  return {
    mode, open: mode === "page", threadId: null, threadTitle: "", messages: [], draft: "",
    streaming: false, controller: null, current: null,
    language: localStorage.getItem("banko.language") || (Banko.shell.languages[0] || { code: "en-US" }).code,
    languages: Banko.shell.languages,
    voiceSupported: !!(window.SpeechRecognition || window.webkitSpeechRecognition),
    canSpeak: "speechSynthesis" in window,
    listening: false, recognition: null,
    autoRead: localStorage.getItem("banko.autoread") === "1",
    // "How CockroachDB ran it" folds to its heading for every answer at once,
    // and stays that way across answers and reloads until opened again.
    plansOpen: localStorage.getItem("banko.plans") !== "closed",

    init() {
      if (this.mode === "panel") {
        localStorage.removeItem("banko.thread");  // the old entry, shared by every user
        this.threadId = localStorage.getItem(threadKey());
      }
      window.addEventListener("banko:ask", (e) => this.ask(e.detail || {}));
      window.addEventListener("banko:receipt", async (e) => { await this.show(); this.upload(e.detail.file); });
      window.addEventListener("banko:pick-receipt", () => { this.show(); this.$nextTick(() => this.$refs.file.click()); });
      this.$watch("language", (v) => localStorage.setItem("banko.language", v));
      this.initVoice();
    },

    initVoice() {
      this.$watch("autoRead", (v) => localStorage.setItem("banko.autoread", v ? "1" : "0"));
      window.addEventListener("keydown", (e) => {
        if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "m" && this.voiceSupported) {
          if (this.mode === "panel" && document.querySelector('[x-data="assistant(\'page\')"]')) return;
          e.preventDefault(); this.show(); this.toggleVoice();
        }
      });
      window.addEventListener("banko:answered", (e) => {
        if (this.autoRead && e.detail.thread === this.threadId && e.detail.bot.answer) this.speak(e.detail.bot.answer);
      });
    },

    toggleVoice() {
      if (!this.voiceSupported) return;
      if (this.listening) { this.recognition && this.recognition.stop(); return; }
      const Rec = window.SpeechRecognition || window.webkitSpeechRecognition;
      const rec = new Rec();
      rec.lang = this.language; rec.interimResults = true; rec.maxAlternatives = 1;
      let finalText = "";
      rec.onresult = (ev) => {
        let interim = "";
        for (let i = ev.resultIndex; i < ev.results.length; i++) {
          const r = ev.results[i];
          if (r.isFinal) finalText += r[0].transcript; else interim += r[0].transcript;
        }
        this.draft = (finalText + interim).trim();
      };
      rec.onerror = () => { this.listening = false; };
      rec.onend = () => { this.listening = false; if (finalText.trim()) this.send(finalText.trim()); };
      this.recognition = rec; this.listening = true; rec.start();
    },

    speak(text) {
      if (!this.canSpeak) return;
      const plain = Banko.stripCitations(text).replace(/[*_`#>|]/g, " ");
      window.speechSynthesis.cancel();
      const u = new SpeechSynthesisUtterance(plain);
      u.lang = this.language;
      const prefix = this.language.slice(0, 2);
      const voice = window.speechSynthesis.getVoices().find((v) => v.lang === this.language) ||
                    window.speechSynthesis.getVoices().find((v) => v.lang.startsWith(prefix));
      if (voice) u.voice = voice;
      window.speechSynthesis.speak(u);
    },

    // Resolves once no thread switch is in flight; never rejects.
    settled() { return pending ? pending.then(() => {}, () => {}) : Promise.resolve(); },

    show() {
      if (this.mode !== "panel") return Promise.resolve();
      if (!this.open) { this.open = true; if (this.threadId && !this.messages.length) this.loadThread(this.threadId); }
      this.$nextTick(() => this.$refs.input && this.$refs.input.focus());
      return this.settled();
    },
    hide() { if (this.mode === "panel") this.open = false; },

    // A palette ask, "Ask Banko about this" or New conversation. Nothing here
    // waits before the question is on screen: send shows it at once.
    ask({ text, context, fresh }) {
      this.show();
      const thread = fresh || context ? this.newThread(context || null) : null;
      if (text) return this.send(text, thread);
      return thread ? thread.then(() => {}, (e) => this.notify(e)) : Promise.resolve();
    },

    notify(e) { Alpine.store("toasts").push({ title: "Assistant", text: Banko.explain(e), kind: "error" }); },

    remember(id, title) {
      this.threadId = id; this.threadTitle = title || this.threadTitle;
      if (this.mode === "panel") localStorage.setItem(threadKey(), id);
      Banko.emit("banko:thread", { id });
    },

    forget() {
      this.threadId = null; this.threadTitle = "";
      if (this.mode === "panel") localStorage.removeItem(threadKey());
    },

    // A thread for a new conversation. A fresh or context thread (replace)
    // takes the conversation off the screen at once; the thread a first
    // question needs keeps that question on screen. If it can't be made, the
    // conversation that was there comes back.
    newThread(context, { replace = true } = {}) {
      const shown = { messages: this.messages, title: this.threadTitle };
      if (replace) { epoch++; this.messages = []; this.threadTitle = ""; }
      const mine = epoch;
      const made = track((async () => {
        try {
          const body = context ? { context } : {};
          const thread = await Banko.api("/api/v1/assistant/threads", { method: "POST", body });
          if (mine === epoch) this.remember(thread.thread_id, thread.title);
          return thread.thread_id;
        } catch (e) {
          if (replace && mine === epoch) {
            this.messages = shown.messages.concat(this.messages); this.threadTitle = shown.title;
            // Its history was still on the way when the screen cleared: ask again.
            if (this.threadId && !shown.messages.length) this.loadThread(this.threadId);
          }
          throw e;
        }
      })());
      creating = made;
      made.then(() => {}, () => {}).then(() => { if (creating === made) creating = null; });
      return made;
    },

    loadThread(id) {
      const mine = epoch;
      const at = this.messages.length;
      return track((async () => {
        try {
          const t = await Banko.api("/api/v1/assistant/threads/" + id);
          // The user moved on while it loaded: another thread or a new conversation.
          if (mine !== epoch || this.threadId !== id) return;
          this.remember(t.thread_id, t.title);
          // What they did meanwhile (a question, a receipt) stays, below its history.
          const added = this.messages.slice(at);
          this.messages = before((t.messages || []).map(fromServer), added).concat(added);
          this.scroll();
        } catch (e) {
          if (mine !== epoch || this.threadId !== id) return;
          // Gone (deleted, or not this user's): start fresh. A blip keeps
          // the thread, so the next question still continues it.
          if (e instanceof Banko.ApiError && e.status === 404) this.forget();
          if (this.mode !== "panel") this.notify(e);
        }
      })());
    },

    // The thread a question goes to: a new conversation on its way decides
    // it (and if that fails, the question fails with it); otherwise the
    // thread on screen once its history is in, or a new one.
    async threadFor(thread) {
      const making = thread || creating;
      if (making) return making;
      await this.settled();
      return this.threadId || this.newThread(null, { replace: false });
    },

    async send(text, thread) {
      text = (text || this.draft).trim();
      if (!text || this.streaming) return;
      // Busy before anything is awaited, so a second send is ignored.
      this.streaming = true;
      this.draft = "";
      this.messages = this.messages.filter((m) => !m.unsent);
      const lang = this.languages.find((l) => l.code === this.language);
      const user = { id: ++seq, role: "user", text, lang: lang ? lang.name : "" };
      const bot = newBot();
      this.messages.push(user, bot);
      this.current = bot;
      this.scroll();
      const t0 = performance.now();
      let tid;
      try { tid = await this.threadFor(thread); }
      catch (e) {
        // It never left: the question goes back to the composer, so trying
        // again is one click, and the bubble says why.
        this.messages = this.messages.filter((m) => m.id !== user.id);
        bot.error = Banko.explain(e); bot.unsent = true;
        if (!this.draft) this.draft = text;
        this.notify(e);
        this.streaming = false; this.current = null;
        return;
      }
      this.controller = new AbortController();
      try {
        await Banko.stream("/api/v1/assistant/threads/" + tid + "/messages", {
          body: { message: text, language: this.language }, signal: this.controller.signal,
          onEvent: (event, data) => this.onAnswerEvent(bot, event, data, t0, tid),
        });
      } catch (e) {
        if (e.name === "AbortError") { if (!bot.answered) bot.stopped = true; }
        else if (!bot.answered) bot.error = Banko.explain(e);
      } finally {
        this.finish(bot, t0, tid);
        // The answer may be out already; the thread was being put away until now.
        bot.done = true;
        this.streaming = false; this.controller = null; this.current = null;
        Banko.emit("banko:turn-end", { thread: tid });
        if (!bot.error && !bot.stopped && !titled.has(tid)) { titled.add(tid); this.refreshTitle(tid); }
      }
    },

    // The bubble as answered: meta line, trace and plan show. Once per bubble,
    // at the answer event, or when the stream ends without one.
    finish(bot, t0, tid) {
      if (bot.answered) return;
      bot.ms = Math.round(performance.now() - t0);
      bot.steps = bot.steps.filter((s) => s.status !== "running");
      // A turn that ended with no answer and no error still failed.
      if (!bot.answer && !bot.error && !bot.stopped) bot.error = Banko.FALLBACK;
      bot.answered = true;
      this.scroll();
      Banko.emit("banko:answered", { thread: tid, bot });
    },

    // The server titles a thread from its first question, so read the
    // title back once the first answer is in.
    async refreshTitle(tid) {
      try {
        const found = ((await Banko.api("/api/v1/assistant/threads")).threads || []).find((t) => t.thread_id === tid);
        if (found && this.threadId === tid) this.threadTitle = found.title;
      } catch (e) { /* the title is only a label */ }
    },

    onAnswerEvent(bot, event, data, t0, tid) {
      if (event === "step") this.onStep(bot, data);
      else if (event === "token") { bot.answer += data.text; this.scroll(); }
      else if (event === "citation") { if (!bot.citations.some((c) => c.expense_id === data.expense_id)) bot.citations.push(data); }
      else if (event === "proposal") bot.proposals.push({ ...data });
      else if (event === "answer") {
        bot.answer = data.answer || bot.answer;
        bot.citations = data.citations && data.citations.length ? data.citations : bot.citations;
        bot.meta = data.meta || {}; bot.route = data.route || ""; bot.cached = !!data.cached;
        paged(bot);
        this.finish(bot, t0, tid);
      } else if (event === "error") {
        bot.error = data.message || Banko.FALLBACK;
        if (data.trace) bot.meta = { ...bot.meta, trace: data.trace };
      }
    },

    // A running step is a chip with a spinner and the status line; the step
    // that finishes it (same key) takes its place.
    onStep(bot, s) {
      const step = { ...s, uid: ++seq };
      const at = s.key && s.status !== "running" ? bot.steps.findIndex((x) => x.key === s.key && x.status === "running") : -1;
      if (at >= 0) bot.steps.splice(at, 1, step);
      else bot.steps.push(step);
    },

    working(m) { return !m.answered && !m.error && !m.stopped; },
    statusLine(m) {
      const running = m.steps.filter((s) => s.status === "running");
      return (running.length ? running[running.length - 1].label : m.answer ? "Writing the answer" : "Thinking") + "…";
    },
    stepNote(s) {
      if (s.status === "running") return "";
      return [s.status === "hit" || s.status === "miss" ? s.status : "", duration(s.ms)].filter(Boolean).join(" · ");
    },
    // "18.2 s · 3 model calls · 2,950 tokens", or "1.2 s · cached · about 350 tokens saved".
    metaLine(m) {
      const meta = m.meta || {};
      const t = meta.tokens || {};
      const ms = m.ms || meta.answer_ms || 0;
      const calls = meta.model_calls || 0;
      const spent = (t.input || 0) + (t.output || 0);
      return [ms ? (ms / 1000).toFixed(1) + " s" : "",
              m.cached || meta.cached ? "cached" : "",
              calls ? calls + " model call" + (calls === 1 ? "" : "s") : "",
              spent ? (t.estimated ? "about " : "") + count(spent) + " tokens" : "",
              t.saved ? "about " + count(t.saved) + " tokens saved" : ""].filter(Boolean).join(" · ");
    },
    hasTrace(m) { return !!((m.meta && m.meta.trace && m.meta.trace.length) || m.steps.some((s) => s.status !== "running")); },
    // The Stop button, until the answer is out (the rest is bookkeeping).
    answering() { return this.streaming && !!this.current && !this.current.answered; },

    stop() { if (this.controller) this.controller.abort(); },

    async upload(file) {
      if (!file) return;
      if (!RECEIPT_TYPES.test(file.type || "")) {
        Alpine.store("toasts").push({ title: "Receipts", text: "Upload a receipt image or PDF.", kind: "error" });
        return;
      }
      await this.settled();
      const card = Alpine.reactive({ id: ++seq, role: "receipt", name: file.name, pipeline: {}, receipt: null, error: "" });
      this.messages.push(card);
      this.scroll();
      const form = new FormData();
      form.append("receipt", file, file.name || "receipt.jpg");
      if (this.threadId) form.append("thread_id", this.threadId);
      try {
        await Banko.stream("/api/v1/receipts", { form, onEvent: (event, data) => {
          if (event === "step") card.pipeline[data.step] = data;
          else if (event === "receipt") { card.receipt = data; if (data.thread_id) this.remember(data.thread_id); }
          else if (event === "error") card.error = data.message || Banko.FALLBACK;
        } });
      } catch (e) { card.error = Banko.explain(e); }
      this.scroll();
    },
    pickFile(e) { const f = e.target.files[0]; e.target.value = ""; this.upload(f); },
    pipelineSteps(card) {
      return Object.keys(STEP_LABELS).filter((k) => card.pipeline[k] && card.pipeline[k].status !== "start")
        .map((k) => ({ key: k, label: STEP_LABELS[k], ...card.pipeline[k] }));
    },

    async decide(p, verb) {
      try {
        const out = await Banko.api("/api/v1/actions/" + p.action_id + "/" + verb, { method: "POST" });
        p.status = out.status;
        if (out.status === "done") Alpine.store("toasts").push({ title: "Done", text: p.summary });
      } catch (e) { p.error = Banko.explain(e); }
    },
    confirm(p) { return this.decide(p, "confirm"); },
    cancel(p) { return this.decide(p, "cancel"); },

    render(text) { return Banko.renderAnswer(text); },
    async copyText(plain, what) {
      try { await navigator.clipboard.writeText(plain); }
      catch (e) {
        const area = document.createElement("textarea");
        area.value = plain; document.body.appendChild(area); area.select(); document.execCommand("copy"); area.remove();
      }
      Alpine.store("toasts").push({ title: "Copied", text: what + " on your clipboard." });
    },
    // The exchange, question and answer, ready to paste into a note.
    copy(m) {
      const asked = this.messages.slice(0, this.messages.indexOf(m)).reverse().find((x) => x.role === "user");
      const answer = Banko.stripCitations(m.answer);
      return asked ? this.copyText("Q: " + asked.text + "\n\nA: " + answer, "The question and answer are")
                   : this.copyText(answer, "The answer is");
    },
    // One message alone: the question, or the answer.
    copyMessage(m) {
      return m.role === "user" ? this.copyText(m.text, "The question is")
                               : this.copyText(Banko.stripCitations(m.answer), "The answer is");
    },
    toggleTrace(m) { m.showTrace = !m.showTrace; Banko.emit("banko:trace", { message: m }); },
    togglePlans() { this.plansOpen = !this.plansOpen; localStorage.setItem("banko.plans", this.plansOpen ? "open" : "closed"); },
    traceSteps(m) { return m.meta && m.meta.trace && m.meta.trace.length ? m.meta.trace : m.steps; },
    stepClass(s) {
      return { running: "run", warning: "warn", error: "warn", hit: "hit", miss: "miss" }[s.status] || "ok";
    },
    openCitation(c) { Banko.emit("banko:open-transaction", { id: c.expense_id }); },
    // The next page of the answer's search, straight from the API: no model,
    // no tokens.
    async showMore(m) {
      const s = m.meta.search;
      if (!s || !m.next || m.paging) return;
      m.paging = true;
      try {
        const p = new URLSearchParams({ q: s.q, limit: String(Banko.shell.search_page_size), cursor: m.next });
        if (s.category) p.set("category", s.category);
        if (s.from) p.set("from", s.from);
        if (s.to) p.set("to", s.to);
        const data = await Banko.api("/api/v1/transactions?" + p);
        m.more = m.more.concat(data.items.map((i) => ({ expense_id: i.expense_id, merchant: i.merchant,
                                                         amount: Math.abs(i.amount), date: i.posted_at })));
        m.next = data.next_cursor;
      } catch (e) { this.notify(e); }
      finally { m.paging = false; }
    },

    async startPage(threadId) {
      if (!threadId) return;
      this.threadId = threadId;
      await this.loadThread(threadId);
    },

    onKey(e) {
      if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); this.send(); }
      else if (e.key === "Escape") this.hide();
    },
    scroll() { this.$nextTick(() => { const box = this.$refs.chat; if (box) box.scrollTop = box.scrollHeight; }); },
  };
  });
});
