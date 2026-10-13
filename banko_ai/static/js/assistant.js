/* The assistant, shared by the slide-over panel (every page) and the full
   Assistant page. Streams answers over SSE, shows tool steps, citations and
   proposals, uploads receipts, and keeps the "How this was answered" trace. */
document.addEventListener("alpine:init", () => {
  "use strict";
  let seq = 0;
  // The same shape answers.strip_citations removes: any [txn:...] marker, well formed or not.
  const MARKER = /[ \t]*\[\s*txn\s*:[^\]]*\]/gi;
  Banko.stripCitations = (text) => String(text || "").replace(MARKER, "");
  const RECEIPT_TYPES = /^(image\/|application\/pdf$)/;
  const STEP_LABELS = { ocr: "OCR", extract: "Extract", category: "Category", duplicate: "Dup check", post: "Posted", fraud: "Fraud", budget: "Budget" };

  Alpine.data("assistant", (mode) => ({
    mode, open: mode === "page", threadId: null, threadTitle: "", messages: [], draft: "",
    streaming: false, controller: null,
    language: localStorage.getItem("banko.language") || (Banko.shell.languages[0] || { code: "en-US" }).code,
    languages: Banko.shell.languages,
    voiceSupported: !!(window.SpeechRecognition || window.webkitSpeechRecognition),
    canSpeak: "speechSynthesis" in window,
    listening: false, recognition: null,
    autoRead: localStorage.getItem("banko.autoread") === "1",

    init() {
      if (this.mode === "panel") this.threadId = localStorage.getItem("banko.thread");
      window.addEventListener("banko:ask", (e) => this.ask(e.detail || {}));
      window.addEventListener("banko:receipt", (e) => { this.show(); this.upload(e.detail.file); });
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

    show() {
      if (this.mode !== "panel") return;
      if (!this.open) { this.open = true; if (this.threadId && !this.messages.length) this.loadThread(this.threadId); }
      this.$nextTick(() => this.$refs.input && this.$refs.input.focus());
    },
    hide() { if (this.mode === "panel") this.open = false; },

    async ask({ text, context, fresh }) {
      this.show();
      if (fresh || context) await this.newThread(context || null);
      if (text) await this.send(text);
    },

    remember(id, title) {
      this.threadId = id; this.threadTitle = title || this.threadTitle;
      if (this.mode === "panel") localStorage.setItem("banko.thread", id);
      Banko.emit("banko:thread", { id });
    },

    async newThread(context) {
      const body = context ? { context } : {};
      const thread = await Banko.api("/api/v1/assistant/threads", { method: "POST", body });
      this.messages = [];
      this.remember(thread.thread_id, thread.title);
      return thread.thread_id;
    },

    async loadThread(id) {
      try {
        const t = await Banko.api("/api/v1/assistant/threads/" + id);
        this.remember(t.thread_id, t.title);
        this.messages = (t.messages || []).map((m) => m.role === "user"
          ? { id: ++seq, role: "user", text: m.content }
          : { id: ++seq, role: "bot", answer: m.content, steps: [], citations: [], proposals: [], meta: {}, showTrace: false });
        this.scroll();
      } catch (e) {
        if (this.mode === "panel") { localStorage.removeItem("banko.thread"); this.threadId = null; this.messages = []; }
        else Alpine.store("toasts").push({ title: "Assistant", text: Banko.explain(e), kind: "error" });
      }
    },

    async ensureThread() { return this.threadId || (await this.newThread(null)); },

    async send(text) {
      text = (text || this.draft).trim();
      if (!text || this.streaming) return;
      this.draft = "";
      const tid = await this.ensureThread();
      const lang = this.languages.find((l) => l.code === this.language);
      this.messages.push({ id: ++seq, role: "user", text, lang: lang ? lang.name : "" });
      const bot = Alpine.reactive({ id: ++seq, role: "bot", steps: [], answer: "", citations: [], proposals: [], meta: {},
                                    route: "", cached: false, error: "", stopped: false, ms: 0, showTrace: false });
      this.messages.push(bot);
      this.scroll();
      this.streaming = true;
      this.controller = new AbortController();
      const t0 = performance.now();
      try {
        await Banko.stream("/api/v1/assistant/threads/" + tid + "/messages", {
          body: { message: text, language: this.language }, signal: this.controller.signal,
          onEvent: (event, data) => this.onAnswerEvent(bot, event, data),
        });
      } catch (e) {
        if (e.name === "AbortError") bot.stopped = true;
        else bot.error = Banko.explain(e);
      } finally {
        bot.ms = Math.round(performance.now() - t0);
        this.streaming = false; this.controller = null;
        this.scroll();
        Banko.emit("banko:answered", { thread: tid, bot });
      }
    },

    onAnswerEvent(bot, event, data) {
      if (event === "step") bot.steps.push(data);
      else if (event === "token") { bot.answer += data.text; this.scroll(); }
      else if (event === "citation") { if (!bot.citations.some((c) => c.expense_id === data.expense_id)) bot.citations.push(data); }
      else if (event === "proposal") bot.proposals.push({ ...data });
      else if (event === "answer") {
        bot.answer = data.answer || bot.answer;
        bot.citations = data.citations && data.citations.length ? data.citations : bot.citations;
        bot.meta = data.meta || {}; bot.route = data.route || ""; bot.cached = !!data.cached;
      } else if (event === "error") {
        bot.error = data.message || Banko.FALLBACK;
        if (data.trace) bot.meta = { ...bot.meta, trace: data.trace };
      }
    },

    stop() { if (this.controller) this.controller.abort(); },

    async upload(file) {
      if (!file) return;
      if (!RECEIPT_TYPES.test(file.type || "")) {
        Alpine.store("toasts").push({ title: "Receipts", text: "Upload a receipt image or PDF.", kind: "error" });
        return;
      }
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

    render(text) {
      const clean = Banko.stripCitations(text);
      return window.DOMPurify.sanitize(window.marked.parse(clean, { breaks: true }));
    },
    async copy(text) {
      const plain = Banko.stripCitations(text);
      try { await navigator.clipboard.writeText(plain); }
      catch (e) {
        const area = document.createElement("textarea");
        area.value = plain; document.body.appendChild(area); area.select(); document.execCommand("copy"); area.remove();
      }
      Alpine.store("toasts").push({ title: "Copied", text: "The answer is on your clipboard." });
    },
    toggleTrace(m) { m.showTrace = !m.showTrace; Banko.emit("banko:trace", { message: m }); },
    traceSteps(m) { return m.meta && m.meta.trace && m.meta.trace.length ? m.meta.trace : m.steps; },
    stepClass(s) { return s.status === "warning" || s.status === "error" ? "warn" : "ok"; },
    openCitation(c) { Banko.emit("banko:open-transaction", { id: c.expense_id }); },

    async startPage(threadId) {
      if (threadId) await this.loadThread(threadId);
    },

    onKey(e) {
      if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); this.send(); }
      else if (e.key === "Escape") this.hide();
    },
    scroll() { this.$nextTick(() => { const box = this.$refs.chat; if (box) box.scrollTop = box.scrollHeight; }); },
  }));
});
