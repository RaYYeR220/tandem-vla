// Tandem — shared frontend helpers. No build step, no framework, no bundler.
// Loaded by both index.html and app.html.

(function (global) {
  "use strict";

  function qs(sel, root) { return (root || document).querySelector(sel); }
  function qsa(sel, root) { return Array.from((root || document).querySelectorAll(sel)); }

  function escapeHtml(s) {
    return String(s == null ? "" : s).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }

  function fmtMs(ms) {
    if (ms == null || Number.isNaN(ms)) return "—";
    if (ms >= 1000) return (ms / 1000).toFixed(2) + " s";
    return Math.round(ms) + " ms";
  }

  async function postJSON(url, body) {
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body || {}),
    });
    let data = null;
    try { data = await res.json(); } catch (e) { /* no body */ }
    if (!res.ok) {
      const detail = (data && data.detail) || res.statusText;
      throw new Error(detail);
    }
    return data;
  }

  async function getJSON(url) {
    const res = await fetch(url);
    return res.json();
  }

  // ---------------------------------------------------------------- SSE client
  //
  // Wraps EventSource: the server already replays a snapshot of the last state/plan/verdict
  // /infer/episode event to every new connection (see tandem/server/hub.py), and EventSource
  // reconnects on its own after a drop, so this is deliberately thin — dispatch by event
  // "type", track connection state, nothing else.

  function TandemStream(url) {
    this.url = url || "/api/stream";
    this.handlers = {};
    this.source = null;
    this.onStatus = null; // function(connected: bool)
  }

  TandemStream.prototype.on = function (type, fn) {
    (this.handlers[type] = this.handlers[type] || []).push(fn);
    return this;
  };

  TandemStream.prototype.connect = function () {
    const self = this;
    const es = new EventSource(this.url);
    this.source = es;
    es.onopen = function () { if (self.onStatus) self.onStatus(true); };
    es.onerror = function () { if (self.onStatus) self.onStatus(false); };
    es.onmessage = function (evt) {
      let data;
      try { data = JSON.parse(evt.data); } catch (e) { return; }
      const type = data && data.type;
      const list = self.handlers[type];
      if (list) list.forEach(function (fn) { try { fn(data); } catch (e) { console.error(e); } });
      const all = self.handlers["*"];
      if (all) all.forEach(function (fn) { try { fn(data); } catch (e) { console.error(e); } });
    };
    return this;
  };

  global.Tandem = {
    qs: qs,
    qsa: qsa,
    escapeHtml: escapeHtml,
    fmtMs: fmtMs,
    postJSON: postJSON,
    getJSON: getJSON,
    Stream: TandemStream,
  };
})(window);
