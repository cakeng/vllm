#!/usr/bin/env python3
"""
Read-only web viewer for gen_history/history.json.
Displays the conversation as a single flowing stream of thought.

Usage:
    python server.py [--port 7000] [--history gen_history/history.json]
"""

import json
import argparse
import ssl
import sys
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

# ---------------------------------------------------------------------------
# Frontend
# ---------------------------------------------------------------------------
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Stream of Thought</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    background: #f7f5f1;
    color: #1c1a17;
    font-family: Georgia, 'Palatino Linotype', Palatino, serif;
    font-size: 19px;
    line-height: 1.84;
    display: flex;
    flex-direction: column;
    height: 100vh;
    overflow: hidden;
  }

  /* ── Header ── */
  #header {
    background: #efece6;
    border-bottom: 1px solid #dbd6ce;
    padding: 9px 28px;
    display: flex;
    align-items: center;
    gap: 10px;
    flex-shrink: 0;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  }
  #header-title {
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #6b6359;
  }
  #status {
    margin-left: auto;
    font-size: 12px;
    color: #9e9488;
    display: flex;
    align-items: center;
    gap: 10px;
  }
  #dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: #82b366;
    flex-shrink: 0;
    animation: pulse 2s infinite;
  }
  #dot.idle { background: #c0b8ad; animation: none; }
  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.25; }
  }

  /* ── Scroll area ── */
  #scroll {
    flex: 1;
    overflow-y: auto;
    padding: 64px 0 96px;
  }

  /* ── Reading column ── */
  #page {
    max-width: 680px;
    margin: 0 auto;
    padding: 0 32px;
  }

  /* ── Seed / user message ── */
  .seed {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    font-size: 13px;
    font-style: italic;
    color: #9e9488;
    border-left: 2px solid #cec8bf;
    padding: 2px 0 2px 14px;
    margin: 52px 0 36px;
    line-height: 1.55;
  }
  .seed:first-child { margin-top: 0; }

  /* ── Thought paragraphs ── */
  .thought p {
    margin-bottom: 1.5em;
    hyphens: auto;
  }
  .thought p:last-child { margin-bottom: 0; }

  /* ── Injection marker (Wikipedia pivot) ── */
  .pivot {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    font-size: 12px;
    color: #b0a898;
    font-style: italic;
    margin: 2em 0 1.6em;
    padding-left: 14px;
    border-left: 2px solid #e0dbd3;
  }

  /* ── Generating cursor ── */
  .cursor::after {
    content: '▋';
    display: inline;
    color: #c0b8ad;
    animation: blink 0.95s steps(1) infinite;
  }
  @keyframes blink { 50% { opacity: 0; } }

  /* ── Empty state ── */
  #empty {
    text-align: center;
    padding: 100px 0;
    color: #bdb5a8;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    font-size: 14px;
    font-style: italic;
  }

  /* ── Scrollbar ── */
  #scroll::-webkit-scrollbar { width: 5px; }
  #scroll::-webkit-scrollbar-track { background: transparent; }
  #scroll::-webkit-scrollbar-thumb { background: #cec8bf; border-radius: 3px; }
</style>
</head>
<body>

<div id="header">
  <div id="dot"></div>
  <div id="header-title">Stream of Thought</div>
  <div id="status">
    <span id="chars-per-sec"></span>
    <span id="char-count"></span>
    <span id="last-update">–</span>
  </div>
</div>

<div id="scroll">
  <div id="page">
    <div id="empty">Waiting for thought stream…</div>
  </div>
</div>

<script>
let lastHash   = null;
let isAtBottom = true;

const scroll = document.getElementById('scroll');
const dot    = document.getElementById('dot');

scroll.addEventListener('scroll', () => {
  isAtBottom = scroll.scrollHeight - scroll.clientHeight - scroll.scrollTop < 80;
});

function timeAgo(ms) {
  const s = Math.floor((Date.now() - ms) / 1000);
  if (s < 5)  return 'live';
  if (s < 60) return s + 's ago';
  return Math.floor(s / 60) + 'm ago';
}

function esc(t) {
  return String(t).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

/* Render text as flowing paragraphs. Double-newlines → <p> breaks.
   Pivot markers ([→ ...]) get lighter styling.
   showCursor appends a blinking cursor to the last paragraph. */
function paragraphs(text, showCursor) {
  const chunks = text.split(/\n\n+/).map(p => p.replace(/\n/g,' ').trim()).filter(Boolean);
  if (!chunks.length) return showCursor ? '<p><span class="cursor"></span></p>' : '';
  return chunks.map((p, i) => {
    const last = i === chunks.length - 1;
    const cur  = (showCursor && last) ? '<span class="cursor"></span>' : '';
    if (/^\[→/.test(p)) return `<div class="pivot">${esc(p)}${cur}</div>`;
    return `<p>${esc(p)}${cur}</p>`;
  }).join('');
}

function combinedText(msg) {
  const t = (msg.thinking || '').trim();
  const c = (msg.content  || '').trim();
  return t + (t && c ? '\n\n' : '') + c;
}

// ── Typewriter ────────────────────────────────────────────────────────────────
let targetText     = '';   // full server-side text for the live entry
let displayedChars = 0;    // how many chars have been "typed" so far
let lastEntryKey   = '';   // resets typewriter when the live entry changes
let liveIsLive     = false; // server says in_progress
let animHandle     = null;
let charAccum      = 0;    // fractional char accumulator for sub-integer speeds

/* Linear interpolation: 0.6 chars/frame at buf=100, 6.0 chars/frame at buf=10000.
   Clamped outside that range. Fractional values handled via charAccum. */
function charsPerFrame(buf) {
  const lo = 0.25, hi = 5.0, bufLo = 5000, bufHi = 10000;
  if (buf >= bufHi) return Infinity;
  return Math.max(lo, lo + (buf - bufLo) * (hi - lo) / (bufHi - bufLo));
}

// ── Chars-per-second tracker ──────────────────────────────────────────────────
let cpsLastLen  = 0;
let cpsLastTime = Date.now();
let cpsSmoothed = 0;

function updateCps() {
  const now = Date.now();
  const dt  = (now - cpsLastTime) / 1000;
  if (dt < 0.5) return;
  const rate  = (targetText.length - cpsLastLen) / dt;
  cpsSmoothed = cpsSmoothed === 0 ? rate : 0.25 * rate + 0.75 * cpsSmoothed;
  cpsLastLen  = targetText.length;
  cpsLastTime = now;
  const el = document.getElementById('chars-per-sec');
  if (el) el.textContent = cpsSmoothed > 1 ? Math.round(cpsSmoothed) + ' c/s' : '';
}

function updateLiveElement() {
  const el = document.getElementById('live-thought');
  if (!el) return;
  const showCursor = liveIsLive || displayedChars < targetText.length;
  el.innerHTML = paragraphs(targetText.slice(0, displayedChars), showCursor);
  if (isAtBottom) scroll.scrollTop = scroll.scrollHeight;
}

function tick() {
  const buf = targetText.length - displayedChars;
  if (buf <= 0) { animHandle = null; return; }
  const speed = charsPerFrame(buf);
  const advance = isFinite(speed) ? Math.floor(charAccum += speed) : buf;
  if (isFinite(speed)) charAccum -= advance;
  if (advance > 0) {
    displayedChars = Math.min(displayedChars + advance, targetText.length);
    updateLiveElement();
  }
  updateCps();
  animHandle = requestAnimationFrame(tick);
}

function kickAnimate() {
  if (!animHandle && displayedChars < targetText.length)
    animHandle = requestAnimationFrame(tick);
}

// ── Rendering ─────────────────────────────────────────────────────────────────

/* Fingerprint of history shape — changes only when entries are added/removed,
   not when the last entry's text grows. Prevents full DOM rebuild every poll. */
function structureKey(history) {
  return history.map(m => m.role).join(',') + ':' + history.length;
}

let lastStructureKey = '';

function rebuildPage(history) {
  const page   = document.getElementById('page');
  const wasBot = isAtBottom;

  if (!history.length) {
    page.innerHTML = '<div id="empty">Waiting for thought stream…</div>';
    return;
  }

  let lastAsstIdx = -1;
  for (let i = history.length - 1; i >= 0; i--) {
    if (history[i].role === 'assistant') { lastAsstIdx = i; break; }
  }

  const parts = [];
  history.forEach((msg, idx) => {
    if (msg.role === 'user') {
      parts.push(`<div class="seed">${esc(msg.content || '')}</div>`);
    } else if (msg.role === 'assistant') {
      if (idx === lastAsstIdx) {
        // Live slot — content is written by updateLiveElement(), not here.
        parts.push(`<div id="live-thought" class="thought"></div>`);
      } else {
        const text = combinedText(msg);
        if (text) parts.push(`<div class="thought">${paragraphs(text, false)}</div>`);
      }
    }
  });

  page.innerHTML = parts.length
    ? parts.join('')
    : '<div id="empty">Waiting for thought stream…</div>';

  if (wasBot) scroll.scrollTop = scroll.scrollHeight;
}

function render(history) {
  const page = document.getElementById('page');

  if (!history.length) {
    page.innerHTML = '<div id="empty">Waiting for thought stream…</div>';
    dot.className  = 'idle';
    return;
  }

  // Rebuild the static DOM only when entries are added or removed.
  const skey = structureKey(history);
  if (skey !== lastStructureKey) {
    lastStructureKey = skey;
    rebuildPage(history);
  }

  // Update liveness indicator.
  const lastMsg = history[history.length - 1];
  liveIsLive    = lastMsg.role === 'assistant' && lastMsg.finish_reason === 'in_progress';
  dot.className = liveIsLive ? '' : 'idle';

  // Update typewriter target for the live entry.
  let lastAsstIdx = -1;
  for (let i = history.length - 1; i >= 0; i--) {
    if (history[i].role === 'assistant') { lastAsstIdx = i; break; }
  }

  let totalChars = 0;
  history.forEach(m => { if (m.role === 'assistant') totalChars += combinedText(m).length; });
  document.getElementById('char-count').textContent =
    totalChars > 0 ? totalChars.toLocaleString() + ' chars' : '';

  if (lastAsstIdx >= 0) {
    const text = combinedText(history[lastAsstIdx]);
    const key  = String(lastAsstIdx);
    if (key !== lastEntryKey) {
      // New live entry — reset the typewriter from zero.
      displayedChars = 0;
      lastEntryKey   = key;
    }
    if (text.length < displayedChars) displayedChars = text.length;
    targetText = text;
    kickAnimate();
  }
}

// ── Poll loop ─────────────────────────────────────────────────────────────────
async function poll() {
  try {
    const res = await fetch('/api/history', { cache: 'no-store' });
    if (res.ok) {
      const text = await res.text();
      if (text !== lastHash) {
        lastHash = text;
        render(JSON.parse(text));
      }
      document.getElementById('last-update').textContent = timeAgo(Date.now());
    }
  } catch (_) {}
  setTimeout(poll, __POLL_MS__);
}

poll();

</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    history_file: Path = None
    poll_ms: int = 500

    def do_GET(self):
        path = urlparse(self.path).path

        if path == "/":
            page = HTML.replace("__POLL_MS__", str(self.poll_ms))
            body = page.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        elif path == "/api/history":
            try:
                raw = self.history_file.read_text(encoding="utf-8")
                json.loads(raw)          # validate — raise if corrupt
            except Exception:
                raw = "[]"
            body = raw.encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, fmt, *args):
        pass  # suppress per-request logs


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Read-only vLLM chat viewer")
    parser.add_argument("history", help="Path to the chat history JSON file")
    parser.add_argument("--port", type=int, default=443,
                        help="Port to listen on (default: 443)")
    parser.add_argument("--cert", default="cert.pem",
                        help="TLS certificate file (e.g. cert.pem) — enables HTTPS")
    parser.add_argument("--key", default="key.pem",
                        help="TLS private key file (e.g. key.pem)")
    args = parser.parse_args()

    Handler.history_file = Path(args.history)
    Handler.poll_ms = 20
    if not Handler.history_file.exists():
        print(f"Warning: history file not found: {Handler.history_file}")

    addr = ("0.0.0.0", args.port)
    server = HTTPServer(addr, Handler)

    if args.cert and args.key:
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(certfile=args.cert, keyfile=args.key)
        server.socket = ctx.wrap_socket(server.socket, server_side=True)
        proto = "https"
    else:
        proto = "http"

    print(f"Serving at {proto}://0.0.0.0:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
