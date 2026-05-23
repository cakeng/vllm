#!/usr/bin/env python3
"""
Read-only web viewer for gen_history/stream.json.
Displays each chunk as a one-sentence summary; click to expand full thinking.

Usage:
    python server.py <history.json> [--port 443] [--cert cert.pem] [--key key.pem]
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
    padding: 48px 0 96px;
  }

  /* ── Reading column ── */
  #page {
    max-width: 720px;
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
    margin: 0 0 32px;
    line-height: 1.55;
  }

  /* ── Chunk row ── */
  .chunk {
    margin: 0;
    border-bottom: 1px solid #ede8e1;
  }
  .chunk:first-of-type { border-top: 1px solid #ede8e1; }

  .chunk-header {
    display: flex;
    align-items: baseline;
    gap: 10px;
    padding: 14px 0;
    cursor: pointer;
    user-select: none;
  }
  .chunk-header:hover .chunk-summary { color: #1c1a17; }

  .chunk-icon {
    font-size: 11px;
    color: #b0a898;
    flex-shrink: 0;
    width: 12px;
    line-height: 1.84;
  }

  .chunk-num {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    font-size: 11px;
    font-weight: 600;
    color: #9e9488;
    flex-shrink: 0;
    min-width: 2.4em;
    text-align: right;
    line-height: 1.55;
    letter-spacing: 0.03em;
  }

  .chunk-summary {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    font-size: 15px;
    color: #1c1a17;
    line-height: 1.55;
    flex: 1;
  }

  /* ── Typewriter cursor on animating summary ── */
  .cursor::after {
    content: '▋';
    display: inline;
    color: #c0b8ad;
    animation: blink 0.95s steps(1) infinite;
  }
  @keyframes blink { 50% { opacity: 0; } }

  /* ── Expanded thinking body ── */
  .chunk-body {
    display: none;
    padding: 4px 0 24px 22px;
    border-left: 2px solid #e0dbd3;
    margin-left: 0;
  }
  .chunk-body.open { display: block; }

  .thought p {
    margin-bottom: 1.4em;
    hyphens: auto;
    color: #2e2a24;
  }
  .thought p:last-child { margin-bottom: 0; }

  /* ── In-progress spinner row ── */
  .chunk.in-progress .chunk-header { cursor: default; }

  .chunk-spinner {
    display: inline-block;
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #82b366;
    flex-shrink: 0;
    animation: pulse 1.4s infinite;
  }
  .chunk-thinking-label {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    font-size: 13px;
    color: #9e9488;
    font-style: italic;
  }

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
// ── State ─────────────────────────────────────────────────────────────────────
let lastHash         = null;
let isAtBottom       = true;
let lastStructureKey = '';
let expandedChunks   = new Set();   // chunk numbers currently expanded
let animatingChunk   = -1;          // chunk number whose summary is animating
let lastSummaryCount = 0;           // detect when a new summary arrives

// ── Typewriter ────────────────────────────────────────────────────────────────
let targetText     = '';
let displayedChars = 0;
let charAccum      = 0;
let animHandle     = null;

// ── CPS tracker ───────────────────────────────────────────────────────────────
let cpsLastLen  = 0;
let cpsLastTime = Date.now();
let cpsSmoothed = 0;

const scroll = document.getElementById('scroll');
const dot    = document.getElementById('dot');

scroll.addEventListener('scroll', () => {
  isAtBottom = scroll.scrollHeight - scroll.clientHeight - scroll.scrollTop < 80;
});

// ── Utilities ─────────────────────────────────────────────────────────────────
function timeAgo(ms) {
  const s = Math.floor((Date.now() - ms) / 1000);
  if (s < 5)  return 'live';
  if (s < 60) return s + 's ago';
  return Math.floor(s / 60) + 'm ago';
}

function esc(t) {
  return String(t)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

function paragraphs(text) {
  const chunks = text.split(/\n\n+/).map(p => p.replace(/\n/g, ' ').trim()).filter(Boolean);
  return chunks.map(p => `<p>${esc(p)}</p>`).join('');
}

// ── Typewriter animation (targets the animating summary span) ─────────────────
function charsPerFrame(buf) {
  const lo = 0.6, hi = 6.0, bufLo = 20, bufHi = 200;
  if (buf >= bufHi) return Infinity;
  return Math.max(lo, lo + (buf - bufLo) * (hi - lo) / (bufHi - bufLo));
}

function updateCps() {
  const now = Date.now();
  const dt  = (now - cpsLastTime) / 1000;
  if (dt < 0.5) return;
  const rate  = (targetText.length - cpsLastLen) / dt;
  cpsSmoothed = cpsSmoothed === 0 ? rate : 0.25 * rate + 0.75 * cpsSmoothed;
  cpsLastLen  = targetText.length;
  cpsLastTime = now;
  const el = document.getElementById('chars-per-sec');
  if (el) el.textContent = cpsSmoothed > 0.5 ? Math.round(cpsSmoothed) + ' c/s' : '';
}

function updateSummaryElement() {
  const el = document.getElementById('live-summary');
  if (!el) return;
  const done = displayedChars >= targetText.length;
  el.innerHTML = esc(targetText.slice(0, displayedChars)) +
    (done ? '' : '<span class="cursor"></span>');
  if (isAtBottom) scroll.scrollTop = scroll.scrollHeight;
}

function tick() {
  const buf = targetText.length - displayedChars;
  if (buf <= 0) { animHandle = null; return; }
  const speed   = charsPerFrame(buf);
  const advance = isFinite(speed) ? Math.floor(charAccum += speed) : buf;
  if (isFinite(speed)) charAccum -= advance;
  if (advance > 0) {
    displayedChars = Math.min(displayedChars + advance, targetText.length);
    updateSummaryElement();
  }
  updateCps();
  animHandle = requestAnimationFrame(tick);
}

function kickAnimate() {
  if (!animHandle && displayedChars < targetText.length)
    animHandle = requestAnimationFrame(tick);
}

// ── Expand / collapse ─────────────────────────────────────────────────────────
function toggleChunk(cnum) {
  const row  = document.querySelector(`.chunk[data-chunk="${cnum}"]`);
  if (!row) return;
  const body = row.querySelector('.chunk-body');
  const icon = row.querySelector('.chunk-icon');
  const num  = row.querySelector('.chunk-num');
  if (expandedChunks.has(cnum)) {
    expandedChunks.delete(cnum);
    body.classList.remove('open');
    icon.textContent = '▸';
    if (num) num.textContent = '#' + cnum;
  } else {
    expandedChunks.add(cnum);
    body.classList.add('open');
    icon.textContent = '▾';
    if (num) num.textContent = '';
  }
}

// ── Page rendering ────────────────────────────────────────────────────────────
function structureKey(history) {
  const chunks   = history.filter(m => m.role === 'chunk');
  const summaries = chunks.filter(m => m.summary !== null && m.summary !== undefined).length;
  return chunks.length + ':' + summaries;
}

function rebuildPage(history) {
  const page   = document.getElementById('page');
  const wasBot = isAtBottom;

  if (!history.length) {
    page.innerHTML = '<div id="empty">Waiting for thought stream…</div>';
    return;
  }

  const parts = [];

  history.forEach(msg => {
    if (msg.role === 'user') {
      parts.push(`<div class="seed">${esc(msg.content || '')}</div>`);
      return;
    }

    if (msg.role !== 'chunk') return;

    const cnum     = msg.chunk;
    const hasSummary = msg.summary !== null && msg.summary !== undefined;

    if (!hasSummary) {
      // In-progress: spinner row
      parts.push(`
        <div class="chunk in-progress" data-chunk="${cnum}">
          <div class="chunk-header">
            <span class="chunk-spinner"></span>
            <span class="chunk-thinking-label">thinking…</span>
          </div>
        </div>`);
      return;
    }

    // Completed chunk
    const isAnimating = cnum === animatingChunk;
    const expanded    = expandedChunks.has(cnum);
    const summaryHtml = isAnimating
      ? `<span id="live-summary"></span>`
      : esc(msg.summary);

    parts.push(`
      <div class="chunk" data-chunk="${cnum}">
        <div class="chunk-header" onclick="toggleChunk(${cnum})">
          <span class="chunk-icon">${expanded ? '▾' : '▸'}</span>
          <span class="chunk-num">${expanded ? '' : '#' + cnum}</span>
          <span class="chunk-summary">${summaryHtml}</span>
        </div>
        <div class="chunk-body${expanded ? ' open' : ''}">
          <div class="thought">${paragraphs(msg.thinking || '')}</div>
        </div>
      </div>`);
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

  const chunks    = history.filter(m => m.role === 'chunk');
  const completed = chunks.filter(m => m.summary !== null && m.summary !== undefined);
  const summaryCount = completed.length;

  // Detect a newly appeared summary → arm the typewriter before rebuilding.
  if (summaryCount > lastSummaryCount && completed.length > 0) {
    const latest    = completed[completed.length - 1];
    animatingChunk  = latest.chunk;
    targetText      = latest.summary || '';
    displayedChars  = 0;
    charAccum       = 0;
    lastSummaryCount = summaryCount;
  }

  // Rebuild DOM when structure changes (chunk added or summary appeared).
  const skey = structureKey(history);
  if (skey !== lastStructureKey) {
    lastStructureKey = skey;
    rebuildPage(history);
  }

  // Liveness dot: green while the last chunk has no summary yet.
  const lastChunk = chunks[chunks.length - 1];
  const isLive    = lastChunk &&
    (lastChunk.summary === null || lastChunk.summary === undefined);
  dot.className   = isLive ? '' : 'idle';

  // Char count (total thinking chars generated).
  let totalChars = 0;
  chunks.forEach(c => { totalChars += (c.thinking || '').length; });
  document.getElementById('char-count').textContent =
    totalChars > 0 ? totalChars.toLocaleString() + ' chars' : '';

  kickAnimate();
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
    parser = argparse.ArgumentParser(description="Read-only vLLM stream viewer")
    parser.add_argument("history", help="Path to the stream history JSON file")
    parser.add_argument("--port", type=int, default=443,
                        help="Port to listen on (default: 443)")
    parser.add_argument("--cert", default="cert.pem",
                        help="TLS certificate file — enables HTTPS")
    parser.add_argument("--key", default="key.pem",
                        help="TLS private key file")
    args = parser.parse_args()

    Handler.history_file = Path(args.history)
    Handler.poll_ms = 20
    if not Handler.history_file.exists():
        print(f"Warning: history file not found: {Handler.history_file}")

    addr   = ("0.0.0.0", args.port)
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
