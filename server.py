#!/usr/bin/env python3
"""
Read-only web viewer for gen_history/history.json.
Displays the conversation in chatbot style and polls for updates.

Usage:
    python server.py [--port 7000] [--history gen_history/history.json]
"""

import json
import argparse
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
<title>vLLM Chat Viewer</title>
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github-dark.min.css">
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    background: #0d1117;
    color: #e6edf3;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    font-size: 14px;
    display: flex;
    flex-direction: column;
    height: 100vh;
    overflow: hidden;
  }

  /* ── Header ── */
  #header {
    background: #161b22;
    border-bottom: 1px solid #30363d;
    padding: 10px 20px;
    display: flex;
    align-items: center;
    gap: 12px;
    flex-shrink: 0;
  }
  #header h1 { font-size: 15px; font-weight: 600; color: #58a6ff; }
  #status {
    margin-left: auto;
    font-size: 12px;
    color: #8b949e;
    display: flex;
    align-items: center;
    gap: 6px;
  }
  #dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #3fb950;
    animation: pulse 2s infinite;
  }
  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.4; }
  }
  #msg-count { color: #8b949e; }

  /* ── Chat area ── */
  #chat {
    flex: 1;
    overflow-y: auto;
    padding: 24px 0;
    scroll-behavior: smooth;
  }

  .msg-row {
    display: flex;
    padding: 6px 24px;
    gap: 14px;
    max-width: 900px;
    margin: 0 auto;
    width: 100%;
  }
  .msg-row.user   { flex-direction: row-reverse; }
  .msg-row.assistant { flex-direction: row; }

  .avatar {
    width: 32px; height: 32px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 14px;
    flex-shrink: 0;
    margin-top: 2px;
  }
  .user     .avatar { background: #1f6feb; }
  .assistant .avatar { background: #21262d; border: 1px solid #30363d; }

  .bubble {
    max-width: 75%;
    min-width: 60px;
  }

  .bubble-inner {
    padding: 10px 14px;
    border-radius: 12px;
    line-height: 1.6;
    word-break: break-word;
  }
  .user .bubble-inner {
    background: #1f6feb;
    color: #fff;
    border-top-right-radius: 4px;
  }
  .assistant .bubble-inner {
    background: #161b22;
    border: 1px solid #30363d;
    color: #e6edf3;
    border-top-left-radius: 4px;
  }

  /* Markdown inside bubbles */
  .bubble-inner p  { margin: 0 0 8px; }
  .bubble-inner p:last-child { margin-bottom: 0; }
  .bubble-inner h1, .bubble-inner h2, .bubble-inner h3 {
    margin: 12px 0 6px; font-size: 1em; color: #58a6ff;
  }
  .bubble-inner ul, .bubble-inner ol { padding-left: 20px; margin: 6px 0; }
  .bubble-inner li { margin: 3px 0; }
  .bubble-inner code {
    background: #0d1117;
    border: 1px solid #30363d;
    border-radius: 4px;
    padding: 1px 5px;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    font-size: 12px;
  }
  .bubble-inner pre {
    background: #0d1117;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 12px;
    overflow-x: auto;
    margin: 8px 0;
  }
  .bubble-inner pre code {
    background: none;
    border: none;
    padding: 0;
    font-size: 12px;
  }
  .bubble-inner strong { color: #f0f6fc; }
  .bubble-inner hr {
    border: none;
    border-top: 1px solid #30363d;
    margin: 10px 0;
  }
  .bubble-inner blockquote {
    border-left: 3px solid #30363d;
    padding-left: 10px;
    color: #8b949e;
    margin: 6px 0;
  }
  .bubble-inner table {
    border-collapse: collapse;
    width: 100%;
    margin: 8px 0;
    font-size: 13px;
  }
  .bubble-inner th, .bubble-inner td {
    border: 1px solid #30363d;
    padding: 5px 10px;
    text-align: left;
  }
  .bubble-inner th { background: #21262d; }

  /* Thinking block */
  .thinking-toggle {
    display: flex;
    align-items: center;
    gap: 6px;
    cursor: pointer;
    color: #8b949e;
    font-size: 12px;
    margin-bottom: 8px;
    user-select: none;
  }
  .thinking-toggle:hover { color: #c9d1d9; }
  .thinking-arrow { font-size: 10px; transition: transform 0.2s; }
  .thinking-arrow.open { transform: rotate(90deg); }
  .thinking-body {
    color: #8b949e;
    font-size: 12px;
    line-height: 1.5;
    border-left: 2px solid #30363d;
    padding-left: 10px;
    margin-bottom: 10px;
    display: none;
    white-space: pre-wrap;
    font-style: italic;
  }
  .thinking-body.open { display: block; }

  /* Meta row (tokens, finish reason) */
  .meta {
    font-size: 11px;
    color: #484f58;
    margin-top: 5px;
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
  }
  .user .meta { justify-content: flex-end; }
  .badge {
    padding: 1px 6px;
    border-radius: 10px;
    background: #21262d;
    border: 1px solid #30363d;
  }
  .badge.stop     { border-color: #238636; color: #3fb950; }
  .badge.length   { border-color: #9e6a03; color: #d29922; }
  .badge.interrupted { border-color: #6e7681; color: #8b949e; }

  /* Generating cursor */
  .generating::after {
    content: '▋';
    display: inline-block;
    color: #58a6ff;
    animation: blink 0.8s steps(1) infinite;
  }
  @keyframes blink { 50% { opacity: 0; } }

  /* Empty state */
  #empty {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 100%;
    color: #484f58;
    gap: 8px;
  }
  #empty .icon { font-size: 40px; }

  /* Scrollbar */
  #chat::-webkit-scrollbar { width: 6px; }
  #chat::-webkit-scrollbar-track { background: transparent; }
  #chat::-webkit-scrollbar-thumb { background: #30363d; border-radius: 3px; }
</style>
</head>
<body>

<div id="header">
  <div id="dot"></div>
  <h1>vLLM Chat Viewer</h1>
  <div id="status">
    <span id="msg-count">0 messages</span>
    &bull;
    <span id="last-update">–</span>
  </div>
</div>

<div id="chat">
  <div id="empty">
    <div class="icon">💬</div>
    <div>Waiting for conversation history…</div>
  </div>
</div>

<script>
marked.setOptions({
  highlight: (code, lang) => {
    if (lang && hljs.getLanguage(lang)) {
      return hljs.highlight(code, { language: lang }).value;
    }
    return hljs.highlightAuto(code).value;
  },
  breaks: true,
  gfm: true,
});

let lastHash = null;
let isAtBottom = true;
const chat = document.getElementById('chat');

chat.addEventListener('scroll', () => {
  isAtBottom = chat.scrollHeight - chat.clientHeight - chat.scrollTop < 60;
});

function scrollToBottom(force) {
  if (force || isAtBottom) {
    chat.scrollTop = chat.scrollHeight;
  }
}

function timeAgo(ms) {
  const s = Math.floor((Date.now() - ms) / 1000);
  if (s < 5)  return 'just now';
  if (s < 60) return s + 's ago';
  return Math.floor(s / 60) + 'm ago';
}

function makeThinkingBlock(thinkingText, uid) {
  if (!thinkingText) return '';
  const escaped = thinkingText.replace(/</g, '&lt;').replace(/>/g, '&gt;');
  return `
    <div class="thinking-toggle" onclick="toggleThinking('${uid}')">
      <span class="thinking-arrow open" id="arr-${uid}">▶</span>
      <span>Thinking (${thinkingText.length} chars)</span>
    </div>
    <div class="thinking-body open" id="body-${uid}">${escaped}</div>
  `;
}

function toggleThinking(uid) {
  const arr  = document.getElementById('arr-' + uid);
  const body = document.getElementById('body-' + uid);
  arr.classList.toggle('open');
  body.classList.toggle('open');
}

function renderMessages(history) {
  if (!history.length) {
    chat.innerHTML = `
      <div id="empty" style="display:flex;flex-direction:column;align-items:center;
           justify-content:center;height:100%;color:#484f58;gap:8px;">
        <div class="icon">💬</div><div>Waiting for conversation history…</div>
      </div>`;
    return;
  }

  const wasAtBottom = isAtBottom;
  const rows = [];

  history.forEach((msg, i) => {
    if (msg.role !== 'user' && msg.role !== 'assistant') return;

    const isUser   = msg.role === 'user';
    const isLast   = i === history.length - 1;
    const isEmpty  = !msg.content;
    const isGenerating = isLast && msg.role === 'assistant' && isEmpty;

    const uid = 'think-' + i;

    // Content
    let contentHtml;
    if (isUser) {
      const escaped = (msg.content || '').replace(/</g, '&lt;').replace(/>/g, '&gt;');
      contentHtml = `<span>${escaped}</span>`;
    } else {
      const md = msg.content || '';
      contentHtml = md ? marked.parse(md) : '';
    }

    // Generating cursor on last empty assistant message
    const cursorClass = isGenerating ? ' generating' : '';

    // Thinking block
    const thinkingHtml = (!isUser && msg.thinking)
      ? makeThinkingBlock(msg.thinking, uid)
      : '';

    // Meta
    let metaHtml = '';
    if (msg.tokens) {
      metaHtml += `<span class="badge">${msg.tokens.toLocaleString()} tokens</span>`;
    }
    if (msg.finish_reason) {
      const cls = msg.finish_reason === 'stop' ? 'stop'
                : msg.finish_reason === 'length' ? 'length'
                : 'interrupted';
      metaHtml += `<span class="badge ${cls}">${msg.finish_reason}</span>`;
    }

    const avatarEmoji = isUser ? '🧑' : '🤖';

    rows.push(`
      <div class="msg-row ${msg.role}">
        <div class="avatar">${avatarEmoji}</div>
        <div class="bubble">
          ${thinkingHtml}
          <div class="bubble-inner${cursorClass}">${contentHtml}</div>
          ${metaHtml ? `<div class="meta">${metaHtml}</div>` : ''}
        </div>
      </div>
    `);
  });

  chat.innerHTML = rows.join('');

  // Re-run highlight.js on any pre>code blocks
  chat.querySelectorAll('pre code').forEach(el => hljs.highlightElement(el));

  scrollToBottom(wasAtBottom);
}

let fetchEpoch = 0;

async function poll() {
  const epoch = ++fetchEpoch;
  try {
    const res = await fetch('/api/history', { cache: 'no-store' });
    if (!res.ok) return;
    const text = await res.text();
    if (epoch !== fetchEpoch) return; // stale response, discard

    // Only re-render if content changed
    if (text !== lastHash) {
      lastHash = text;
      const history = JSON.parse(text);
      renderMessages(history);
      const n = history.length;
      document.getElementById('msg-count').textContent =
        n + ' message' + (n !== 1 ? 's' : '');
    }
    document.getElementById('last-update').textContent = timeAgo(Date.now());
  } catch (e) {
    // ignore network errors
  }
}

// Poll interval set via --poll server argument
setInterval(poll, __POLL_MS__);
poll();

// Update "last updated" timestamp every 5s
setInterval(() => {
  if (lastHash !== null) {
    document.getElementById('last-update').textContent = timeAgo(Date.now());
  }
}, 5000);
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
    parser.add_argument("--port", type=int, default=7000,
                        help="Port to listen on (default: 7000)")
    parser.add_argument("--history", default="gen_history/history.json",
                        help="Path to history JSON file")
    parser.add_argument("--poll", type=int, default=200,
                        help="Browser poll interval in milliseconds (default: 200)")
    args = parser.parse_args()

    Handler.history_file = Path(args.history)
    Handler.poll_ms = args.poll
    if not Handler.history_file.exists():
        print(f"Warning: history file not found: {Handler.history_file}")

    addr = ("0.0.0.0", args.port)
    server = HTTPServer(addr, Handler)
    print(f"Serving at http://localhost:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
