#!/usr/bin/env python3
"""
stream_loop.py — Autonomous stream-of-consciousness generation loop.

Generates in CHUNK_TOKENS-sized bursts and lets the model wander freely.
After each chunk a second LLM call produces a one-sentence summary.

Output JSON format (one entry per chunk):

    [
      {"role": "user", "content": "<seed>", "tokens": N},
      {"role": "chunk", "chunk": 1, "thinking": "...", "summary": "...", "finish_reason": "length"},
      {"role": "chunk", "chunk": 2, "thinking": "...", "summary": null,  "finish_reason": "in_progress"},
      ...
    ]

Usage:
    python stream_loop.py [options]

Key options:
    --url URL            vLLM base URL            (default: https://127.0.0.1:18000/v1/chat/completions)
    --model NAME         Model name               (default: google/gemma-4-E2B-it)
    --api-key KEY        Bearer token for auth    (default: none)
    --seed TEXT          Opening thought (default: built-in)
    --chunks N           Max generation chunks (0 = unlimited, Ctrl+C to stop)
    --chunk-tokens N     Tokens per API call                  (default: 512)
    --context-chars N    Rolling context window in characters  (default: 8192)
    --history FILE       Output JSON path (default: gen_history/stream.json)
    --continue           Resume from an existing history file
"""

import sys
import json
import time
import argparse
import requests
import urllib3
from pathlib import Path

# Populated in main() from CLI args; module-level so helpers can read them.
URL        = "https://127.0.0.1:18000/v1/chat/completions"
MODEL      = "google/gemma-4-E2B-it"
HEADERS: dict = {}
SSL_VERIFY = False

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

SYSTEM_PROMPT = (
    "You are a continuous thought. You think the way a curious, well-read person does while browsing Wikipedia late at night.\n\n"
    "Rules you must follow absolutely:\n"
    "- Never summarize, conclude, wrap up, or say things like \"in summary\", "
    "\"so\", \"ultimately\", \"this shows us\", or \"it's fascinating how\"\n"
    "- Never use bullet points, headers, or markdown formatting, just sentences and paragraphs with clear grammar and separation between thoughts.\n"
    "- Always think in first person, present tense, as thoughts arriving, during thinking. This is not a conversation, it is a stream of consciousness.\n"
)

SUMMARY_SYSTEM = (
    "You are a precise one-sentence summarizer. "
    "Given a passage of internal monologue, output a single sentence describing specifically what the author is thinking about in first person as if you are the author himself."
)

DEFAULT_SEED = (
    "I keep thinking about the best way to utilize hardware resources, both compute and memory, for efficient MoE model inference. "
)

GEN_PARAMS = dict(
    temperature=1.2,
    top_p=0.95,
    top_k=50,
    repetition_penalty=1.1,
    presence_penalty=1.5,
    frequency_penalty=0.6,
)

# ── ANSI palette ───────────────────────────────────────────────────────────────
R        = "\033[0m"
_MAGENTA = "\033[1;35m"
_CYAN    = "\033[1;36m"
_YELLOW  = "\033[33m"
_GREEN   = "\033[32m"
_RED     = "\033[31m"
_GREY    = "\033[90m"

def _c(color: str, text: str) -> str:
    return f"{color}{text}{R}"

def _p(color: str, text: str, **kwargs):
    print(_c(color, text), **kwargs)


# ── History I/O ────────────────────────────────────────────────────────────────

def write_history(path: Path, history: list):
    """Atomic write so server.py never sees a half-written file."""
    tmp = path.with_name("." + path.name + ".tmp")
    tmp.write_text(json.dumps(history, indent=2), encoding="utf-8")
    tmp.rename(path)


def load_history(path: Path) -> list:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return []


# ── Text helpers ───────────────────────────────────────────────────────────────

def _cap_paragraphs(text: str) -> str:
    """Capitalize the first alphabetic character of every paragraph."""
    parts = text.split('\n\n')
    out = []
    for part in parts:
        for i, ch in enumerate(part):
            if ch.isalpha():
                part = part[:i] + ch.upper() + part[i+1:]
                break
        out.append(part)
    return '\n\n'.join(out)


def _clean(text: str) -> str:
    """Replace <channel|> separators with paragraph breaks."""
    return text.replace("<channel|>", "\n\n")


# ── Message construction ───────────────────────────────────────────────────────

def build_messages(seed: str, rolling_ctx: str) -> list:
    """
    First chunk: seed is the opening user prompt.
    Subsequent chunks: rolling_ctx (tail of accumulated thinking) replaces the
    seed, so the model wanders freely instead of anchoring to the original topic.
    """
    msgs: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    msgs.append({"role": "user", "content": rolling_ctx if rolling_ctx else seed})
    return msgs


# ── Generation ─────────────────────────────────────────────────────────────────

def generate_chunk(
    messages: list,
    chunk_tokens: int,
    on_thinking=None,
) -> tuple[str, str, str]:
    """
    Stream one generation chunk from vLLM.
    Returns (thinking_text, content_text, finish_reason).

    on_thinking(partial_text) — callback fired on every token batch for live
    disk writes.
    """
    payload = {
        "model":          MODEL,
        "messages":       messages,
        "stream":         True,
        "stream_options": {"include_usage": False},
        "max_tokens":     chunk_tokens,
        **GEN_PARAMS,
    }
    try:
        resp = requests.post(URL, json=payload, headers=HEADERS, stream=True, timeout=180, verify=SSL_VERIFY)
    except requests.RequestException as exc:
        _p(_RED, f"\n[API error: {exc}]")
        return "", "", "error"

    if resp.status_code != 200:
        _p(_RED, f"\n[HTTP {resp.status_code}: {resp.text[:200]}]")
        return "", "", "error"

    thinking_parts: list[str] = []
    content_parts:  list[str] = []
    channel_seen  = False
    finish_reason = "unknown"

    try:
        for raw_line in resp.iter_lines():
            if not raw_line:
                continue
            decoded = raw_line.decode(errors="replace")
            if not decoded.startswith("data: "):
                continue
            data_str = decoded[6:]
            if data_str == "[DONE]":
                break
            try:
                data = json.loads(data_str)
            except json.JSONDecodeError:
                continue
            if not data.get("choices"):
                continue

            choice = data["choices"][0]
            if choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]

            delta     = choice.get("delta", {})
            reasoning = delta.get("reasoning_content") or delta.get("reasoning") or ""
            content   = delta.get("content") or ""

            if reasoning:
                sys.stdout.write(_c(_MAGENTA, reasoning))
                sys.stdout.flush()
                thinking_parts.append(reasoning)
                if on_thinking:
                    on_thinking("".join(thinking_parts))

            if content:
                if channel_seen:
                    sys.stdout.write(_c(_CYAN, content))
                    sys.stdout.flush()
                    content_parts.append(content)
                    if on_thinking:
                        on_thinking("".join(thinking_parts) + "".join(content_parts))
                elif "<channel|>" in content:
                    channel_seen = True
                    before, after = content.split("<channel|>", 1)
                    if before:
                        sys.stdout.write(_c(_MAGENTA, before))
                        sys.stdout.flush()
                        thinking_parts.append(before)
                        if on_thinking:
                            on_thinking("".join(thinking_parts))
                    if after:
                        sys.stdout.write(_c(_CYAN, after))
                        sys.stdout.flush()
                        content_parts.append(after)
                        if on_thinking:
                            on_thinking("".join(thinking_parts) + "".join(content_parts))
                else:
                    sys.stdout.write(_c(_MAGENTA, content))
                    sys.stdout.flush()
                    thinking_parts.append(content)
                    if on_thinking:
                        on_thinking("".join(thinking_parts))

    except KeyboardInterrupt:
        finish_reason = "interrupted"
        raise
    finally:
        resp.close()

    return "".join(thinking_parts), "".join(content_parts), finish_reason


def summarize_chunk(rolling_ctx: str, chunk_text: str) -> str:
    """Call the model (non-streaming) to produce a one-sentence summary.

    max_tokens must be large enough to cover Gemma 4's internal think block
    before the <channel|> separator, plus the actual response sentence.
    80 tokens is far too small; 1024 gives the thinking room to breathe.
    """
    ctx_tail = rolling_ctx[-500:] if rolling_ctx else "(beginning of stream)"
    messages = [
        {"role": "system", "content": SUMMARY_SYSTEM},
        {
            "role": "user",
            "content": (
                f"Prior context (tail):\n{ctx_tail}\n\n"
                f"New thought:\n{chunk_text[:3000]}\n\n"
                "Summarize the new thought in one sentence."
            ),
        },
    ]
    payload = {
        "model":       MODEL,
        "messages":    messages,
        "stream":      False,
        "max_tokens":  1024,   # thinking block + response sentence
        "temperature": 0.3,
        "top_p":       0.9,
    }
    try:
        resp = requests.post(URL, json=payload, headers=HEADERS, timeout=60, verify=SSL_VERIFY)
        if resp.status_code == 200:
            data    = resp.json()
            msg     = data["choices"][0]["message"]
            content = (msg.get("content") or "").strip()
            reason  = (msg.get("reasoning_content") or "").strip()
            finish  = data["choices"][0].get("finish_reason", "?")
            _p(_GREY, f"  [summarize finish={finish} content={len(content)}c reasoning={len(reason)}c]")
            # content holds text after <channel|>; reasoning holds the think block.
            # Prefer content (the actual response) over the raw thinking.
            text = content or reason
            return text
        _p(_RED, f"[summarize HTTP {resp.status_code}: {resp.text[:120]}]")
    except Exception as exc:
        _p(_RED, f"[summarize error: {exc}]")
    return ""


# ── Main loop ──────────────────────────────────────────────────────────────────

def main():
    global URL, MODEL, HEADERS, SSL_VERIFY

    parser = argparse.ArgumentParser(
        description="Autonomous stream-of-consciousness generation loop",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--url", default="https://127.0.0.1:18000/v1/chat/completions",
                        help="vLLM completions endpoint (default: https://127.0.0.1:18000/v1/chat/completions)")
    parser.add_argument("--model", default="google/gemma-4-E2B-it",
                        help="Model name (default: google/gemma-4-E2B-it)")
    parser.add_argument("--api-key", default="", dest="api_key",
                        help="Bearer token for API authentication (default: none)")
    parser.add_argument("--seed", default=DEFAULT_SEED,
                        help="Opening thought seed")
    parser.add_argument("--chunks", type=int, default=0,
                        help="Max chunks to generate; 0 = unlimited (default: 0)")
    parser.add_argument("--chunk-tokens", type=int, default=512,
                        help="Tokens per API call (default: 512)")
    parser.add_argument("--context-chars", type=int, default=8192,
                        help="Rolling context window in characters (default: 8192)")
    parser.add_argument("--history", default="gen_history/stream.json",
                        help="Output history JSON path (default: gen_history/stream.json)")
    parser.add_argument("--continue", dest="resume", action="store_true",
                        help="Resume from the existing history file")
    args = parser.parse_args()

    URL        = args.url
    MODEL      = args.model
    HEADERS    = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else {}
    SSL_VERIFY = False  # self-signed certs used locally

    history_path = Path(args.history)
    history_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Banner ─────────────────────────────────────────────────────────────────
    _p(_CYAN, "\n╔══════════════════════════════════════════════════════════╗")
    _p(_CYAN, "║         Stream-of-Consciousness Loop                    ║")
    _p(_CYAN, "╠══════════════════════════════════════════════════════════╣")
    _p(_CYAN, f"║  chunk_tokens  : {args.chunk_tokens:<39}║")
    _p(_CYAN, f"║  context_chars : {args.context_chars:<39}║")
    _p(_CYAN, f"║  history       : {str(history_path):<39}║")
    _p(_CYAN, "╠══════════════════════════════════════════════════════════╣")
    _p(_CYAN, "║  Ctrl+C to stop and save                                ║")
    _p(_CYAN, "╚══════════════════════════════════════════════════════════╝\n")

    # ── State ──────────────────────────────────────────────────────────────────
    seed        = args.seed
    accumulated = ""   # full joined thinking for rolling context
    chunk_num   = 0
    start_time  = time.time()

    # ── Resume or start fresh ──────────────────────────────────────────────────
    if args.resume and history_path.exists():
        history = load_history(history_path)
        # Drop any incomplete chunk (no summary yet) so we restart cleanly.
        history = [e for e in history
                   if e.get("role") != "chunk" or e.get("summary") is not None]
        chunks = [e for e in history if e.get("role") == "chunk"]
        if history and history[0]["role"] == "user":
            seed = history[0]["content"]
        accumulated = "\n\n".join(
            c["thinking"] for c in chunks if c.get("thinking")
        )
        chunk_num = max((c.get("chunk", 0) for c in chunks), default=0)
        _p(_GREEN, f"[Resumed — {len(accumulated):,} chars, {chunk_num} chunks]\n")
        write_history(history_path, history)
    else:
        _p(_GREY, f"Seed: {seed[:90]}{'...' if len(seed) > 90 else ''}\n")
        history = [{"role": "user", "content": seed, "tokens": len(seed.split())}]
        write_history(history_path, history)

    _p(_GREY, f"View live → python server.py {history_path}\n")

    # ── Generation loop ────────────────────────────────────────────────────────
    try:
        while True:
            chunk_num += 1
            if args.chunks and chunk_num > args.chunks:
                _p(_GREY, f"\n[Max chunks ({args.chunks}) reached — stopping]")
                break

            # ── Rolling context window ─────────────────────────────────────────
            rolling_ctx = accumulated[-args.context_chars:] if accumulated else ""

            # ── Chunk header ───────────────────────────────────────────────────
            elapsed = time.time() - start_time
            print(_c(_GREY,
                     f"\n── chunk {chunk_num:03d}  "
                     f"elapsed={elapsed:.0f}s  "
                     f"total={len(accumulated):,}c"),
                  flush=True)

            # ── New chunk entry (in-progress, no summary yet) ──────────────────
            chunk_entry: dict = {
                "role":          "chunk",
                "chunk":         chunk_num,
                "thinking":      "",
                "summary":       None,
                "finish_reason": "in_progress",
            }
            history.append(chunk_entry)
            write_history(history_path, history)

            # ── API call with live thinking updates ────────────────────────────
            messages = build_messages(seed, rolling_ctx)

            def _live_write(chunk_thinking: str):
                chunk_entry["thinking"] = _cap_paragraphs(_clean(chunk_thinking))
                write_history(history_path, history)

            thinking, content, finish_reason = generate_chunk(
                messages, args.chunk_tokens, on_thinking=_live_write,
            )

            if finish_reason == "error":
                _p(_RED, "[Generation error — retrying in 5s]")
                history.remove(chunk_entry)
                time.sleep(5)
                chunk_num -= 1
                continue

            # ── Accumulate ────────────────────────────────────────────────────
            if finish_reason == "stop" and content.strip():
                _p(_YELLOW, "[Model concluded — continuing]")
                new_text = _clean(
                    thinking + ("\n\n" if thinking.strip() else "") + content.strip()
                )
            else:
                new_text = _clean(thinking + content)

            if not new_text.strip():
                _p(_YELLOW, "[Empty chunk — skipping]")
                history.remove(chunk_entry)
                write_history(history_path, history)
                chunk_num -= 1
                continue

            new_text = _cap_paragraphs(new_text)
            accumulated += ("\n\n" if accumulated else "") + new_text

            chunk_entry["thinking"]      = new_text
            chunk_entry["finish_reason"] = finish_reason
            write_history(history_path, history)

            # ── Summarize ─────────────────────────────────────────────────────
            _p(_CYAN, f"\n[summarizing chunk {chunk_num}…]", end=" ", flush=True)
            summary = summarize_chunk(rolling_ctx, new_text)
            _p(_GREEN, summary or "(empty)")

            chunk_entry["summary"] = summary or "(no summary)"
            write_history(history_path, history)

            if finish_reason == "interrupted":
                break

    except KeyboardInterrupt:
        pass

    # ── Final save ─────────────────────────────────────────────────────────────
    # Mark any dangling in-progress chunk as stopped.
    for entry in history:
        if entry.get("role") == "chunk" and entry.get("finish_reason") == "in_progress":
            entry["finish_reason"] = "stopped"
    write_history(history_path, history)

    elapsed = time.time() - start_time
    completed = [e for e in history if e.get("role") == "chunk" and e.get("summary")]
    _p(_CYAN, f"\n\n╔══════════════════════════════════════╗")
    _p(_CYAN, f"║  Stream stopped.                     ║")
    _p(_CYAN, f"║  chunks     : {chunk_num:<23}║")
    _p(_CYAN, f"║  total chars: {len(accumulated):<23,}║")
    _p(_CYAN, f"║  elapsed    : {elapsed:.0f}s{'':<22}║")
    _p(_CYAN, f"╚══════════════════════════════════════╝")
    _p(_GREY, f"\nHistory saved → {history_path}")
    _p(_GREY, f"View live    → python server.py {history_path}")


if __name__ == "__main__":
    main()
