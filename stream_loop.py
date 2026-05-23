#!/usr/bin/env python3
"""
stream_loop.py — Autonomous stream-of-consciousness generation loop.

Generates in CHUNK_TOKENS-sized bursts and lets the model wander freely.
Writes to gen_history/stream.json (same format as chat_test.py history)
so server.py can display the live stream in the browser:

    python server.py --history gen_history/stream.json

Usage:
    python stream_loop.py [options]

Key options:
    --seed TEXT          Opening thought (default: built-in)
    --chunks N           Max generation chunks (0 = unlimited, Ctrl+C to stop)
    --chunk-tokens N     Tokens per API call                  (default: 512)
    --context-chars N    Rolling context window in characters  (default: 2048)
    --history FILE       Output JSON path (default: gen_history/stream.json)
    --continue           Resume from an existing history file
"""

import sys
import json
import time
import argparse
import requests
from pathlib import Path

# ── Server / model ─────────────────────────────────────────────────────────────
URL   = "http://127.0.0.1:8000/v1/chat/completions"
MODEL = "gemma4"

SYSTEM_PROMPT = (
    "You are a continuous thought. You think the way a curious, well-read person does while browsing Wikipedia late at night — one question leads to the next, with hypothesis, thought experiments, and example scenarios simulated in your mind. Not just think about specific topics, but also think about general priciples and concepts that govern the field, and the trend that is shaping the field.\n\n"
    "Rules you must follow absolutely:\n"
    "- Never summarize, conclude, wrap up, or say things like \"in summary\", "
    "\"so\", \"ultimately\", \"this shows us\", or \"it's fascinating how\"\n"
    "- Never use bullet points, headers, or markdown formatting, just sentences and paragraphs with clear grammar and separation between thoughts.\n"
    "- Use normal punctuation within sentences — capital letters, commas, periods, semicolons.\n"
    "- Always write in first person, present tense, as thoughts arriving, during thinking. This is not a conversation, it is a stream of consciousness.\n"
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


# ── Message construction ───────────────────────────────────────────────────────

def build_messages(seed: str, rolling_ctx: str) -> list:
    """
    Construct the OpenAI-format messages list for the next chunk.

    First chunk (nudge=None): seed is the opening user prompt.
    Subsequent chunks: rolling_ctx replaces the seed so the model is free
    to wander — keeping the seed permanently anchors it to the original topic.
    """
    msgs: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]
    if rolling_ctx:
        msgs.append({"role": "user", "content": rolling_ctx})
    else:
        msgs.append({"role": "user", "content": seed})
    return msgs

# ── Single-chunk generation ────────────────────────────────────────────────────

def generate_chunk(
    messages: list,
    chunk_tokens: int,
    on_thinking=None,
) -> tuple[str, str, str]:
    """
    Stream one generation chunk from vLLM.
    Returns (thinking_text, content_text, finish_reason).

    on_thinking(chunk_thinking_so_far) — callback fired on every token batch;
    used to write live updates to disk without waiting for the chunk to finish.
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
        resp = requests.post(URL, json=payload, stream=True, timeout=180)
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


# ── Main loop ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Autonomous stream-of-consciousness generation loop",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--seed", default=DEFAULT_SEED,
                        help="Opening thought seed")
    parser.add_argument("--chunks", type=int, default=0,
                        help="Max chunks to generate; 0 = unlimited (default: 0)")
    parser.add_argument("--chunk-tokens", type=int, default=256,
                        help="Tokens per API call (default: 1024)")
    parser.add_argument("--context-chars", type=int, default=8192,
                        help="Rolling context window in characters (default: 8192)")
    parser.add_argument("--history", default="gen_history/stream.json",
                        help="Output history JSON path (default: gen_history/stream.json)")
    parser.add_argument("--continue", dest="resume", action="store_true",
                        help="Resume from the existing history file")
    args = parser.parse_args()

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
    accumulated = ""
    chunk_num   = 0
    start_time  = time.time()

    # ── Resume or start fresh ──────────────────────────────────────────────────
    if args.resume and history_path.exists():
        history = load_history(history_path)
        if len(history) >= 2 and history[0]["role"] == "user":
            seed = history[0]["content"]
            last_asst = next(
                (e for e in reversed(history) if e["role"] == "assistant"), None
            )
            if last_asst:
                accumulated = last_asst.get("thinking", "") or last_asst.get("content", "")
        _p(_GREEN, f"[Resumed — {len(accumulated):,} chars of prior stream loaded]\n")
        asst_entry = last_asst if last_asst else {
            "role": "assistant", "content": "", "thinking": accumulated,
            "tokens": 0, "finish_reason": "in_progress",
        }
        if last_asst not in history:
            history.append(asst_entry)
    else:
        _p(_GREY, f"Seed: {seed[:90]}{'...' if len(seed) > 90 else ''}\n")
        asst_entry = {
            "role": "assistant", "content": "", "thinking": "",
            "tokens": 0, "finish_reason": "in_progress",
        }
        history = [
            {"role": "user", "content": seed, "tokens": len(seed.split())},
            asst_entry,
        ]
        write_history(history_path, history)

    _p(_GREY, f"View live → python server.py --history {history_path}\n")

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

            # ── Inter-chunk pause ──────────────────────────────────────────────
            # if chunk_num > 1:
            #     time.sleep(1)

            # ── API call ───────────────────────────────────────────────────────
            messages = build_messages(seed, rolling_ctx)
            _prefix  = accumulated

            def _live_write(chunk_thinking: str):
                asst_entry["thinking"]      = _prefix + _cap_paragraphs(_clean(chunk_thinking))
                asst_entry["finish_reason"] = "in_progress"
                write_history(history_path, history)

            thinking, content, finish_reason = generate_chunk(
                messages, args.chunk_tokens, on_thinking=_live_write
            )

            nudge = None

            if finish_reason == "error":
                _p(_RED, "[Generation error — retrying in 5s]")
                time.sleep(5)
                chunk_num -= 1
                continue

            # ── Accumulate ────────────────────────────────────────────────────
            if finish_reason == "stop" and content.strip():
                _p(_YELLOW, "[Model concluded — continuing]")
                nudge = "continue"
                new_text = _clean(thinking + ("\n\n" if thinking.strip() else "") + content.strip())
            else:
                new_text = _clean(thinking + content)

            if not new_text.strip():
                _p(_YELLOW, "[Empty chunk — retrying]")
                nudge = "continue"
            else:
                accumulated += ("\n\n" if accumulated else "") + _cap_paragraphs(new_text)

            # ── Final update for this chunk ────────────────────────────────────
            asst_entry["thinking"]      = accumulated
            asst_entry["finish_reason"] = finish_reason
            write_history(history_path, history)

            if finish_reason == "interrupted":
                break

    except KeyboardInterrupt:
        pass

    # ── Final save ─────────────────────────────────────────────────────────────
    asst_entry["finish_reason"] = "stopped"
    write_history(history_path, history)

    elapsed = time.time() - start_time
    _p(_CYAN, f"\n\n╔══════════════════════════════════════╗")
    _p(_CYAN, f"║  Stream stopped.                     ║")
    _p(_CYAN, f"║  chunks     : {chunk_num:<23}║")
    _p(_CYAN, f"║  total chars: {len(accumulated):<23,}║")
    _p(_CYAN, f"║  elapsed    : {elapsed:.0f}s{'':<22}║")
    _p(_CYAN, f"╚══════════════════════════════════════╝")
    _p(_GREY, f"\nHistory saved → {history_path}")
    _p(_GREY, f"View live    → python server.py --history {history_path}")


if __name__ == "__main__":
    main()
