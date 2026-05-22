#!/usr/bin/env python3
"""
stream_loop.py — Autonomous stream-of-consciousness generation loop.

Generates in CHUNK_TOKENS-sized bursts, detects repetition / degeneration
using n-gram overlap, and injects Wikipedia-sourced topic stimuli to steer
the stream into fresh territory before it collapses.

Writes to gen_history/stream.json (same format as chat_test.py history)
so server.py can display the live stream in the browser:

    python server.py --history gen_history/stream.json

Usage:
    python stream_loop.py [options]

Key options:
    --seed TEXT          Opening thought (default: built-in)
    --chunks N           Max generation chunks (0 = unlimited, Ctrl+C to stop)
    --chunk-tokens N     Tokens per API call                  (default: 350)
    --context-chars N    Rolling context window in characters  (default: 6000)
    --inject-every N     Inject Wikipedia stimulus every N chunks (default: 5)
    --rep-threshold F    N-gram overlap ratio that triggers injection (default: 0.25)
    --no-wiki            Skip Wikipedia; use simple pivot phrases only
    --history FILE       Output JSON path (default: gen_history/stream.json)
    --continue           Resume from an existing history file
"""

import re
import sys
import json
import time
import argparse
import requests
from pathlib import Path
from urllib.parse import quote as urlquote

# ── Server / model ─────────────────────────────────────────────────────────────
URL   = "http://127.0.0.1:8000/v1/chat/completions"
MODEL = "gemma4"

SYSTEM_PROMPT = (
    "You are a continuous thought. You think the way a curious, well-read person does while browsing Wikipedia late at night — one question leads to the next.\n\n"
    "Rules you must follow absolutely:\n"
    "- Never summarize, conclude, wrap up, or say things like \"in summary\", "
    "\"so\", \"ultimately\", \"this shows us\", or \"it's fascinating how\"\n"
    "- Never use bullet points, headers, or markdown formatting,just sentences and paragraphs with clear grammar and separation between thoughts.\n"
    "- Always start a new sentence with a capital letter and end it with a period. Always start a new paragraph with a capital letter and a space between the previous paragraph.\n"
    "- Use normal punctuation within sentences — capital letters, commas, periods, semicolons.\n"
    "- Write in first person, present tense, as thoughts arriving.\n"
    # "- The thought should feel like it could go on forever because it can."
)

DEFAULT_SEED = (
    "I keep thinking about the best way to utilize hardware resources, both compute and memory, for efficient MoE model inference. "
)

# Generation params — frequency_penalty is key: it scales with per-token count
# so it bites harder the more a token has been repeated, unlike presence_penalty
# which is a flat one-time penalty. Both together cover different failure modes.
GEN_PARAMS = dict(
    temperature=1.2,
    top_p=0.95,
    top_k=50,
    repetition_penalty=1.1,    # slightly softer than chat_test; loop handles rest
    presence_penalty=1.5,
    frequency_penalty=0.6,     # proportional-to-count penalty; not in chat_test
)

# ── Wikipedia ──────────────────────────────────────────────────────────────────
WIKI_SUMMARY = "https://en.wikipedia.org/api/rest_v1/page/summary/{}"
WIKI_SEARCH  = "https://en.wikipedia.org/w/api.php"
WIKI_HEADERS = {"User-Agent": "stream-loop/1.0 (autonomous-thought-experiment)"}

# ── ANSI palette ───────────────────────────────────────────────────────────────
R = "\033[0m"
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


# ── Degeneration detection ─────────────────────────────────────────────────────

def ngram_overlap_ratio(text: str, n: int = 5, window_words: int = 300) -> float:
    """
    Split the last `2 * window_words` words into two equal halves.
    Return the fraction of n-grams in the second half that appear in the first.
    High ratio → repetitive degeneration.
    """
    words = text.lower().split()
    if len(words) < n * 4:
        return 0.0
    sample = words[-2 * window_words:] if len(words) > 2 * window_words else words
    mid = len(sample) // 2
    prior  = set(tuple(sample[i:i+n]) for i in range(mid - n + 1))
    recent = [tuple(sample[i:i+n]) for i in range(mid, len(sample) - n + 1)]
    if not recent:
        return 0.0
    return sum(1 for ng in recent if ng in prior) / len(recent)


# ── Topic extraction ───────────────────────────────────────────────────────────

def extract_topic(text: str) -> str:
    """
    Pull a concrete topic from the tail of the text for Wikipedia lookup.
    Prefers capitalized proper-noun phrases; falls back to long content words.
    """
    tail = text[-700:]
    # Multi-word capitalized phrases first (e.g. "Mercator Projection", "Roman Empire")
    caps = re.findall(r'\b[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})*\b', tail)
    if caps:
        return caps[-1]
    # Single capitalized word
    caps = re.findall(r'\b[A-Z][a-z]{3,}\b', tail)
    if caps:
        return caps[-1]
    # Long lowercase content word
    words = re.findall(r'\b[a-z]{7,}\b', tail)
    return words[-1] if words else "stream of consciousness"


# ── Wikipedia lookup ───────────────────────────────────────────────────────────

def _wiki_summary_for(title: str) -> str | None:
    """Fetch 1–2 sentences from the Wikipedia REST summary endpoint."""
    try:
        r = requests.get(
            WIKI_SUMMARY.format(urlquote(title)),
            timeout=6,
            headers=WIKI_HEADERS,
        )
        if r.status_code == 200:
            extract = r.json().get("extract", "")
            if len(extract) > 60:
                sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', extract) if s.strip()]
                snippet = " ".join(sentences[:2])
                return snippet if len(snippet) > 50 else None
    except Exception:
        pass
    return None


def fetch_wiki_snippet(query: str) -> str | None:
    """
    Try direct Wikipedia lookup for `query`, then fall back to a search
    and retry with the best search result title.
    Returns a 1–2 sentence snippet, or None on failure.
    """
    snippet = _wiki_summary_for(query)
    if snippet:
        return snippet
    # Search fallback
    try:
        r = requests.get(
            WIKI_SEARCH,
            params={
                "action": "query", "list": "search",
                "srsearch": query, "format": "json", "srlimit": 1,
            },
            timeout=6,
            headers=WIKI_HEADERS,
        )
        if r.status_code == 200:
            results = r.json().get("query", {}).get("search", [])
            if results:
                return _wiki_summary_for(results[0]["title"])
    except Exception:
        pass
    return None


# ── Message construction ───────────────────────────────────────────────────────

def build_messages(seed: str, rolling_ctx: str, nudge: str | None) -> list:
    """
    Construct the OpenAI-format messages list for the next chunk.

    rolling_ctx  — last CONTEXT_CHARS of accumulated thinking fed as a prior
                   assistant turn so the model sees what it's been thinking.
    nudge        — user message to append; if None the prior turn is omitted
                   entirely (used for the very first chunk).

    Gemma 4 thinking format: the assistant content is
        <thinking_text><channel|><response_text>
    When there is no response yet we send  "<thinking_text><channel|>"  so
    the model sees that the assistant was in mid-thought and should continue.
    """
    msgs: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": seed},
    ]
    if rolling_ctx and nudge is not None:
        msgs.append({"role": "assistant", "content": rolling_ctx + "<channel|>"})
        msgs.append({"role": "user",      "content": nudge})
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

    on_thinking(chunk_thinking_so_far: str) — optional callback fired after
    every new token batch with the full within-chunk thinking accumulated so
    far.  Use it to write live updates to disk without waiting for the chunk
    to complete.

    Thinking tokens arrive in delta.reasoning_content (vLLM reasoning parser)
    or, for Gemma 4, can also arrive as delta.content before <channel|>.
    Both paths are handled.  Text is printed to stdout as it arrives.
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

            # reasoning_content field → pure thinking token stream
            if reasoning:
                sys.stdout.write(_c(_MAGENTA, reasoning))
                sys.stdout.flush()
                thinking_parts.append(reasoning)
                if on_thinking:
                    on_thinking("".join(thinking_parts))

            # content field — may contain thinking (pre-channel|) or response
            if content:
                if channel_seen:
                    sys.stdout.write(_c(_CYAN, content))
                    sys.stdout.flush()
                    content_parts.append(content)
                    if on_thinking:
                        combined = "".join(thinking_parts) + "".join(content_parts)
                        on_thinking(combined)
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
                            combined = "".join(thinking_parts) + "".join(content_parts)
                            on_thinking(combined)
                else:
                    # Pre-channel content is thinking
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
    """Capitalize the first alphabetic character of every paragraph (split on \\n\\n)."""
    parts = text.split('\n\n')
    out = []
    for part in parts:
        for i, ch in enumerate(part):
            if ch.isalpha():
                part = part[:i] + ch.upper() + part[i+1:]
                break
        out.append(part)
    return '\n\n'.join(out)


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
    parser.add_argument("--chunk-tokens", type=int, default=512,
                        help="Tokens per API call (default: 512)")
    parser.add_argument("--context-chars", type=int, default=2048,
                        help="Rolling context window in characters (default: 2048)")
    parser.add_argument("--inject-every", type=int, default=999999,
                        help="Inject Wikipedia stimulus every N chunks (default: 5)")
    parser.add_argument("--rep-threshold", type=float, default=999999,
                        help="N-gram overlap ratio that triggers stimulus injection (default: 0.25)")
    parser.add_argument("--no-wiki", action="store_true",
                        help="Skip Wikipedia; use a simple pivot phrase instead")
    parser.add_argument("--history", default="gen_history/stream.json",
                        help="Output history JSON path (default: gen_history/stream.json)")
    parser.add_argument("--continue", dest="resume", action="store_true",
                        help="Resume from the existing history file")
    args = parser.parse_args()

    history_path = Path(args.history)
    history_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Banner ─────────────────────────────────────────────────────────────────
    wiki_status = "off (--no-wiki)" if args.no_wiki else "on"
    _p(_CYAN, "\n╔══════════════════════════════════════════════════════════╗")
    _p(_CYAN, "║         Stream-of-Consciousness Loop                    ║")
    _p(_CYAN, "╠══════════════════════════════════════════════════════════╣")
    _p(_CYAN, f"║  chunk_tokens  : {args.chunk_tokens:<39}║")
    _p(_CYAN, f"║  context_chars : {args.context_chars:<39}║")
    _p(_CYAN, f"║  inject_every  : {args.inject_every:<39}║")
    _p(_CYAN, f"║  rep_threshold : {args.rep_threshold:<39}║")
    _p(_CYAN, f"║  wikipedia     : {wiki_status:<39}║")
    _p(_CYAN, f"║  history       : {str(history_path):<39}║")
    _p(_CYAN, "╠══════════════════════════════════════════════════════════╣")
    _p(_CYAN, "║  Ctrl+C to stop and save                                ║")
    _p(_CYAN, "╚══════════════════════════════════════════════════════════╝\n")

    # ── State ──────────────────────────────────────────────────────────────────
    seed          = args.seed
    accumulated   = ""       # full accumulated thinking stream (written to history)
    inject_log: list[dict] = []
    chunk_num     = 0
    start_time    = time.time()

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
        seed = args.seed
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

    _p(_GREY, f"Serving via: python server.py --history {history_path}\n")

    # ── Generation loop ────────────────────────────────────────────────────────
    nudge: str | None = None   # user message to inject (None = first chunk)

    try:
        while True:
            chunk_num += 1
            if args.chunks and chunk_num > args.chunks:
                _p(_GREY, f"\n[Max chunks ({args.chunks}) reached — stopping]")
                break

            # ── Degeneration check ─────────────────────────────────────────────
            rep_ratio   = ngram_overlap_ratio(accumulated) if len(accumulated) > 400 else 0.0
            degenerated = rep_ratio >= args.rep_threshold

            # ── Injection decision ─────────────────────────────────────────────
            scheduled = chunk_num > 1 and (chunk_num - 1) % args.inject_every == 0
            should_inject = degenerated or scheduled

            if should_inject:
                topic   = extract_topic(accumulated) if accumulated else "consciousness"
                snippet = None if args.no_wiki else fetch_wiki_snippet(topic)

                if snippet:
                    nudge = (
                        f"[a fragment of memory surfaces, pulling the thought sideways: "
                        f"{snippet}] continue"
                    )
                    inject_log.append({
                        "chunk": chunk_num, "topic": topic,
                        "snippet": snippet[:100], "reason": "degenerated" if degenerated else "scheduled",
                    })
                    reason_label = (
                        _c(_RED, f"DEGENERATED rep={rep_ratio:.2f}")
                        if degenerated else
                        _c(_YELLOW, "scheduled")
                    )
                    _p(_YELLOW, f"\n\n[INJECT #{len(inject_log)} {reason_label} → '{topic}']")
                    _p(_GREY,   f"  wiki: {snippet[:110]}")
                else:
                    # No Wikipedia result — simple pivot
                    nudge = "continue — let the thought drift somewhere entirely different"
                    if degenerated:
                        _p(_YELLOW, f"\n\n[PIVOT rep={rep_ratio:.2f}, no wiki result for '{topic}']")

                # On degeneration: trim the rolling context to its first half so
                # the model doesn't see the repetitive tail and mirror it again.
                if degenerated and accumulated:
                    trim_to = max(args.context_chars // 3, 500)
                    # Keep earlier coherent part + just a tiny bit of recent
                    accumulated_for_ctx = accumulated[-(args.context_chars):-(args.context_chars // 3)] or accumulated[:trim_to]
                else:
                    accumulated_for_ctx = accumulated
            else:
                accumulated_for_ctx = accumulated

            # ── Rolling context window ─────────────────────────────────────────
            rolling_ctx = accumulated_for_ctx[-args.context_chars:] if accumulated_for_ctx else ""

            # ── Chunk header ───────────────────────────────────────────────────
            elapsed = time.time() - start_time
            rep_str = (_c(_RED, f"{rep_ratio:.2f}!") if degenerated else _c(_GREY, f"{rep_ratio:.2f}"))
            inject_tag = _c(_YELLOW, " [INJECTING]") if (should_inject and nudge) else ""
            print(
                _c(_GREY,
                   f"\n── chunk {chunk_num:03d}  "
                   f"elapsed={elapsed:.0f}s  "
                   f"total={len(accumulated):,}c  "
                   f"rep={rep_ratio:.2f}") + inject_tag,
                flush=True,
            )

            # For the very first chunk, nudge stays None → no prior-turn context.
            # After that, default nudge is plain "continue".
            if chunk_num > 1 and nudge is None:
                nudge = "continue"

            # ── API call ───────────────────────────────────────────────────────
            messages  = build_messages(seed, rolling_ctx, nudge)
            _prefix   = accumulated  # snapshot before this chunk starts

            def _live_write(chunk_thinking: str):
                clean = chunk_thinking.replace("<channel|>", "\n\n")
                asst_entry["thinking"]      = _prefix + _cap_paragraphs(clean)
                asst_entry["finish_reason"] = "in_progress"
                write_history(history_path, history)

            thinking, content, finish_reason = generate_chunk(
                messages, args.chunk_tokens, on_thinking=_live_write
            )

            nudge = None  # reset; will be set again if next cycle needs injection

            if finish_reason == "error":
                _p(_RED, "[Generation error — retrying in 5s]")
                time.sleep(5)
                chunk_num -= 1   # don't count the failed chunk
                continue

            # ── Accumulate ────────────────────────────────────────────────────
            # If the model concluded its thinking and produced a response
            # (finish_reason="stop", content non-empty), keep the content but
            # put it on its own paragraph, then force a pivot away from conclusion mode.
            if finish_reason == "stop" and content.strip():
                _p(_YELLOW, "[Model concluded — keeping response, forcing pivot]")
                nudge = "continue — the thought isn't finished, keep going"
                new_text = (thinking + ("\n\n" if thinking.strip() else "") + content.strip()).replace("<channel|>", "\n\n")
            else:
                new_text = thinking + content

            if not new_text.strip():
                _p(_YELLOW, "[Empty chunk — forcing pivot next cycle]")
                nudge = "continue — something new is surfacing"
            else:
                accumulated += ("\n\n" if accumulated else "") + _cap_paragraphs(new_text.replace("<channel|>", "\n\n"))

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
    _p(_CYAN, f"║  injections : {len(inject_log):<23}║")
    _p(_CYAN, f"╚══════════════════════════════════════╝")
    if inject_log:
        _p(_GREY, "\nInjection log:")
        for inj in inject_log:
            _p(_GREY, f"  chunk {inj['chunk']:03d}  [{inj['reason']}]  topic='{inj['topic']}'")
            _p(_GREY, f"           {inj['snippet']}...")
    _p(_GREY, f"\nHistory saved → {history_path}")
    _p(_GREY, f"View live    → python server.py --history {history_path}")


if __name__ == "__main__":
    main()
