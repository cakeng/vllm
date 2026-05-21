#!/usr/bin/env python3
"""
Standalone vLLM Chat Client — streaming, colored output, thinking visibility.
"""

import re
import requests
import json
import sys
import argparse
from pathlib import Path

try:
    from prompt_toolkit import prompt as pt_prompt
    HAS_PROMPT_TOOLKIT = True
except ImportError:
    HAS_PROMPT_TOOLKIT = False
    import readline


URL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL = "google/gemma-4-E4B-it"
API_KEY = ""

MAX_TOKENS = 32_768          # max generation length
CONTEXT_WINDOW = 96_000      # history budget = 128k - 32k generation headroom

HEADERS = {"Authorization": f"Bearer {API_KEY}"} if API_KEY else {}

SYSTEM_PROMPT = """"""


def print_colored(msg, color=None):
    if color:
        sys.stdout.write("\033[%sm%s\033[0m" % (color, msg))
    else:
        sys.stdout.write(msg)
    sys.stdout.flush()


def strip_thinking_from_content(content):
    """Strip thinking blocks from content before saving to history.

    Handles both the <channel|> separator Gemma4 uses in the content field
    and the 'Here's a thinking process:' pattern that can leak through.
    """
    if not content:
        return content
    if "<channel|>" in content:
        content = content.split("<channel|>", 1)[1]
    cleaned = re.sub(r"Here's a thinking process:[\s\S]*?(?=\n\n|\Z)", "", content)
    return cleaned.strip()


def chat(messages, verbose=False, on_chunk=None):
    """Stream tokens with diagnostics. If verbose, also dump raw chunks.

    on_chunk(partial_content: str) is called after every content delta with
    the full content accumulated so far, enabling live file updates.
    """
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0.9,
        "max_tokens": MAX_TOKENS,
        "top_p": 0.95,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    response = requests.post(URL, json=payload, headers=HEADERS, stream=True)
    if response.status_code != 200:
        print("Error: HTTP %d: %s\n" % (response.status_code, response.text))
        return "", payload, {}, None

    content_parts = []
    reasoning_parts = []
    content_buffer = ""   # holds content before <channel|> separator
    channel_seen = False  # True once <channel|> has been found
    thinking_started = False
    response_started = False
    raw_chunks = [] if verbose else None
    finish_reason = None
    usage = {}

    for line in response.iter_lines():
        if not line:
            continue
        decoded = line.decode()
        if not decoded.startswith("data: "):
            continue
        data_str = decoded.split("data: ", 1)[1]
        if data_str == "[DONE]":
            break
        if raw_chunks is not None:
            raw_chunks.append(data_str)
        try:
            data = json.loads(data_str)
        except json.JSONDecodeError:
            continue
        if data.get("usage"):
            usage = data["usage"]
            continue
        if not data.get("choices"):
            continue
        choice = data["choices"][0]
        if choice.get("finish_reason"):
            finish_reason = choice["finish_reason"]
        delta = choice.get("delta", {})
        content = delta.get("content", "")
        reasoning = delta.get("reasoning", "") or delta.get("reasoning_content", "")
        if reasoning:
            if not thinking_started:
                sys.stdout.write("\033[90m── Thinking ──────────────────────────────────\033[0m\n")
                thinking_started = True
            sys.stdout.write("\033[1;35m" + reasoning + "\033[0m")
            reasoning_parts.append(reasoning)
            sys.stdout.flush()
        if content:
            if channel_seen:
                if not response_started:
                    sys.stdout.write("\n\033[1;36m── Response ───────────────────────────────────\033[0m\n")
                    response_started = True
                sys.stdout.write(content)
                content_parts.append(content)
                sys.stdout.flush()
                if on_chunk:
                    on_chunk("".join(content_parts))
            else:
                # Buffer until <channel|> separator is found.
                content_buffer += content
                if "<channel|>" in content_buffer:
                    channel_seen = True
                    before, after = content_buffer.split("<channel|>", 1)
                    if before:
                        if not thinking_started:
                            sys.stdout.write("\033[90m── Thinking ──────────────────────────────────\033[0m\n")
                            thinking_started = True
                        sys.stdout.write("\033[1;35m" + before + "\033[0m")
                        reasoning_parts.append(before)
                    sys.stdout.write("\n\033[1;36m── Response ───────────────────────────────────\033[0m\n")
                    response_started = True
                    if after:
                        sys.stdout.write(after)
                        content_parts.append(after)
                        if on_chunk:
                            on_chunk("".join(content_parts))
                    sys.stdout.flush()
                # else: still buffering, nothing to print yet

    sys.stdout.write("\n")

    if verbose:
        print("\033[1;33m\n========== RAW API CHUNKS ==========\033[0m")
        for i, chunk in enumerate(raw_chunks or []):
            print("  [%d] %s\n" % (i + 1, chunk))
        print("\033[1;33m========== END RAW CHUNKS (total: %d) ==========\033[0m" % len(raw_chunks or []))

    reasoning_text = "".join(reasoning_parts)
    if finish_reason and finish_reason != "stop":
        sys.stdout.write("\033[33m  Finish reason: %s\033[0m\n" % finish_reason)

    return "".join(content_parts), reasoning_text, usage, finish_reason


def load_history(history_file):
    history_file = Path(history_file)
    if history_file.exists():
        with open(history_file, "r") as f:
            return json.load(f)
    return []


def write_history(history_file, history):
    with open(Path(history_file), "w") as f:
        json.dump(history, f, indent=2)



def build_messages(history, system_prompt):
    """Walk history newest-to-oldest, keeping messages whose exact stored token
    counts fit within CONTEXT_WINDOW.  Always prepends the system prompt.

    Returns:
        messages          – list ready to POST to vLLM
        sent_history_tokens – sum of stored tokens for the history slice sent
                              (does NOT include system-prompt tokens)
    """
    budget = CONTEXT_WINDOW
    kept = []
    for entry in reversed(history):
        # Only user messages carry the meaningful turn token cost.
        # Assistant message costs are already counted in the following user entry.
        if entry["role"] == "user":
            t = entry.get("tokens", 0)
            if budget - t < 0:
                break
            budget -= t
        if entry["role"] == "assistant" and entry.get("thinking"):
            full_content = entry["thinking"] + "<channel|>" + entry["content"]
        else:
            full_content = entry["content"]
        kept.append({"role": entry["role"], "content": full_content})

    included = len(history) - len(kept)
    if included:
        sys.stdout.write(
            "\033[33m[CTX] Keeping %d of %d history messages to stay within %dk-token window\033[0m\n"
            % (len(kept), len(history), CONTEXT_WINDOW // 1000)
        )

    kept.reverse()
    sent_history_tokens = CONTEXT_WINDOW - budget
    messages = [{"role": "system", "content": system_prompt}] + kept
    return messages, sent_history_tokens


def main():
    parser = argparse.ArgumentParser(description="vLLM Chat Client")
    parser.add_argument("--continue", dest="continue_chat", action="store_true",
                       help="Continue chat - load messages from history.json")
    parser.add_argument("--history", default="gen_history/history.json",
                       help="Path to history file (default: gen_history/history.json)")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose mode - dump raw API chunks after each response")
    args = parser.parse_args()
    history_file = Path(args.history)

    # history is the authoritative record: [{role, content, tokens, thinking?}, ...]
    # It is persisted to disk and rebuilt into messages before each request.
    # tokens on user messages = prompt_tokens_N - prompt_tokens_{N-1}: the exact
    # incremental prompt cost for (prev assistant response + this user message).
    # tokens on assistant messages = completion_tokens (for display only).
    history = []
    prev_prompt_tokens = 0      # prompt_tokens from the previous turn
    prev_completion_tokens = 0  # completion_tokens from the previous turn
    if args.continue_chat:
        history = load_history(history_file)

    print("")
    print("\033[1;34m╔══════════════════════════════════════════════════════════════╗\033[0m")
    print("\033[1;34m║     vLLM Chat Client (Streaming + Thinking + Color)         ║\033[0m")
    print("\033[1;34m║     Model: %s            ║\033[0m" % MODEL)
    print("\033[1;34m║                                                             ║\033[0m")
    print("\033[1;22;34m║     :reset  - Clear history                                   ║\033[0m")
    print("\033[1;22;34m║     :quit   - Exit                                          ║\033[0m")
    print("\033[1;34m║     Use --verbose on startup for raw API chunks             ║\033[0m")
    print("\033[1;34m╚══════════════════════════════════════════════════════════════╝\033[0m")

    while True:
        try:
            if HAS_PROMPT_TOOLKIT:
                user_input = pt_prompt('You: ')
            else:
                user_input = input("You: ")
        except (EOFError, KeyboardInterrupt):
            print("\nBye!\n")
            sys.exit(0)

        if not user_input.strip():
            continue

        if user_input == ":quit":
            print("Bye!\n")
            break

        if user_input == ":reset":
            history = []
            prev_prompt_tokens = 0
            prev_completion_tokens = 0
            with open(history_file, "w") as f:
                json.dump([], f)
            print("\033[32m[RESET]\033[0m History cleared.\n")
            continue

        # Build the context window from history (newest → oldest) using exact
        # stored token counts, then append the new user message.
        messages, sent_history_tokens = build_messages(history, SYSTEM_PROMPT)
        messages.append({"role": "user", "content": user_input})

        # Seed a live entry in history so the file updates during streaming.
        live_entry = {"role": "assistant", "content": "", "thinking": "", "tokens": 0}
        history.append({"role": "user", "content": user_input, "tokens": 0})
        history.append(live_entry)
        write_history(history_file, history)

        def _live_update(partial: str):
            live_entry["content"] = partial
            write_history(history_file, history)

        print("\n\033[1;36mAssistant:\033[0m\n")
        text, reasoning, usage, finish_reason = chat(messages, verbose=args.verbose, on_chunk=_live_update)

        if text and usage:
            prompt_tokens     = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)

            # With thinking kept in context, completion_tokens is the exact token
            # cost of the assistant message in the next prompt.
            # user_tokens = what vLLM counted for the new user message alone.
            user_tokens = prompt_tokens - prev_prompt_tokens - prev_completion_tokens
            history[-2]["tokens"] = user_tokens

            # Split completion_tokens into thinking vs content by char ratio.
            thinking_chars = len(reasoning)
            content_chars  = len(text)
            total_chars    = thinking_chars + content_chars
            if total_chars > 0:
                thinking_tokens = round(completion_tokens * thinking_chars / total_chars)
            else:
                thinking_tokens = 0
            content_tokens = completion_tokens - thinking_tokens

            # Finalise assistant entry; keep thinking for context inclusion.
            live_entry["content"]  = text
            live_entry["thinking"] = reasoning
            live_entry["tokens"]   = completion_tokens

            prev_prompt_tokens     = prompt_tokens
            prev_completion_tokens = completion_tokens

            sys.stdout.write(
                "\033[90m  User: %d tokens │ Thinking: ~%d tokens │ Content: ~%d tokens │ Context: %d / %d\033[0m\n"
                % (user_tokens, thinking_tokens, content_tokens, prompt_tokens, CONTEXT_WINDOW + MAX_TOKENS)
            )
        else:
            # No response — remove the placeholder pair.
            history.pop()
            history.pop()

        write_history(history_file, history)


if __name__ == "__main__":
    main()
