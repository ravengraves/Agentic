# agentic_client.py
# -------------------------------------------------------------
#  Dependency-free OpenAI-compatible chat completions streaming.
#  Shared by the Sublime plugin and the standalone CLI.
#
#  Minimal Python 3.8+ (no f-strings, no type-heavy interfaces).
# -------------------------------------------------------------
import json
import re
import urllib.request
import urllib.error


TOOL_CALL_EVENT = "__tool_calls__"


def parse_metrics(resp):
    """
    Return (cache_n, prompt_n, prompt_per_sec, predicted_per_sec)
    from either llama.cpp "timings" or OpenAI-style "usage".
    """
    t = resp.get("timings")
    if t:
        return (
            t.get("cache_n", 0),
            t.get("prompt_n", 0),
            t.get("prompt_per_second", 0),
            t.get("predicted_per_second", 0),
        )

    u = resp.get("usage")
    if u:
        cache = u.get("prompt_tokens_details", {}).get("cached_tokens", 0)
        prompt = u.get("prompt_tokens", 0) - cache
        pt = u.get("prompt_time", 0) or 1e12
        ct = u.get("completion_time", 0) or 1e12
        return (
            cache,
            prompt,
            u.get("prompt_tokens", 0) / pt,
            u.get("completion_tokens", 0) / ct,
        )
    return None


def _copy_messages(messages, no_system_prompt):
    """Copy messages before provider compatibility rewrites."""
    copied = []
    for msg in messages:
        m = dict(msg)
        if no_system_prompt and m.get("role") == "developer":
            m["role"] = "user"
        copied.append(m)
    return copied


def _request_body(messages, model, tools=None, include_reasoning=None):
    body = dict(model.get("options", {}))
    body.update({
        "messages": _copy_messages(messages, model.get("no_system_prompt", False)),
        "model": model.get("model"),
    })
    if tools:
        body["tools"] = tools
    if include_reasoning is not None and body.get("include_reasoning"):
        body["include_reasoning"] = include_reasoning
    return body


def _request(model, body, user_agent=None):
    return urllib.request.Request(
        model.get("url"),
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer {}".format(model.get("token", "")),
            "User-Agent": user_agent or "Agentic/Python",
        },
    )


def normalize_tool_calls(tool_calls):
    """Return the minimal OpenAI tool-call shape used by callers."""
    out = []
    for i, tc in enumerate(tool_calls or []):
        tc = tc if isinstance(tc, dict) else {}
        fn = tc.get("function", {})
        fn = fn if isinstance(fn, dict) else {}
        arguments = fn.get("arguments", "") or "{}"
        if not isinstance(arguments, str):
            try:
                arguments = json.dumps(arguments, separators=(",", ":"))
            except Exception:
                arguments = str(arguments)
        out.append({
            "id": tc.get("id") or "tc_{}".format(i),
            "function": {
                "name": fn.get("name", "") or "",
                "arguments": arguments,
            },
        })
    return out


def _parse_pseudo_tool_call(inner, call_id="tc_0"):
    """
    Parse gpt-oss-style text tool calls:
        <|tool_call>call: read {"file_path": "/tmp/a"}<tool_call|>
    Uses JSONDecoder so braces inside strings and nested objects work.
    """
    text = inner.strip()
    m = re.search(r"call:\s*", text)
    if m:
        text = text[m.end():].strip()

    brace = text.find("{")
    if brace == -1:
        name = text.strip()
        args_text = "{}"
    else:
        name = text[:brace].strip()
        raw_args = text[brace:].lstrip()
        try:
            args, _end = json.JSONDecoder().raw_decode(raw_args)
        except (TypeError, ValueError):
            return None
        args_text = json.dumps(args, separators=(",", ":"))

    # Existing parser allowed an optional non-space prefix ending in ":".
    if ":" in name:
        name = name.rsplit(":", 1)[-1].strip()
    name = name.strip()
    if not name:
        return None

    return {
        "id": call_id,
        "function": {
            "name": name,
            "arguments": args_text,
        },
    }


def _consume_pseudo_tool(content, call_id="tc_0"):
    start = content.find("<|tool_call>")
    if start == -1:
        return None
    end = content.find("<tool_call|>", start)
    if end == -1:
        return {"pending": True}

    prefix = content[:start]
    inner = content[start + len("<|tool_call>"):end]
    suffix = content[end + len("<tool_call|>"):]
    tc = _parse_pseudo_tool_call(inner, call_id=call_id)
    return {
        "pending": False,
        "prefix": prefix,
        "tool_call": tc,
        "raw": content[:end + len("<tool_call|>")],
        "suffix": suffix,
    }


def _accumulate_tool_calls(tool_accums, fragments):
    for tc in fragments or []:
        if not isinstance(tc, dict):
            continue
        idx = tc.get("index", 0)
        if idx not in tool_accums:
            tool_accums[idx] = {
                "id": "",
                "function": {"name": "", "arguments": ""},
            }
        acc = tool_accums[idx]
        if "id" in tc:
            acc["id"] = tc["id"]
        if "function" in tc:
            fn = tc["function"]
            if not isinstance(fn, dict):
                continue
            if "name" in fn:
                acc["function"]["name"] = fn["name"] or ""
            if "arguments" in fn:
                acc["function"]["arguments"] += str(fn["arguments"] or "")


def _ordered_tool_calls(tool_accums):
    calls = []
    for idx in sorted(tool_accums):
        tc = tool_accums[idx]
        if not tc.get("id"):
            tc["id"] = "tc_{}".format(idx)
        calls.append(tc)
    return normalize_tool_calls(calls)


def _yield_message(message):
    if message.get("reasoning_content"):
        yield (True, message["reasoning_content"])
    elif message.get("reasoning"):
        yield (True, message["reasoning"])

    content = message.get("content")
    if content:
        parsed = _consume_pseudo_tool(content)
        if parsed and not parsed.get("pending"):
            if parsed.get("prefix", "").strip():
                yield (False, parsed["prefix"])
            if parsed.get("tool_call"):
                yield (TOOL_CALL_EVENT, [parsed["tool_call"]])
                return
        yield (False, content)

    if message.get("tool_calls"):
        yield (TOOL_CALL_EVENT, normalize_tool_calls(message["tool_calls"]))


def stream_chat(messages, model, cancel=None, tools=None,
                include_reasoning=None, user_agent=None):
    """
    Query an OpenAI-compatible chat completion endpoint.

    Yields:
      (False, text) for response content
      (True, text) for reasoning content
      ("__tool_calls__", tool_calls) for tool calls
      (cache_n, prompt_n, prompt_per_sec, predicted_per_sec) for metrics
    """
    body = _request_body(messages, model, tools=tools,
                         include_reasoning=include_reasoning)
    req = _request(model, body, user_agent=user_agent)
    stream = body.get("stream", False)

    if not stream:
        try:
            resp = urllib.request.urlopen(req)
        except urllib.error.HTTPError:
            raise
        resp = json.loads("".join(r.decode("utf-8") for r in resp))
        message = resp["choices"][0]["message"]
        for event in _yield_message(message):
            yield event
            if isinstance(event, tuple) and event[0] == TOOL_CALL_EVENT:
                return
        metrics = parse_metrics(resp)
        if metrics:
            yield metrics
        return

    tool_accums = {}
    content_buf = ""
    with urllib.request.urlopen(req) as resp:
        for raw in resp:
            if cancel and cancel.is_set():
                return
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                if tool_accums:
                    yield (TOOL_CALL_EVENT, _ordered_tool_calls(tool_accums))
                elif content_buf:
                    yield (False, content_buf)
                return

            try:
                evt = json.loads(payload)
            except ValueError:
                continue

            for choice in evt.get("choices", []):
                delta = choice.get("delta", {})

                if "tool_calls" in delta:
                    _accumulate_tool_calls(tool_accums, delta["tool_calls"])
                    continue

                if choice.get("finish_reason") == "tool_calls" and tool_accums:
                    yield (TOOL_CALL_EVENT, _ordered_tool_calls(tool_accums))
                    return

                if delta.get("reasoning_content"):
                    yield (True, delta["reasoning_content"])
                elif delta.get("reasoning"):
                    yield (True, delta["reasoning"])
                elif delta.get("content"):
                    content_buf += delta["content"]
                    parsed = _consume_pseudo_tool(content_buf)
                    if not parsed:
                        yield (False, content_buf)
                        content_buf = ""
                    elif parsed.get("pending"):
                        continue
                    else:
                        if parsed.get("prefix", "").strip():
                            yield (False, parsed["prefix"])
                        if parsed.get("tool_call"):
                            yield (TOOL_CALL_EVENT, [parsed["tool_call"]])
                        else:
                            yield (False, parsed["raw"])
                        content_buf = parsed.get("suffix", "")
                elif not delta:
                    if choice.get("finish_reason") == "tool_calls" and tool_accums:
                        yield (TOOL_CALL_EVENT, _ordered_tool_calls(tool_accums))
                        return
                    metrics = parse_metrics(evt)
                    if metrics:
                        yield metrics
