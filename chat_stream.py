# Packages/Agentic/chat_stream.py
# -------------------------------------------------------------
#  Agentic AI plugin: four primary commands
#
#  * AI Agent        - Execute an arbitrary prompt providing a code
#                       snippet or file
#
#  * AI Agent Action - Execute user-defined custom agent action
#                       on a custom code section.
#
#  * AI Agent Model Chat  - Start a chat with a specific model,
#                            optionally with a code snippet
#
#  * AI Agent Chat   - Stream on an existing chat view that already
#                       contains the tags "# --- System ---",
#                       "# --- User ---", "# --- Agent ---".
#                       Run with [ctrl/cmd] + [enter] in a chat file.
#
#  Supplemental chat management commands
#
#  * AI Agent Clear Reasoning  - Erase reasoning output from a chat
#
#  * AI Agent Clone Chat  - Create a copy of the current chat
#
#  * AI Agent New Chat    - Prepare a new chat file
#
#  * AI Agent Set Model   - Set the model for a chat file
#
#  * AI Agent Insert File - Insert a file into the chat
#
#  All API call logic lives in `chat_stream()` - the single code
#  path used by all three commands.
#
#  The plugin uses only Python features available in older
#  Sublime Text builds (no f-string syntax, only .format()).
# -------------------------------------------------------------
import sys
import os
import time
import json
import urllib.request
import urllib.error
import threading
import random
import re

from .agentic_tools import TOOLS, execute_tool

import sublime
import sublime_plugin

CHARS_PER_TOKEN = 4.4  # Based on python+english data

# Registry of ongoing streams: view.id() -> StreamingTask
_ACTIVE_STREAMERS = {}

# Tags for chat file
TAG_MAP = {
    "developer": "# --- System ---",
    "user": "# --- User ---",
    "assistant": "# --- Agent ---",
    "tool": "# --- Tool ---",
}

_LAST_SANITIZE_DICT_RAW = None
_SANITIZE_DICT = None
_SANITIZE_RE = None
_LAST_MODEL_IDX = {}

def _printstatus(msg):
    sublime.status_message(msg)
    if sublime.load_settings("Agentic.sublime-settings").get("console_log", False):
        print(msg)

def _load_agents_md(window):
    """Load AGENTS.md from the first project folder, if enabled."""
    settings = sublime.load_settings("Agentic.sublime-settings")
    if not settings.get("enable_agents_md"):
        return None
    for folder in window.folders():
        path = os.path.join(folder, "AGENTS.md")
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    return f.read()
            except Exception:
                continue
    return None

def _pick_model(capability=None):
    """Return a random model from the provided model list string"""
    settings = sublime.load_settings("Agentic.sublime-settings")
    if capability is None:
        capability = settings.get("default_models")
    capable_models = settings.get(capability)
    models = settings.get("models")

    # round robin request load balancing
    if capability not in _LAST_MODEL_IDX:
        idx = 0
    else:
        idx = _LAST_MODEL_IDX[capability]
    model = capable_models[idx % len(capable_models)]
    _LAST_MODEL_IDX[capability] = (idx + 1) % len(capable_models)

    return model


def _parse_metrics(r):
    """
    Return (cache_n, prompt_n, prompt_per_sec, predicted_per_sec)
    from either a "timings" block or the older "usage" block.
    """
    t = r.get("timings")
    if t:
        return (t["cache_n"], t["prompt_n"],
                t["prompt_per_second"], t["predicted_per_second"])
    u = r.get("usage")  # groq / openai
    if u:
        cache = u.get("prompt_tokens_details", {}).get("cached_tokens", 0)
        prompt = u.get("prompt_tokens", 0) - cache
        pt = u.get("prompt_time", 0) or 1e12
        ct = u.get("completion_time", 0) or 1e12
        return (cache, prompt,
                u.get("prompt_tokens", 0) / pt,
                u.get("completion_tokens", 0) / ct)
    return None


def chat_stream(messages, model, cancel=None, tools=None):
    """
    Query an OpenAI server given messages and a model configuration.
    Yields `(is_reasoning, text)` for incremental stream chunks.
    Yields `("__tool_calls__", tool_calls_list)` if tool calls are detected.
    At the end yields a 4-tuple of timing metrics:
        (cache_n, prompt_n, prompt_per_second, predicted_per_second).
    """
    url = model.get("url")
    token = model.get("token")
    body = dict(model.get("options", {}))
    body.update({
        "messages": messages,
        "model": model.get("model"),
    })

    if tools:
        body["tools"] = tools

    # some local models (EXAONE) do not support "developer"
    if model.get("no_system_prompt", False):
        for m in body.get("messages"):
            if m['role'] == "developer":
                m['role'] = "user"

    if "include_reasoning" in body and body["include_reasoning"]:
        body.update({  # match the package setting if True
            "include_reasoning":
            sublime.load_settings("Agentic.sublime-settings").get("show_reasoning")
        })

    stream = body.get("stream", False)

    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer {}".format(token),
            "User-Agent": "AsyncAPIClient/Python 2.14.0"
        },
    )

    if not stream:
        try:
            resp = urllib.request.urlopen(req)
        except urllib.error.HTTPError as e:
            print("HTTP Error {}: {}".format(e.code, e.reason))
            raise e
        resp = json.loads(''.join([r.decode("utf-8") for r in resp]))
        m = resp["choices"][0]["message"]
        if "reasoning_content" in m and m["reasoning_content"]:
            yield (True, m["reasoning_content"])
        elif "reasoning" in m and m["reasoning"]:  # groq
            yield (True, m["reasoning"])
        if "content" in m:
            content = m["content"]
            start = content.find("<|tool_call>")
            if start != -1:
                end = content.find("<tool_call|>", start)
                if end != -1:
                    prefix = content[:start]
                    if prefix.strip():
                        yield (False, prefix)
                    inner = content[start + len("<|tool_call>"):end]
                    mm = re.search(r'call:\s*(?:\S+:)?([^{]+)\s*(\{[^}]*\})?', inner)
                    if mm:
                        name = mm.group(1).strip()
                        args = mm.group(2) if mm.group(2) else "{}"
                        try:
                            json.loads(args)
                            yield ("__tool_calls__", [{"id": "tc_0", "function": {"name": name, "arguments": args}}])
                            return
                        except (ValueError, Exception):
                            pass
            yield (False, content)

        # Tool calls in non-streaming response
        if m.get("tool_calls"):
            yield ("__tool_calls__", [
                {
                    "id": tc["id"],
                    "function": {
                        "name": tc["function"]["name"],
                        "arguments": tc["function"]["arguments"],
                    },
                }
                for tc in m["tool_calls"]
            ])
            return

        t = _parse_metrics(resp)
        if t:
            yield (t)
        return

    with urllib.request.urlopen(req) as resp:
        tool_accums = {}  # index -> {id, function: {name, arguments}}
        _content_buf = ""
        for raw in resp:
            if cancel and cancel.is_set():
                return
            line = raw.decode("utf-8").strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                return

            try:
                evt = json.loads(payload)
            except ValueError:
                continue

            for choice in evt.get("choices", []):
                delta = choice.get("delta", {})

                # Accumulate streaming tool_call fragments by index
                if "tool_calls" in delta:
                    for tc in delta["tool_calls"]:
                        idx = tc["index"]
                        if idx not in tool_accums:
                            tool_accums[idx] = {
                                "id": "",
                                "function": {"name": "", "arguments": ""},
                            }
                        acc = tool_accums[idx]
                        if "id" in tc:
                            acc["id"] = tc["id"]
                        if "function" in tc:
                            if "name" in tc["function"]:
                                acc["function"]["name"] = tc["function"]["name"]
                            if "arguments" in tc["function"]:
                                acc["function"]["arguments"] += tc["function"]["arguments"]
                    continue

                # Harvest content / reasoning tokens
                if "reasoning_content" in delta and delta["reasoning_content"]:
                    yield (True, delta["reasoning_content"])
                elif "reasoning" in delta and delta["reasoning"]:  # groq
                    yield (True, delta["reasoning"])
                elif "content" in delta and delta["content"]:
                    _content_buf += delta["content"]
                    start = _content_buf.find("<|tool_call>")
                    if start == -1:
                        yield (False, _content_buf)
                        _content_buf = ""
                        continue
                    end = _content_buf.find("<tool_call|>", start)
                    if end == -1:
                        continue
                    prefix = _content_buf[:start]
                    if prefix.strip():
                        yield (False, prefix)
                    inner = _content_buf[start + len("<|tool_call>"):end]
                    m = re.search(r'call:\s*(?:\S+:)?([^{]+)\s*(\{[^}]*\})?', inner)
                    if m:
                        name = m.group(1).strip()
                        args = m.group(2) if m.group(2) else "{}"
                        try:
                            json.loads(args)
                            yield ("__tool_calls__", [{"id": "tc_0", "function": {"name": name, "arguments": args}}])
                        except (ValueError, Exception):
                            yield (False, _content_buf[:end + len("<tool_call|>")])
                    _content_buf = _content_buf[end + len("<tool_call|>"):]
                elif not delta:  # Final timing report or tool_calls finish
                    if choice.get("finish_reason") == "tool_calls" and tool_accums:
                        yield ("__tool_calls__", list(tool_accums.values()))
                        return
                    t = _parse_metrics(evt)
                    if t:
                        yield (t)


class AgentStreamingTask(threading.Thread):
    """Background worker - streams into the view"""
    def __init__(self, view, messages, registry):
        super().__init__(daemon=True)
        self.view = view
        self.messages = messages
        self.registry = registry
        self._cancel_event = threading.Event()
        self.sanitize = sublime.load_settings(
            "Agentic.sublime-settings").get("sanitize_output")
        if self.sanitize:
            _update_sanitize_dict()
        self.show_reasoning = sublime.load_settings(
            "Agentic.sublime-settings").get("show_reasoning")
        self._buffer = []       # pending writes
        self._pending = False   # a flush is already scheduled?
        self.start_time = time.time()

    def cancel(self):
        self._cancel_event.set()
        self.view.settings().set("agentic_is_streaming", False)

    def run(self):
        prev_reasoning = False
        cache, tps = (0.0, None)  # default empty performance

        # Inject AGENTS.md from project folder into system prompt (once)
        window = self.view.window()
        agents_md = _load_agents_md(window) if window else None
        if agents_md and self.messages:
            sys_msg = self.messages[0].get("content", "")
            if "AGENTS.md instructions" not in sys_msg:
                self.messages[0]["content"] = (
                    sys_msg
                    + "\n\n"
                    + "Project AGENTS.md instructions:\n\n"
                    + "`````````\n"
                    + agents_md
                    + "\n`````````"
                )

        # Detect project root for relative path resolution
        cwd = None
        if window:
            folders = window.folders()
            if folders:
                cwd = folders[0]

        if self.view.settings().get("agent_model") is None:
            self.view.settings().set("agent_model", _pick_model())

        # Load latest model settings
        models = sublime.load_settings("Agentic.sublime-settings").get("models")
        model_name = self.view.settings().get("agent_model")
        model = models[model_name]

        input_tokens = int(sum([len(m.get("content") or "") for m in self.messages]) / CHARS_PER_TOKEN)
        fraction_used = input_tokens / model.get("context", float("NaN"))
        _printstatus("Streaming {}. Context: {} ({:.0f}%).".format(
            model_name, input_tokens, fraction_used * 100))

        # Determine which tools to send (opt-in per model)
        tooling = TOOLS if model.get("tool_calling") else None
        max_rounds = 10

        try:
            for _round in range(max_rounds):
                tool_calls = None
                prev_reasoning = False
                streamed_content = ""
                self._write("\n\n# --- Agent ---\n")
                for chunk in chat_stream(self.messages, model, self._cancel_event, tools=tooling):
                    if isinstance(chunk, tuple) and chunk[0] == "__tool_calls__":
                        tool_calls = chunk[1]
                        break

                    # Normal (2-tuple) chunk vs. metrics (4-tuple)
                    if len(chunk) == 2:
                        is_reasoning, text = chunk
                    else:
                        cache, prompt, pps, tps = chunk
                        break

                    if self._cancel_event.is_set() or not self.is_valid():
                        self._cancel_event.set()
                        _printstatus("Interrupted {}".format(model_name))
                        break

                    if is_reasoning:
                        if not self.show_reasoning:
                            continue
                        if not prev_reasoning:
                            self._write("\n## --- Thinking ---\n>")
                            prev_reasoning = True
                        text = text.replace("\n", "\n> ")
                    else:
                        if prev_reasoning:
                            self._write("\n\n## --- Response ---\n")
                            prev_reasoning = False
                        streamed_content += text
                    self._write(text)

                if self._cancel_event.is_set() or not tool_calls:
                    break

                # Insert assistant message with tool_calls (required by OpenAI API protocol)
                self.messages.append({
                    "role": "assistant",
                    "content": streamed_content.strip() or "",
                    "tool_calls": [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["function"]["name"],
                                "arguments": tc["function"]["arguments"],
                            },
                        }
                        for tc in tool_calls
                    ],
                })

                # Execute accumulated tool calls
                for tc in tool_calls:
                    if self._cancel_event.is_set():
                        break
                    name = tc["function"]["name"]
                    try:
                        args = json.loads(tc["function"]["arguments"]) if isinstance(
                            tc["function"]["arguments"], str
                        ) else (tc["function"]["arguments"] or {})
                    except Exception:
                        args = tc["function"]["arguments"]

                    output, is_error = execute_tool(
                        name, args,
                        sanitize_fn=_sanitize_text if self.sanitize else None,
                        cancel=self._cancel_event,
                        cwd=cwd,
                        session_id=str(self.view.id()))

                    args_str = json.dumps(args) if isinstance(args, dict) else str(args)
                    self._write("\n# --- Tool ---\ntool_call_id: {}\nname: {}\nargs: {}\n".format(
                        tc["id"], name, args_str))
                    self._write(output)
                    self._write("\n")

                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "name": name,
                        "arguments": args_str,
                        "content": output,
                    })

        except Exception as e:
            error_msg = "Streaming Error: {}".format(str(e))
            error_details = ""
            try:
                error_details = "\n" + e.read().decode('utf-8')
            except Exception:
                pass
            print(
                "Could not stream from {}:\n"
                "  model: {}\n  url: {}\n  options: {}\nError: {}{}".format(
                    model_name,
                    model.get("model", '"model" field missing"'),
                    model.get("url", '"url" field missing'),
                    model.get("options", {}),
                    str(e),
                    error_details))
            sublime.status_message(error_msg)
            self._write("\n\n## --- Error ---\n{}\n".format(str(e)))
            self.messages.append({
                "role": "assistant",
                "content": error_msg,
            })
            sublime.set_timeout(self._finalize, 0)
            return

        # Auto-compact if context exceeds 70% of model limit
        if self.messages and model.get("auto_compact"):
            est_tokens = sum(len(m.get("content") or "") for m in self.messages) // 4
            context_limit = model.get("context", 128000)
            if est_tokens > context_limit * 0.7:
                _printstatus("Compacting context ({:.0f}% used)...".format(
                    float(est_tokens) / context_limit * 100))
                self.messages = _compact_history(self.messages, model)

        ## Estimate and print usage based on elapsed time
        if not tps:
            tps = model.get("speed", float("NaN"))

        # Estimate generated tokens and measure input.
        gen_estimate = (time.time() - self.start_time) * tps

        # Compute total context usage and percentage of model's context
        used_context = input_tokens + gen_estimate
        fraction_used = used_context / model.get("context", float("NaN"))

        # Calculate cost based on inputs
        token_cost = model.get("cost", float("NaN")) * 1e-6
        cost = cache * 0.02 * token_cost \
                + (input_tokens - cache) * 0.2 * token_cost \
                + gen_estimate * token_cost

        # Log status
        _printstatus("Streaming Done. {}. Tk/s: {}. Context: {} ({:.0f}%). Cost: {:.3}".format(
            model_name, int(tps), int(used_context), fraction_used*100, cost))
        sublime.set_timeout(self._finalize, 0)

    def _write(self, txt):
        self._buffer.append(txt)            # atomic under the GIL
        time = 25 if len(self._buffer) < 96 else 0
        if not self._pending:               # schedule a debounced flush
            self._pending = True
            sublime.set_timeout(self.__flush, time)  # balance responsiveness and power

    def is_valid(self):
        return self.view and self.view.is_valid() \
                and self.view.window() and self.view.window().is_valid()

    def __flush(self):
        """Do not run outside of sublime.set_timeout"""
        self._pending = False
        if not self.is_valid():
            self._buffer.clear()
            return

        parts = []
        while self._buffer:                 # pop everything (O(1) each)
            parts.append(self._buffer.pop())
        if not parts:
            return

        out_string = "".join(parts[::-1])
        if self.sanitize:
            out_string = _sanitize_text(out_string)

        self.view.run_command(
            "append",
            {"characters": out_string}  # restore original order
        )

    def _finalize(self):
        if not self.is_valid():
            return
        self.registry.pop(self.view.id(), None)
        self._write("\n\n# --- User ---\n")
        self.view.settings().set("agentic_is_streaming", False)


def start_streaming(view, messages, model_name=None):
    """Public helper - start a streaming task on a view"""
    view.settings().set("agentic_is_streaming", True)
    if model_name:
        view.settings().set("agent_model", model_name)
    task = AgentStreamingTask(view, messages, _ACTIVE_STREAMERS)
    _ACTIVE_STREAMERS[view.id()] = task
    task.start()


def _build_messages_from_text(text):
    """Parse a view into a list of chat messages"""
    lines = text.splitlines(True)

    messages = []
    current_role = None
    content_lines = []
    in_reasoning = False

    for line in lines:
        stripped = line.strip()

        # Block start
        if stripped.startswith("# --- "):
            if current_role and content_lines:
                msg = "".join(content_lines).rstrip("\n")
                if current_role == "tool":
                    # Parse tool header lines: tool_call_id + name, rest is output
                    parts = msg.split("\n", 3)
                    tc_id = parts[0].replace("tool_call_id: ", "").strip() if len(parts) > 0 else ""
                    tname = parts[1].replace("name: ", "").strip() if len(parts) > 1 else ""
                    targs = parts[2].replace("args: ", "").strip() if len(parts) > 2 else ""
                    content = parts[3] if len(parts) > 3 else ""
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "name": tname,
                        "content": content,
                    })
                    # Inject tool_calls into the preceding assistant message
                    for i in range(len(messages) - 2, -1, -1):
                        if messages[i]["role"] == "assistant":
                            if "tool_calls" not in messages[i]:
                                messages[i]["tool_calls"] = []
                            messages[i]["tool_calls"].append({
                                "id": tc_id,
                                "type": "function",
                                "function": {
                                    "name": tname,
                                    "arguments": targs,
                                },
                            })
                            break
                else:
                    messages.append({"role": current_role, "content": msg})

            if "System" in stripped:
                current_role = "developer"
            elif "User" in stripped:
                current_role = "user"
            elif "Agent" in stripped:
                current_role = "assistant"
            elif "Tool" in stripped:
                current_role = "tool"
            else:
                current_role = None

            content_lines = []
            continue

        # Reasoning section toggles
        if stripped.startswith("## --- "):
            if "Thinking" in stripped:
                in_reasoning = True
            elif "Response" in stripped:
                in_reasoning = False
            continue

        if in_reasoning:
            continue

        if current_role:
            content_lines.append(line)

    # Finalise the last block
    if current_role and content_lines:
        msg = "".join(content_lines).rstrip("\n")
        if current_role == "tool":
            parts = msg.split("\n", 3)
            tc_id = parts[0].replace("tool_call_id: ", "").strip() if len(parts) > 0 else ""
            tname = parts[1].replace("name: ", "").strip() if len(parts) > 1 else ""
            targs = parts[2].replace("args: ", "").strip() if len(parts) > 2 else ""
            content = parts[3] if len(parts) > 3 else ""
            messages.append({"role": "tool", "tool_call_id": tc_id, "name": tname, "content": content})
            for i in range(len(messages) - 2, -1, -1):
                if messages[i]["role"] == "assistant":
                    if "tool_calls" not in messages[i]:
                        messages[i]["tool_calls"] = []
                    messages[i]["tool_calls"].append({
                        "id": tc_id, "type": "function",
                        "function": {"name": tname, "arguments": targs},
                    })
                    break
        else:
            messages.append({"role": current_role, "content": msg})

    return messages


def _rebuild_text(messages, strip_active=False):
    """Reconstructs chat text from message list"""
    if not messages:
        return ""
    parts = []
    for m in messages:
        tag = TAG_MAP.get(m["role"])
        if not tag:
            continue
        if m["role"] == "tool":
            header = "tool_call_id: {}\nname: {}\nargs: {}\n".format(
                m.get("tool_call_id", ""),
                m.get("name", ""),
                m.get("arguments", "{}"),
            )
            parts.append("{}\n{}{}\n".format(tag, header, m["content"].strip()))
        else:
            parts.append("{}\n{}\n".format(tag, m["content"].strip()))
    if strip_active and parts[-1].startswith(TAG_MAP["assistant"]) \
            and len(parts[-1]) < 19:
        parts = parts[:-1]
    return "\n".join(parts)[:-1]


COMPACT_SYSTEM = "\n".join([
    "You are a context summarization assistant. Task: summarize the conversation",
    "between a user and an AI coding agent into a structured snapshot that",
    "another LLM will use to continue the work. Do NOT continue the conversation.",
    "Do NOT respond to any questions in the conversation. ONLY output the",
    "structured summary below.",
    "",
    "First, reason privately in <scratchpad>. Then produce this exact format:",
    "",
    "<state_snapshot>",
    "  <overall_goal>",
    "    A single sentence describing the high-level objective.",
    "  </overall_goal>",
    "  <key_knowledge>",
    "    - Crucial facts, conventions, build commands, constraints.",
    "    - File paths that were read or modified.",
    "  </key_knowledge>",
    "  <file_system_state>",
    "    - Files read, created, modified with paths and status.",
    "  </file_system_state>",
    "  <recent_actions>",
    "    - Last few significant actions and their outcomes.",
    "  </recent_actions>",
    "  <current_plan>",
    "    Step-by-step plan. Mark: [DONE], [IN PROGRESS], [TODO].",
    "  </current_plan>",
    "</state_snapshot>",
])


def _compact_history(messages, model):
    """LLM-based compaction: preserve last ~30%, summarize older ~70%.
    Returns a new message list with the summary replacing old messages."""
    n = len(messages)
    keep_start = int(n * 0.7)
    if keep_start < 2 or keep_start >= n:
        return messages

    to_compact = list(messages[:keep_start])
    to_keep = list(messages[keep_start:])

    # Serialize old messages using the same markdown format
    compact_text = _rebuild_text(to_compact)
    summary_request = [
        {"role": "developer", "content": COMPACT_SYSTEM},
        {"role": "user", "content": "Summarize this conversation history:\n\n" + compact_text},
    ]

    # Call model non-streaming, no tools, to get the summary
    summary_text = ""
    try:
        for chunk in chat_stream(summary_request, model, tools=None):
            if isinstance(chunk, tuple) and len(chunk) == 2:
                is_reasoning, text = chunk
                if not is_reasoning:
                    summary_text += text
    except Exception:
        pass

    # Extract snapshot from response
    start = summary_text.find("<state_snapshot>")
    end = summary_text.find("</state_snapshot>")
    if start != -1 and end != -1:
        summary_text = summary_text[start:end + len("</state_snapshot>")]
    else:
        summary_text = summary_text[:2000]

    result = [{
        "role": "user",
        "content": "[CONTEXT SUMMARY]\n" + summary_text.strip() + "\n[/CONTEXT SUMMARY]",
    }]
    result.extend(to_keep)
    return result


def _create_chat(window, name, initial="",
                 temporary=True, create_pane=True, pane_dir="right"):
    view = window.new_file()
    if create_pane:
        window.set_layout({
            "cols":  [0.0, 0.5, 1.0],
            "rows":  [0.0, 1.0],
            "cells": [[0, 0, 1, 1],  # left group
                      [1, 0, 2, 1]]  # right group
        })
        right_group = window.num_groups() - 1
        window.set_view_index(view, right_group, 0)  # move file
        window.focus_group(right_group)
        # # Origami:
        # window.run_command("create_pane_with_file", {"direction": pane_dir})
    if temporary:
        view.set_scratch(True)
    view.set_name(name)
    view.settings().set("agentic_is_chat", True)
    view.set_syntax_file("Packages/Markdown/Markdown.sublime-syntax")
    if initial:
        view.run_command("append", {"characters": initial})
        sublime.set_timeout(
            lambda: view.run_command("move_to", {"to": "eof", "extend": False}),
            0)
    return view


def _read_selection(view):
    """Load user-selected data or file"""
    sel = view.sel()
    if any(not r.empty() for r in sel):
        content = "\n...\n".join(view.substr(r)
                                 for r in sel if not r.empty())
    else:  # load entire file:
        content = view.substr(sublime.Region(0, view.size()))
    return content


class PromptInputHandler(sublime_plugin.TextInputHandler):
    """Input handler - free-form prompt for AgenticCodeCommand"""
    def placeholder(self):
        return "Enter command string"

    def initial_text(self):
        return ""


class AgenticCodeCommand(sublime_plugin.WindowCommand):
    """Start a new chat based on highlighted text"""
    def input(self, args):
        return PromptInputHandler()

    def run(self, prompt):
        # New window + scratch view
        old = self.window.active_view()

        content = _read_selection(old)
        user_prompt = "File: {}\n```\n{}\n```\n{}".format(
            old.file_name(), content, prompt)
        new_chat = "# --- System ---\n{}\n\n# --- User ---\n{}\n".format(
            sublime.load_settings("Agentic.sublime-settings").get("default_prompt"),
            user_prompt
        )

        view = _create_chat(self.window, "Chat " + prompt[:12],
                            initial=new_chat)

        messages = _build_messages_from_text(new_chat)

        # Start agent
        if prompt:  # only launch automatically if user provided a command
            start_streaming(view, messages)
            sublime.status_message("Submitting prompt")


class AgenticChatCommand(sublime_plugin.WindowCommand):
    """Interact with an existing chat file"""
    def run(self):
        view = self.window.active_view()
        if not view:
            return

        if view.settings().get("agentic_is_streaming"):
            self.window.run_command("cancel_stream")
            return

        text = view.substr(sublime.Region(0, view.size()))
        messages = _build_messages_from_text(text)
        if not messages:
            sublime.status_message("Chat data not found")
            self.window.run_command("agent_new_chat")
            return

        view.settings().set("agentic_is_streaming", True)
        view.settings().set("agentic_is_chat", True)
        view.set_syntax_file("Packages/Markdown/Markdown.sublime-syntax")

        start_streaming(view, messages)
        sublime.status_message("Submitting prompt")


class AgenticNewChatCommand(sublime_plugin.WindowCommand):
    """Open a new chat window"""
    def run(self):
        if not self.window.active_view():
            return
        new_chat = "# --- System ---\n{}\n\n# --- User ---\n".format(
            sublime.load_settings("Agentic.sublime-settings").get("default_prompt")
        )
        view = _create_chat(self.window, "Chat", new_chat, create_pane=False)


class AgenticCloneChatCommand(sublime_plugin.WindowCommand):
    """Create a new chat from an existing one"""
    def run(self):
        view = self.window.active_view()
        if not view:
            return
        text = view.substr(sublime.Region(0, view.size()))
        messages = _build_messages_from_text(text)
        if not messages:
            sublime.status_message("Chat data not found")
            self.window.run_command("agent_new_chat")
            return
        cleaned = _rebuild_text(messages, strip_active=True)
        view = _create_chat(self.window, "Chat", cleaned, create_pane=False)


class AgenticCancelStreamCommand(sublime_plugin.WindowCommand):
    """Cancel an active stream command"""
    def run(self):
        view = self.window.active_view()
        if not view or not view.settings().get("agentic_is_streaming"):
            return
        task = _ACTIVE_STREAMERS.get(view.id())
        if task:
            task.cancel()
            _printstatus("Streaming cancelled.")
        else:
            view.settings().set("agentic_is_streaming", False)


class AgenticClearReasoningCommand(sublime_plugin.TextCommand):
    """Clear reasoning sections from a finished chat"""
    def run(self, edit):
        view = self.view
        if view.id() in _ACTIVE_STREAMERS:
            return
        text = view.substr(sublime.Region(0, view.size()))
        messages = _build_messages_from_text(text)
        if not messages:
            sublime.status_message("Chat data not found")
            return
        cleaned = _rebuild_text(messages)
        view.replace(edit, sublime.Region(0, view.size()), cleaned)
        sublime.status_message("Chat reasoning cleared")


class AgenticCompactCommand(sublime_plugin.WindowCommand):
    """Compact chat history via LLM summarization, overwriting the view."""
    def run(self):
        view = self.window.active_view()
        if not view or not view.settings().get("agentic_is_chat"):
            return
        if view.settings().get("agentic_is_streaming"):
            self.window.run_command("cancel_stream")

        text = view.substr(sublime.Region(0, view.size()))
        messages = _build_messages_from_text(text)
        if not messages or len(messages) < 4:
            return

        models = sublime.load_settings("Agentic.sublime-settings").get("models")
        model_name = view.settings().get("agent_model")
        model = models.get(model_name, {}) if model_name else _pick_model()
        if not model_name:
            model_name = _pick_model()
            model = models.get(model_name, {})

        compacted = _compact_history(messages, model)
        rebuilt = _rebuild_text(compacted)
        view.run_command("select_all")
        view.run_command("insert", {"characters": rebuilt + "\n\n# --- User ---\n"})
        _printstatus("Chat compacted")


class AgenticActionCommand(sublime_plugin.WindowCommand):
    """Run a user-defined action - see Agentic.sublime-settings"""
    def run(self):
        self.actions = self._load_actions()
        if not self.actions:
            sublime.error_message(
                "Agentic.sublime-settings contains no actions.\n"
                'Add a JSON array under the key `"actions"`.'
            )
            return

        self.window.show_quick_panel(
            list(self.actions.keys()),
            self.run_action  # callback called with the index chosen by the user
        )

    def run_action(self, index):
        """Called when the user picks an item (or cancels)."""
        if index == -1:  # user pressed Escape
            return
        action_name = list(self.actions.keys())[index]
        chosen = self.actions[action_name]
        settings = sublime.load_settings("Agentic.sublime-settings")
        models_list = chosen["models"]
        system_prompt = chosen["system"]
        prompt = chosen.get("prompt", "")

        old = self.window.active_view()
        sel = any(not r.empty() for r in old.sel())
        if prompt or sel:
            user_prompt = "File: {}\n```\n{}\n```\n\n{}".format(
                old.file_name(),  _read_selection(old), prompt)
        else:
            user_prompt = ""

        new_chat = "# --- System ---\n{}\n\n# --- User ---\n{}".format(
            system_prompt, user_prompt)

        view = _create_chat(self.window, "Chat " + action_name[:12], new_chat)

        if prompt:
            messages = _build_messages_from_text(new_chat)
            model = _pick_model(models_list)
            start_streaming(view, messages, model)
            sublime.status_message("Submitting prompt")

    def _load_actions(self):
        return sublime.load_settings("Agentic.sublime-settings").get("actions")


class AgenticModelChatCommand(sublime_plugin.WindowCommand):
    """Run a user-defined action - see Agentic.sublime-settings"""
    def run(self):
        self.models = self._load_models()
        if not self.models:
            sublime.error_message(
                "Agentic.sublime-settings contains no models.\n"
                'Add a JSON array under the key `"models"`.'
            )
            return

        self.window.show_quick_panel(
            list(self.models.keys()),
            self.model_chat
        )

    def model_chat(self, index):
        """Called when the user picks an item (or cancels)."""
        if index == -1:  # user pressed Escape
            return
        model_name = list(self.models.keys())[index]
        model = self.models[model_name]

        old = self.window.active_view()
        sel = old.sel()
        if any(not r.empty() for r in sel):  # load active selection
            content = "\n...\n".join(old.substr(r)
                                     for r in sel if not r.empty())
            user_prompt = "File: {}\n```\n{}\n```\n".format(
                old.file_name(), content)
        else:  # empty new chat
            user_prompt = ""

        new_chat = "# --- System ---\n{}\n\n# --- User ---\n{}".format(
            sublime.load_settings("Agentic.sublime-settings").get("default_prompt"),
            user_prompt)

        view = _create_chat(self.window, "Chat " + model_name[:12], new_chat)

        messages = _build_messages_from_text(new_chat)
        view.settings().set("agent_model", model_name)

    def _load_models(self):
        return sublime.load_settings("Agentic.sublime-settings").get("models")


class AgenticSetModelCommand(sublime_plugin.WindowCommand):
    """Pick a model from your settings and store it in the view."""
    def run(self):
        self._models = sublime.load_settings("Agentic.sublime-settings").get("models")
        if not self._models:
            sublime.error_message("Agentic.sublime-settings contains no models.")
            return

        self._keys = list(self._models.keys())
        self.window.show_quick_panel(self._keys, self._on_done)

    def _on_done(self, index):
        if index == -1:
            return

        model_name = self._keys[index]
        view = self.window.active_view()
        if view:
            view.settings().set("agent_model", model_name)
            sublime.status_message("Model set to %s" % model_name)


class AgenticViewCloseHandler(sublime_plugin.EventListener):
    """
    Close stream when tab (view) closes
    """
    def on_close(self, view):
        if view.settings().get("agentic_is_streaming"):
            task = _ACTIVE_STREAMERS.get(view.id())
            if task:
                task.cancel()
            else:
                view.settings().set("agentic_is_streaming", False)


def _update_sanitize_dict():
    global _LAST_SANITIZE_DICT_RAW
    global _SANITIZE_DICT
    global _SANITIZE_RE

    all_sanitize = sublime.load_settings("Agentic.sublime-settings").get("sanitize_dict")
    # {canonical: [look-alikes]}

    # Do nothing if dict is the same.
    if _LAST_SANITIZE_DICT_RAW is not None \
            and all_sanitize == _LAST_SANITIZE_DICT_RAW:
        return

    _SANITIZE_DICT = {}
    for _canonical, _alts in all_sanitize.items():
        for _ch in _alts:
            _SANITIZE_DICT[_ch] = _canonical

    _SANITIZE_RE = re.compile(
        "|".join(sorted(map(re.escape, _SANITIZE_DICT), key=len, reverse=True))
    )
    _LAST_SANITIZE_DICT_RAW = all_sanitize


def _sanitize_text(text: str) -> str:
    """
    Replace Unicode look-alike in *text* with its ASCII canonical value
    """
    # `sub` receives each match; we look up the canonical character
    # in the flat map we built above.
    return _SANITIZE_RE.sub(lambda m: _SANITIZE_DICT[m.group(0)], text)


class AgenticSanitizeCommand(sublime_plugin.TextCommand):
    """Sanitize the whole document or the selected text."""
    def run(self, edit):
        _update_sanitize_dict()
        view = self.view

        #  1.  If there is at least one non-empty selection - sanitize it.
        sel = view.sel()
        if any(not r.empty() for r in sel):
            # Replace each selected region independently.
            # Iterate from the end so that earlier offsets stay valid.
            for r in reversed(list(sel)):
                if r.empty():
                    continue
                selected_text = view.substr(r)
                sanitized = _sanitize_text(selected_text)
                view.replace(edit, r, sanitized)
            sublime.status_message("Selection sanitized")
            return

        #  2. No selection - sanitize the entire document.
        region = sublime.Region(0, view.size())
        text = view.substr(region)

        if not text:
            sublime.status_message("Document is empty - nothing to sanitize")
            return

        sanitized = _sanitize_text(text)
        view.replace(edit, region, sanitized)
        sublime.status_message("Document sanitized")


class AgenticCopyCommand(sublime_plugin.TextCommand):
    """Sanitizes AI outputs (removes extra unicode based on settings)"""
    def run(self, edit):
        do_clean = sublime.load_settings("Agentic.sublime-settings").get("sanitize_on_copy")

        # Standard copy if sanitize_on_copy is false
        if not do_clean:
            self.view.run_command("copy")
            return

        _update_sanitize_dict()
        view = self.view
        pieces = []
        for region in view.sel():
            if region.empty():
                raw = view.substr(view.line(region))
            else:
                raw = view.substr(region)
            pieces.append(raw)
        out = _sanitize_text("\n".join(pieces))
        sublime.set_clipboard(out)
        end = "" if len(out) == 1 else "s"
        sublime.status_message("Sanitized + Copied {} character{}".format(len(out), end))


class AgenticCutCommand(sublime_plugin.TextCommand):
    """Sanitizes AI outputs and cuts the selection (copy + delete)."""
    def run(self, edit):
        # Use the same setting that controls copy sanitization
        do_clean = sublime.load_settings("Agentic.sublime-settings").get("sanitize_on_copy")

        # Standard cut if sanitization is disabled
        if not do_clean:
            self.view.run_command("cut")
            return

        # Sanitize the selected text and put it on the clipboard
        _update_sanitize_dict()
        view = self.view
        pieces = []
        for region in view.sel():
            if region.empty():
                raw = view.substr(view.line(region))
            else:
                raw = view.substr(region)
            pieces.append(raw)
        out = _sanitize_text("\n".join(pieces))
        sublime.set_clipboard(out)

        # Delete the original selection(s)
        delete_regions = []
        for region in view.sel():
            delete_regions.append(view.line(region) if region.empty() else region)

        # Delete from the end to the start so offsets don't shift
        for region in sorted(delete_regions, key=lambda r: r.begin(), reverse=True):
            view.erase(edit, region)

        # Status feedback
        end = "" if len(out) == 1 else "s"
        sublime.status_message("Sanitized + Cut {} character{}".format(len(out), end))


class AgenticInsertFileCommand(sublime_plugin.WindowCommand):
    """
    Insert a file's contents (chosen from the current project) at the
    current caret position as a new user block.
    """
    def run(self):
        self.view = self.window.active_view()
        if not self.view or self.view.settings().get("agentic_is_streaming"):
            return

        # Gather visible files from all project folders
        self._paths, self._names = [], []
        for folder in self.window.folders():
            for root, _, files in os.walk(folder):
                for f in files:
                    full = os.path.join(root, f)
                    rel = os.path.relpath(full, folder)
                    if rel.startswith('.git'):
                        continue
                    self._paths.append(full)
                    self._names.append(rel)

        self._names, self._paths = zip(
            *sorted(zip(self._names, self._paths), key=lambda x: x[1])
        )

        if not self._paths:
            return

        # Let the user pick a file
        self.window.show_quick_panel(self._names, self._on_done)

    def _on_done(self, idx):
        if idx < 0:
            return  # user cancelled

        path = self._paths[idx]
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                content = fh.read()
        except Exception:
            sublime.error_message("Could not read file: {}".format(path))
            return

        # Language hint for markdown fence (extension without dot)
        lang = os.path.splitext(path)[1].lstrip('.')
        insertion = "\nFile: {}\n```{}\n{}\n```\n".format(
            path, lang, content)

        # Insert at each caret (default behaviour leaves cursor after the text)
        auto_indent = self.view.settings().get('auto_indent', False)
        self.view.settings().set('auto_indent', False)
        self.view.run_command('insert', {'characters': insertion})
        self.view.settings().set('auto_indent', auto_indent)

