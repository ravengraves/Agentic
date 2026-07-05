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
import os
import time
import json
import threading
import re
import importlib

from .lib import agentic_client as _client
from .lib import agentic_compaction as _compaction
from .lib import agentic_markdown as _markdown
from .lib import agentic_project as _project
from .lib import agentic_tools as _tools

for _module in (_markdown, _client, _compaction, _project):
    importlib.reload(_module)

TOOLS = _tools.TOOLS
execute_tool = _tools.execute_tool
stream_chat = _client.stream_chat
COMPACT_SYSTEM = _compaction.COMPACT_SYSTEM
SUMMARY_PREFIX = _compaction.SUMMARY_PREFIX
SUMMARY_SUFFIX = _compaction.SUMMARY_SUFFIX
build_summary_prompt = _compaction.build_summary_prompt
compact_history = _compaction.compact_history
copy_compaction_settings = _compaction.copy_compaction_settings
clip_tool_outputs = _compaction.clip_tool_outputs
split_history = _compaction.split_history
MarkdownContentEscaper = _markdown.MarkdownContentEscaper
assistant_tool_message = _markdown.assistant_tool_message
escape_content = _markdown.escape_content
format_chat = _markdown.format_chat
format_tool_block = _markdown.format_tool_block
parse_chat = _markdown.parse_chat
tool_result_message = _markdown.tool_result_message
apply_project_agents_md = _project.apply_agents_md
project_folder_for_path = _project.project_folder_for_path

import sublime
import sublime_plugin

CHARS_PER_TOKEN = 4.4  # Based on python+english data

# Registry of cancellable background tasks: view.id() -> task
_ACTIVE_STREAMERS = {}

_LAST_SANITIZE_DICT_RAW = None
_SANITIZE_DICT = None
_SANITIZE_RE = None
_LAST_MODEL_IDX = {}

def _unregister_task(registry, view, task):
    if registry.get(view.id()) is task:
        registry.pop(view.id(), None)

def _printstatus(msg):
    sublime.status_message(msg)
    if sublime.load_settings("Agentic.sublime-settings").get("console_log", False):
        print(msg)

def _project_folder_for_view(view):
    window = view.window() if view else None
    if not window:
        return None
    return project_folder_for_path(view.file_name(), window.folders())

def _apply_agents_md(messages, view):
    settings = sublime.load_settings("Agentic.sublime-settings")
    return apply_project_agents_md(
        messages,
        _project_folder_for_view(view),
        enabled=settings.get("enable_agents_md", True))

def _pick_model(capability=None):
    """Return a random model from the provided model list string"""
    settings = sublime.load_settings("Agentic.sublime-settings")
    if capability is None:
        capability = settings.get("default_models")
    capable_models = settings.get(capability)

    # round robin request load balancing
    if capability not in _LAST_MODEL_IDX:
        idx = 0
    else:
        idx = _LAST_MODEL_IDX[capability]
    model = capable_models[idx % len(capable_models)]
    _LAST_MODEL_IDX[capability] = (idx + 1) % len(capable_models)

    return model


def _pick_compact_model(models, fallback_name=None):
    settings = sublime.load_settings("Agentic.sublime-settings")
    capability = settings.get("default_models_compact", "models_low")
    model_name = _pick_model(capability) if settings.get(capability) else None
    if model_name not in models:
        model_name = fallback_name
    if model_name not in models:
        model_name = _pick_model()
    return model_name, copy_compaction_settings(models.get(model_name, {}), settings)


def chat_stream(messages, model, cancel=None, tools=None):
    """
    Query an OpenAI server given messages and a model configuration.
    Yields `(is_reasoning, text)` for incremental stream chunks.
    Yields `("__tool_calls__", tool_calls_list)` if tool calls are detected.
    At the end yields a 4-tuple of timing metrics:
        (cache_n, prompt_n, prompt_per_second, predicted_per_second).
    """
    include_reasoning = None
    options = model.get("options", {})
    if options.get("include_reasoning"):
        include_reasoning = sublime.load_settings(
            "Agentic.sublime-settings").get("show_reasoning")
    return stream_chat(
        messages,
        model,
        cancel=cancel,
        tools=tools,
        include_reasoning=include_reasoning,
        user_agent="AsyncAPIClient/Python 2.14.0")


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

        # Inject project AGENTS.md into the developer/system prompt once.
        _apply_agents_md(self.messages, self.view)

        # Detect project root for relative path resolution
        cwd = _project_folder_for_view(self.view)

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
                streamed_reasoning = ""
                content_escaper = MarkdownContentEscaper()
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
                        streamed_reasoning += text
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
                        text = content_escaper.feed(text)
                    self._write(text)

                pending_content = content_escaper.flush()
                if pending_content:
                    self._write(pending_content)

                if self._cancel_event.is_set() or not tool_calls:
                    break

                # Insert assistant message with tool_calls (required by OpenAI API protocol)
                self.messages.append(assistant_tool_message(
                    tool_calls,
                    content=streamed_content.strip(),
                    reasoning=streamed_reasoning.strip(),
                ))

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

                    tool_msg = tool_result_message(tc, args, output)
                    self._write("\n")
                    self._write(format_tool_block(
                        tool_msg.get("tool_call_id", ""),
                        tool_msg.get("name", ""),
                        tool_msg.get("arguments", "{}"),
                        tool_msg.get("content", ""),
                    ))
                    self._write("\n")

                    self.messages.append(tool_msg)

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
                compact_model_name, compact_model = _pick_compact_model(
                    models, model_name)
                _printstatus("Compacting context with {} ({:.0f}% used)...".format(
                    compact_model_name, float(est_tokens) / context_limit * 100))
                self.messages = _compact_history(self.messages, compact_model)

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
        _unregister_task(self.registry, self.view, self)
        if not self.is_valid():
            return
        self._write("\n\n# --- User ---\n")
        self.view.settings().set("agentic_is_streaming", False)


class AgentCompactionTask(threading.Thread):
    """Background worker - compacts a chat without blocking Sublime."""
    operation = "Compaction"

    def __init__(self, source_view, target_view, head, middle, tail,
                 model_name, model, behavior, registry, stream_to_view):
        super().__init__(daemon=True)
        self.source_view = source_view
        self.view = target_view
        self.head = head
        self.middle = middle
        self.tail = tail
        self.model_name = model_name
        self.model = model
        self.behavior = behavior
        self.registry = registry
        self.stream_to_view = stream_to_view
        self._cancel_event = threading.Event()
        self._buffer = []
        self._pending = False
        self._closed = False

    def cancel(self):
        self._cancel_event.set()
        self.view.settings().set("agentic_is_streaming", False)

    def is_valid(self):
        return self.view and self.view.is_valid() \
                and self.view.window() and self.view.window().is_valid()

    def run(self):
        summary_text = ""
        content_escaper = MarkdownContentEscaper()
        summary_request = [
            {"role": "developer", "content": COMPACT_SYSTEM},
            {"role": "user", "content": build_summary_prompt(self.middle, self.model)},
        ]
        try:
            for chunk in chat_stream(
                    summary_request, self.model, self._cancel_event, tools=None):
                if self._cancel_event.is_set():
                    break
                if isinstance(chunk, tuple) and len(chunk) == 2:
                    is_reasoning, text = chunk
                    if is_reasoning:
                        continue
                    summary_text += text
                    if self.stream_to_view:
                        self._write(content_escaper.feed(text))
        except Exception as e:
            sublime.set_timeout(lambda: self._finish(None, str(e)), 0)
            return

        if self.stream_to_view:
            pending = content_escaper.flush()
            if pending:
                self._write(pending)

        compacted = _compacted_messages(
            self.head, summary_text, self.tail, self.model)
        sublime.set_timeout(lambda: self._finish(compacted, None), 0)

    def _write(self, text):
        if not text or self._closed:
            return
        self._buffer.append(text)
        if not self._pending:
            self._pending = True
            sublime.set_timeout(self.__flush, 25)

    def __flush(self):
        self._pending = False
        if self._closed or not self.is_valid():
            self._buffer = []
            return
        parts = []
        while self._buffer:
            parts.append(self._buffer.pop())
        if parts:
            self.view.run_command("append", {"characters": "".join(parts[::-1])})

    def _finish(self, compacted, error):
        self._closed = True
        self._buffer = []
        if self.is_valid():
            self.view.settings().set("agentic_is_streaming", False)
        _unregister_task(self.registry, self.view, self)

        if self._cancel_event.is_set():
            _printstatus("Compaction cancelled.")
            return
        if not self.is_valid():
            return
        if error:
            if self.stream_to_view and self.is_valid():
                self.view.run_command(
                    "append",
                    {"characters": "\n\n[Compaction failed: {}]\n".format(error)})
            _printstatus("Chat compaction failed: {}".format(error))
            return
        if not compacted:
            _printstatus("Chat compaction did not change context")
            return

        continuation = _rebuild_text(compacted) + "\n\n# --- User ---\n"
        if self.behavior == "replace":
            self.source_view.run_command("select_all")
            self.source_view.run_command("insert", {"characters": continuation})
            _printstatus("Chat compacted")
            return

        self.view.run_command("select_all")
        self.view.run_command("insert", {"characters": continuation})
        self.view.set_name("Chat compacted")
        if self.model_name:
            self.view.settings().set("agent_model", self.model_name)
        _printstatus("Compacted chat created")


def start_streaming(view, messages, model_name=None):
    """Public helper - start a streaming task on a view"""
    view.settings().set("agentic_is_streaming", True)
    if model_name:
        view.settings().set("agent_model", model_name)
    task = AgentStreamingTask(view, messages, _ACTIVE_STREAMERS)
    _ACTIVE_STREAMERS[view.id()] = task
    task.start()


def _build_messages_from_text(text):
    """Parse a view into a list of chat messages."""
    return parse_chat(text)


def _rebuild_text(messages, strip_active=False):
    """Reconstruct chat text from a message list."""
    return format_chat(messages, strip_active=strip_active)


def _compacted_messages(head, summary_text, tail, model):
    summary_text = summary_text or ""
    start = summary_text.find("<state_snapshot>")
    end = summary_text.find("</state_snapshot>")
    if start != -1 and end != -1:
        summary_text = summary_text[start:end + len("</state_snapshot>")].strip()
    else:
        summary_text = summary_text[:4000].strip()
    if not summary_text:
        return None
    result = clip_tool_outputs(head, model)
    result.append({
        "role": "user",
        "content": SUMMARY_PREFIX + "\n" + summary_text + "\n" + SUMMARY_SUFFIX,
    })
    result.extend(clip_tool_outputs(tail, model))
    return result


def _compact_history(messages, model, cancel=None):
    """LLM-based compaction: keep head/tail, summarize middle."""
    settings = sublime.load_settings("Agentic.sublime-settings")
    compact_model = copy_compaction_settings(model, settings)

    def summarize(system_prompt, user_prompt):
        summary_text = ""
        summary_request = [
            {"role": "developer", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        try:
            for chunk in chat_stream(
                    summary_request, compact_model, cancel=cancel, tools=None):
                if cancel and cancel.is_set():
                    return ""
                if isinstance(chunk, tuple) and len(chunk) == 2:
                    is_reasoning, text = chunk
                    if not is_reasoning:
                        summary_text += text
        except Exception:
            return ""
        return summary_text

    return compact_history(messages, compact_model, summarize)


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


def _initial_chat_text(source_view, system_prompt, user_prompt):
    messages = [
        {"role": "developer", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    _apply_agents_md(messages, source_view)
    return format_chat(messages)


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
        new_chat = _initial_chat_text(
            old,
            sublime.load_settings("Agentic.sublime-settings").get("default_prompt"),
            user_prompt) + "\n"

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
            self.window.run_command("agentic_cancel_stream")
            return

        text = view.substr(sublime.Region(0, view.size()))
        messages = _build_messages_from_text(text)
        if not messages:
            sublime.status_message("Chat data not found")
            self.window.run_command("agent_new_chat")
            return
        if _apply_agents_md(messages, view):
            view.run_command("select_all")
            view.run_command("insert", {"characters": _rebuild_text(messages)})

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
        new_chat = _initial_chat_text(
            self.window.active_view(),
            sublime.load_settings("Agentic.sublime-settings").get("default_prompt"),
            "") + "\n"
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
            _printstatus("{} cancelled.".format(
                getattr(task, "operation", "Streaming")))
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
    """Compact chat history via LLM summarization."""
    def run(self):
        view = self.window.active_view()
        if not view:
            _printstatus("No active view to compact")
            return
        if view.settings().get("agentic_is_streaming"):
            self.window.run_command("agentic_cancel_stream")
            return

        text = view.substr(sublime.Region(0, view.size()))
        messages = _build_messages_from_text(text)
        if not messages:
            _printstatus("Chat data not found")
            return
        if len(messages) < 4:
            _printstatus("Chat compaction needs more history")
            return
        view.settings().set("agentic_is_chat", True)

        settings = sublime.load_settings("Agentic.sublime-settings")
        models = settings.get("models") or {}
        model_name = view.settings().get("agent_model")
        if not model_name or model_name not in models:
            model_name = _pick_model()
        compact_model_name, compact_model = _pick_compact_model(
            models, model_name)
        head, middle, tail = split_history(messages, compact_model)
        if not middle:
            _printstatus("Chat compaction did not change context")
            return

        behavior = settings.get("compact_chat_behavior", "new_chat")
        stream_to_view = behavior != "replace"
        target_view = view
        if stream_to_view:
            initial = _rebuild_text(clip_tool_outputs(head, compact_model))
            if initial:
                initial += "\n\n"
            initial += "# --- User ---\n" + escape_content(SUMMARY_PREFIX + "\n")
            target_view = _create_chat(
                self.window,
                "Chat compacting",
                initial,
                create_pane=False)
            target_view.settings().set("agent_model", model_name)

        target_view.settings().set("agentic_is_streaming", True)
        target_view.settings().set("agentic_is_chat", True)
        task = AgentCompactionTask(
            view,
            target_view,
            head,
            middle,
            tail,
            model_name,
            compact_model,
            behavior,
            _ACTIVE_STREAMERS,
            stream_to_view)
        _ACTIVE_STREAMERS[target_view.id()] = task
        _printstatus("Compaction starting with {}...".format(compact_model_name))
        task.start()


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

        new_chat = _initial_chat_text(old, system_prompt, user_prompt)

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
        self.models = sublime.load_settings("Agentic.sublime-settings").get("models")
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

        new_chat = _initial_chat_text(
            old,
            sublime.load_settings("Agentic.sublime-settings").get("default_prompt"),
            user_prompt)

        view = _create_chat(self.window, "Chat " + model_name[:12], new_chat)

        messages = _build_messages_from_text(new_chat)
        view.settings().set("agent_model", model_name)

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
            path, lang, escape_content(content))

        # Insert at each caret (default behaviour leaves cursor after the text)
        auto_indent = self.view.settings().get('auto_indent', False)
        self.view.settings().set('auto_indent', False)
        self.view.run_command('insert', {'characters': insertion})
        self.view.settings().set('auto_indent', auto_indent)
