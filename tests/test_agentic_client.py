import json
import importlib.machinery
import importlib.util
import contextlib
import io
import os
import sys
import tempfile
import types
import urllib.error
import unittest


PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_DIR = os.path.dirname(PACKAGE_DIR)
sys.path.insert(0, PACKAGE_DIR)
sys.path.insert(0, REPO_DIR)

from lib import agentic_client
from lib import agentic_compaction
from lib import agentic_markdown
from lib import agentic_project
from lib import agentic_tools


class FakeResponse(object):
    def __init__(self, lines):
        self._lines = lines

    def __iter__(self):
        return iter(self._lines)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def sse(obj):
    return ("data: " + json.dumps(obj) + "\n").encode("utf-8")


class InterruptingInput(object):
    def readline(self):
        raise KeyboardInterrupt()


class AgenticClientTests(unittest.TestCase):
    def setUp(self):
        self._old_urlopen = agentic_client.urllib.request.urlopen

    def tearDown(self):
        agentic_client.urllib.request.urlopen = self._old_urlopen

    def install_response(self, lines, capture=None):
        def fake_urlopen(req):
            if capture is not None:
                capture["body"] = json.loads(req.data.decode("utf-8"))
                capture["headers"] = dict(req.header_items())
            return FakeResponse(lines)
        agentic_client.urllib.request.urlopen = fake_urlopen

    def test_streaming_content(self):
        self.install_response([
            sse({"choices": [{"delta": {"content": "hi"}}]}),
            b"data: [DONE]\n",
        ])
        model = {
            "url": "http://example.test/v1/chat/completions",
            "model": "local",
            "options": {"stream": True},
        }
        events = list(agentic_client.stream_chat(
            [{"role": "user", "content": "hello"}], model))
        self.assertEqual(events, [(False, "hi")])

    def test_streaming_tool_call_fragments(self):
        self.install_response([
            sse({"choices": [{"delta": {"tool_calls": [{
                "index": 0,
                "id": "call_1",
                "function": {"name": "read", "arguments": "{\"file_"},
            }]}}]}),
            sse({"choices": [{"delta": {"tool_calls": [{
                "index": 0,
                "function": {"arguments": "path\":\"a.py\"}"},
            }]}}]}),
            b"data: [DONE]\n",
        ])
        model = {
            "url": "http://example.test/v1/chat/completions",
            "model": "local",
            "options": {"stream": True},
        }
        events = list(agentic_client.stream_chat(
            [{"role": "user", "content": "read"}], model))
        self.assertEqual(events[0][0], agentic_client.TOOL_CALL_EVENT)
        self.assertEqual(events[0][1][0]["id"], "call_1")
        self.assertEqual(events[0][1][0]["function"]["name"], "read")
        self.assertEqual(
            events[0][1][0]["function"]["arguments"],
            "{\"file_path\":\"a.py\"}")

    def test_pseudo_tool_call_handles_nested_json_and_braces(self):
        content = (
            "before <|tool_call>call: patch "
            "{\"file_path\":\"a.py\","
            "\"old_string\":\"x { y\","
            "\"new_string\":{\"nested\":true}}"
            "<tool_call|>"
        )
        response = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": content,
                }
            }],
            "usage": {"prompt_tokens": 4, "completion_tokens": 2},
        }
        self.install_response([json.dumps(response).encode("utf-8")])
        model = {
            "url": "http://example.test/v1/chat/completions",
            "model": "local",
            "options": {"stream": False},
        }
        events = list(agentic_client.stream_chat(
            [{"role": "user", "content": "patch"}], model))
        self.assertEqual(events[0], (False, "before "))
        self.assertEqual(events[1][0], agentic_client.TOOL_CALL_EVENT)
        args = json.loads(events[1][1][0]["function"]["arguments"])
        self.assertEqual(args["old_string"], "x { y")
        self.assertEqual(args["new_string"], {"nested": True})

    def test_no_system_prompt_does_not_mutate_input_messages(self):
        capture = {}
        response = {
            "choices": [{"message": {"role": "assistant", "content": "ok"}}],
        }
        self.install_response([json.dumps(response).encode("utf-8")], capture)
        messages = [{"role": "developer", "content": "rules"}]
        model = {
            "url": "http://example.test/v1/chat/completions",
            "model": "local",
            "options": {"stream": False},
            "no_system_prompt": True,
        }
        list(agentic_client.stream_chat(messages, model))
        self.assertEqual(messages[0]["role"], "developer")
        self.assertEqual(capture["body"]["messages"][0]["role"], "user")

    def test_normalize_tool_calls_handles_malformed_entries(self):
        calls = agentic_client.normalize_tool_calls([
            None,
            {"id": "call_1", "function": None},
            {"function": {"name": "read", "arguments": ""}},
            {"function": {"name": "patch", "arguments": {"file_path": "a.py"}}},
        ])
        self.assertEqual(calls[0]["id"], "tc_0")
        self.assertEqual(calls[0]["function"]["arguments"], "{}")
        self.assertEqual(calls[1]["id"], "call_1")
        self.assertEqual(calls[1]["function"]["name"], "")
        self.assertEqual(calls[2]["function"]["name"], "read")
        self.assertEqual(calls[2]["function"]["arguments"], "{}")
        self.assertEqual(calls[3]["function"]["arguments"], "{\"file_path\":\"a.py\"}")


class AgenticToolOutputTests(unittest.TestCase):
    def test_tool_output_clips_middle_by_default_lines(self):
        content = "".join("line {}\n".format(i) for i in range(400))
        clipped, did_clip = agentic_tools._clip_tool_output(content, {})
        self.assertTrue(did_clip)
        self.assertIn("line 0\n", clipped)
        self.assertIn("line 200\n", clipped)
        self.assertIn("line 301\n", clipped)
        self.assertIn("line 399\n", clipped)
        self.assertIn("tool output truncated", clipped)
        self.assertNotIn("line 250\n", clipped)

    def test_tool_output_flag_allows_more_lines(self):
        content = "".join("x{}\n".format(i) for i in range(400))
        by_lines, line_clipped = agentic_tools._clip_tool_output(
            content, {"max_output_lines": 500})
        self.assertFalse(line_clipped)
        self.assertEqual(by_lines, content)

    def test_tool_schema_exposes_output_controls(self):
        ls_tool = [t for t in agentic_tools.TOOLS
                   if t["function"]["name"] == "ls"][0]
        props = ls_tool["function"]["parameters"]["properties"]
        self.assertIn("max_output_lines", props)

    def test_clipped_read_does_not_count_as_full_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "large.txt")
            with open(path, "w", encoding="utf-8") as f:
                f.write("".join("line {}\n".format(i) for i in range(400)))
            output, is_error = agentic_tools.execute_tool(
                "read",
                {"file_path": path, "limit": 500, "max_output_lines": 20})
            ok, message = agentic_tools._check_prior_read(
                path, require_full=True)

        self.assertFalse(is_error)
        self.assertIn("tool output truncated", output)
        self.assertFalse(ok)
        self.assertIn("only partially read", message)

    def test_read_output_is_not_byte_clipped_under_line_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "wide.txt")
            with open(path, "w", encoding="utf-8") as f:
                f.write("".join("{}\n".format("x" * 1000) for _ in range(80)))
            output, is_error = agentic_tools.execute_tool(
                "read",
                {"file_path": path, "limit": 80, "max_output_lines": 100})
            ok, message = agentic_tools._check_prior_read(
                path, require_full=True)

        self.assertFalse(is_error)
        self.assertIn("80: ", output)
        self.assertNotIn("tool output truncated", output)
        self.assertTrue(ok, message)


class AgenticCliConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        loader = importlib.machinery.SourceFileLoader(
            "agentic_cli_for_tests",
            os.path.join(PACKAGE_DIR, "agentic_cli"))
        spec = importlib.util.spec_from_loader(loader.name, loader)
        module = importlib.util.module_from_spec(spec)
        loader.exec_module(module)
        cls.cli = module

    def run_cli_main(self, stdin, argv=None, run_agent=None):
        old_load_config = self.cli._load_config
        old_run_agent = self.cli._run_agent
        old_argv = sys.argv[:]
        old_cwd = os.getcwd()
        old_stdin = sys.stdin

        def fake_load_config():
            return {
                "default_prompt": "base",
                "enable_agents_md": False,
                "models": {
                    "local": {
                        "model": "local",
                        "url": "http://example.test",
                        "options": {"stream": True},
                        "context": 1000,
                    }
                },
            }

        try:
            with tempfile.TemporaryDirectory() as tmp:
                os.chdir(tmp)
                sys.argv = argv or ["agentic_cli"]
                sys.stdin = stdin
                self.cli._load_config = fake_load_config
                if run_agent:
                    self.cli._run_agent = run_agent
                out = io.StringIO()
                with contextlib.redirect_stdout(out):
                    rc = self.cli.main()
                return rc, out.getvalue()
        finally:
            os.chdir(old_cwd)
            sys.argv = old_argv
            sys.stdin = old_stdin
            self.cli._load_config = old_load_config
            self.cli._run_agent = old_run_agent

    def test_user_models_replace_default_models_with_key_fallback(self):
        base = {
            "default_prompt": "base prompt",
            "models": {"base-model": {"model": "base"}},
        }
        user = {
            "models": {"user-model": {"model": "user"}},
        }
        cfg = self.cli._overlay_config(base, user)
        self.assertEqual(cfg["default_prompt"], "base prompt")
        self.assertEqual(list(cfg["models"].keys()), ["user-model"])
        name, model = self.cli._select_model(cfg, "user-model")
        self.assertEqual(name, "user-model")
        self.assertEqual(model["model"], "user")

    def test_compact_model_defaults_to_low_model_list(self):
        cfg = {
            "default_models_compact": "models_low",
            "models_low": ["low-model"],
            "models": {
                "high-model": {"model": "high"},
                "low-model": {"model": "low"},
            },
        }
        model = self.cli._apply_config_to_model(
            dict(cfg["models"]["high-model"]), cfg)

        self.assertEqual(model["model"], "high")
        self.assertEqual(model["_compact_model"]["model"], "low")
        self.assertEqual(
            model["_compact_model"]["_compact_model_name"], "low-model")

    def test_run_agent_handles_http_errors_without_traceback(self):
        old_chat_stream = self.cli.chat_stream

        def fake_chat_stream(messages, model, cancel=None, tools=None):
            raise urllib.error.HTTPError(
                "http://example.test/v1/chat/completions",
                429,
                "Too Many Requests",
                {},
                io.BytesIO(b'{"error":{"message":"rate limited"}}'))
            yield

        try:
            self.cli.chat_stream = fake_chat_stream
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                messages, had_error = self.cli._run_agent(
                    [{"role": "user", "content": "hi"}],
                    {"options": {"stream": True}},
                    None)
        finally:
            self.cli.chat_stream = old_chat_stream

        self.assertTrue(had_error)
        self.assertIn("HTTP 429 Too Many Requests: rate limited", out.getvalue())
        self.assertIn("Error: HTTP 429 Too Many Requests", messages[-1]["content"])

    def test_compact_slash_prints_full_new_context(self):
        old_compact_history = self.cli._compact_history
        compacted = [
            {"role": "developer", "content": "rules"},
            {"role": "user", "content": "summary"},
        ]

        def fake_compact_history(messages, model):
            return list(compacted)

        try:
            self.cli._compact_history = fake_compact_history
            messages = [
                {"role": "developer", "content": "rules"},
                {"role": "user", "content": "old"},
                {"role": "assistant", "content": "old response"},
            ]
            out = io.StringIO()
            err = io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                done = self.cli._handle_slash("/compact", messages, {}, {}, ".")
        finally:
            self.cli._compact_history = old_compact_history

        self.assertFalse(done)
        self.assertEqual(messages, compacted)
        self.assertIn("Compaction starting with current", err.getvalue())
        printed = out.getvalue()
        self.assertIn("[Context compacted]", printed)
        self.assertIn("# --- System ---\nrules", printed)
        self.assertIn("# --- User ---\nsummary", printed)

    def test_compact_slash_noop_does_not_print_context(self):
        old_compact_history = self.cli._compact_history

        def fake_compact_history(messages, model):
            return list(messages)

        try:
            self.cli._compact_history = fake_compact_history
            messages = [
                {"role": "developer", "content": "rules"},
                {"role": "user", "content": "unchanged"},
            ]
            out = io.StringIO()
            err = io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                self.cli._handle_slash("/compact", messages, {}, {}, ".")
        finally:
            self.cli._compact_history = old_compact_history

        self.assertIn("Compaction starting with current", err.getvalue())
        printed = out.getvalue()
        self.assertIn("[Context unchanged]", printed)
        self.assertNotIn("# --- System ---", printed)

    def test_auto_compact_prints_full_new_context(self):
        old_chat_stream = self.cli.chat_stream
        old_compact_history = self.cli._compact_history
        compacted = [
            {"role": "developer", "content": "rules"},
            {"role": "user", "content": "auto summary"},
        ]

        def fake_chat_stream(messages, model, cancel=None, tools=None):
            if False:
                yield None

        def fake_compact_history(messages, model):
            return list(compacted)

        try:
            self.cli.chat_stream = fake_chat_stream
            self.cli._compact_history = fake_compact_history
            out = io.StringIO()
            err = io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                messages, had_error = self.cli._run_agent(
                    [{"role": "developer", "content": "rules" + ("x" * 100)}],
                    {"options": {"stream": True}, "auto_compact": True, "context": 1},
                    None)
        finally:
            self.cli.chat_stream = old_chat_stream
            self.cli._compact_history = old_compact_history

        self.assertFalse(had_error)
        self.assertEqual(messages, compacted)
        self.assertIn("Compaction starting with current", err.getvalue())
        printed = out.getvalue()
        self.assertIn("[Context compacted automatically]", printed)
        self.assertIn("# --- User ---\nauto summary", printed)

    def test_cli_loads_agents_md_from_current_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "Agents.md"), "w", encoding="utf-8") as f:
                f.write("Use project-local instructions.")
            messages = [{"role": "developer", "content": "base"}]
            changed = self.cli._apply_agents_md(
                messages,
                {"enable_agents_md": True},
                tmp)

        self.assertTrue(changed)
        self.assertIn("Project AGENTS.md instructions:", messages[0]["content"])
        self.assertIn("Use project-local instructions.", messages[0]["content"])
        self.assertFalse(agentic_project.inject_agents_md(messages, "again"))

    def test_cli_intro_prints_agents_md(self):
        old_load_config = self.cli._load_config
        old_run_agent = self.cli._run_agent
        old_argv = sys.argv[:]
        old_cwd = os.getcwd()

        def fake_load_config():
            return {
                "default_prompt": "base",
                "enable_agents_md": True,
                "models": {
                    "local": {
                        "model": "local",
                        "url": "http://example.test",
                        "options": {"stream": True},
                        "context": 1000,
                    }
                },
            }

        def fake_run_agent(messages, model, cancel=None):
            return messages, False

        try:
            with tempfile.TemporaryDirectory() as tmp:
                with open(os.path.join(tmp, "AGENTS.md"), "w", encoding="utf-8") as f:
                    f.write("Visible CLI instructions.")
                os.chdir(tmp)
                sys.argv = ["agentic_cli", "--query", "hi"]
                self.cli._load_config = fake_load_config
                self.cli._run_agent = fake_run_agent
                out = io.StringIO()
                with contextlib.redirect_stdout(out):
                    rc = self.cli.main()
        finally:
            os.chdir(old_cwd)
            sys.argv = old_argv
            self.cli._load_config = old_load_config
            self.cli._run_agent = old_run_agent

        self.assertEqual(rc, 0)
        printed = out.getvalue()
        self.assertIn("# --- System ---", printed)
        self.assertIn("Project AGENTS.md instructions:", printed)
        self.assertIn("Visible CLI instructions.", printed)

    def test_cli_eof_at_prompt_exits_cleanly(self):
        rc, printed = self.run_cli_main(io.StringIO(""))
        self.assertEqual(rc, 0)
        self.assertIn("Goodbye.", printed)

    def test_cli_eof_during_multiline_exits_without_submit(self):
        def fail_run_agent(messages, model, cancel=None):
            raise AssertionError("_run_agent should not be called")

        rc, printed = self.run_cli_main(
            io.StringIO("hello\n"),
            run_agent=fail_run_agent)
        self.assertEqual(rc, 0)
        self.assertIn("Goodbye.", printed)
        self.assertNotIn("# --- User ---\nhello", printed)

    def test_cli_keyboard_interrupt_from_input_returns_130(self):
        rc, printed = self.run_cli_main(InterruptingInput())
        self.assertEqual(rc, 130)
        self.assertIn("[Interrupted]", printed)

    def test_cli_keyboard_interrupt_from_run_agent_returns_130(self):
        def interrupt_run_agent(messages, model, cancel=None):
            raise KeyboardInterrupt()

        rc, printed = self.run_cli_main(
            io.StringIO(""),
            argv=["agentic_cli", "--query", "hi"],
            run_agent=interrupt_run_agent)
        self.assertEqual(rc, 130)
        self.assertIn("[Interrupted]", printed)

    def test_sublime_loads_agents_md_from_active_project_root(self):
        fake_settings = types.SimpleNamespace(
            get=lambda key, default=None: True if key == "enable_agents_md" else default)
        fake_sublime = types.SimpleNamespace(
            load_settings=lambda name: fake_settings,
            status_message=lambda msg: None,
            set_timeout=lambda fn, delay=0: None,
            set_clipboard=lambda text: None,
        )
        fake_plugin = types.SimpleNamespace(
            WindowCommand=object,
            TextCommand=object,
            EventListener=object,
            TextInputHandler=object,
        )
        old_sublime = sys.modules.get("sublime")
        old_plugin = sys.modules.get("sublime_plugin")
        sys.modules["sublime"] = fake_sublime
        sys.modules["sublime_plugin"] = fake_plugin
        try:
            from Agentic import chat_stream

            with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
                with open(os.path.join(first, "AGENTS.md"), "w", encoding="utf-8") as f:
                    f.write("wrong root")
                with open(os.path.join(second, "AGENTS.md"), "w", encoding="utf-8") as f:
                    f.write("right root")
                active_file = os.path.join(second, "main.py")
                with open(active_file, "w", encoding="utf-8") as f:
                    f.write("print('hi')\n")

                window = types.SimpleNamespace(folders=lambda: [first, second])
                view = types.SimpleNamespace(
                    window=lambda: window,
                    file_name=lambda: active_file)
                messages = [{"role": "developer", "content": "base"}]

                self.assertTrue(chat_stream._apply_agents_md(messages, view))
                self.assertIn("right root", messages[0]["content"])
                self.assertNotIn("wrong root", messages[0]["content"])

                intro = chat_stream._initial_chat_text(view, "base", "hello")
                self.assertIn("# --- System ---", intro)
                self.assertIn("Project AGENTS.md instructions:", intro)
                self.assertIn("right root", intro)
                self.assertIn("# --- User ---\nhello", intro)
        finally:
            if old_sublime is None:
                sys.modules.pop("sublime", None)
            else:
                sys.modules["sublime"] = old_sublime
            if old_plugin is None:
                sys.modules.pop("sublime_plugin", None)
            else:
                sys.modules["sublime_plugin"] = old_plugin


class AgenticMarkdownTests(unittest.TestCase):
    def test_basic_round_trip(self):
        messages = [
            {"role": "developer", "content": "rules"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        self.assertEqual(
            agentic_markdown.parse_chat(agentic_markdown.format_chat(messages)),
            messages)

    def test_developer_serializes_as_system_and_parses_back(self):
        text = agentic_markdown.format_chat([
            {"role": "developer", "content": "rules"},
        ])
        self.assertIn("# --- System ---", text)
        self.assertNotIn("# --- Developer ---", text)
        self.assertEqual(agentic_markdown.parse_chat(text), [
            {"role": "developer", "content": "rules"},
        ])

    def test_assistant_tool_call_and_result_round_trip(self):
        tc = {
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "read",
                "arguments": "{\"file_path\":\"a.py\"}",
            },
        }
        messages = [
            {"role": "user", "content": "read"},
            agentic_markdown.assistant_tool_message([tc]),
            agentic_markdown.tool_result_message(
                tc, "{\"file_path\":\"a.py\"}", "1: print('hi')"),
        ]
        self.assertEqual(
            agentic_markdown.parse_chat(agentic_markdown.format_chat(messages)),
            messages)

    def test_multiple_tool_calls_and_results_preserve_order(self):
        calls = [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "read", "arguments": "{\"file_path\":\"a.py\"}"},
            },
            {
                "id": "call_2",
                "type": "function",
                "function": {"name": "ls", "arguments": "{\"path\":\".\"}"},
            },
        ]
        messages = [
            {"role": "user", "content": "inspect"},
            agentic_markdown.assistant_tool_message(calls),
            agentic_markdown.tool_result_message(
                calls[0], "{\"file_path\":\"a.py\"}", "a"),
            agentic_markdown.tool_result_message(
                calls[1], "{\"path\":\".\"}", "b"),
        ]
        parsed = agentic_markdown.parse_chat(agentic_markdown.format_chat(messages))
        assistant = parsed[1]
        self.assertEqual(
            [tc["id"] for tc in assistant["tool_calls"]],
            ["call_1", "call_2"])
        self.assertEqual(
            [m["tool_call_id"] for m in parsed if m["role"] == "tool"],
            ["call_1", "call_2"])
        self.assertEqual(parsed, messages)

    def test_empty_assistant_without_tool_calls_is_dropped(self):
        text = "\n".join([
            "# --- User ---",
            "hello",
            "",
            "# --- Agent ---",
            "",
        ])
        self.assertEqual(agentic_markdown.parse_chat(text), [
            {"role": "user", "content": "hello"},
        ])

    def test_empty_assistant_followed_by_tool_is_preserved(self):
        text = "\n".join([
            "# --- Agent ---",
            "",
            "# --- Tool ---",
            "tool_call_id: call_1",
            "name: read",
            "args: {\"file_path\":\"a.py\"}",
            "1: print('hi')",
        ])
        messages = agentic_markdown.parse_chat(text)
        self.assertEqual(messages[0]["role"], "assistant")
        self.assertEqual(messages[0]["content"], "")
        self.assertEqual(
            messages[0]["tool_calls"][0]["function"]["arguments"],
            "{\"file_path\":\"a.py\"}")

    def test_empty_assistant_tool_turn_round_trips_arguments(self):
        text = "\n".join([
            "# --- System ---",
            "rules",
            "",
            "# --- User ---",
            "read file",
            "",
            "# --- Agent ---",
            "",
            "# --- Tool ---",
            "tool_call_id: call_1",
            "name: read",
            "args: {\"file_path\":\"a.py\"}",
            "1: print('hi')",
        ])
        messages = agentic_markdown.parse_chat(text)
        assistant = [m for m in messages if m["role"] == "assistant"][0]
        tool = [m for m in messages if m["role"] == "tool"][0]
        self.assertEqual(
            assistant["tool_calls"][0]["function"]["arguments"],
            "{\"file_path\":\"a.py\"}")
        self.assertEqual(tool["arguments"], "{\"file_path\":\"a.py\"}")

        rebuilt = agentic_markdown.format_chat(messages)
        self.assertIn("# --- Agent ---", rebuilt)
        self.assertIn("args: {\"file_path\":\"a.py\"}", rebuilt)

    def test_tool_output_with_block_tags_round_trips_escaped(self):
        tc = {
            "id": "call_1",
            "type": "function",
            "function": {"name": "read", "arguments": "{}"},
        }
        output = "\n".join([
            "before",
            "# --- System ---",
            "# --- User ---",
            "# --- Agent ---",
            "# --- Tool ---",
            "after",
        ])
        messages = [
            agentic_markdown.assistant_tool_message([tc]),
            agentic_markdown.tool_result_message(tc, "{}", output),
        ]
        text = agentic_markdown.format_chat(messages)
        self.assertIn("\\# --- System ---", text)
        parsed = agentic_markdown.parse_chat(text)
        self.assertEqual(parsed[1]["content"], output)

    def test_user_content_with_block_tags_round_trips_escaped(self):
        content = "\n".join([
            "File: example.md",
            "# --- System ---",
            "# --- User ---",
            "# --- Agent ---",
            "# --- Tool ---",
        ])
        messages = [{"role": "user", "content": content}]
        text = agentic_markdown.format_chat(messages)
        self.assertIn("\\# --- User ---", text)
        self.assertEqual(agentic_markdown.parse_chat(text), messages)

    def test_escaped_backslash_tag_line_round_trips(self):
        messages = [{"role": "user", "content": "\\# --- User ---"}]
        text = agentic_markdown.format_chat(messages)
        self.assertIn("\\\\# --- User ---", text)
        self.assertEqual(agentic_markdown.parse_chat(text), messages)

    def test_strip_active_only_removes_plain_empty_trailing_assistant(self):
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": ""},
        ]
        self.assertNotIn(
            "# --- Agent ---",
            agentic_markdown.format_chat(messages, strip_active=True))

        tc = {
            "id": "call_1",
            "type": "function",
            "function": {"name": "read", "arguments": "{}"},
        }
        text = agentic_markdown.format_chat(
            [agentic_markdown.assistant_tool_message([tc])],
            strip_active=True)
        self.assertIn("# --- Agent ---", text)


class AgenticCompactionTests(unittest.TestCase):
    def test_compaction_preserves_head_and_tail_summarizes_middle(self):
        messages = [{"role": "developer", "content": "rules"}]
        for i in range(12):
            messages.append({"role": "user", "content": "u{}".format(i)})
            messages.append({"role": "assistant", "content": "a{}".format(i)})

        seen = {}

        def summarize(system_prompt, user_prompt):
            seen["system"] = system_prompt
            seen["user"] = user_prompt
            return "<state_snapshot>\n  <overall_goal>middle summary</overall_goal>\n</state_snapshot>"

        compacted = agentic_compaction.compact_history(
            messages,
            {"compact_keep_head": 2, "compact_keep_tail": 4},
            summarize)

        self.assertEqual(compacted[0], messages[0])
        self.assertEqual(compacted[1], messages[1])
        self.assertIn("middle summary", compacted[2]["content"])
        self.assertEqual(compacted[-4:], messages[-4:])
        self.assertIn("u1", seen["user"])
        self.assertNotIn("u0", seen["user"])
        self.assertNotIn("u11", seen["user"])

    def test_compaction_preserves_assistant_tool_group_boundary(self):
        tc = {
            "id": "call_1",
            "type": "function",
            "function": {"name": "read", "arguments": "{}"},
        }
        messages = [
            {"role": "developer", "content": "rules"},
            {"role": "user", "content": "start"},
            agentic_markdown.assistant_tool_message([tc]),
            agentic_markdown.tool_result_message(tc, "{}", "tool output"),
            {"role": "user", "content": "end"},
            {"role": "assistant", "content": "done"},
        ]

        compacted = agentic_compaction.compact_history(
            messages,
            {"compact_keep_head": 3, "compact_keep_tail": 1},
            lambda _system, _user: "<state_snapshot>summary</state_snapshot>")

        self.assertEqual(compacted[2]["role"], "assistant")
        self.assertEqual(compacted[3]["role"], "tool")
        self.assertEqual(
            compacted[2]["tool_calls"][0]["id"],
            compacted[3]["tool_call_id"])

    def test_tool_output_clipping_preserves_first_and_last_lines(self):
        content = "".join("line {}\n".format(i) for i in range(30))
        clipped = agentic_compaction.clip_text_middle(
            content, 120, 3, 3, label="tool output")
        self.assertIn("line 0\nline 1\nline 2\n", clipped)
        self.assertIn("line 27\nline 28\nline 29\n", clipped)
        self.assertIn("tool output clipped", clipped)
        self.assertNotIn("line 12\n", clipped)

    def test_compaction_failure_preserves_original_messages(self):
        messages = [
            {"role": "developer", "content": "rules"},
            {"role": "user", "content": "one"},
            {"role": "assistant", "content": "two"},
            {"role": "user", "content": "three"},
            {"role": "assistant", "content": "four"},
        ]
        compacted = agentic_compaction.compact_history(
            messages,
            {"compact_keep_head": 1, "compact_keep_tail": 1},
            lambda _system, _user: "")
        self.assertEqual(compacted, messages)


if __name__ == "__main__":
    unittest.main()
