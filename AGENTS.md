# Agentic Development Notes

## Project Shape

Agentic is a small Sublime Text package for local-first LLM agents. Keep the
Python surface minimal because Sublime package loading effectively makes the
package behave like one concatenated plugin surface. Avoid sprawling module
graphs, import side effects, modern syntax that older Sublime Python builds may
reject, and duplicate implementations that can drift.

Important files:

- `chat_stream.py`: the only root-level Sublime plugin module; owns Sublime
  commands, streaming UI, chat creation, and sanitizing commands.
- `lib/agentic_client.py`: dependency-free OpenAI-compatible streaming client
  shared by the Sublime plugin and CLI.
- `lib/agentic_tools.py`: OpenAI-format tool schemas plus handlers for `read`,
  `patch`, `python`, `reset`, `find`, `grep`, and `ls`.
- `lib/_repl.py`: persistent subprocess used by the `python` tool.
- `agentic_cli`: standalone terminal agent. It reuses `lib/` helpers; supports
  one-shot `--query` runs for testing.
- `Agentic.sublime-settings`: defaults, model definitions, actions, sanitizing
  map, and feature flags.
- `Default*.sublime-keymap`, `Default.sublime-commands`, `Main.sublime-menu`:
  Sublime integration.

## Minimal Philosophy

- One visible interaction model: editable markdown chats with clear block tags.
- One model API assumption: OpenAI-style `/chat/completions`, with small
  provider shims only where necessary.
- One editing safety model: `read` before `patch`, exact replacement first,
  fuzzy replacement only as a recovery path, and unified diffs as feedback.
- Local LLMs are first-class: tools should be simple, explicit, low dependency,
  and robust to imperfect tool-call formatting.
- Prefer fewer tools with predictable behavior over a large tool suite.
- Keep shared logic shared. If CLI and Sublime behavior must match, it should
  live in one implementation that both surfaces call.

## Current Tool/Cache Risks

- `lib/agentic_tools.py` stores `_read_cache` and `_write_cache` globally by file
  path. Different chats can share read/patch authority accidentally. Make these
  session-scoped before relying on concurrent tool use.
- Successful `patch` updates `_write_cache` but not `_read_cache`. This allows
  sequential patches without a fresh read by design, but it weakens the stated
  read-before-edit invariant. Either update the read cache to the new file state
  or require a re-read after each patch.
- Fuzzy patching calculates a match index in normalized text and applies that
  index to the original text. If normalization changes length, the edit can land
  at the wrong position.
- Empty assistant tool-call turns and parsed tool block `arguments` are now
  preserved when chat markdown is rebuilt.
- Pseudo tool-call parsing now uses `json.JSONDecoder`, so nested JSON and
  braces in patch strings are handled by the shared client.

## Simplification Targets

- Keep provider-agnostic chat client/tool-call parsing in `lib/agentic_client.py`
  and avoid reintroducing duplicate streaming code.
- Keep markdown chat parse/rebuild behavior in `lib/agentic_markdown.py`; do not
  add second parsers in CLI or Sublime code.
- Make tool execution state explicit: `ToolSession` or equivalent holding REPL
  state, read cache, write cache, cwd, and sanitizer.
- Replace repeated command/action chat creation code with one helper that builds
  a tagged chat from system, user, model, and optional auto-submit inputs.
- Keep sanitizing as an explicit copy/output feature, not mixed into tool safety
  logic unless the implications for exact patch strings are clear.
- Add small local tests for parser round-trips, tool-cache invariants, fuzzy
  patch edge cases, and pseudo tool-call parsing. These do not need Sublime.
