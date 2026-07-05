# agentic_compaction.py
# -------------------------------------------------------------
#  Shared history compaction helpers for Agentic.
# -------------------------------------------------------------
from .agentic_markdown import format_chat
COMPACT_SYSTEM = "\n".join([
    "You are a context summarization assistant. Task: summarize the middle",
    "portion of a conversation between a user and an AI coding agent into a",
    "structured snapshot that another LLM will use to continue the work.",
    "Do NOT continue the conversation. Do NOT respond to any questions in the",
    "conversation. Preserve concrete facts, decisions, constraints, file paths,",
    "commands, tool results, errors, and unresolved tasks. Prefer precise",
    "bullet points over general prose.",
    "",
    "Produce this exact format:",
    "",
    "<state_snapshot>",
    "  <overall_goal>",
    "    A single sentence describing the high-level objective.",
    "  </overall_goal>",
    "  <key_knowledge>",
    "    - Crucial facts, conventions, build commands, constraints.",
    "    - Important user preferences and decisions.",
    "  </key_knowledge>",
    "  <file_system_state>",
    "    - Files read, created, modified with paths and status.",
    "    - Tool outputs or command results that matter later.",
    "  </file_system_state>",
    "  <recent_actions>",
    "    - Significant actions from the summarized middle section.",
    "  </recent_actions>",
    "  <current_plan>",
    "    Step-by-step plan. Mark: [DONE], [IN PROGRESS], [TODO].",
    "  </current_plan>",
    "</state_snapshot>",
])

SUMMARY_PREFIX = "[CONTEXT SUMMARY - MIDDLE OF CONVERSATION]"
SUMMARY_SUFFIX = "[/CONTEXT SUMMARY]"

DEFAULT_KEEP_HEAD = 4
DEFAULT_KEEP_TAIL = 16
DEFAULT_TOOL_MAX_CHARS = 12000
DEFAULT_TOOL_HEAD_LINES = 80
DEFAULT_TOOL_TAIL_LINES = 80
COMPACTION_SETTING_KEYS = (
    "compact_keep_head",
    "compact_keep_tail",
    "compact_tool_max_chars",
    "compact_tool_head_lines",
    "compact_tool_tail_lines",
)


def copy_compaction_settings(model, settings):
    copied = dict(model or {})
    for key in COMPACTION_SETTING_KEYS:
        try:
            value = settings.get(key)
        except Exception:
            value = None
        if value is not None:
            copied[key] = value
    return copied


def _as_int(value, default):
    try:
        value = int(value)
    except Exception:
        return default
    if value < 0:
        return default
    return value


def compaction_options(model):
    model = model or {}
    return {
        "keep_head": _as_int(
            model.get("compact_keep_head", model.get("compact_keep_head_messages")),
            DEFAULT_KEEP_HEAD),
        "keep_tail": _as_int(
            model.get("compact_keep_tail", model.get("compact_keep_tail_messages")),
            DEFAULT_KEEP_TAIL),
        "tool_max_chars": _as_int(
            model.get("compact_tool_max_chars"),
            DEFAULT_TOOL_MAX_CHARS),
        "tool_head_lines": _as_int(
            model.get("compact_tool_head_lines"),
            DEFAULT_TOOL_HEAD_LINES),
        "tool_tail_lines": _as_int(
            model.get("compact_tool_tail_lines"),
            DEFAULT_TOOL_TAIL_LINES),
    }


def _take_first_lines(lines, n):
    if n <= 0:
        return []
    return lines[:n]


def _take_last_lines(lines, n):
    if n <= 0:
        return []
    return lines[-n:]


def clip_text_middle(text, max_chars, head_lines, tail_lines, label="content"):
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    if max_chars <= 0 or len(text) <= max_chars:
        return text

    lines = text.splitlines(True)
    head = "".join(_take_first_lines(lines, head_lines))
    tail = "".join(_take_last_lines(lines, tail_lines))
    omitted = max(0, len(text) - len(head) - len(tail))
    marker = "\n[... {} clipped: {} chars omitted from middle ...]\n".format(
        label, omitted)

    clipped = head + marker + tail
    if len(clipped) <= max_chars:
        return clipped

    marker = "\n[... {} clipped to {} chars ...]\n".format(label, max_chars)
    budget = max(0, max_chars - len(marker))
    head_budget = budget // 2
    tail_budget = budget - head_budget
    return text[:head_budget] + marker + text[-tail_budget:]


def _clip_message(msg, options):
    if msg.get("role") != "tool":
        return dict(msg)
    clipped = dict(msg)
    clipped["content"] = clip_text_middle(
        clipped.get("content", ""),
        options["tool_max_chars"],
        options["tool_head_lines"],
        options["tool_tail_lines"],
        label="tool output")
    return clipped


def clip_tool_outputs(messages, model=None):
    options = compaction_options(model)
    return [_clip_message(m, options) for m in messages]


def _message_groups(messages):
    groups = []
    i = 0
    while i < len(messages):
        msg = messages[i]
        group = [msg]
        i += 1
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            call_ids = set()
            for tc in msg.get("tool_calls") or []:
                if isinstance(tc, dict) and tc.get("id"):
                    call_ids.add(tc.get("id"))
            while i < len(messages) and messages[i].get("role") == "tool":
                if call_ids and messages[i].get("tool_call_id") not in call_ids:
                    break
                group.append(messages[i])
                i += 1
        elif msg.get("role") == "tool":
            while i < len(messages) and messages[i].get("role") == "tool":
                group.append(messages[i])
                i += 1
        groups.append(group)
    return groups


def _flatten(groups):
    out = []
    for group in groups:
        out.extend(group)
    return out


def split_history(messages, model=None):
    options = compaction_options(model)
    groups = _message_groups(list(messages or []))
    keep_head = options["keep_head"]
    keep_tail = options["keep_tail"]
    if keep_head + keep_tail >= len(groups) - 1:
        return list(messages or []), [], []
    head_groups = groups[:keep_head]
    tail_groups = groups[len(groups) - keep_tail:] if keep_tail else []
    middle_groups = groups[keep_head:len(groups) - keep_tail if keep_tail else len(groups)]
    return _flatten(head_groups), _flatten(middle_groups), _flatten(tail_groups)


def extract_snapshot(summary_text):
    summary_text = summary_text or ""
    start = summary_text.find("<state_snapshot>")
    end = summary_text.find("</state_snapshot>")
    if start != -1 and end != -1:
        return summary_text[start:end + len("</state_snapshot>")].strip()
    return summary_text[:4000].strip()


def compacted_messages(head, summary_text, tail, model=None):
    summary_text = extract_snapshot(summary_text)
    if not summary_text:
        return None
    result = []
    result.extend(clip_tool_outputs(head, model))
    result.append({
        "role": "user",
        "content": SUMMARY_PREFIX + "\n" + summary_text + "\n" + SUMMARY_SUFFIX,
    })
    result.extend(clip_tool_outputs(tail, model))
    return result


def build_summary_prompt(middle_messages, model=None):
    clipped_middle = clip_tool_outputs(middle_messages, model)
    compact_text = format_chat(clipped_middle)
    return "Summarize this middle conversation segment:\n\n" + compact_text


def compact_history(messages, model, summarize_fn):
    head, middle, tail = split_history(messages, model)
    if not middle:
        return list(messages or [])

    prompt = build_summary_prompt(middle, model)
    result = compacted_messages(
        head, summarize_fn(COMPACT_SYSTEM, prompt) or "", tail, model)
    if not result:
        return list(messages or [])
    return result
