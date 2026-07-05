# agentic_markdown.py
# -------------------------------------------------------------
#  Pure helpers for Agentic editable markdown chat history.
#  Shared by the Sublime plugin and the standalone CLI.
# -------------------------------------------------------------
import json


TAG_TO_ROLE = {
    "# --- System ---": "developer",
    "# --- User ---": "user",
    "# --- Agent ---": "assistant",
    "# --- Tool ---": "tool",
}

ROLE_TO_TAG = {
    "developer": "# --- System ---",
    "system": "# --- System ---",
    "user": "# --- User ---",
    "assistant": "# --- Agent ---",
    "tool": "# --- Tool ---",
}

THINKING_TAG = "## --- Thinking ---"
RESPONSE_TAG = "## --- Response ---"


def _needs_escape(line):
    return line.startswith("# --- ") or line.startswith("\\# --- ")


def _should_unescape(line):
    if not line.startswith("\\"):
        return False
    rest = line[1:]
    return rest.startswith("# --- ") or rest.startswith("\\# --- ")


def escape_content(text):
    """Escape content lines that could be mistaken for block headings."""
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    return "".join(
        ("\\" + line) if _needs_escape(line) else line
        for line in text.splitlines(True)
    )


def unescape_content(text):
    """Reverse escape_content for message/tool content."""
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    return "".join(
        line[1:] if _should_unescape(line) else line
        for line in text.splitlines(True)
    )


class MarkdownContentEscaper(object):
    """Incrementally escape streamed content without delaying whole lines."""
    _MATCHES = ("# --- ", "\\# --- ")

    def __init__(self):
        self._prefix = ""
        self._decided = False

    def feed(self, text):
        if text is None:
            return ""
        if not isinstance(text, str):
            text = str(text)

        out = []
        for ch in text:
            if self._decided:
                out.append(ch)
                if ch == "\n":
                    self._decided = False
                    self._prefix = ""
                continue

            self._prefix += ch
            if self._prefix in self._MATCHES:
                out.append("\\" + self._prefix)
                self._prefix = ""
                self._decided = True
                continue

            possible = False
            for match in self._MATCHES:
                if match.startswith(self._prefix):
                    possible = True
                    break

            if possible:
                continue

            out.append(self._prefix)
            if ch == "\n":
                self._prefix = ""
                self._decided = False
            else:
                self._prefix = ""
                self._decided = True

        return "".join(out)

    def flush(self):
        pending = self._prefix
        self._prefix = ""
        self._decided = False
        return pending


def _argument_text(arguments):
    if arguments is None:
        return "{}"
    if isinstance(arguments, str):
        return arguments
    try:
        return json.dumps(arguments)
    except Exception:
        return str(arguments)


def _normal_tool_call(tool_call, index=0):
    tc = tool_call if isinstance(tool_call, dict) else {}
    fn = tc.get("function", {})
    fn = fn if isinstance(fn, dict) else {}
    return {
        "id": tc.get("id") or "tc_{}".format(index),
        "type": tc.get("type", "function"),
        "function": {
            "name": fn.get("name", ""),
            "arguments": _argument_text(fn.get("arguments", "{}")),
        },
    }


def assistant_tool_message(tool_calls, content="", reasoning=""):
    """Build an assistant message with OpenAI-compatible tool_calls."""
    msg = {
        "role": "assistant",
        "tool_calls": [
            _normal_tool_call(tc, i)
            for i, tc in enumerate(tool_calls or [])
        ],
    }
    if content or msg["tool_calls"]:
        msg["content"] = content or ""
    if reasoning:
        msg["reasoning_content"] = reasoning
    return msg


def tool_result_message(tool_call, args, output):
    """Build a tool result message matching the visible Tool block."""
    fn = tool_call.get("function", {}) if isinstance(tool_call, dict) else {}
    return {
        "role": "tool",
        "tool_call_id": tool_call.get("id", "") if isinstance(tool_call, dict) else "",
        "name": fn.get("name", ""),
        "arguments": _argument_text(args),
        "content": output or "",
    }


def format_tool_block(tool_call_id, name, arguments, content):
    """Return one visible Tool block."""
    header = "tool_call_id: {}\nname: {}\nargs: {}\n".format(
        tool_call_id or "",
        name or "",
        _argument_text(arguments),
    )
    return "# --- Tool ---\n{}{}".format(header, escape_content(content or ""))


def _structural_role(line):
    return TAG_TO_ROLE.get(line.rstrip("\r\n"))


def _finish_block(messages, role, content_lines):
    if not role:
        return
    if not content_lines and role != "assistant":
        return

    raw = "".join(content_lines).rstrip("\n")
    if role == "tool":
        parts = raw.split("\n", 3)
        tc_id = parts[0][len("tool_call_id: "):].strip() \
            if len(parts) > 0 and parts[0].startswith("tool_call_id: ") else ""
        name = parts[1][len("name: "):].strip() \
            if len(parts) > 1 and parts[1].startswith("name: ") else ""
        args = parts[2][len("args: "):].strip() \
            if len(parts) > 2 and parts[2].startswith("args: ") else ""
        content = unescape_content(parts[3] if len(parts) > 3 else "")
        tool_msg = {
            "role": "tool",
            "tool_call_id": tc_id,
            "name": name,
            "arguments": args,
            "content": content,
        }
        messages.append(tool_msg)
        _inject_tool_call(messages, tool_msg)
        return

    messages.append({
        "role": role,
        "content": unescape_content(raw),
    })


def _inject_tool_call(messages, tool_msg):
    for i in range(len(messages) - 2, -1, -1):
        if messages[i].get("role") == "assistant":
            calls = messages[i].setdefault("tool_calls", [])
            calls.append({
                "id": tool_msg.get("tool_call_id", ""),
                "type": "function",
                "function": {
                    "name": tool_msg.get("name", ""),
                    "arguments": tool_msg.get("arguments", ""),
                },
            })
            break


def _drop_blank_assistants(messages):
    out = []
    for msg in messages:
        if msg.get("role") == "assistant" \
                and not (msg.get("content") or "").strip() \
                and not msg.get("tool_calls"):
            continue
        out.append(msg)
    return out


def parse_chat(text):
    """Parse Agentic markdown chat text into OpenAI-style messages."""
    messages = []
    current_role = None
    content_lines = []
    in_reasoning = False

    for line in (text or "").splitlines(True):
        role = _structural_role(line)
        if role:
            _finish_block(messages, current_role, content_lines)
            current_role = role
            content_lines = []
            in_reasoning = False
            continue

        marker = line.rstrip("\r\n")
        if current_role == "assistant" and marker == THINKING_TAG:
            in_reasoning = True
            continue
        if current_role == "assistant" and marker == RESPONSE_TAG:
            in_reasoning = False
            continue
        if in_reasoning:
            continue

        if current_role:
            content_lines.append(line)

    _finish_block(messages, current_role, content_lines)
    return _drop_blank_assistants(messages)


def _empty_assistant(msg):
    return msg.get("role") == "assistant" \
        and not (msg.get("content") or "").strip() \
        and not msg.get("tool_calls")


def format_chat(messages, strip_active=False):
    """Serialize messages into Agentic markdown chat text."""
    serializable = []
    for msg in messages or []:
        role = msg.get("role")
        if role not in ROLE_TO_TAG:
            continue
        serializable.append(msg)

    if strip_active and serializable and _empty_assistant(serializable[-1]):
        serializable = serializable[:-1]

    blocks = []
    for msg in serializable:
        role = msg.get("role")
        tag = ROLE_TO_TAG.get(role)
        if not tag:
            continue
        if role == "assistant" and not (msg.get("content") or "").strip() \
                and not msg.get("tool_calls"):
            continue
        if role == "tool":
            blocks.append(format_tool_block(
                msg.get("tool_call_id", ""),
                msg.get("name", ""),
                msg.get("arguments", "{}"),
                msg.get("content", ""),
            ))
        else:
            blocks.append("{}\n{}".format(
                tag,
                escape_content(msg.get("content", "") or ""),
            ))

    return "\n\n".join(blocks)
