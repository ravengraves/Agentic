#!/usr/bin/env python3
"""
_repl.py -- Persistent Python REPL server for the Agentic plugin.

Wire protocol: NDJSON (one JSON object per line) on stdin/stdout.

  Incoming (stdin):
    {"type": "exec", "code": "<python source>"}
    {"type": "clear"}

  Outgoing (stdout):
    {"type": "result", "stdout": "<captured print output>", "error": null|"msg"}
    {"type": "error", "message": "<subprocess fatal error>"}

Real stdout is reserved for protocol only.  User print() is intercepted
into an io.StringIO and returned in the "stdout" field of the response.
Namespace lives in-process -- no disk state, no pickle.
"""

import io
import json
import math
import os
import re
import signal
import subprocess
import sys


# ---------------------------------------------------------------------------
# Fatal-exit handler -- write a protocol-level error and exit so the parent
# can auto-restart cleanly.
# ---------------------------------------------------------------------------
def _die(message):
    """Send a top-level error message and exit with code 1."""
    try:
        json.dump({"type": "error", "message": message}, sys.stdout)
        sys.stdout.write("\n")
        sys.stdout.flush()
    except Exception:
        pass
    sys.exit(1)


# ---------------------------------------------------------------------------
# SIGTERM / SIGHUP handler -- exit without corrupting protocol
# ---------------------------------------------------------------------------
def _sigterm_handler(signum, frame):
    sys.exit(0)


signal.signal(signal.SIGTERM, _sigterm_handler)
try:
    signal.signal(signal.SIGHUP, _sigterm_handler)
except AttributeError:
    pass  # Windows has no SIGHUP

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
_ns = {}  # persistent namespace (isolated from globals)

# Pre-inject well-known modules
_NS_MODULES = ("json", "os", "math", "subprocess", "re")
for _mod_name in _NS_MODULES:
    _ns[_mod_name] = sys.modules[_mod_name]

del _mod_name


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _flush_stdout():
    sys.stdout.flush()


def _send_result(stdout_text, error_text=None):
    """Write a result JSON line to stdout."""
    resp = {
        "type": "result",
        "stdout": stdout_text,
        "error": error_text,
    }
    json.dump(resp, sys.stdout)
    sys.stdout.write("\n")
    _flush_stdout()


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main():
    """Read NDJSON commands from stdin, process, write NDJSON responses."""
    # stdin as text for NDJSON reading
    stdin = sys.stdin

    for raw_line in stdin:
        raw_line = raw_line.strip()
        if not raw_line:
            continue

        # Parse the command
        try:
            cmd = json.loads(raw_line)
        except (json.JSONDecodeError, ValueError):
            _send_result("", "invalid JSON command")
            continue

        ctype = cmd.get("type")

        if ctype == "exec":
            code = cmd.get("code", "")
            _exec_code(code)

        elif ctype == "clear":
            _clear_ns()
            _send_result("Python repl state cleared")

        else:
            _send_result("", "unknown command type: " + str(ctype))


def _exec_code(code):
    """Execute Python code with captured print() and return result."""
    # Create a StringIO to capture user print() output
    captured = io.StringIO()

    try:
        # Save original stdout/stderr so user code can still write to them
        # but we capture stdout into StringIO
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = captured
        sys.stderr = captured  # stderr also goes into captured

        try:
            exec(code, _ns)
        except Exception:
            # Capture the traceback into our StringIO
            import traceback
            tb_text = "".join(traceback.format_exception(*sys.exc_info()))
            captured.write(tb_text)
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            user_output = captured.getvalue()
            # The last part is the exception trace -- include as error field too
            _send_result(user_output.rstrip("\n"), tb_text.rstrip("\n"))
        else:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            user_output = captured.getvalue()
            _send_result(user_output.rstrip("\n"))

    except Exception as e:
        # Should never reach here, but be safe
        try:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
        except Exception:
            pass
        _send_result("", str(e))


def _clear_ns():
    """Reset the namespace, re-injecting built-in modules."""
    _ns.clear()
    for _mod_name in _NS_MODULES:
        _ns[_mod_name] = sys.modules[_mod_name]


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        _die("uncaught error in repl server: " + str(e))
    finally:
        try:
            _flush_stdout()
        except Exception:
            pass
