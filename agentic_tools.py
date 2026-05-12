# agentic_tools.py
# -------------------------------------------------------------
#  Tool definitions and handlers for Agentic AI plugin.
#  Provides: read, patch, python, reset, find, grep, ls
#
#  Minimal Python 3.8+ (no f-strings, no type hints).
# -------------------------------------------------------------
import io
import os
import re
import sys
import time
import json
import glob
import atexit
import difflib
import signal
import subprocess
import threading
import unicodedata

# -------------------------------------------------------------
#  Tool definitions (OpenAI format)
# -------------------------------------------------------------
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read",
            "description": (
                "Read file contents with line-number prefixes (1: ...). "
                "If file_path is a directory, list entries with '/' suffix for subdirectories. "
                "Use offset/limit for large files. Line numbers are 1-indexed. "
                "Binary files (.exe, .png, .pdf, .zip, etc.) are rejected. "
                "Examples: "
                '{"file_path": "/abs/path/file.py"} reads entire file. '
                '{"file_path": "/abs/path/file.py", "offset": 50, "limit": 100} reads lines 50-149. '
                '{"file_path": "/abs/path/src/"} lists the directory (entries with "/" are subdirs). '
                "Use read to examine files instead of cat, head, tail, or sed."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Absolute path to the file or directory.",
                    },
                    "offset": {
                        "type": "number",
                        "description": "1-indexed line number to start from (defaults to 1).",
                    },
                    "limit": {
                        "type": "number",
                        "description": "Maximum lines to return (defaults to 2000).",
                    },
                },
                "required": ["file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "patch",
            "description": (
                "Edit a file using exact then fuzzy string replacement. "
                "ALWAYS use read first to get the exact old_string including all whitespace and indentation. "
                "Include at least 3 lines of surrounding context for unique matching. "
                "Returns a unified diff of the changes made. "
                "Examples: "
                '{"file_path": "/abs/file.py", "old_string": "x = 1", "new_string": "x = 2"} '
                "edits a single line. "
                '{"file_path": "/abs/file.py", "old_string": "def foo():\\n    pass", '
                '"new_string": "def foo():\\n    return 42"} edits multiple lines with exact indentation. '
                '{"file_path": "/abs/new.py", "old_string": "", "new_string": "print(\'hello\')"} '
                "creates a new file. "
                "If exact match fails, unicode-normalized fuzzy match is attempted "
                "(smart quotes, em-dashes, non-breaking spaces are matched to their ASCII equivalents). "
                "Use patch to edit files instead of sed or awk."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Absolute path to the file to modify.",
                    },
                    "old_string": {
                        "type": "string",
                        "description": "Exact literal text to find and replace.",
                    },
                    "new_string": {
                        "type": "string",
                        "description": "Replacement text.",
                    },
                },
                "required": ["file_path", "old_string", "new_string"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "python",
            "description": (
                "Execute Python code in a persistent session. "
                "Variables, imports, and function definitions persist across calls "
                "(state is saved to disk between invocations). "
                "Use reset to clear all state and start fresh. "
                "json, os, math, subprocess, and re are pre-imported and always available. "
                "30s timeout. For file I/O prefer the read/patch tools instead. "
                "Examples: "
                '{"code": "x = 42"} stores x. {"code": "print(x)"} prints 42. '
                '{"code": "import math\\nprint(math.sqrt(81))"} one-shot calculation. '
                '{"code": "r = subprocess.run([\'git\', \'status\'], capture_output=True, '
                'text=True, cwd=\'.\'); print(r.stdout)"} runs a binary and prints output. '
                '{"code": "import json\\nprint(json.dumps({\\"a\\":1}))"} outputs JSON. '
                "Output is combined stdout + stderr."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute. Output is returned.",
                    },
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "reset",
            "description": (
                "Reset the Python REPL, clearing all variables, imports, and function definitions. "
                "Use when you need a clean interpreter or want to start over. "
                "Example: {} (empty object, no arguments needed)."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find",
            "description": (
                "Search for files by glob pattern, sorted by modification time (newest first). "
                "Use this instead of bash find or ls for file search. "
                "Supports recursive patterns with **/ prefix. Results capped at 500. "
                "Use a narrower pattern if results are truncated. "
                "Examples: "
                '{"pattern": "**/*.py"} finds all Python files recursively from project root. '
                '{"pattern": "*.ts", "path": "src"} searches src/ only. '
                "Results are sorted newest-first by modification time."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern (e.g. '*.py', '**/*.js').",
                    },
                    "path": {
                        "type": "string",
                        "description": "Directory to search in (defaults to project root).",
                    },
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": (
                "Search file contents for a regex pattern, returning matching lines "
                "with 3 lines of surrounding context. Matched lines are marked with '>'. "
                "Uses Python re syntax. Max 100 matches. Binary files automatically skipped. "
                "Use this instead of bash grep or rg for content search. "
                "Examples: "
                '{"pattern": "TODO", "path": "src"} finds all TODO comments in src/. '
                '{"pattern": "from\\\\s+typing\\\\s+import", "include": "*.py"} filters to Python files. '
                "Escape backslashes: \\\\s for whitespace, \\\\d for digits, \\\\w for word chars. "
                "Output format: path, then --- separator, then matching blocks with line numbers."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Regex pattern to search for.",
                    },
                    "path": {
                        "type": "string",
                        "description": "Directory to search in (defaults to project root).",
                    },
                    "include": {
                        "type": "string",
                        "description": "File glob filter (e.g. '*.py', '*.{ts,js}').",
                    },
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ls",
            "description": (
                "List directory contents sorted alphabetically. "
                "Entries with '/' suffix are subdirectories. "
                "Use this instead of bash ls for directory listing. "
                'Examples: {"path": "."} lists the project root. '
                '{"path": "src/components"} lists a subdirectory.'
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory to list (defaults to '.')",
                    },
                },
            },
        },
    },
]

# -------------------------------------------------------------
#  Persistent Python REPL subprocess (NDJSON protocol)
# -------------------------------------------------------------

_REPL_EXE = sys.executable
_repl_proc = None
_repl_lock = threading.Lock()
_REPL_TIMEOUT = 30


def _cleanup_repl():
    """Terminate the REPL subprocess on plugin unload or process exit."""
    global _repl_proc
    if _repl_proc is not None:
        try:
            _repl_proc.stdin.close()
        except (OSError, AttributeError):
            pass
        try:
            _repl_proc.terminate()
            _repl_proc.wait(timeout=3)
        except Exception:
            try:
                _repl_proc.kill()
            except OSError:
                pass
        _repl_proc = None


atexit.register(_cleanup_repl)


def _get_repl_script_path():
    """Return the absolute path to the REPL server script bundled alongside this module."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "_repl.py")


def _start_repl():
    """Launch the persistent REPL subprocess. Returns the Popen object."""
    global _repl_proc
    _repl_proc = subprocess.Popen(
        [_REPL_EXE, _get_repl_script_path()],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
        cwd=os.getcwd(),
    )
    # Drain any startup noise on stderr (protocol uses JSON on stdout only)
    return _repl_proc


def _ensure_repl():
    """Make sure the REPL subprocess is alive; restart if necessary."""
    with _repl_lock:
        if _repl_proc is None:
            return _start_repl()
        rv = _repl_proc.poll()
        if rv is not None:
            return _start_repl()
        return _repl_proc


def _send_to_repl(cmd_dict):
    """
    Send a NDJSON command to the REPL subprocess and read the next NDJSON response.
    Returns (stdout_text, error_flag).
    """
    global _repl_proc

    proc = _ensure_repl()

    payload = json.dumps(cmd_dict) + "\n"
    try:
        proc.stdin.write(payload.encode("utf-8"))
        proc.stdin.flush()
    except (BrokenPipeError, OSError):
        # Pipe broken -> restart inside lock
        with _repl_lock:
            _repl_proc = None
        proc = _ensure_repl()
        proc.stdin.write(payload.encode("utf-8"))
        proc.stdin.flush()

    # Read response line by line until we get a valid JSON protocol message
    start_time = time.time()
    partial = b""
    while True:
        if time.time() - start_time > _REPL_TIMEOUT:
            try:
                proc.terminate()
                proc.wait(timeout=2)
            except Exception:
                try:
                    proc.kill()
                except OSError:
                    pass
            return (
                "python error: execution timed out after 30s. "
                "Simplify the code or split into smaller chunks.",
                True,
            )

        try:
            byte = proc.stdout.read(1)
        except (IOError, OSError):
            break

        if byte == b"":
            # EOF -- process died
            with _repl_lock:
                _repl_proc = None
            return (
                "python error: subprocess died unexpectedly. "
                "It will be restarted on next call.",
                True,
            )

        partial += byte

        if b"\n" in partial:
            line_end = partial.index(b"\n")
            line = partial[:line_end].decode("utf-8", errors="replace").strip()
            partial = partial[line_end + 1:]

            if not line:
                continue

            try:
                resp = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue  # not a protocol line; ignore

            rtype = resp.get("type")
            if rtype == "result":
                output = resp.get("stdout", "")
                err = resp.get("error")
                if err:
                    if output:
                        output = output + "\n"
                    output = output + "Error: " + err
                return output, False
            elif rtype == "error":
                return resp.get("message", "subprocess error"), True

    return "python error: no response from subprocess.", True

# -------------------------------------------------------------
#  File read cache (TOCTOU enforcement)
# -------------------------------------------------------------
_read_cache = {}  # file_path -> (mtime, size, full_read: bool)

# Write cache: tracks the last mtime/size when patch wrote a file.
# Used to suppress stale-mtime errors when the model itself last
# modified the file (e.g. sequential patches without re-read).
_write_cache = {}  # file_path -> (mtime, size)

# Directories excluded from find/ls/read listings
_TOOL_IGNORE = {".Agentic", ".git", "__pycache__", "node_modules"}

# -------------------------------------------------------------
#  Tool dispatch
# -------------------------------------------------------------
def _resolve_path(file_path, cwd=None):
    """Resolve relative paths against cwd. Absolute paths pass through."""
    if cwd and file_path and not os.path.isabs(file_path):
        return os.path.abspath(os.path.join(cwd, file_path))
    return file_path

def execute_tool(name, args, sanitize_fn=None, cancel=None, cwd=None):
    """
    Execute a tool by name.  args is a dict of parameters.
    Returns (output_string, is_error).
    """
    if name == "read":
        return _read(args, sanitize_fn, cwd)
    if name == "patch":
        return _patch(args, sanitize_fn, cwd)
    if name == "python":
        return _python(args, cancel)
    if name == "reset":
        return _python_reset(args)
    if name == "find":
        return _find(args, cwd)
    if name == "grep":
        return _grep(args, cwd)
    if name == "ls":
        return _ls(args, cwd)
    return "tool error: unknown tool '{}'. Available: read, patch, python, reset, find, grep, ls.".format(name), True

# -------------------------------------------------------------
#  TOCTOU helpers
# -------------------------------------------------------------
def _mark_read(file_path, full_read=False):
    """Record that a file was read successfully."""
    try:
        apath = os.path.abspath(file_path)
        stat = os.stat(apath)
        _read_cache[apath] = (stat.st_mtime, stat.st_size, full_read)
    except OSError:
        pass

def _check_prior_read(file_path, require_full=True):
    """Verify file was read before editing. Returns (ok: bool, error_message: str)."""
    apath = os.path.abspath(file_path)
    if apath not in _read_cache:
        return False, (
            "Safety: {} was never read. "
            "Use the read tool first to examine the file before editing.".format(file_path)
        )
    read_cached = _read_cache[apath]
    try:
        stat = os.stat(apath)
    except OSError:
        return False, (
            "Safety: {} no longer exists. Re-read the file to verify its current state.".format(file_path)
        )

    write_cached = _write_cache.get(apath)

    if stat.st_mtime != read_cached[0] or stat.st_size != read_cached[1]:
        # File has changed since last read.  If the model was the last
        # writer and the file state matches that write, the model is the
        # current author, so allow the edit (it may be a follow-up patch).
        if write_cached and \
           stat.st_mtime == write_cached[0] and stat.st_size == write_cached[1]:
            pass  # model was last writer, file unchanged since then
        else:
            return False, (
                "Safety: {} has changed since last read (mtime or size mismatch). "
                "Re-read the file before editing.".format(file_path)
            )

    if require_full and not read_cached[2]:
        return False, (
            "Safety: {} was only partially read. "
            "Use read without offset/limit to read the full file before editing.".format(file_path)
        )
    return True, None

# -------------------------------------------------------------
#  read
# -------------------------------------------------------------
_BINARY_EXTS = {
    ".exe", ".dll", ".so", ".dylib", ".bin", ".obj", ".o", ".a",
    ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z", ".rar",
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".ico", ".webp",
    ".mp3", ".wav", ".ogg", ".mp4", ".avi", ".mov", ".mkv",
    ".pyc", ".pyo", ".class", ".wasm",
    ".ttf", ".otf", ".woff", ".woff2",
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
}

_MAX_READ_BYTES = 50 * 1024

def _read(args, sanitize_fn=None, cwd=None):
    file_path = _resolve_path(args.get("file_path", ""), cwd)
    offset = int(args.get("offset", 1))
    limit = int(args.get("limit", 2000))

    if not os.path.exists(file_path):
        return (
            "read error: file not found: {}. "
            "Check the path with ls or find first.".format(file_path),
            True,
        )

    if os.path.isdir(file_path):
        try:
            entries = os.listdir(file_path)
            entries.sort()
            lines = []
            for e in entries:
                if e.endswith("~") or e.startswith("#"):
                    continue
                if e in _TOOL_IGNORE:
                    continue
                full = os.path.join(file_path, e)
                suffix = "/" if os.path.isdir(full) else ""
                lines.append(e + suffix)
            result = "\n".join(lines[:limit])
            if sanitize_fn:
                result = sanitize_fn(result)
            return result, False
        except Exception as e:
            return "read error: listing directory failed {}: {}".format(file_path, str(e)), True

    ext = os.path.splitext(file_path)[1].lower()
    if ext in _BINARY_EXTS:
        return (
            "read error: {} is a binary file ({}). "
            "Use find or ls to locate text files.".format(file_path, ext),
            True,
        )

    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            all_lines = f.readlines()
    except Exception as e:
        return "read error: reading file failed {}: {}".format(file_path, str(e)), True

    # Mark file as read (TOCTOU enforcement)
    full_read = (offset == 1 and limit >= len(all_lines))
    _mark_read(file_path, full_read=full_read)

    # Slice
    start = max(0, offset - 1)
    end = start + limit
    chunk = all_lines[start:end]

    # Number and trim each line
    out = []
    for i, line in enumerate(chunk, start=start + 1):
        if len(line) > 2000:
            line = line[:2000] + "... [truncated]"
        out.append("{}: {}".format(i, line.rstrip("\n")))

    result = "\n".join(out)
    if sanitize_fn:
        result = sanitize_fn(result)
    if len(result) > _MAX_READ_BYTES:
        result = result[:_MAX_READ_BYTES] + "\n... [truncated]"
    return result, False

# -------------------------------------------------------------
#  patch  -- exact match, then fuzzy (pi-style)
# -------------------------------------------------------------
def _fuzzy_normalize(s):
    """NFKC normalize + strip trailing whitespace per line + unicode quotes/dashes to ASCII."""
    s = unicodedata.normalize("NFKC", s)
    # strip trailing whitespace per line
    s = "\n".join(line.rstrip() for line in s.split("\n"))
    # smart single quotes -> '
    s = re.sub("[\u2018\u2019\u201A\u201B]", "'", s)
    # smart double quotes -> "
    s = re.sub("[\u201C\u201D\u201E\u201F]", '"', s)
    # dashes -> -
    s = re.sub("[\u2010\u2011\u2012\u2013\u2014\u2015\u2212]", "-", s)
    # special spaces -> regular space
    s = re.sub("[\u00A0\u2002-\u200A\u202F\u205F\u3000]", " ", s)
    return s

def _patch(args, sanitize_fn=None, cwd=None):
    file_path = _resolve_path(args.get("file_path", ""), cwd)
    old_string = str(args.get("old_string", ""))
    new_string = str(args.get("new_string", ""))

    if sanitize_fn:
        old_string = sanitize_fn(old_string)
        new_string = sanitize_fn(new_string)

    is_new_file = (not old_string and not os.path.exists(file_path))
    if not is_new_file:
        ok, msg = _check_prior_read(file_path, require_full=True)
        if not ok:
            return msg, True

    if is_new_file:
        # empty old_string + new file: create it
        try:
            os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(new_string)
            return "Created new file: {}".format(file_path), False
        except Exception as e:
            return "patch error: creating file failed {}: {}".format(file_path, str(e)), True

    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            raw = f.read()
    except Exception as e:
        return "patch error: reading file failed {}: {}".format(file_path, str(e)), True

    # Detect line endings
    has_crlf = "\r\n" in raw
    content = raw.replace("\r\n", "\n")

    # Try exact match
    idx = content.find(old_string)
    used_fuzzy = False

    # Check for multiple occurrences (warn on exact but not fuzzy)
    if idx != -1:
        second = content.find(old_string, idx + len(old_string))
        if second != -1:
            return (
                "patch error: old_string matches multiple locations in {}. "
                "Include more surrounding context (5+ lines) to make it unique, "
                "or use the exact text from a read call.".format(file_path),
                True,
            )

    # Try fuzzy match
    if idx == -1:
        fuzzy_content = _fuzzy_normalize(content)
        fuzzy_old = _fuzzy_normalize(old_string)
        idx = fuzzy_content.find(fuzzy_old)
        if idx == -1:
            return (
                "patch error: old_string not found in {}. "
                "Use read to verify the file's EXACT content (whitespace, indentation, line endings). "
                "Include at least 3 lines of context around the change. "
                "The file may have changed since last read -- re-read before retrying.".format(
                    file_path
                ),
                True,
            )
        used_fuzzy = True

    new_content = content[:idx] + new_string + content[idx + len(old_string):]

    # Restore line endings
    if has_crlf:
        new_content = new_content.replace("\n", "\r\n")

    try:
        with open(file_path, "w", encoding="utf-8", newline="") as f:
            f.write(new_content)
        # Update write cache for TOCTOU: track that the model wrote this file,
        # so subsequent patches from the model don't trigger stale-mtime errors.
        try:
            apath = os.path.abspath(file_path)
            wstat = os.stat(apath)
            _write_cache[apath] = (wstat.st_mtime, wstat.st_size)
        except OSError:
            pass
    except Exception as e:
        return "patch error: writing file failed {}: {}".format(file_path, str(e)), True

    # Generate unified diff
    lines_a = content.splitlines(keepends=True)
    lines_b = new_content.splitlines(keepends=True)
    a_label = "a/" + file_path
    b_label = "b/" + file_path
    diff = "".join(difflib.unified_diff(lines_a, lines_b, fromfile=a_label, tofile=b_label))

    extra = ""
    if used_fuzzy:
        extra = " (fuzzy match)"

    return "Patch applied{}.\n\n{}".format(extra, diff), False

# -------------------------------------------------------------
#  python / reset  -- delegates to persistent subprocess
# -------------------------------------------------------------
def _python(args, cancel=None):
    code = args.get("code", "")
    if not code or not code.strip():
        return "python error: code parameter required and must be non-empty", True

    result, is_error = _send_to_repl({"type": "exec", "code": code})
    return result, is_error


def _python_reset(args):
    result, is_error = _send_to_repl({"type": "clear"})
    return result, is_error

# -------------------------------------------------------------
#  find  -- glob search sorted by mtime
# -------------------------------------------------------------
def _find(args, cwd=None):
    pattern = args.get("pattern", "*")
    root = _resolve_path(args.get("path", "."), cwd)
    _MAX_FIND = 500

    try:
        matches = glob.glob(os.path.join(root, "**", pattern), recursive=True)
    except Exception as e:
        return "find error: search failed '{}' in {}: {}".format(pattern, root, str(e)), True

    # Filter out ignored directories
    matches = [p for p in matches if not any(seg in _TOOL_IGNORE for seg in p.split(os.sep))]

    if not matches:
        return (
            "find: no files matching '{}' in {}. "
            "Try a broader pattern or check the directory with ls.".format(pattern, root),
            False,
        )

    # Sort by modification time, newest first
    matches.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    truncated = len(matches) > _MAX_FIND
    if truncated:
        matches = matches[:_MAX_FIND]

    out = "\n".join(matches)
    if truncated:
        out = "find: truncated at {} results (use a narrower pattern).\n".format(
            _MAX_FIND) + out
    return out, False

# -------------------------------------------------------------
#  grep  -- regex search in files
# -------------------------------------------------------------
def _grep(args, cwd=None):
    pattern = args.get("pattern", "")
    root = _resolve_path(args.get("path", "."), cwd)
    include = args.get("include", None)
    context = 3

    if not pattern:
        return "grep error: pattern parameter required", True

    try:
        regex = re.compile(pattern)
    except re.error as e:
        return (
            "grep error: invalid regex '{}': {}. "
            "Remember to escape backslashes (\\\\s for whitespace, \\\\d for digits).".format(
                pattern, str(e)
            ),
            True,
        )

    # Build include extension list
    globs = None
    if include:
        globs = []
        for g in include.split(","):
            g = g.strip()
            if not g.startswith("*"):
                g = "*" + g
            globs.append(g)

    sections = []  # per-file output blocks
    total_matches = 0
    truncated = False

    for cwd, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for filename in files:
            if filename.startswith("."):
                continue
            ext = os.path.splitext(filename)[1].lower()
            if ext in _BINARY_EXTS:
                continue
            if globs:
                matched = False
                for g in globs:
                    if glob.fnmatch.fnmatch(filename, g):
                        matched = True
                        break
                if not matched:
                    continue
            filepath = os.path.join(cwd, filename)
            try:
                with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                    lines = f.readlines()
            except Exception:
                continue

            # Find all matching line numbers (1-based)
            hit_lines = set()
            for i, line in enumerate(lines):
                if regex.search(line):
                    hit_lines.add(i + 1)

            if not hit_lines:
                continue

            # Build context ranges: each match gets +/- context lines
            want = set()
            for h in hit_lines:
                for offset in range(-context, context + 1):
                    ln = h + offset
                    if 1 <= ln <= len(lines):
                        want.add(ln)

            # Emit contiguous blocks with --- separators
            ordered = sorted(want)
            blocks = []
            block = []
            block_start = ordered[0]
            for ln in ordered:
                if block and ln > block[-1] + 1:
                    blocks.append((block_start, block))
                    block = []
                    block_start = ln
                block.append(ln)
            if block:
                blocks.append((block_start, block))

            file_out = [filepath]
            for start_ln, group in blocks:
                file_out.append("---")
                for ln in group:
                    marker = ">" if ln in hit_lines else " "
                    raw = lines[ln - 1].rstrip("\n")
                    if len(raw) > 500:
                        raw = raw[:500] + "..."
                    file_out.append("{} {}: {}".format(marker, ln, raw))
            sections.append("\n".join(file_out))
            total_matches += len(hit_lines)
            if total_matches >= 100:
                truncated = True
                break
        if total_matches >= 100:
            truncated = True
            break

    if truncated:
        sections.append("grep: truncated at 100 matches. Narrow scope with path/include or use a simpler pattern.")

    if not sections:
        return (
            "grep: no matches for '{}' in {}. "
            "Try a simpler pattern or verify the file exists with read.".format(pattern, root),
            False,
        )

    return "\n\n".join(sections), False

# -------------------------------------------------------------
#  ls  -- directory listing
# -------------------------------------------------------------
def _ls(args, cwd=None):
    path = _resolve_path(args.get("path", "."), cwd)

    if not os.path.isdir(path):
        return "ls error: {} is not a directory. Use read to view file contents.".format(path), True

    try:
        entries = os.listdir(path)
        entries.sort()
        lines = []
        for e in entries:
            if e.endswith("~") or e.startswith("#"):
                continue
            if e in _TOOL_IGNORE:
                continue
            full = os.path.join(path, e)
            try:
                if os.path.isdir(full):
                    lines.append(e + "/")
                else:
                    lines.append(e)
            except OSError:
                lines.append(e)
        return "\n".join(lines), False
    except Exception as e:
        return "ls error: listing directory failed {}: {}".format(path, str(e)), True
