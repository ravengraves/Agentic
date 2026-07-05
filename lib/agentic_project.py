# agentic_project.py
# -------------------------------------------------------------
#  Shared project-local instruction helpers.
#  Pure Python, no Sublime dependencies.
# -------------------------------------------------------------
import os


AGENTS_MD_NAMES = ("AGENTS.md", "Agents.md", "agents.md")
AGENTS_MD_MARKER = "Project AGENTS.md instructions:"


def find_agents_md(folder):
    if not folder:
        return None
    for name in AGENTS_MD_NAMES:
        path = os.path.join(folder, name)
        if os.path.isfile(path):
            return path
    return None


def load_agents_md(folder):
    path = find_agents_md(folder)
    if not path:
        return None
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except Exception:
        return None


def inject_agents_md(messages, agents_md):
    if not agents_md or messages is None:
        return False
    if not messages or messages[0].get("role") not in ("developer", "system"):
        messages.insert(0, {"role": "developer", "content": ""})
    content = messages[0].get("content", "")
    if AGENTS_MD_MARKER in content:
        return False
    messages[0]["content"] = (
        content
        + "\n\n"
        + AGENTS_MD_MARKER
        + "\n\n"
        + "`````````\n"
        + agents_md
        + "\n`````````"
    )
    return True


def apply_agents_md(messages, folder, enabled=True):
    if not enabled:
        return False
    return inject_agents_md(messages, load_agents_md(folder))


def project_folder_for_path(file_name, folders):
    folders = folders or []
    if file_name:
        file_name = os.path.abspath(file_name)
        best = None
        for folder in folders:
            folder = os.path.abspath(folder)
            try:
                common = os.path.commonpath([file_name, folder])
            except Exception:
                continue
            if common == folder and (best is None or len(folder) > len(best)):
                best = folder
        if best:
            return best
    return folders[0] if folders else None
