import os
import re
import shutil
import subprocess

from agent_cli.config import WORKSPACE, MAX_TOOL_RESULT_CHARS

os.makedirs(WORKSPACE, exist_ok=True)

# Directories we never want to list or search through — noise for the agent.
_IGNORE_DIRS = {
    ".git", "__pycache__", ".venv", "venv", "node_modules",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", ".idea", ".vscode",
}


def truncate_result(text: str) -> str:
    if len(text) <= MAX_TOOL_RESULT_CHARS:
        return text
    half = MAX_TOOL_RESULT_CHARS // 2
    dropped = len(text) - MAX_TOOL_RESULT_CHARS
    marker = f"\n\n[... truncated {dropped} characters from the middle ...]\n\n"
    return text[:half] + marker + text[-half:]


def _workspace_path(path: str) -> str:
    if path in ("", "."):
        path = ""
    if path.startswith("workspace/") or path.startswith("workspace\\"):
        path = path[len("workspace/"):]
    # realpath resolves symlinks BEFORE we check containment, so a symlink
    # inside the workspace that points outside it can no longer be used to escape.
    full = os.path.realpath(os.path.join(WORKSPACE, path))
    ws = os.path.realpath(WORKSPACE)
    if not (full == ws or full.startswith(ws + os.sep)):
        raise ValueError(f"Path '{path}' escapes the workspace directory.")
    return full


def write_file(path: str, content: str) -> str:
    full = _workspace_path(path)
    parent = os.path.dirname(full)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(full, "w") as f:
        f.write(content)
    return f"Wrote {len(content)} bytes to {full}"


def read_file(path: str) -> str:
    full = _workspace_path(path)
    try:
        with open(full, "r") as f:
            return f.read()
    except FileNotFoundError:
        return f"ERROR: file not found: {full}"
    except OSError as e:
        return f"ERROR: could not read {full}: {e}"


def run_bash(command: str) -> str:
    docker_cmd = [
        "docker", "run", "--rm",
        "--network=none",
        "-v", f"{WORKSPACE}:/work",
        "-w", "/work",
        "python:3.11-alpine",
        "sh", "-c", command,
    ]
    try:
        result = subprocess.run(docker_cmd, capture_output=True, text=True, timeout=30)
        output = result.stdout + result.stderr
        return output if output else "(no output)"
    except subprocess.TimeoutExpired:
        return "ERROR: command timed out after 30 seconds."
    except FileNotFoundError:
        return "ERROR: Docker is not installed or not found in PATH."


def str_replace(path: str, old_str: str, new_str: str) -> str:
    full = _workspace_path(path)
    try:
        with open(full, "r") as f:
            content = f.read()
    except FileNotFoundError:
        return f"ERROR: file not found: {full}"
    except OSError as e:
        return f"ERROR: could not read {full}: {e}"

    count = content.count(old_str)
    if count == 0:
        return f"ERROR: old_str not found in {full}. Nothing was changed."
    if count > 1:
        return (
            f"ERROR: old_str appears {count} times in {full}. "
            "It must be unique. Add more surrounding context to make it match only once."
        )
    with open(full, "w") as f:
        f.write(content.replace(old_str, new_str, 1))
    return f"Successfully replaced text in {full}."


def _rel(full: str) -> str:
    """Path relative to the workspace, for clean agent-facing output."""
    ws = os.path.realpath(WORKSPACE)
    return os.path.relpath(full, ws)


def list_files(path: str = ".", recursive: bool = True, max_entries: int = 200) -> str:
    root = _workspace_path(path)
    if not os.path.exists(root):
        return f"ERROR: path not found: {path}"
    if os.path.isfile(root):
        return _rel(root)

    lines: list[str] = []
    if recursive:
        for dirpath, dirnames, filenames in os.walk(root):
            # prune ignored dirs in place so os.walk doesn't descend into them
            dirnames[:] = sorted(d for d in dirnames if d not in _IGNORE_DIRS)
            for fn in sorted(filenames):
                lines.append(_rel(os.path.join(dirpath, fn)))
                if len(lines) >= max_entries:
                    lines.append(f"... (stopped at {max_entries} entries — narrow the path)")
                    return "\n".join(lines)
    else:
        for entry in sorted(os.listdir(root)):
            if entry in _IGNORE_DIRS:
                continue
            full = os.path.join(root, entry)
            lines.append(entry + ("/" if os.path.isdir(full) else ""))
    return "\n".join(lines) if lines else "(empty directory)"


def search_files(
    pattern: str,
    path: str = ".",
    glob: str | None = None,
    ignore_case: bool = False,
    max_results: int = 100,
) -> str:
    root = _workspace_path(path)
    if not os.path.exists(root):
        return f"ERROR: path not found: {path}"

    rg = shutil.which("rg")
    if rg:
        cmd = [rg, "--line-number", "--no-heading", "--color", "never", "--max-count", "50"]
        if ignore_case:
            cmd.append("--ignore-case")
        if glob:
            cmd += ["--glob", glob]
        cmd += ["--", pattern, root]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        except subprocess.TimeoutExpired:
            return "ERROR: search timed out after 20 seconds."
        if res.returncode not in (0, 1):  # rg returns 1 for "no matches", which is fine
            return f"ERROR: search failed: {res.stderr.strip()}"
        out = res.stdout.replace(root + os.sep, "").replace(os.path.realpath(WORKSPACE) + os.sep, "")
        lines = out.splitlines()
        if not lines:
            return f"No matches for '{pattern}'."
        if len(lines) > max_results:
            extra = len(lines) - max_results
            lines = lines[:max_results] + [f"... ({extra} more matches — refine your pattern)"]
        return "\n".join(lines)

    # Pure-Python fallback when ripgrep isn't installed.
    flags = re.IGNORECASE if ignore_case else 0
    try:
        rx = re.compile(pattern, flags)
    except re.error as e:
        return f"ERROR: invalid regex: {e}"
    results: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _IGNORE_DIRS]
        for fn in filenames:
            full = os.path.join(dirpath, fn)
            try:
                with open(full, "r", errors="ignore") as f:
                    for i, line in enumerate(f, 1):
                        if rx.search(line):
                            results.append(f"{_rel(full)}:{i}:{line.rstrip()}")
                            if len(results) >= max_results:
                                results.append("... (more matches — refine your pattern)")
                                return "\n".join(results)
            except OSError:
                continue
    return "\n".join(results) if results else f"No matches for '{pattern}'."


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write text content to a file at the given path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to write to"},
                    "content": {"type": "string", "description": "Text content to write"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read and return the text contents of a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to read"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_bash",
            "description": "Run a shell command inside a sandboxed Docker container and return its output.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to run"},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "str_replace",
            "description": (
                "Replace an exact string in a file with new text. "
                "old_str must appear EXACTLY once — add surrounding context to make it unique. "
                "Prefer this over write_file when editing existing files."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File to edit"},
                    "old_str": {"type": "string", "description": "Exact text to find (must be unique)"},
                    "new_str": {"type": "string", "description": "Replacement text"},
                },
                "required": ["path", "old_str", "new_str"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": (
                "List files in the workspace so you can see what exists before reading or editing. "
                "Use this first when exploring an unfamiliar project. Skips noise like .git and __pycache__."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Folder to list. Defaults to the workspace root."},
                    "recursive": {"type": "boolean", "description": "List all nested files (default true)."},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": (
                "Search file contents across the workspace for a regex pattern (like grep/ripgrep). "
                "Use this to find where something is defined or used — e.g. a function, a config key, "
                "a database connection. Returns matching lines as 'path:line:text'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Regex or plain text to search for"},
                    "path": {"type": "string", "description": "Folder to search in. Defaults to the workspace root."},
                    "glob": {"type": "string", "description": "Optional file filter, e.g. '*.py'"},
                    "ignore_case": {"type": "boolean", "description": "Case-insensitive search (default false)."},
                },
                "required": ["pattern"],
            },
        },
    },
]

TOOL_MAP = {
    "write_file": write_file,
    "read_file": read_file,
    "run_bash": run_bash,
    "str_replace": str_replace,
    "list_files": list_files,
    "search_files": search_files,
}