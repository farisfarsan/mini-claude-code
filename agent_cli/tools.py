import os
import subprocess

from agent_cli.config import WORKSPACE, MAX_TOOL_RESULT_CHARS

os.makedirs(WORKSPACE, exist_ok=True)


def truncate_result(text: str) -> str:
    if len(text) <= MAX_TOOL_RESULT_CHARS:
        return text
    half = MAX_TOOL_RESULT_CHARS // 2
    dropped = len(text) - MAX_TOOL_RESULT_CHARS
    marker = f"\n\n[... truncated {dropped} characters from the middle ...]\n\n"
    return text[:half] + marker + text[-half:]


def _workspace_path(path: str) -> str:
    if path.startswith("workspace/") or path.startswith("workspace\\"):
        path = path[len("workspace/"):]
    full = os.path.normpath(os.path.join(WORKSPACE, path))
    if not (full == WORKSPACE or full.startswith(WORKSPACE + os.sep)):
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
        "--read-only",
        "--memory=256m",
        "--pids-limit=128",
        "--cpus=1",
        "--tmpfs=/tmp",
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
]

TOOL_MAP = {
    "write_file": write_file,
    "read_file": read_file,
    "run_bash": run_bash,
    "str_replace": str_replace,
}
