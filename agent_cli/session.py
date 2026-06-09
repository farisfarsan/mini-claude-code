import json
import os
import sys

from rich.console import Console

from agent_cli.config import SESSIONS_DIR

console = Console()

os.makedirs(SESSIONS_DIR, exist_ok=True)


def normalize_messages(messages: list) -> list:
    clean = []
    for m in messages:
        if isinstance(m, dict):
            clean.append(m)
            continue
        entry = {"role": getattr(m, "role", "assistant")}
        content = getattr(m, "content", None)
        if content is not None:
            entry["content"] = content
        tool_calls = getattr(m, "tool_calls", None)
        if tool_calls:
            entry["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in tool_calls
            ]
        clean.append(entry)
    return clean


def save_session(session_id: str, messages: list) -> None:
    path = os.path.join(SESSIONS_DIR, f"{session_id}.json")
    with open(path, "w") as f:
        json.dump(normalize_messages(messages), f, indent=2)


def load_session(session_id: str) -> list:
    path = os.path.join(SESSIONS_DIR, f"{session_id}.json")
    if not os.path.exists(path):
        console.print(f"[red]No session found: {session_id}[/red]")
        sys.exit(1)
    with open(path, "r") as f:
        return json.load(f)


def list_sessions() -> None:
    files = sorted(f for f in os.listdir(SESSIONS_DIR) if f.endswith(".json"))
    if not files:
        console.print("[dim]No saved sessions yet.[/dim]")
        return
    console.print("[bold]Saved sessions:[/bold]")
    for f in files:
        console.print(f"  {f[:-5]}")
