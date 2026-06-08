import os
import json
import tiktoken
from openai import OpenAI
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
import subprocess
import sys
import uuid
from datetime import datetime

load_dotenv()
client = OpenAI()
console = Console()


HISTORY_DIR = ".history"
os.makedirs(HISTORY_DIR, exist_ok=True)

# gpt-4o-mini and gpt-4o share the same tokenizer ("o200k_base")
encoding = tiktoken.get_encoding("o200k_base")

# USD per 1M tokens — update if OpenAI changes pricing
PRICE_INPUT_PER_1M = 0.15
PRICE_OUTPUT_PER_1M = 0.60

usage = {"input": 0, "output": 0}

CONTEXT_LIMIT = 8000
COMPACT_THRESHOLD = 0.80


def normalize_messages(messages):
    # The SDK returns ChatCompletionMessage objects; JSON serialization requires plain dicts.
    clean = []
    for m in messages:
        if isinstance(m, dict):
            clean.append(m)
        else:
            entry = {"role": getattr(m, "role", "assistant")}
            content = getattr(m, "content", None)
            if content is not None:
                entry["content"] = content
            # Preserve tool_calls so resumed sessions can replay the full turn correctly.
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

def save_session(session_id, messages):
    path = os.path.join(HISTORY_DIR, f"{session_id}.json")
    with open(path, "w") as f:
        json.dump(normalize_messages(messages), f, indent=2)

def load_session(session_id):
    path = os.path.join(HISTORY_DIR, f"{session_id}.json")
    if not os.path.exists(path):
        console.print(f"[red]No session found with id: {session_id}[/red]")
        sys.exit(1)
    with open(path, "r") as f:
        return json.load(f)

def list_sessions():
    files = sorted(f for f in os.listdir(HISTORY_DIR) if f.endswith(".json"))
    if not files:
        console.print("[dim]No saved sessions yet.[/dim]")
        return
    console.print("[bold]Saved sessions:[/bold]")
    for f in files:
        console.print(f"  {f[:-5]}")


def count_tokens(messages):
    total = 0
    for m in messages:
        # Handle both plain dicts and ChatCompletionMessage objects.
        if isinstance(m, dict):
            content = m.get("content")
        else:
            content = getattr(m, "content", None)

        if isinstance(content, str):
            total += len(encoding.encode(content))
    return total


def compact_history(messages):
    system_msg = messages[0]

    convo_text = ""
    for m in messages[1:]:
        if isinstance(m, dict):
            role = m.get("role", "?")
            content = m.get("content")
        else:
            role = getattr(m, "role", "?")
            content = getattr(m, "content", None)
        if isinstance(content, str) and content.strip():
            convo_text += f"{role}: {content}\n"

    console.print("[bold yellow]⟳ context near limit — compacting conversation...[/bold yellow]")

    summary_resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": (
                "Summarize this agent conversation concisely but completely. "
                "PRESERVE: the user's current goal/task, any file names and paths, "
                "decisions made, and what has been done so far vs. what still remains. "
                "Write it so the agent can seamlessly continue the task."
            )},
            {"role": "user", "content": convo_text},
        ],
    )
    summary = summary_resp.choices[0].message.content

    # Inject the summary as a user message so the next turn starts with a valid role sequence.
    new_messages = [
        system_msg,
        {"role": "user", "content": f"[Summary of earlier conversation so far]\n{summary}\n\n(Continue helping based on this context.)"},
    ]
    return new_messages

WORKSPACE = os.path.abspath("workspace")
os.makedirs(WORKSPACE, exist_ok=True)


MAX_TOOL_RESULT_CHARS = 5000

def truncate_result(text):
    if len(text) <= MAX_TOOL_RESULT_CHARS:
        return text

    half = MAX_TOOL_RESULT_CHARS // 2
    head = text[:half]
    tail = text[-half:]
    dropped = len(text) - MAX_TOOL_RESULT_CHARS
    # Explicit marker so the model knows content was removed and can re-fetch if needed.
    marker = f"\n\n[... truncated {dropped} characters from the middle ...]\n\n"
    return head + marker + tail


def write_file(path, content):
    with open(path, "w") as f:
        f.write(content)
    return f"Wrote {len(content)} bytes to {path}"

def read_file(path):
    with open(path, "r") as f:
        return f.read()

def run_bash(command):
    # Runs inside a disposable Alpine container with no network access and a mounted workspace.
    docker_cmd = [
        "docker", "run", "--rm",
        "--network=none",
        "-v", f"{WORKSPACE}:/work",
        "-w", "/work",
        "alpine",
        "sh", "-c", command,
    ]
    try:
        result = subprocess.run(
            docker_cmd, capture_output=True, text=True, timeout=30
        )
        output = result.stdout + result.stderr
        return output if output else "(no output)"
    except subprocess.TimeoutExpired:
        return "ERROR: command timed out after 30 seconds."


def str_replace(path, old_str, new_str):
    with open(path, "r") as f:
        content = f.read()

    count = content.count(old_str)

    if count == 0:
        return f"ERROR: old_str not found in {path}. Nothing was changed."
    if count > 1:
        return (f"ERROR: old_str appears {count} times in {path}. "
                f"It must be unique. Add more surrounding context to make it match only once.")

    new_content = content.replace(old_str, new_str)
    with open(path, "w") as f:
        f.write(new_content)
    return f"Successfully replaced text in {path}."


tools = [
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write text content to a file at the given path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "The file path to write to"},
                    "content": {"type": "string", "description": "The text to write"},
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
                    "path": {"type": "string", "description": "The file path to read"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_bash",
            "description": "Run a bash/shell command and return its output. Use for listing files, running scripts, etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The shell command to run"},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "str_replace",
            "description": ("Replace an exact string in a file with new text. "
                            "old_str must appear EXACTLY once in the file. "
                            "Include enough surrounding context to make it unique. "
                            "Prefer this over write_file for editing existing files."),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "The file to edit"},
                    "old_str": {"type": "string", "description": "The exact text to find (must be unique)"},
                    "new_str": {"type": "string", "description": "The text to replace it with"},
                },
                "required": ["path", "old_str", "new_str"],
            },
        },
    },
]


available_tools = {
    "write_file": write_file,
    "read_file": read_file,
    "run_bash": run_bash,
    "str_replace": str_replace,
}

messages = [
    {"role": "system", "content": "You are a helpful coding assistant with file tools. All work happens inside the 'workspace' directory. When using run_bash, paths are relative to the workspace (mounted as /work in a sandbox). When reading or writing files directly, prefix paths with 'workspace/'."}
]


if "--list-sessions" in sys.argv:
    list_sessions()
    sys.exit(0)

if "--resume" in sys.argv:
    idx = sys.argv.index("--resume")
    session_id = sys.argv[idx + 1]
    messages = load_session(session_id)
    console.print(f"[green]Resumed session: {session_id}[/green]")
else:
    session_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    console.print(f"[dim]New session: {session_id}[/dim]")

console.print(Panel("Mini Claude Code — v0.2 (type 'exit' to quit)", style="bold green"))


global session_input_tokens, session_output_tokens

while True:
    user_input = console.input("[bold cyan]you > [/bold cyan]")
    if user_input.strip().lower() in ("exit", "quit"):
        console.print("[dim]bye[/dim]")
        break

    messages.append({"role": "user", "content": user_input})

    loop_count = 0
    while True:
        loop_count += 1
        if loop_count > 15:
            console.print("[bold red]⚠ loop cap reached (15 iterations) — stopping to prevent runaway execution.[/bold red]")
            break

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )

        msg = response.choices[0].message

        if msg.tool_calls:
            messages.append(msg)

            for tool_call in msg.tool_calls:
                fn_name = tool_call.function.name
                fn_args = json.loads(tool_call.function.arguments)

                console.print(f"[yellow]→ calling {fn_name}({fn_args})[/yellow]")

                fn = available_tools[fn_name]
                result = fn(**fn_args)

                console.print(f"[dim]  result: {result[:200]}...[/dim]" if len(result) > 200 else f"[dim]  result: {result}[/dim]")

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": truncate_result(result),
                })

            continue

        else:
            console.print(f"[bold magenta]ai  > [/bold magenta]{msg.content}")
            messages.append({"role": "assistant", "content": msg.content})

            input_toks = count_tokens(messages)
            output_toks = len(encoding.encode(msg.content or ""))
            usage["input"] += input_toks
            usage["output"] += output_toks

            cost = (usage["input"] / 1_000_000 * PRICE_INPUT_PER_1M
                    + usage["output"] / 1_000_000 * PRICE_OUTPUT_PER_1M)

            console.print(
                f"[dim]  tokens this turn: ~{input_toks} in / {output_toks} out | "
                f"session cost: ~${cost:.4f}[/dim]"
            )

            # Compact only at a resting point (after a final answer) to avoid
            # cutting context mid-tool-call, and only when history is large enough
            # that summarization yields a meaningful reduction.
            current_tokens = count_tokens(messages)
            if current_tokens > CONTEXT_LIMIT * COMPACT_THRESHOLD and len(messages) > 4:
                messages = compact_history(messages)
                after = count_tokens(messages)
                console.print(f"[green]  compacted: {current_tokens} → {after} tokens[/green]")

            save_session(session_id, messages)
            break
