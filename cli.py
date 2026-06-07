import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
import subprocess  
load_dotenv()
client = OpenAI()
console = Console()

# ───────────────────────────────────────────────────────────
# 1. THE ACTUAL TOOL (a normal Python function)
# This is what really runs on your machine when the AI asks.
# ───────────────────────────────────────────────────────────
def write_file(path, content):
    with open(path, "w") as f:
        f.write(content)
    return f"Wrote {len(content)} bytes to {path}"

def read_file(path):
    with open(path, "r") as f:
        return f.read()


def run_bash(command):
    # Run a shell command and capture its output.
    # capture_output grabs stdout/stderr; text=True returns strings not bytes.
    result = subprocess.run(
        command, shell=True, capture_output=True, text=True, timeout=30
    )
    # Combine normal output and error output so the AI sees everything.
    output = result.stdout + result.stderr
    return output if output else "(no output)"

def str_replace(path, old_str, new_str):
    # Read the current file contents
    with open(path, "r") as f:
        content = f.read()

    # Count how many times old_str appears — this is the precision check
    count = content.count(old_str)

    if count == 0:
        return f"ERROR: old_str not found in {path}. Nothing was changed."
    if count > 1:
        return (f"ERROR: old_str appears {count} times in {path}. "
                f"It must be unique. Add more surrounding context to make it match only once.")

    # Exactly one match — safe to replace
    new_content = content.replace(old_str, new_str)
    with open(path, "w") as f:
        f.write(new_content)
    return f"Successfully replaced text in {path}."


# ───────────────────────────────────────────────────────────
# 2. THE TOOL MENU (what we tell the AI it's allowed to ask for)
# This describes the tool: its name, purpose, and arguments.
# The AI reads this to know HOW to request the tool.
# ───────────────────────────────────────────────────────────
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




# A lookup so we can find the real function by its name string.
available_tools = {
    "write_file": write_file,
    "read_file": read_file,
    "run_bash": run_bash,
    "str_replace": str_replace,
}

messages = [
    {"role": "system", "content": "You are a helpful coding assistant with file tools."}
]

console.print(Panel("Mini Claude Code — v0.2 (type 'exit' to quit)", style="bold green"))

while True:
    user_input = console.input("[bold cyan]you > [/bold cyan]")
    if user_input.strip().lower() in ("exit", "quit"):
        console.print("[dim]bye[/dim]")
        break

    messages.append({"role": "user", "content": user_input})

    # ───────────────────────────────────────────────────────
    # 3. THE INNER LOOP (the heart of the agent)
    # Keep calling the API until the AI stops requesting tools.
    # ───────────────────────────────────────────────────────
    while True:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=tools,          # <-- hand over the tool menu
            tool_choice="auto",   # <-- let the AI decide: text OR tool call
        )

        msg = response.choices[0].message

        # Did the AI request any tools?
        if msg.tool_calls:
            # IMPORTANT: add the AI's tool-request message to history first.
            messages.append(msg)

            # The AI can request multiple tools at once — handle each.
            for tool_call in msg.tool_calls:
                fn_name = tool_call.function.name
                fn_args = json.loads(tool_call.function.arguments)  # arguments arrive as a JSON string

                console.print(f"[yellow]→ calling {fn_name}({fn_args})[/yellow]")

                # Run the REAL function
                fn = available_tools[fn_name]
                result = fn(**fn_args)

                console.print(f"[dim]  result: {result}[/dim]")

                # Send the result back to the AI as a 'tool' message.
                # tool_call_id links this result to the specific request.
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(result),
                })

            # Loop again so the AI can see the results and decide next step.
            continue

        else:
            # No tool calls = the AI is done, this is its final text answer.
            console.print(f"[bold magenta]ai  > [/bold magenta]{msg.content}")
            messages.append({"role": "assistant", "content": msg.content})
            break  # exit inner loop, go back to waiting for user input