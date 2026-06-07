import os
import json
import tiktoken
from openai import OpenAI
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
import subprocess  
load_dotenv()
client = OpenAI()
console = Console()

# gpt-4o-mini uses the same encoding as gpt-4o ("o200k_base").
encoding = tiktoken.get_encoding("o200k_base")

# gpt-4o-mini pricing (USD per 1 MILLION tokens) — check current prices, these are approximate.
PRICE_INPUT_PER_1M = 0.15    # cost of tokens we SEND
PRICE_OUTPUT_PER_1M = 0.60   # cost of tokens we RECEIVE

# Running totals across the whole session.
usage = {"input": 0, "output": 0}

CONTEXT_LIMIT = 800
COMPACT_THRESHOLD = 0.80

def count_tokens(messages):
    total = 0
    for m in messages:
        # Messages are a mix: dicts (our messages) and ChatCompletionMessage
        # objects (the AI's tool-request turns). Handle both.
        if isinstance(m, dict):
            content = m.get("content")
        else:
            content = getattr(m, "content", None)  # object attribute access

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
        if isinstance(content, str):
            convo_text += f"{role}: {content}\n"

    console.print("[bold yellow]⟳ context near limit — compacting conversation...[/bold yellow]")

    summary_resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Summarize the following agent conversation concisely. Preserve key facts, decisions, file names, and any task state needed to continue the work. Be compact."},
            {"role": "user", "content": convo_text},
        ],
    )
    summary = summary_resp.choices[0].message.content

    return [
        system_msg,
        {"role": "user", "content": f"[Summary of earlier conversation]\n{summary}"},
    ]


WORKSPACE = os.path.abspath("workspace")
os.makedirs(WORKSPACE, exist_ok=True)



MAX_TOOL_RESULT_CHARS = 5000  # cap on how much of a tool result we keep

def truncate_result(text):
    # If it's already small enough, leave it alone.
    if len(text) <= MAX_TOOL_RESULT_CHARS:
        return text

    # Otherwise keep the head and the tail, drop the middle.
    half = MAX_TOOL_RESULT_CHARS // 2
    head = text[:half]
    tail = text[-half:]
    dropped = len(text) - MAX_TOOL_RESULT_CHARS
    # The marker tells the MODEL that content was removed, so it can re-fetch
    # more specifically if it needs the missing part.
    marker = f"\n\n[... truncated {dropped} characters from the middle ...]\n\n"
    return head + marker + tail


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
    # Run the command inside a disposable Alpine container.
    # Breakdown of the docker flags:
    #   run --rm           -> start a container, delete it when done
    #   --network=none     -> no internet access (stronger isolation)
    #   -v WORKSPACE:/work -> mount our workspace folder as /work inside the container
    #   -w /work           -> set the working directory to /work
    #   alpine             -> the tiny Linux image to use
    #   sh -c "command"    -> run the AI's command inside a shell
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
    {"role": "system", "content": "You are a helpful coding assistant with file tools. All work happens inside the 'workspace' directory. When using run_bash, paths are relative to the workspace (mounted as /work in a sandbox). When reading or writing files directly, prefix paths with 'workspace/'."}
]

console.print(Panel("Mini Claude Code — v0.2 (type 'exit' to quit)", style="bold green"))


global session_input_tokens, session_output_tokens

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
        if count_tokens(messages) >= CONTEXT_LIMIT * COMPACT_THRESHOLD:
            messages = compact_history(messages)

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

                console.print(f"[dim]  result: {result[:200]}...[/dim]" if len(result) > 200 else f"[dim]  result: {result}[/dim]")

                # Send the result back to the AI as a 'tool' message.
                # tool_call_id links this result to the specific request.
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                })

            # Loop again so the AI can see the results and decide next step.
            continue

        else:
            console.print(f"[bold magenta]ai  > [/bold magenta]{msg.content}")
            messages.append({"role": "assistant", "content": msg.content})

            # --- token + cost accounting ---
            input_toks = count_tokens(messages)          # everything we've sent
            output_toks = len(encoding.encode(msg.content or ""))  # this reply
            usage["input"] += input_toks
            usage["output"] += output_toks

            cost = (usage["input"] / 1_000_000 * PRICE_INPUT_PER_1M
                    + usage["output"] / 1_000_000 * PRICE_OUTPUT_PER_1M)

            console.print(
                f"[dim]  tokens this turn: ~{input_toks} in / {output_toks} out | "
                f"session cost: ~${cost:.4f}[/dim]"
            )

            # --- compaction check ---
            current_tokens = count_tokens(messages)
            if current_tokens > CONTEXT_LIMIT * COMPACT_THRESHOLD:
                messages = compact_history(messages)
                after = count_tokens(messages)
                console.print(f"[green]  compacted: {current_tokens} → {after} tokens[/green]")

            break