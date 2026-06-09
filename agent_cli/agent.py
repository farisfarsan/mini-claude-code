import json
import os

from openai import OpenAI
from rich.console import Console

from agent_cli.config import PRICE_INPUT_PER_1M, PRICE_OUTPUT_PER_1M
from agent_cli.context import compact_history, count_tokens, should_compact
from agent_cli.session import save_session
from agent_cli.tools import TOOL_MAP, TOOL_SCHEMAS, truncate_result

console = Console()
client = OpenAI()

_PROMPT_PATH = os.path.join(os.path.dirname(__file__), "..", "prompts", "system.md")
with open(_PROMPT_PATH) as _f:
    SYSTEM_PROMPT = _f.read().strip()

MAX_LOOP_ITERATIONS = 15

# Tools that can change the system — these prompt the user before running.
# Read-only tools (read_file, list_files, search_files) are never gated.
REQUIRES_APPROVAL = {"run_bash", "write_file", "str_replace", "multi_edit"}

# Tools the user has chosen to "always" allow, remembered for this session only.
_approved_tools: set[str] = set()


def check_permission(fn_name: str, approvals: set, ask) -> bool:
    """Decide whether a tool call may run.

    Read-only (or already 'always'-approved) tools pass straight through.
    Destructive tools ask the user. `ask` is a callable returning the user's
    raw answer string — it's injected so this logic can be tested without a
    real terminal prompt.
    """
    if fn_name not in REQUIRES_APPROVAL or fn_name in approvals:
        return True
    answer = ask().strip().lower()
    if answer in ("a", "always"):
        approvals.add(fn_name)
        return True
    if answer in ("n", "no"):
        return False
    return True  # default (y / yes / Enter) = allow once


def run_turn(session_id: str, messages: list, usage: dict) -> list:
    for _ in range(MAX_LOOP_ITERATIONS):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",
        )
        msg = response.choices[0].message

        if msg.tool_calls:
            messages.append(msg)
            for tool_call in msg.tool_calls:
                fn_name = tool_call.function.name
                fn_args = json.loads(tool_call.function.arguments)
                console.print(f"[yellow]→ calling {fn_name}({fn_args})[/yellow]")

                allowed = check_permission(
                    fn_name,
                    _approved_tools,
                    ask=lambda: console.input(
                        "  Allow? [Y]es / [n]o / [a]lways this session: ", markup=False
                    ),
                )
                if not allowed:
                    console.print("[red]  ✗ denied by user[/red]")
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": (
                            "DENIED by the user. Do not retry this exact action; "
                            "ask how to proceed or try a different approach."
                        ),
                    })
                    continue

                try:
                    result = TOOL_MAP[fn_name](**fn_args)
                except Exception as e:  # noqa: BLE001 — surface any tool error to the model
                    result = f"ERROR: {fn_name} failed: {e}"
                console.print(
                    f"[dim]  result: {result[:200]}...[/dim]"
                    if len(result) > 200
                    else f"[dim]  result: {result}[/dim]"
                )
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": truncate_result(result),
                })
            continue

        console.print(f"[bold magenta]ai  > [/bold magenta]{msg.content}")
        messages.append({"role": "assistant", "content": msg.content})

        input_toks = response.usage.prompt_tokens
        output_toks = response.usage.completion_tokens
        usage["input"] += input_toks
        usage["output"] += output_toks

        cost = (
            usage["input"] / 1_000_000 * PRICE_INPUT_PER_1M
            + usage["output"] / 1_000_000 * PRICE_OUTPUT_PER_1M
        )
        console.print(
            f"[dim]  tokens this turn: ~{input_toks} in / {output_toks} out | "
            f"session cost: ~${cost:.4f}[/dim]"
        )

        if should_compact(messages):
            before = count_tokens(messages)
            messages = compact_history(messages, client, usage, console)
            after = count_tokens(messages)
            console.print(f"[green]  compacted: {before} → {after} tokens[/green]")

        save_session(session_id, messages)
        return messages

    console.print(
        f"[bold red]⚠ loop cap reached ({MAX_LOOP_ITERATIONS} iterations) — "
        "stopping to prevent runaway execution.[/bold red]"
    )
    return messages