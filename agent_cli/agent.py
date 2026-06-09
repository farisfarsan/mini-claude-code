import json
import os
import time

from openai import APIConnectionError, OpenAI, RateLimitError
from rich.console import Console

from agent_cli.config import MODEL, PRICE_INPUT_PER_1M, PRICE_OUTPUT_PER_1M
from agent_cli.context import compact_history, count_tokens, should_compact
from agent_cli.session import save_session
from agent_cli.tools import TOOL_MAP, TOOL_SCHEMAS, truncate_result

console = Console()
client = OpenAI()

_PROMPT_PATH = os.path.join(os.path.dirname(__file__), "..", "prompts", "system.md")
with open(_PROMPT_PATH) as _f:
    SYSTEM_PROMPT = _f.read().strip()

MAX_LOOP_ITERATIONS = 15


def _chat(**kwargs):
    for attempt in range(4):
        try:
            return client.chat.completions.create(**kwargs)
        except (RateLimitError, APIConnectionError):
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


def run_turn(session_id: str, messages: list, usage: dict) -> list:
    for _ in range(MAX_LOOP_ITERATIONS):
        response = _chat(
            model=MODEL,
            messages=messages,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",
        )
        msg = response.choices[0].message
        usage["input"] += response.usage.prompt_tokens
        usage["output"] += response.usage.completion_tokens

        if msg.tool_calls:
            messages.append(msg)
            for tool_call in msg.tool_calls:
                fn_name = tool_call.function.name
                fn_args = json.loads(tool_call.function.arguments)
                console.print(f"[yellow]→ calling {fn_name}({fn_args})[/yellow]")
                result = TOOL_MAP[fn_name](**fn_args)
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

        cost = (
            usage["input"] / 1_000_000 * PRICE_INPUT_PER_1M
            + usage["output"] / 1_000_000 * PRICE_OUTPUT_PER_1M
        )
        console.print(
            f"[dim]  tokens this turn: ~{response.usage.prompt_tokens} in / "
            f"{response.usage.completion_tokens} out | "
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
