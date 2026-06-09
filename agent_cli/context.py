from agent_cli.config import encoding, CONTEXT_LIMIT, COMPACT_THRESHOLD


def count_tokens(messages: list) -> int:
    total = 0
    for m in messages:
        content = m.get("content") if isinstance(m, dict) else getattr(m, "content", None)
        if isinstance(content, str):
            total += len(encoding.encode(content))
    return total


def should_compact(messages: list) -> bool:
    return count_tokens(messages) > CONTEXT_LIMIT * COMPACT_THRESHOLD and len(messages) > 4


def compact_history(messages: list, client, usage: dict, console) -> list:
    system_msg = messages[0]

    convo_text = ""
    for m in messages[1:]:
        role = m.get("role", "?") if isinstance(m, dict) else getattr(m, "role", "?")
        content = m.get("content") if isinstance(m, dict) else getattr(m, "content", None)
        tool_calls = m.get("tool_calls") if isinstance(m, dict) else getattr(m, "tool_calls", None)

        if isinstance(content, str) and content.strip():
            convo_text += f"{role}: {content}\n"
        elif role == "assistant" and tool_calls:
            names = [
                (tc["function"]["name"] if isinstance(tc, dict) else tc.function.name)
                for tc in tool_calls
            ]
            convo_text += f"assistant: [called tools: {', '.join(names)}]\n"

    console.print("[bold yellow]⟳ context near limit — compacting conversation...[/bold yellow]")

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "Summarize this agent conversation concisely but completely. "
                    "PRESERVE: the user's current goal/task, any file names and paths, "
                    "decisions made, and what has been done vs. what still remains. "
                    "Write it so the agent can seamlessly continue the task."
                ),
            },
            {"role": "user", "content": convo_text},
        ],
    )
    summary = resp.choices[0].message.content
    usage["input"] += resp.usage.prompt_tokens
    usage["output"] += resp.usage.completion_tokens

    return [
        system_msg,
        {
            "role": "user",
            "content": f"[Summary of earlier conversation]\n{summary}\n\n(Continue helping based on this context.)",
        },
    ]
