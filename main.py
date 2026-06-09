import sys
from datetime import datetime

from dotenv import load_dotenv

# must run before agent_cli imports — OpenAI client is initialised at module load time
load_dotenv()

from rich.console import Console
from rich.panel import Panel

from agent_cli.agent import SYSTEM_PROMPT, run_turn
from agent_cli.session import list_sessions, load_session

console = Console()


def main() -> None:
    if "--list-sessions" in sys.argv:
        list_sessions()
        return

    if "--resume" in sys.argv:
        idx = sys.argv.index("--resume")
        if idx + 1 >= len(sys.argv):
            console.print("[red]--resume requires a session id argument.[/red]")
            sys.exit(1)
        session_id = sys.argv[idx + 1]
        messages = load_session(session_id)
        console.print(f"[green]Resumed session: {session_id}[/green]")
    else:
        session_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        console.print(f"[dim]New session: {session_id}[/dim]")

    usage = {"input": 0, "output": 0}

    console.print(Panel("Mini Claude Code — v0.6  (type 'exit' to quit)", style="bold green"))

    while True:
        user_input = console.input("[bold cyan]you > [/bold cyan]")
        if user_input.strip().lower() in ("exit", "quit"):
            console.print("[dim]bye[/dim]")
            break

        messages.append({"role": "user", "content": user_input})
        messages = run_turn(session_id, messages, usage)


if __name__ == "__main__":
    main()
