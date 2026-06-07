import os
from openai import OpenAI
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel

# Load the OPENAI_API_KEY from your .env file into the program
load_dotenv()

# Create the OpenAI client. It automatically reads OPENAI_API_KEY from the environment.
client = OpenAI()

# `rich` gives us pretty terminal output (colored boxes, etc.)
console = Console()

# This list holds the whole conversation so the AI remembers what was said.
# Each message is a dict with a "role" (system/user/assistant) and "content".
messages = [
    {"role": "system", "content": "You are a helpful assistant."}
]

console.print(Panel("Mini Claude Code — v0.1 (type 'exit' to quit)", style="bold green"))

# The main loop: keep chatting until the user types 'exit'
while True:
    # Get input from the user
    user_input = console.input("[bold cyan]you > [/bold cyan]")

    if user_input.strip().lower() in ("exit", "quit"):
        console.print("[dim]bye[/dim]")
        break

    # Add the user's message to the conversation history
    messages.append({"role": "user", "content": user_input})

    # Ask the AI, with stream=True so tokens arrive one at a time
    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        stream=True,
    )

    # We'll collect the full reply here as it streams in
    full_reply = ""

    console.print("[bold magenta]ai  > [/bold magenta]", end="")

    # Loop over each chunk as it arrives from OpenAI
    for chunk in stream:
        # Each chunk may or may not contain a piece of text
        token = chunk.choices[0].delta.content
        if token:
            console.print(token, end="")  # print it immediately, no newline
            full_reply += token            # also save it

    console.print()  # newline after the reply finishes

    # Add the AI's full reply to history so it remembers next turn
    messages.append({"role": "assistant", "content": full_reply})