# mini-claude-code

A command-line AI agent built from scratch in Python — no LangChain, no frameworks, just the raw tool-use loop. It runs bash, reads and edits files, sandboxes every command in Docker, manages its own context window, and resumes work across sessions.

The point of this project wasn't the LLM. It was understanding how agentic systems actually work underneath the abstractions: the tool-call loop, context management, sandboxing, and the failure modes you only discover by building one.

<!-- TODO: record a 30-second demo and embed it here -->
<!-- ![demo](docs/demo.gif) -->

## What it does

- **Tool-use loop** — the agent calls the model, which requests tools (bash, read, write, edit); the host executes them and feeds results back, looping until the task is done.
- **Four tools** — `run_bash`, `read_file`, `write_file`, and `str_replace` (surgical single-match editing).
- **Docker sandboxing** — every bash command runs in a disposable, network-isolated Alpine container with a mounted workspace and a 30-second timeout. The agent cannot touch the host filesystem.
- **Tool-result truncation** — large outputs are trimmed head-and-tail (not just the first N chars) so the important parts at both ends survive.
- **Token counting + cost** — running token and cost estimate per turn via tiktoken.
- **Context compaction** — at 80% of the context limit, the conversation is summarized and restarted from the summary, enabling arbitrarily long sessions.
- **Loop cap** — a hard bound on inner-loop iterations prevents runaway execution.
- **Session persistence** — every turn is saved to disk; `--resume` continues an earlier session, `--list-sessions` shows saved ones.

## Install

```bash
git clone https://github.com/farisfarsan/mini-claude-code.git
cd mini-claude-code
uv sync   # or: pip install openai rich python-dotenv tiktoken
```

Copy the env template and add your OpenAI API key:

```bash
cp .env.example .env
# then edit .env and set OPENAI_API_KEY=sk-...
```

Docker must be installed and running (used for bash sandboxing).

## Usage

```bash
uv run main.py                          # start a new session
uv run main.py --list-sessions          # show saved sessions
uv run main.py --resume 20260608-092331 # resume a saved session
```

## Example sessions

**Multi-step file task**

```
you > list the files in the workspace, then create notes.txt with a todo list
→ calling run_bash({'command': 'ls'})
→ calling write_file({'path': 'workspace/notes.txt', ...})
ai  > Done — I listed the workspace and created notes.txt with your todo list.
```

**Surgical edit with self-correction**

```
you > in sample.py, replace "return" with "yield"
→ calling str_replace(old_str="return", ...)
  result: ERROR: old_str appears 2 times. It must be unique.
→ calling read_file(...)
→ calling str_replace(old_str='return "hi there"', ...)   # now unique
ai  > Replaced both occurrences with precise, unique edits.
```

**Resume across a restart**

```
$ uv run cli.py --resume 20260608-092331
Resumed session: 20260608-092331
you > what's my favorite color?
ai  > Your favorite color is teal.
```

## Architecture

The core is a tool-agnostic loop. Adding a capability is three steps — write a function, describe it in the tool schema, register it — and the loop never changes.

```
user input
   │
   ▼
┌─────────────────────────────────────────────┐
│  agent loop                                  │
│   call model ──▶ tool calls? ──yes──▶ run    │
│        ▲                              tool   │
│        │                               │     │
│        └────── feed result back ◀──────┘     │
│   no tool calls ──▶ final answer ──▶ exit    │
└─────────────────────────────────────────────┘
   │
   ▼
truncate result · count tokens · compact if >80% · save session
```

Bash execution is isolated:

```
agent's bash command
   │
   ▼
docker run --rm --network=none -v workspace:/work -w /work python:3.11-alpine sh -c "<command>"
   │
   ▼
output (host filesystem untouched)
```

## The hardest engineering problem

Context compaction was the trickiest part, because it rewrites the conversation history *while the agent loop is actively reading it*. I hit three failure modes in sequence:

1. **Mid-task compaction** — an early version compacted in the middle of a task, and the agent lost track of what it had been asked to do.
2. **Role confusion** — storing the summary under the wrong message role made the model treat its own summary as the new instruction and spiral into talking to itself.
3. **The real culprit** — a stray, duplicated compaction check at the top of the inner loop fired on every iteration, compacting constantly and eating the user's request.

The fix was to compact only at a resting point (after a final answer, between turns), preserve the user's goal explicitly in the summary, and add a hard loop cap so the agent can never run away again regardless of what compaction does.

The lesson generalizes: **when state is mutated mid-loop, *when* you mutate it matters as much as *how*.**

## Why I built this

I wanted to understand how agentic coding tools work without the abstraction layers in the way. So I built one from scratch — the loop, the sandbox, the context management, the persistence — and hit the real problems firsthand: tool-result truncation blowing up the context, compaction interacting badly with loop control, Docker isolation. Reading about these is one thing; debugging them in your own code is another.

## Stack

Python 3.11 · OpenAI API · Docker · rich · tiktoken
