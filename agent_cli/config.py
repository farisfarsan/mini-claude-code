import os
import tiktoken

SESSIONS_DIR = "sessions"
WORKSPACE = os.path.abspath("workspace")

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

encoding = tiktoken.get_encoding("o200k_base")

PRICE_INPUT_PER_1M = 0.15
PRICE_OUTPUT_PER_1M = 0.60

CONTEXT_LIMIT = 8000
COMPACT_THRESHOLD = 0.80

MAX_TOOL_RESULT_CHARS = 5000
