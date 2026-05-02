"""
core/guardrails/validation.py
------------------------------
Input validation and safe execution guardrails.

Three responsibilities:
  1. validate_input()    – rejects empty, oversized, or unsafe queries
  2. UNSAFE_PATTERNS     – keyword blocklist for obvious malicious prompts
  3. safe_tool_runner()  – wraps any tool call in a try/except so a single
                           bad tool never crashes the whole graph

These are intentionally simple and transparent — no hidden magic.
Add more patterns or rules as the platform grows.
"""

import re
from typing import Callable

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

MAX_INPUT_LENGTH = 5_000   # characters — prevents prompt-injection via giant inputs
MIN_INPUT_LENGTH = 2       # characters — rejects blank/whitespace-only queries

# Patterns that indicate prompt-injection or clearly malicious intent.
# Each entry is a compiled regex for fast matching.
UNSAFE_PATTERNS: list[re.Pattern] = [
    re.compile(r"ignore\s+(all\s+)?(previous\s+|prior\s+)?(instructions?|prompts?)", re.I),
    re.compile(r"you are now", re.I),
    re.compile(r"act as (a |an )?(different|new|evil|unrestricted)", re.I),
    re.compile(r"jailbreak", re.I),
    re.compile(r"(system|hidden) prompt", re.I),
    re.compile(r"<\s*script", re.I),          # HTML script injection
    re.compile(r"(drop|delete|truncate)\s+table", re.I),   # SQL injection
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class ValidationError(ValueError):
    """Raised when a user input fails guardrail checks."""
    pass


def validate_input(text: str) -> str:
    """
    Validate and sanitise a user query before it enters the agent graph.

    Checks performed (in order):
      1. Not empty / too short
      2. Not exceeding max length
      3. Does not match any unsafe pattern

    Args:
        text: Raw user input string.

    Returns:
        Stripped, validated input string.

    Raises:
        ValidationError: with a human-readable message on any failure.
    """
    stripped = text.strip()

    # ── Length checks ────────────────────────────────────────────────────────
    if len(stripped) < MIN_INPUT_LENGTH:
        raise ValidationError("Input is too short. Please enter a meaningful query.")

    if len(stripped) > MAX_INPUT_LENGTH:
        raise ValidationError(
            f"Input exceeds the maximum allowed length of {MAX_INPUT_LENGTH} characters. "
            "Please shorten your query."
        )

    # ── Unsafe pattern check ─────────────────────────────────────────────────
    for pattern in UNSAFE_PATTERNS:
        if pattern.search(stripped):
            raise ValidationError(
                "Your query contains content that cannot be processed. "
                "Please rephrase and try again."
            )

    return stripped


def safe_tool_runner(tool_fn: Callable, state: dict) -> dict:
    """
    Execute a tool function inside a protective try/except wrapper.

    If the tool raises an unexpected exception, the error is captured
    and returned as a graceful output rather than crashing the graph.

    Args:
        tool_fn : A tool function with signature (state: dict) -> dict.
        state   : The current AgentState dict.

    Returns:
        Tool result dict, or a dict with an error message in "output".
    """
    try:
        return tool_fn(state)
    except Exception as exc:
        error_msg = f"Tool execution error ({tool_fn.__name__}): {exc}"
        steps = list(state.get("intermediate_steps") or [])
        steps.append(f"❌ [Guardrail] {error_msg}")
        return {"output": error_msg, "intermediate_steps": steps}