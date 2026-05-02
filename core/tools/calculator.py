"""
core/tools/calculator.py
------------------------
Safe arithmetic evaluation tool.

Handles two input forms:
  1. Pure expression : "125 * 8 + 42 / 3"  → evaluate directly
  2. Natural language: "what is result of 125 * 8 + 42 / 3"
                       → extract expression first, then evaluate

Extraction uses a simple regex scan first (fast, no LLM cost).
Only falls back to LLM extraction when no valid expression is found inline.
"""

import ast
import operator
import re
from typing import Any

from langchain_core.messages import HumanMessage
from core.models.llm import get_llm

# ---------------------------------------------------------------------------
# Allowed operators for safe AST evaluation
# ---------------------------------------------------------------------------
_ALLOWED_OPS: dict = {
    ast.Add:      operator.add,
    ast.Sub:      operator.sub,
    ast.Mult:     operator.mul,
    ast.Div:      operator.truediv,
    ast.Pow:      operator.pow,
    ast.USub:     operator.neg,
    ast.Mod:      operator.mod,
    ast.FloorDiv: operator.floordiv,
}

# Regex to pull a math expression from a natural language string
# Matches sequences of digits, operators, spaces, parentheses, and dots
_EXPR_PATTERN = re.compile(
    r'[\d\s\.\(\)]+(?:[+\-\*\/\%\^]{1,2}[\d\s\.\(\)]+)+'
)


# ---------------------------------------------------------------------------
# Safe AST evaluator
# ---------------------------------------------------------------------------
def _safe_eval(expr: str) -> Any:
    """
    Parse and evaluate a math expression using AST — no eval() called.

    Raises:
        ValueError: for invalid syntax or unsupported operations.
    """
    try:
        tree = ast.parse(expr.strip(), mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Invalid expression syntax: {expr}") from exc

    def _eval(node: ast.AST) -> Any:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type not in _ALLOWED_OPS:
                raise ValueError(f"Operator '{op_type.__name__}' is not permitted.")
            return _ALLOWED_OPS[op_type](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type not in _ALLOWED_OPS:
                raise ValueError(f"Operator '{op_type.__name__}' is not permitted.")
            return _ALLOWED_OPS[op_type](_eval(node.operand))
        raise ValueError(f"Unsupported element: {type(node).__name__}")

    return _eval(tree)


# ---------------------------------------------------------------------------
# Expression extractor
# ---------------------------------------------------------------------------
def _extract_expression(text: str) -> str:
    """
    Extract a math expression from a natural language string.

    Strategy (in order — stops at first success):
      1. If the text is already a valid expression → return as-is
      2. Regex scan for digit+operator patterns
      3. LLM extraction as final fallback

    Args:
        text: Raw user input string.

    Returns:
        Extracted math expression string.

    Raises:
        ValueError: if no valid expression can be found.
    """
    stripped = text.strip()

    # ── Step 1: try direct evaluation first ────────────────────────────────
    try:
        _safe_eval(stripped)
        return stripped   # already a pure expression
    except (ValueError, SyntaxError):
        pass

    # ── Step 2: regex extraction ────────────────────────────────────────────
    match = _EXPR_PATTERN.search(stripped)
    if match:
        candidate = match.group(0).strip()
        try:
            _safe_eval(candidate)
            return candidate
        except (ValueError, SyntaxError):
            pass

    # ── Step 3: LLM extraction fallback ────────────────────────────────────
    llm = get_llm()
    prompt = (
        "Extract ONLY the mathematical expression from the sentence below.\n"
        "Return the raw expression only — no words, no explanation, no formatting.\n"
        "Examples:\n"
        "  Input : 'what is result of 125 * 8 + 42 / 3'\n"
        "  Output: 125 * 8 + 42 / 3\n\n"
        "  Input : 'calculate 10 to the power of 3 minus 5'\n"
        "  Output: 10 ** 3 - 5\n\n"
        f"Input : {stripped}\n"
        "Output:"
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    extracted = response.content.strip()

    # Validate what the LLM returned
    try:
        _safe_eval(extracted)
        return extracted
    except (ValueError, SyntaxError):
        raise ValueError(
            f"Could not extract a valid math expression from: {stripped!r}"
        )


# ---------------------------------------------------------------------------
# Public tool entry point
# ---------------------------------------------------------------------------
def run(state: dict) -> dict:
    """
    Calculator tool — handles both pure expressions and natural language math.

    Args:
        state: AgentState dict; reads `input` and `intermediate_steps`.

    Returns:
        Partial state dict with `output` and updated `intermediate_steps`.
    """
    raw_input = state["input"]
    steps = list(state.get("intermediate_steps") or [])
    steps.append(f"⚙️ [Calculator] Received: `{raw_input}`")

    try:
        # Extract the math expression (handles natural language inputs)
        expression = _extract_expression(raw_input)

        # Log if extraction changed the input
        if expression != raw_input.strip():
            steps.append(f"🔎 [Calculator] Extracted expression: `{expression}`")

        result = _safe_eval(expression)

        # Display whole numbers without a decimal point
        if isinstance(result, float) and result.is_integer():
            result = int(result)

        output = f"Result: {result}"

    except Exception as exc:
        output = f"Calculator error: {exc}"

    steps.append(f"✅ [Calculator] Output → {output}")
    return {"output": output, "intermediate_steps": steps}