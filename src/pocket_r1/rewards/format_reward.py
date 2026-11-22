"""Format reward R_format with Hard Gate for illegal/missing tags and Length Penalty."""

from __future__ import annotations

from pocket_r1.prompts.parser import ParseResult

_HARD_ERRORS = frozenset({"MISSING_STEPS", "MISSING_ACTION", "UNKNOWN_ACTION", "ILLEGAL_ACTION"})


def compute_format_reward(
    parsed: ParseResult,
    soft_max_tokens: int = 550,
    hard_max_tokens: int = 800,
) -> tuple[float, bool]:
    """Return (r_format in [-1.0, +1.0], hard_gate_failed).

    Protects DeBERTa-v3's 1024-token context via linear length penalty above `soft_max_tokens`.
    """
    errors = set(parsed.error_codes)
    if errors & _HARD_ERRORS or parsed.action is None:
        return -1.0, True

    if "WRONG_STEP_COUNT" in errors:
        return -0.6, True

    # Base score: +1.0 if completely clean, +0.2 if minor extra text outside tags
    score = 1.0 if not (errors - {"LENGTH_OVERFLOW"}) else 0.2

    # Smooth length penalty beyond soft_max_tokens
    if parsed.response_tokens > soft_max_tokens:
        span = max(1, hard_max_tokens - soft_max_tokens)
        excess_frac = min(1.0, (parsed.response_tokens - soft_max_tokens) / span)
        score -= 0.5 * excess_frac

    return round(max(-1.0, min(1.0, score)), 4), False
