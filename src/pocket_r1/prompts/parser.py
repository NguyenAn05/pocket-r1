"""Strict parser and error taxonomy for 5-step Structured Poker CoT responses."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from pocket_r1.env.action_abstraction import DiscreteAction
from pocket_r1.prompts.state_formatter import estimate_tokens

_STEP_RE = re.compile(r"<step>\s*(.*?)\s*</step>", re.IGNORECASE | re.DOTALL)
_ACTION_RE = re.compile(r"<action>\s*(.*?)\s*</action>", re.IGNORECASE | re.DOTALL)
_ALL_TAGS_RE = re.compile(
    r"(?:<step>.*?</step>|<action>.*?</action>)", re.IGNORECASE | re.DOTALL
)

_ACTION_LOOKUP: dict[str, DiscreteAction] = {a.value: a for a in DiscreteAction}


@dataclass
class ParseResult:
    is_valid: bool
    steps: list[str] = field(default_factory=list)
    action: DiscreteAction | None = None
    raw_action_str: str | None = None
    error_codes: list[str] = field(default_factory=list)
    response_tokens: int = 0
    fallback_action: DiscreteAction | None = None

    @property
    def resolved_action(self) -> DiscreteAction:
        if self.action is not None and "ILLEGAL_ACTION" not in self.error_codes:
            return self.action
        assert self.fallback_action is not None
        return self.fallback_action


def _choose_safe_fallback(legal_actions: list[DiscreteAction]) -> DiscreteAction:
    if not legal_actions:
        return DiscreteAction.CHECK_CALL
    # If FOLD is not legal, CHECK_CALL is free; otherwise FOLD is safest fallback
    if DiscreteAction.FOLD in legal_actions:
        return DiscreteAction.FOLD
    if DiscreteAction.CHECK_CALL in legal_actions:
        return DiscreteAction.CHECK_CALL
    return legal_actions[0]


def parse_poker_cot(
    raw_text: str,
    legal_actions: list[DiscreteAction],
    expected_steps: int = 5,
    max_tokens: int = 650,
) -> ParseResult:
    """Parse `<step>` and `<action>` tags and classify structural errors."""
    text = raw_text or ""
    tokens = estimate_tokens(text)
    errors: list[str] = []

    steps = [m.group(1).strip() for m in _STEP_RE.finditer(text)]
    if len(steps) == 0:
        errors.append("MISSING_STEPS")
    elif len(steps) != expected_steps:
        errors.append("WRONG_STEP_COUNT")

    action_matches = [m.group(1).strip() for m in _ACTION_RE.finditer(text)]
    raw_action_str: str | None = None
    parsed_action: DiscreteAction | None = None

    if len(action_matches) == 0:
        errors.append("MISSING_ACTION")
    else:
        if len(action_matches) > 1:
            errors.append("MULTIPLE_ACTIONS")
        raw_action_str = action_matches[-1].strip().upper()
        parsed_action = _ACTION_LOOKUP.get(raw_action_str)
        if parsed_action is None:
            errors.append("UNKNOWN_ACTION")
        elif legal_actions and parsed_action not in legal_actions:
            errors.append("ILLEGAL_ACTION")

    leftover = _ALL_TAGS_RE.sub("", text).strip()
    if len(leftover) > 10:
        errors.append("EXTRA_TEXT_OUTSIDE_TAGS")

    if tokens > max_tokens:
        errors.append("LENGTH_OVERFLOW")

    fallback = _choose_safe_fallback(legal_actions)
    is_valid = len(errors) == 0 and parsed_action is not None

    return ParseResult(
        is_valid=is_valid,
        steps=steps,
        action=parsed_action,
        raw_action_str=raw_action_str,
        error_codes=errors,
        response_tokens=tokens,
        fallback_action=fallback,
    )
