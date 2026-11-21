"""Unit tests for StateFormatter, Templates, and Structured CoT Parser."""

from __future__ import annotations

import random

from pocket_r1.env import DiscreteAction, HeadsUpNLHEEnv
from pocket_r1.prompts import (
    ONE_SHOT_ASSISTANT,
    build_chat_messages,
    estimate_tokens,
    format_observation,
    parse_poker_cot,
)


def test_parse_valid_five_step_cot() -> None:
    legal = [DiscreteAction.FOLD, DiscreteAction.CHECK_CALL, DiscreteAction.BET_75]
    res = parse_poker_cot(ONE_SHOT_ASSISTANT, legal_actions=legal)
    assert res.is_valid
    assert len(res.steps) == 5
    assert res.action == DiscreteAction.CHECK_CALL
    assert res.resolved_action == DiscreteAction.CHECK_CALL
    assert res.error_codes == []


def test_parse_error_taxonomy() -> None:
    legal = [DiscreteAction.CHECK_CALL, DiscreteAction.BET_75, DiscreteAction.ALL_IN]

    # Missing steps & action
    r1 = parse_poker_cot("I think we should call here.", legal)
    assert not r1.is_valid
    assert "MISSING_STEPS" in r1.error_codes
    assert "MISSING_ACTION" in r1.error_codes
    assert r1.resolved_action == DiscreteAction.CHECK_CALL

    # Wrong step count & illegal action (FOLD when check is free)
    bad = "<step>1. A</step><step>2. B</step><action>FOLD</action>"
    r2 = parse_poker_cot(bad, legal)
    assert not r2.is_valid
    assert "WRONG_STEP_COUNT" in r2.error_codes
    assert "ILLEGAL_ACTION" in r2.error_codes
    assert r2.resolved_action == DiscreteAction.CHECK_CALL

    # Unknown action + extra text
    unknown = (
        "Here is my thought process:\n"
        + "\n".join(f"<step>{i}. ok</step>" for i in range(1, 6))
        + "\n<action>RAISE_BIG</action>"
    )
    r3 = parse_poker_cot(unknown, legal)
    assert not r3.is_valid
    assert "UNKNOWN_ACTION" in r3.error_codes
    assert "EXTRA_TEXT_OUTSIDE_TAGS" in r3.error_codes

    # Length overflow
    long_body = (
        "\n".join(f"<step>{i}. {'word ' * 150}</step>" for i in range(1, 6))
        + "\n<action>CHECK_CALL</action>"
    )
    r4 = parse_poker_cot(long_body, legal, max_tokens=650)
    assert not r4.is_valid
    assert "LENGTH_OVERFLOW" in r4.error_codes


def test_state_formatter_token_budget_on_1000_states() -> None:
    rng = random.Random(77)
    env = HeadsUpNLHEEnv()
    token_counts: list[int] = []

    hand_seed = 0
    while len(token_counts) < 1000:
        obs = env.reset(seed=hand_seed)
        hand_seed += 1
        while not env.is_terminal and len(token_counts) < 1000:
            assert obs is not None
            text = format_observation(obs)
            token_counts.append(estimate_tokens(text))
            msgs = build_chat_messages(obs, few_shot=False)
            assert len(msgs) == 2
            obs, _, _, _ = env.step(rng.choice(obs.legal_actions))

    under_300 = sum(1 for t in token_counts if t < 300)
    assert under_300 / len(token_counts) >= 0.99
