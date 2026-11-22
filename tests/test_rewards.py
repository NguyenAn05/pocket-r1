"""Unit tests for Phase 4 Hybrid Reward Suite, Anti-Hacking Gate, and HackingMonitor."""

from __future__ import annotations

import pytest

from pocket_r1.env import DiscreteAction
from pocket_r1.prompts import ONE_SHOT_ASSISTANT, parse_poker_cot
from pocket_r1.rewards import (
    HackingMonitor,
    HybridRewardAggregator,
    HybridRewardConfig,
    aggregate_prm_score,
    compute_format_reward,
    compute_relative_ev_reward,
)


def test_format_reward_hard_gate_and_length_penalty() -> None:
    legal = [DiscreteAction.FOLD, DiscreteAction.CHECK_CALL, DiscreteAction.BET_75]
    valid = parse_poker_cot(ONE_SHOT_ASSISTANT, legal)
    r_fmt, failed = compute_format_reward(valid)
    assert r_fmt == pytest.approx(1.0)
    assert not failed

    # Length penalty above 550 tokens
    valid.response_tokens = 675
    r_long, failed_long = compute_format_reward(valid, soft_max_tokens=550, hard_max_tokens=800)
    assert not failed_long
    assert r_long == pytest.approx(0.75)

    # Illegal action triggers hard gate
    illegal = parse_poker_cot(ONE_SHOT_ASSISTANT, [DiscreteAction.BET_75])
    r_bad, failed_bad = compute_format_reward(illegal)
    assert r_bad == pytest.approx(-1.0)
    assert failed_bad


def test_relative_ev_and_prm_aggregation() -> None:
    ev_table = {
        DiscreteAction.FOLD: -0.5,
        DiscreteAction.CHECK_CALL: 4.5,
        DiscreteAction.BET_75: 6.0,
    }
    r_best, reg_best = compute_relative_ev_reward(DiscreteAction.BET_75, ev_table, pot_bb=12.0)
    r_fold, reg_fold = compute_relative_ev_reward(DiscreteAction.FOLD, ev_table, pot_bb=12.0)
    assert r_best > r_fold
    assert reg_best == pytest.approx(0.0)
    assert reg_fold == pytest.approx(-6.5)

    probs = [0.95, 0.92, 0.90, 0.88, 0.10, 0.95]
    assert aggregate_prm_score(probs, mode="min") == pytest.approx(0.10)
    assert 0.5 < aggregate_prm_score(probs, mode="geom_mean") < 0.8


def test_hybrid_aggregator_anti_hacking_gate_and_monitor() -> None:
    legal = [DiscreteAction.FOLD, DiscreteAction.CHECK_CALL, DiscreteAction.BET_75]
    valid = parse_poker_cot(ONE_SHOT_ASSISTANT, legal)

    agg = HybridRewardAggregator(HybridRewardConfig(mode="hybrid", alpha_ev=0.6, beta_prm=0.3, gamma_format=0.1))

    # Good action + high PRM -> no anti-hack trigger
    good_evs = {DiscreteAction.FOLD: -6.0, DiscreteAction.CHECK_CALL: 4.0, DiscreteAction.BET_75: 2.0}
    b_good = agg.score(valid, pot_bb=12.0, ev_table=good_evs, step_probs=[0.95] * 6)
    assert not b_good.anti_hack_triggered
    assert b_good.r_prm > 0.9

    # Reward-hacking attempt: eloquent high PRM (0.95) on a catastrophic -8BB regret call!
    bad_evs = {DiscreteAction.FOLD: 0.0, DiscreteAction.CHECK_CALL: -8.0, DiscreteAction.BET_75: -10.0}
    b_hack = agg.score(valid, pot_bb=12.0, ev_table=bad_evs, step_probs=[0.95] * 6)
    assert b_hack.anti_hack_triggered
    assert b_hack.r_prm < 0.30
    assert b_good.total > b_hack.total

    mon = HackingMonitor()
    mon.update(b_good)
    mon.update(b_hack)
    assert mon.summary()["count"] == 2.0
