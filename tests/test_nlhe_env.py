"""Unit tests for HeadsUpNLHEEnv and 6-action discrete abstraction."""

from __future__ import annotations

import random

import pytest

from pocket_r1.env import DiscreteAction, HeadsUpNLHEEnv


def test_nlhe_initial_positions_blinds_and_pot_odds() -> None:
    env = HeadsUpNLHEEnv(starting_stack_bb=100.0)
    obs = env.reset(seed=42)
    assert obs.street == "PREFLOP"
    assert obs.seat == 1
    assert obs.position == "BTN"
    assert obs.pot_bb == pytest.approx(1.5)
    assert obs.to_call_bb == pytest.approx(0.5)
    assert obs.hero_stack_bb == pytest.approx(99.5)
    assert obs.villain_stack_bb == pytest.approx(99.0)
    assert obs.pot_odds == pytest.approx(0.5 / (1.5 + 0.5))
    assert DiscreteAction.FOLD in obs.legal_actions
    assert DiscreteAction.CHECK_CALL in obs.legal_actions
    assert DiscreteAction.ALL_IN in obs.legal_actions


def test_nlhe_no_fold_when_check_is_free() -> None:
    env = HeadsUpNLHEEnv()
    env.reset(seed=7)
    # BTN limps -> BB has to_call == 0
    obs, _, done, _ = env.step(DiscreteAction.CHECK_CALL)
    assert not done and obs is not None
    assert obs.position == "BB"
    assert obs.to_call_bb == pytest.approx(0.0)
    assert DiscreteAction.FOLD not in obs.legal_actions


def test_nlhe_all_in_and_split_pot() -> None:
    # Identical Broadway straight on board -> exact split pot
    deck = ["2c", "3d", "4c", "5d", "Ts", "Js", "Qs", "Ks", "As"]
    env = HeadsUpNLHEEnv(starting_stack_bb=100.0)
    env.reset(seed=99, deck=deck)
    obs, _, done, _ = env.step(DiscreteAction.ALL_IN)
    assert not done and obs is not None
    assert obs.legal_actions == [DiscreteAction.FOLD, DiscreteAction.CHECK_CALL]
    obs, payoffs, done, _ = env.step(DiscreteAction.CHECK_CALL)
    assert done and obs is None
    assert payoffs == [pytest.approx(0.0), pytest.approx(0.0)]


def test_nlhe_clone_counterfactual_deck() -> None:
    deck_a = ["As", "Ah", "7c", "2d", "Kd", "8s", "3c", "9h", "4s"]
    deck_b = ["As", "Ah", "Ks", "Kh", "Kd", "8s", "3c", "9h", "4s"]
    env = HeadsUpNLHEEnv()
    env.reset(seed=10, deck=deck_a)
    env.step(DiscreteAction.ALL_IN)

    branch_a = env.clone()
    _, payoffs_a, done_a, _ = branch_a.step(DiscreteAction.CHECK_CALL)
    assert done_a and payoffs_a[0] == pytest.approx(100.0)  # Seat 0 (AA) beats 72o

    branch_b = env.clone(override_deck=deck_b)
    _, payoffs_b, done_b, _ = branch_b.step(DiscreteAction.CHECK_CALL)
    assert done_b and payoffs_b[0] == pytest.approx(-100.0)  # Seat 1 (KK sets on K) beats AA


def test_nlhe_10000_random_hands_zero_sum_no_crash() -> None:
    rng = random.Random(2026)
    env = HeadsUpNLHEEnv(starting_stack_bb=100.0)
    for i in range(10000):
        obs = env.reset(seed=i)
        steps = 0
        while not env.is_terminal:
            assert obs is not None
            action = rng.choice(obs.legal_actions)
            obs, payoffs, done, _ = env.step(action)
            steps += 1
            assert steps < 40
        assert abs(sum(env.payoffs_bb)) < 1e-9
        assert -100.0 <= env.payoffs_bb[0] <= 100.0
