"""Unit tests for prototype LeducPokerEnv."""

from __future__ import annotations

import random

import pytest

from pocket_r1.env import DiscreteAction, LeducPokerEnv


def test_leduc_fold_payoff_zero_sum() -> None:
    env = LeducPokerEnv()
    obs = env.reset(seed=1, deck=["Ks", "Jh", "Kh"])
    assert obs.street == "PREFLOP"
    assert obs.pot_bb == 2.0
    assert DiscreteAction.FOLD not in obs.legal_actions  # cannot fold when to_call == 0

    obs, payoffs, done, _ = env.step(DiscreteAction.BET_75)
    assert not done
    assert obs is not None and obs.to_call_bb == 2.0
    assert DiscreteAction.FOLD in obs.legal_actions

    obs, payoffs, done, _ = env.step(DiscreteAction.FOLD)
    assert done and obs is None
    assert payoffs == [1.0, -1.0]
    assert sum(payoffs) == pytest.approx(0.0)


def test_leduc_showdown_pair_beats_high_card_and_split_pot() -> None:
    # Pair of Jacks beats King high
    env = LeducPokerEnv()
    env.reset(seed=2, deck=["Js", "Ks", "Jh"])
    env.step(DiscreteAction.CHECK_CALL)
    obs, _, done, _ = env.step(DiscreteAction.CHECK_CALL)
    assert not done and obs is not None and obs.street == "FLOP" and obs.board == ["Jh"]
    env.step(DiscreteAction.CHECK_CALL)
    _, payoffs, done, _ = env.step(DiscreteAction.CHECK_CALL)
    assert done
    assert payoffs == [1.0, -1.0]

    # Exact tie splits pot -> [0.0, 0.0]
    env.reset(seed=3, deck=["Ks", "Kh", "Js"])
    for _ in range(4):
        _, payoffs, done, _ = env.step(DiscreteAction.CHECK_CALL)
    assert done
    assert payoffs == [0.0, 0.0]


def test_leduc_clone_and_1000_random_hands() -> None:
    rng = random.Random(123)
    env = LeducPokerEnv()
    for i in range(1000):
        obs = env.reset(seed=i)
        cloned = False
        while not env.is_terminal:
            assert obs is not None
            action = rng.choice(obs.legal_actions)
            if not cloned and rng.random() < 0.2:
                branch = env.clone()
                branch_obs, _, _, _ = branch.step(action)
                assert branch_obs is None or branch_obs.hand_id == obs.hand_id
                cloned = True
            obs, payoffs, done, _ = env.step(action)
        assert abs(sum(env.payoffs_bb)) < 1e-9
