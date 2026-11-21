"""Unit tests and speed/accuracy sanity checks for EVCalculator."""

from __future__ import annotations

import time

from pocket_r1.agents import create_bot
from pocket_r1.env import DiscreteAction, HeadsUpNLHEEnv, LeducPokerEnv
from pocket_r1.verifiers import EVCalculator


def test_ev_calculator_leduc_exact_ordering() -> None:
    env = LeducPokerEnv()
    # Hero (seat 0) holds Ks, board is Kh (pair of Kings = nuts!)
    env.reset(seed=1, deck=["Ks", "Js", "Kh"])
    env.step(DiscreteAction.CHECK_CALL)
    env.step(DiscreteAction.CHECK_CALL)  # advance to FLOP with Kh
    env.step(DiscreteAction.CHECK_CALL)  # seat 0 checks
    # Now seat 1 bets; seat 0 faces bet with the nuts
    env.step(DiscreteAction.BET_75)
    assert env.current_seat == 0

    calc = EVCalculator()
    opp = create_bot("calling_station", seed=1)
    evs = calc.compute_action_evs(env, hero_seat=0, opponent=opp, seed=42)
    assert evs[DiscreteAction.BET_75] > evs[DiscreteAction.FOLD]
    assert evs[DiscreteAction.CHECK_CALL] > evs[DiscreteAction.FOLD]


def test_ev_calculator_nlhe_aa_vs_72o_and_speed() -> None:
    env = HeadsUpNLHEEnv()
    opp = create_bot("calling_station", seed=7, equity_samples=20)
    calc = EVCalculator(default_rollouts=24, rollout_equity_samples=16)

    # Hero (seat 1 = BTN) holds AA preflop
    deck_aa = ["7c", "2d", "As", "Ah", "Kd", "8s", "3c", "9h", "4s", "5d", "6c", "Td"]
    env.reset(seed=10, deck=deck_aa)
    t0 = time.perf_counter()
    evs_aa = calc.compute_action_evs(env, hero_seat=1, opponent=opp, n_rollouts=24, seed=99)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    # With AA, playing/betting must strongly beat folding (-0.5 BB)
    assert max(evs_aa.values()) > evs_aa[DiscreteAction.FOLD] + 1.0
    assert evs_aa[DiscreteAction.FOLD] == -0.5
    # Reasonable local CPU latency for full 6-action table
    assert elapsed_ms < 2500.0
