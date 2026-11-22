"""Unit tests for Tier-3 GTO Bot, PFSP League Manager, GRPO Advantages, and Co-evolution."""

from __future__ import annotations

import pytest

from pocket_r1.agents.gto_bot import GTOMDFBot
from pocket_r1.agents.league_manager import LeagueManager
from pocket_r1.env import HeadsUpNLHEEnv
from pocket_r1.training.coevolution import (
    mine_false_positives,
    rejection_sample_top_trajectories,
)
from pocket_r1.training.grpo_trainer import GRPOTrainConfig, compute_group_advantages


def test_gto_mdf_bot_plays_legally() -> None:
    env = HeadsUpNLHEEnv()
    gto = GTOMDFBot(seed=42, equity_samples=30)
    opp = GTOMDFBot(seed=99, equity_samples=30)
    for h in range(40):
        obs = env.reset(seed=500 + h)
        while not env.is_terminal:
            assert obs is not None
            ag = gto if obs.seat == 0 else opp
            res = ag.act(obs)
            assert res.action in obs.legal_actions
            obs, _, _, _ = env.step(res.action)


def test_pfsp_league_manager_prioritizes_hard_opponents_and_keeps_tier2_floor() -> None:
    lm = LeagueManager(include_gto=True, min_tier2_weight=0.30, seed=42)
    # Hero crushes all Tier-2 bots but loses heavily to gto_mdf
    for bot in ("calling_station", "nit", "maniac", "tag"):
        for _ in range(50):
            lm.record_result(bot, +2.0)
    for _ in range(50):
        lm.record_result("gto_mdf", -3.0)

    probs = lm.sampling_probabilities()
    assert sum(probs.values()) == pytest.approx(1.0)
    assert probs["gto_mdf"] > probs["tag"]
    tier2_total = sum(probs[b] for b in ("calling_station", "nit", "maniac", "tag"))
    assert tier2_total >= 0.299  # >= 30% anti-forgetting floor


def test_grpo_advantages_and_coevolution_mining() -> None:
    adv = compute_group_advantages([1.0, 0.0, -1.0, 0.0])
    assert sum(adv) == pytest.approx(0.0, abs=1e-5)
    assert adv[0] > 0 and adv[2] < 0
    assert compute_group_advantages([0.5, 0.5, 0.5]) == [0.0, 0.0, 0.0]

    logs = [
        {
            "hand_id": "h1",
            "prompt_text": "P1",
            "steps": ["1", "2", "3", "4", "5"],
            "action": "CHECK_CALL",
            "total_reward": 0.72,
            "r_prm": 0.92,
            "regret_bb": -0.1,
            "hard_gate_failed": False,
        },
        {
            "hand_id": "h2",
            "prompt_text": "P2",
            "steps": ["1", "2", "3", "4", "5"],
            "action": "ALL_IN",
            "total_reward": -0.30,
            "r_prm": 0.88,
            "regret_bb": -7.5,
            "hard_gate_failed": False,
        },
    ]
    fps = mine_false_positives(logs)
    assert len(fps) == 1 and fps[0].hand_id == "h2" and fps[0].labels == [1.0, 1.0, 1.0, 1.0, 0.0, 0.0]

    rft = rejection_sample_top_trajectories(logs)
    assert len(rft) == 1 and rft[0]["hand_id"] == "h1"

    cfg = GRPOTrainConfig.from_yaml("configs/rl/exp4_grpo_hybrid.yaml")
    assert cfg.reward_mode == "hybrid" and cfg.alpha_ev == 0.6 and cfg.beta_prm == 0.3
