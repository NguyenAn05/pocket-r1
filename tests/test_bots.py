"""Unit tests for the 4 Tier-2 personality bots and MockSLMAgent."""

from __future__ import annotations

from collections import defaultdict

from pocket_r1.agents import MockSLMAgent, create_bot
from pocket_r1.env import DiscreteAction, HeadsUpNLHEEnv


def _run_quick_match(bot_a_name: str, bot_b_name: str, n_hands: int = 200, seed: int = 42) -> dict:
    env = HeadsUpNLHEEnv()
    bot_a = create_bot(bot_a_name, seed=seed, equity_samples=25)
    bot_b = create_bot(bot_b_name, seed=seed + 100, equity_samples=25)
    stats = {bot_a_name: defaultdict(int), bot_b_name: defaultdict(int)}
    for h in range(n_hands):
        seats = [bot_a, bot_b] if h % 2 == 0 else [bot_b, bot_a]
        obs = env.reset(seed=seed * 1000 + h)
        vpip_done = [False, False]
        pfr_done = [False, False]
        while not env.is_terminal:
            assert obs is not None
            seat = obs.seat
            act = seats[seat].act(obs).action
            if obs.street == "PREFLOP":
                if act != DiscreteAction.FOLD and (obs.to_call_bb > 0 or act != DiscreteAction.CHECK_CALL):
                    vpip_done[seat] = True
                if act in (
                    DiscreteAction.BET_33,
                    DiscreteAction.BET_75,
                    DiscreteAction.BET_150,
                    DiscreteAction.ALL_IN,
                ):
                    pfr_done[seat] = True
            obs, _, _, _ = env.step(act)
        for idx, ag in enumerate(seats):
            stats[ag.name]["hands"] += 1
            if vpip_done[idx]:
                stats[ag.name]["vpip"] += 1
            if pfr_done[idx]:
                stats[ag.name]["pfr"] += 1
    return stats


def test_all_four_bots_play_legal_actions() -> None:
    env = HeadsUpNLHEEnv()
    bots = [create_bot(name, seed=10, equity_samples=30) for name in ("calling_station", "nit", "maniac", "tag")]
    for idx, bot in enumerate(bots):
        opp = bots[(idx + 1) % len(bots)]
        for h in range(50):
            obs = env.reset(seed=idx * 100 + h)
            while not env.is_terminal:
                assert obs is not None
                agent = bot if obs.seat == 0 else opp
                res = agent.act(obs)
                assert res.action in obs.legal_actions
                obs, _, _, _ = env.step(res.action)


def test_bot_personality_separation() -> None:
    stats = _run_quick_match("calling_station", "nit", n_hands=200, seed=7)
    cs_vpip = stats["calling_station"]["vpip"] / stats["calling_station"]["hands"]
    nit_vpip = stats["nit"]["vpip"] / stats["nit"]["hands"]
    assert cs_vpip > nit_vpip + 0.20

    stats2 = _run_quick_match("maniac", "calling_station", n_hands=200, seed=11)
    man_pfr = stats2["maniac"]["pfr"] / stats2["maniac"]["hands"]
    cs_pfr = stats2["calling_station"]["pfr"] / stats2["calling_station"]["hands"]
    assert man_pfr > cs_pfr + 0.15


def test_mock_slm_emits_valid_cot() -> None:
    env = HeadsUpNLHEEnv()
    agent = MockSLMAgent(seed=42)
    obs = env.reset(seed=5)
    res = agent.act(obs)
    assert res.parsed is not None
    assert res.parsed.is_valid
    assert len(res.parsed.steps) == 5
    assert res.action in obs.legal_actions
