"""Round-robin sanity check for the 4 Tier-2 personality bots."""

from __future__ import annotations

import argparse
from collections import defaultdict

from pocket_r1.agents.rule_bots import create_bot
from pocket_r1.env import DiscreteAction, HeadsUpNLHEEnv


def run_match(bot_a_name: str, bot_b_name: str, n_hands: int = 1000, seed: int = 42) -> dict:
    env = HeadsUpNLHEEnv(starting_stack_bb=100.0)
    bot_a = create_bot(bot_a_name, seed=seed, equity_samples=40)
    bot_b = create_bot(bot_b_name, seed=seed + 1000, equity_samples=40)

    stats = {
        bot_a_name: defaultdict(int),
        bot_b_name: defaultdict(int),
    }
    total_a_bb = 0.0

    for h in range(n_hands):
        # Alternate seats every hand for fairness
        if h % 2 == 0:
            seats = [bot_a, bot_b]
            a_seat = 0
        else:
            seats = [bot_b, bot_a]
            a_seat = 1

        obs = env.reset(seed=seed * 10000 + h)
        vpip_done = [False, False]
        pfr_done = [False, False]

        while not env.is_terminal:
            assert obs is not None
            seat = obs.seat
            agent = seats[seat]
            res = agent.act(obs)
            act = res.action
            s = stats[agent.name]

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
            else:
                if act in (
                    DiscreteAction.BET_33,
                    DiscreteAction.BET_75,
                    DiscreteAction.BET_150,
                    DiscreteAction.ALL_IN,
                ):
                    s["post_bets"] += 1
                elif act == DiscreteAction.CHECK_CALL and obs.to_call_bb > 0:
                    s["post_calls"] += 1
                if obs.to_call_bb > 0:
                    s["faced_bet"] += 1
                    if act == DiscreteAction.FOLD:
                        s["folded_to_bet"] += 1

            obs, _, _, _ = env.step(act)

        for seat_idx, agent in enumerate(seats):
            stats[agent.name]["hands"] += 1
            if vpip_done[seat_idx]:
                stats[agent.name]["vpip"] += 1
            if pfr_done[seat_idx]:
                stats[agent.name]["pfr"] += 1

        total_a_bb += env.payoffs_bb[a_seat]

    return {
        "bb_per_100_a": (total_a_bb / n_hands) * 100.0,
        "stats": stats,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Sanity check 4 personality bots round-robin.")
    parser.add_argument("--hands", type=int, default=600, help="Hands per matchup")
    args = parser.parse_args()

    bots = ["calling_station", "nit", "maniac", "tag"]
    agg_stats: dict[str, defaultdict[str, int]] = {b: defaultdict(int) for b in bots}
    print(f"=== Round-Robin Bot Sanity Check ({args.hands} hands/matchup) ===")
    for i, a in enumerate(bots):
        for b in bots[i + 1 :]:
            res = run_match(a, b, n_hands=args.hands, seed=42 + i)
            print(f"  {a:16s} vs {b:16s} -> {a}: {res['bb_per_100_a']:+7.2f} BB/100")
            for name in (a, b):
                for k, v in res["stats"][name].items():
                    agg_stats[name][k] += v

    print("\n=== Aggregate Bot Personality Profiles ===")
    for b in bots:
        s = agg_stats[b]
        hands = max(1, s["hands"])
        vpip = s["vpip"] / hands * 100.0
        pfr = s["pfr"] / hands * 100.0
        af = s["post_bets"] / max(1, s["post_calls"])
        ftb = s["folded_to_bet"] / max(1, s["faced_bet"]) * 100.0
        print(
            f"  {b:16s} | VPIP: {vpip:5.1f}% | PFR: {pfr:5.1f}% | "
            f"AF: {af:4.2f} | Fold-to-Bet: {ftb:5.1f}%"
        )


if __name__ == "__main__":
    main()
