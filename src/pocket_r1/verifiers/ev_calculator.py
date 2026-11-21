"""Action EV Calculator using Common Random Numbers (CRN) Monte Carlo Rollouts & Exact Leduc EV."""

from __future__ import annotations

from pocket_r1.agents.base import BaseAgent
from pocket_r1.env.action_abstraction import DiscreteAction
from pocket_r1.env.base import BasePokerEnv
from pocket_r1.env.leduc_env import LEDUC_DECK, LeducPokerEnv
from pocket_r1.env.nlhe_env import STANDARD_DECK, HeadsUpNLHEEnv
from pocket_r1.utils.seeding import make_rng


class EVCalculator:
    """Computes low-variance Expected Value (EV in BB) for legal actions at a decision point.

    Key optimizations:
    - Common Random Numbers (CRN): reuses the exact same sampled unseen cards across all
      legal actions at state `s` to drastically reduce variance when comparing actions.
    - Single-pass action table: computes EV for all <= 6 legal actions in one call so GRPO
      can look up EV in O(1) across all G rollouts.
    """

    def __init__(
        self,
        default_rollouts: int = 32,
        rollout_equity_samples: int = 24,
    ) -> None:
        self.default_rollouts = default_rollouts
        self.rollout_equity_samples = rollout_equity_samples

    def compute_action_evs(
        self,
        env: BasePokerEnv,
        hero_seat: int,
        opponent: BaseAgent,
        hero_rollout_policy: BaseAgent | None = None,
        n_rollouts: int | None = None,
        seed: int = 42,
    ) -> dict[DiscreteAction, float]:
        """Compute EV (net hand payoff in BB for hero_seat) for every legal action."""
        if env.is_terminal:
            return {}
        obs = env.get_observation(hero_seat)
        if not obs.legal_actions:
            return {}

        if isinstance(env, LeducPokerEnv):
            return self._compute_leduc_evs(env, hero_seat, opponent, hero_rollout_policy, seed)
        if isinstance(env, HeadsUpNLHEEnv):
            k = n_rollouts if n_rollouts is not None else self.default_rollouts
            return self._compute_nlhe_evs(env, hero_seat, opponent, hero_rollout_policy, k, seed)
        raise TypeError(f"Unsupported env type: {type(env)}")

    def compute_single_action_ev(
        self,
        env: BasePokerEnv,
        hero_seat: int,
        action: DiscreteAction,
        opponent: BaseAgent,
        hero_rollout_policy: BaseAgent | None = None,
        n_rollouts: int | None = None,
        seed: int = 42,
    ) -> float:
        evs = self.compute_action_evs(
            env=env,
            hero_seat=hero_seat,
            opponent=opponent,
            hero_rollout_policy=hero_rollout_policy,
            n_rollouts=n_rollouts,
            seed=seed,
        )
        return evs[action]

    def _rollout_to_terminal(
        self,
        sim_env: BasePokerEnv,
        hero_seat: int,
        hero_policy: BaseAgent,
        opp_policy: BaseAgent,
    ) -> float:
        steps = 0
        while not sim_env.is_terminal and steps < 40:
            seat = sim_env.current_seat
            assert seat is not None
            obs = sim_env.get_observation(seat)
            agent = hero_policy if seat == hero_seat else opp_policy
            act = agent.act(obs).action
            sim_env.step(act)
            steps += 1
        return sim_env.payoffs_bb[hero_seat]

    def _compute_leduc_evs(
        self,
        env: LeducPokerEnv,
        hero_seat: int,
        opponent: BaseAgent,
        hero_rollout_policy: BaseAgent | None,
        seed: int,
    ) -> dict[DiscreteAction, float]:
        obs = env.get_observation(hero_seat)
        if hero_rollout_policy is None:
            from pocket_r1.agents.rule_bots import create_bot

            hero_policy = create_bot(
                "tag", seed=seed + 7, equity_samples=self.rollout_equity_samples
            )
        else:
            hero_policy = hero_rollout_policy
        known = set(obs.hero_cards) | set(obs.board)
        unseen = [c for c in LEDUC_DECK if c not in known]

        # Enumerate all possible unseen (opp_card, board_card) completions
        completions: list[list[str]] = []
        if len(obs.board) == 0:
            for opp_c in unseen:
                for brd_c in unseen:
                    if brd_c != opp_c:
                        deck = ["", "", brd_c]
                        deck[hero_seat] = obs.hero_cards[0]
                        deck[1 - hero_seat] = opp_c
                        completions.append(deck)
        else:
            for opp_c in unseen:
                deck = ["", "", obs.board[0]]
                deck[hero_seat] = obs.hero_cards[0]
                deck[1 - hero_seat] = opp_c
                completions.append(deck)

        evs: dict[DiscreteAction, float] = {}
        for action in obs.legal_actions:
            total = 0.0
            for deck in completions:
                sim = env.clone()
                sim._deck = list(deck)
                sim._hole_cards = [deck[0], deck[1]]
                sim.step(action)
                total += self._rollout_to_terminal(sim, hero_seat, hero_policy, opponent)
            evs[action] = total / max(1, len(completions))
        return evs

    def _compute_nlhe_evs(
        self,
        env: HeadsUpNLHEEnv,
        hero_seat: int,
        opponent: BaseAgent,
        hero_rollout_policy: BaseAgent | None,
        n_rollouts: int,
        seed: int,
    ) -> dict[DiscreteAction, float]:
        obs = env.get_observation(hero_seat)
        rng = make_rng(seed)
        if hero_rollout_policy is None:
            from pocket_r1.agents.rule_bots import create_bot

            hero_policy = create_bot(
                "tag", seed=seed + 13, equity_samples=self.rollout_equity_samples
            )
        else:
            hero_policy = hero_rollout_policy

        known = set(obs.hero_cards) | set(obs.board)
        unseen = [c for c in STANDARD_DECK if c not in known]
        n_board_known = len(obs.board)
        n_board_missing = 5 - n_board_known

        # Build K counterfactual decks shared across all legal actions (CRN)
        cf_decks: list[list[str]] = []
        for _ in range(n_rollouts):
            # Sample 2 villain hole cards + remaining board cards + 3 burn cards
            sampled = rng.sample(unseen, 2 + n_board_missing + 3)
            opp_cards = sampled[:2]
            rem_board = sampled[2 : 2 + n_board_missing]
            burns = sampled[2 + n_board_missing :]

            deck = [""] * 12
            if hero_seat == 0:
                deck[0], deck[1] = obs.hero_cards[0], obs.hero_cards[1]
                deck[2], deck[3] = opp_cards[0], opp_cards[1]
            else:
                deck[0], deck[1] = opp_cards[0], opp_cards[1]
                deck[2], deck[3] = obs.hero_cards[0], obs.hero_cards[1]

            full_board = list(obs.board) + rem_board
            for idx, card in enumerate(full_board):
                deck[4 + idx] = card
            for idx, card in enumerate(burns):
                deck[9 + idx] = card
            cf_decks.append(deck)

        evs: dict[DiscreteAction, float] = {}
        for action in obs.legal_actions:
            if action == DiscreteAction.FOLD:
                # Exact deterministic payoff on fold: no rollout needed
                sim = env.clone()
                sim.step(action)
                evs[action] = sim.payoffs_bb[hero_seat]
                continue

            total = 0.0
            for deck in cf_decks:
                sim = env.clone(override_deck=deck)
                sim.step(action)
                total += self._rollout_to_terminal(sim, hero_seat, hero_policy, opponent)
            evs[action] = total / max(1, len(cf_decks))
        return evs
