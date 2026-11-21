"""Duplicate Deals Arena for low-variance BB/100 evaluation and trajectory logging."""

from __future__ import annotations

import json
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

from tqdm import tqdm

from pocket_r1.agents.base import BaseAgent
from pocket_r1.env.nlhe_env import STANDARD_DECK, HeadsUpNLHEEnv
from pocket_r1.eval.metrics import EvaluationSummary, compute_eval_summary
from pocket_r1.prompts.state_formatter import format_observation
from pocket_r1.utils.logging import ParsedOutputSchema, TrajectoryLogger, TrajectoryRecord
from pocket_r1.utils.seeding import make_rng
from pocket_r1.verifiers.ev_calculator import EVCalculator
from pocket_r1.verifiers.step_verifier import verify_cot_steps


@dataclass(frozen=True)
class DuplicateHandSpec:
    pair_id: str
    seed: int
    opponent_type: str
    deck: list[str]  # 12 cards: 4 hole + 5 board + 3 burn


def generate_duplicate_suite(
    opponents: list[str],
    pairs_per_opponent: int = 250,
    base_seed: int = 202601,
) -> list[DuplicateHandSpec]:
    """Generate deterministic fixed decks for Duplicate Deals evaluation."""
    specs: list[DuplicateHandSpec] = []
    for opp_idx, opp in enumerate(opponents):
        for p in range(pairs_per_opponent):
            hand_seed = base_seed + opp_idx * 100000 + p
            rng = make_rng(hand_seed)
            deck = list(STANDARD_DECK)
            rng.shuffle(deck)
            specs.append(
                DuplicateHandSpec(
                    pair_id=f"hu_nlhe_{opp}_{p:04d}",
                    seed=hand_seed,
                    opponent_type=opp,
                    deck=deck[:12],
                )
            )
    return specs


def save_eval_suite(specs: list[DuplicateHandSpec], path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for spec in specs:
            f.write(json.dumps(asdict(spec)) + "\n")


def load_eval_suite(path: str | Path) -> list[DuplicateHandSpec]:
    specs: list[DuplicateHandSpec] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                specs.append(DuplicateHandSpec(**json.loads(line)))
    return specs


class DuplicateArena:
    """Runs Duplicate Deal matches (both seats with identical deck) to cut variance."""

    def __init__(
        self,
        starting_stack_bb: float = 100.0,
        compute_ev: bool = False,
        ev_rollouts: int = 24,
        trajectory_logger: TrajectoryLogger | None = None,
    ) -> None:
        self.starting_stack_bb = starting_stack_bb
        self.compute_ev = compute_ev
        self.ev_calc = EVCalculator(default_rollouts=ev_rollouts) if compute_ev else None
        self.trajectory_logger = trajectory_logger

    def _play_single_hand(
        self,
        hero: BaseAgent,
        opponent: BaseAgent,
        hero_seat: int,
        spec: DuplicateHandSpec,
        hand_suffix: str,
    ) -> tuple[float, list[TrajectoryRecord]]:
        env = HeadsUpNLHEEnv(starting_stack_bb=self.starting_stack_bb)
        hand_id = f"{spec.pair_id}_{hand_suffix}"
        obs = env.reset(seed=spec.seed, deck=spec.deck, hand_id=hand_id)

        hand_records: list[TrajectoryRecord] = []
        decision_idx = 0

        while not env.is_terminal:
            assert obs is not None
            seat = obs.seat
            if seat == hero_seat:
                res = hero.act(obs)
                ev_bb: float | None = None
                if self.ev_calc is not None:
                    ev_bb = self.ev_calc.compute_single_action_ev(
                        env=env,
                        hero_seat=hero_seat,
                        action=res.action,
                        opponent=opponent,
                        seed=spec.seed + decision_idx,
                    )

                auto_checks: dict = {}
                if res.parsed is not None:
                    v = verify_cot_steps(obs, res.parsed.steps, res.parsed.action, equity_samples=60)
                    auto_checks = v.to_dict()
                    parsed_schema = ParsedOutputSchema(
                        is_valid=res.parsed.is_valid,
                        steps=res.parsed.steps,
                        action=res.parsed.action.value if res.parsed.action else None,
                        error_codes=res.parsed.error_codes,
                        response_tokens=res.parsed.response_tokens,
                    )
                else:
                    parsed_schema = ParsedOutputSchema(
                        is_valid=True,
                        steps=[],
                        action=res.action.value,
                        error_codes=[],
                        response_tokens=0,
                    )

                hand_records.append(
                    TrajectoryRecord(
                        hand_id=hand_id,
                        env_name=env.env_name,
                        seed=spec.seed,
                        opponent_type=spec.opponent_type,
                        street=obs.street,
                        decision_idx=decision_idx,
                        state_json=obs.to_state_json(),
                        prompt_text=res.prompt_text or format_observation(obs),
                        raw_response=res.raw_response,
                        parsed=parsed_schema,
                        auto_checks=auto_checks,
                        ev_bb=round(ev_bb, 3) if ev_bb is not None else None,
                        hand_outcome_bb=None,
                    )
                )
                decision_idx += 1
                action = res.action
            else:
                action = opponent.act(obs).action

            obs, _, _, _ = env.step(action)

        payoff_bb = env.payoffs_bb[hero_seat]
        for rec in hand_records:
            rec.hand_outcome_bb = round(payoff_bb, 3)

        if self.trajectory_logger is not None:
            self.trajectory_logger.log_many(hand_records)

        return payoff_bb, hand_records

    def evaluate(
        self,
        hero: BaseAgent,
        opponent: BaseAgent,
        specs: list[DuplicateHandSpec],
        show_progress: bool = False,
        max_workers: int = 1,
        opponent_factory: Callable[[int], BaseAgent] | None = None,
    ) -> tuple[EvaluationSummary, list[float], list[TrajectoryRecord]]:
        """Evaluate hero vs opponent on duplicate pairs (supports concurrent vLLM batching)."""
        if max_workers <= 1:
            pair_mean_bbs: list[float] = []
            all_records: list[TrajectoryRecord] = []
            iterator = tqdm(specs, desc=f"vs {opponent.name}", disable=not show_progress)
            for spec in iterator:
                p_a, rec_a = self._play_single_hand(hero, opponent, 0, spec, "seat0")
                p_b, rec_b = self._play_single_hand(hero, opponent, 1, spec, "seat1")
                pair_mean_bbs.append((p_a + p_b) / 2.0)
                all_records.extend(rec_a)
                all_records.extend(rec_b)
            summary = compute_eval_summary(pair_mean_bbs, all_records)
            return summary, pair_mean_bbs, all_records

        # Concurrent execution: each pair gets an isolated opponent instance for thread safety
        orig_logger = self.trajectory_logger
        self.trajectory_logger = None  # log deterministically on main thread

        def _run_pair(idx_spec: tuple[int, DuplicateHandSpec]) -> tuple[int, float, list[TrajectoryRecord]]:
            idx, spec = idx_spec
            opp_inst = opponent_factory(spec.seed) if opponent_factory is not None else opponent
            p_a, rec_a = self._play_single_hand(hero, opp_inst, 0, spec, "seat0")
            p_b, rec_b = self._play_single_hand(hero, opp_inst, 1, spec, "seat1")
            return idx, (p_a + p_b) / 2.0, rec_a + rec_b

        results: dict[int, tuple[float, list[TrajectoryRecord]]] = {}
        try:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = [pool.submit(_run_pair, (i, s)) for i, s in enumerate(specs)]
                for fut in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc=f"vs {opponent.name} ({max_workers}w)",
                    disable=not show_progress,
                ):
                    idx, mean_bb, recs = fut.result()
                    results[idx] = (mean_bb, recs)
        finally:
            self.trajectory_logger = orig_logger

        pair_mean_bbs = []
        all_records = []
        for i in range(len(specs)):
            mean_bb, recs = results[i]
            pair_mean_bbs.append(mean_bb)
            all_records.extend(recs)
            if self.trajectory_logger is not None:
                self.trajectory_logger.log_many(recs)

        summary = compute_eval_summary(pair_mean_bbs, all_records)
        return summary, pair_mean_bbs, all_records
