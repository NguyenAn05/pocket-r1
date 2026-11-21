"""Adapter to convert external PokerBench-style solver records into 5-step Verified CoT."""

from __future__ import annotations

import re
from typing import Any

from pocket_r1.data.cot_synthesizer import SFTExample, synthesize_verified_cot
from pocket_r1.env import DiscreteAction, Observation
from pocket_r1.prompts.state_formatter import estimate_tokens, format_observation
from pocket_r1.prompts.templates import SYSTEM_PROMPT
from pocket_r1.utils.seeding import make_rng

_BET_AMOUNT_RE = re.compile(r"(?:bet|raise)\s+(\d+(?:\.\d+)?)", re.IGNORECASE)


def map_pokerbench_action(
    raw_action: str,
    pot_bb: float,
    to_call_bb: float,
    hero_stack_bb: float,
    legal_actions: list[DiscreteAction],
) -> DiscreteAction:
    """Map a continuous PokerBench action string ('fold', 'call', 'check', 'bet 6') to DiscreteAction."""
    s = raw_action.strip().lower()
    if "fold" in s:
        return DiscreteAction.FOLD if DiscreteAction.FOLD in legal_actions else DiscreteAction.CHECK_CALL
    if "check" in s or "call" in s:
        return DiscreteAction.CHECK_CALL
    if "all" in s:
        return DiscreteAction.ALL_IN if DiscreteAction.ALL_IN in legal_actions else DiscreteAction.BET_150

    m = _BET_AMOUNT_RE.search(s)
    if m:
        amt = float(m.group(1))
        if amt >= 0.85 * hero_stack_bb and DiscreteAction.ALL_IN in legal_actions:
            return DiscreteAction.ALL_IN
        pot_after_call = max(1.0, pot_bb + to_call_bb)
        ratio = amt / pot_after_call
        if ratio < 0.55 and DiscreteAction.BET_33 in legal_actions:
            return DiscreteAction.BET_33
        if ratio < 1.15 and DiscreteAction.BET_75 in legal_actions:
            return DiscreteAction.BET_75
        if DiscreteAction.BET_150 in legal_actions:
            return DiscreteAction.BET_150
    for fallback in (DiscreteAction.BET_75, DiscreteAction.BET_33, DiscreteAction.ALL_IN):
        if fallback in legal_actions:
            return fallback
    return DiscreteAction.CHECK_CALL


def convert_pokerbench_records(
    records: list[dict[str, Any]],
    seed: int = 900000,
) -> list[SFTExample]:
    """Convert structured PokerBench-like spot dicts into verified 5-step SFTExamples."""
    rng = make_rng(seed)
    out: list[SFTExample] = []

    for idx, rec in enumerate(records):
        hero_cards = list(rec["hero_cards"])
        board = list(rec.get("board", []))
        pot_bb = float(rec.get("pot_bb", 3.0))
        to_call_bb = float(rec.get("to_call_bb", 0.0))
        hero_stack_bb = float(rec.get("hero_stack_bb", 95.0))
        villain_stack_bb = float(rec.get("villain_stack_bb", 95.0))
        position = str(rec.get("position", "BTN")).upper()
        street = str(
            rec.get(
                "street",
                "PREFLOP" if len(board) == 0 else ("FLOP" if len(board) == 3 else ("TURN" if len(board) == 4 else "RIVER")),
            )
        ).upper()

        legal = [DiscreteAction.CHECK_CALL, DiscreteAction.BET_33, DiscreteAction.BET_75, DiscreteAction.BET_150, DiscreteAction.ALL_IN]
        if to_call_bb > 0:
            legal = [DiscreteAction.FOLD] + legal

        obs = Observation(
            env_name="hu_nlhe",
            hand_id=f"pokerbench_{idx:06d}",
            seed=seed + idx,
            street=street,
            seat=1 if position == "BTN" else 0,
            position=position,
            hero_cards=hero_cards,
            board=board,
            pot_bb=pot_bb,
            to_call_bb=to_call_bb,
            hero_stack_bb=hero_stack_bb,
            villain_stack_bb=villain_stack_bb,
            hero_street_bet_bb=0.0,
            villain_street_bet_bb=to_call_bb,
            legal_actions=legal,
        )
        act = map_pokerbench_action(
            str(rec["action"]), pot_bb, to_call_bb, hero_stack_bb, legal
        )
        cot = synthesize_verified_cot(obs, act, opponent_type="tag", rng=rng, equity_samples=80)
        if cot is None:
            continue
        raw_response, steps = cot
        user_prompt = format_observation(obs)
        total_tokens = estimate_tokens(SYSTEM_PROMPT + user_prompt + raw_response)
        if total_tokens > 850:
            continue

        out.append(
            SFTExample(
                example_id=f"pokerbench_{idx:06d}",
                hand_id=obs.hand_id,
                env_name="hu_nlhe",
                task_type="standard_cot",
                seed=obs.seed,
                street=street,
                action=act.value,
                opponent_type="gto_pokerbench",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": raw_response},
                ],
                steps=steps,
                meta={
                    "hero_cards": hero_cards,
                    "board": board,
                    "position": position,
                    "source": "pokerbench",
                },
            )
        )
    return out
