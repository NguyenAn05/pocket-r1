"""Evaluation metrics (BB/100 with 95% CI + Reasoning Integrity) and Markdown reporting."""

from __future__ import annotations

import math

import numpy as np
from pydantic import BaseModel

from pocket_r1.utils.logging import TrajectoryRecord


class EvaluationSummary(BaseModel):
    total_hands: int
    total_decisions: int
    bb_per_100: float
    bb_per_100_ci95: float
    format_validity_rate: float
    illegal_action_rate: float
    card_hallucination_rate: float
    step2_hand_eval_acc: float
    pot_odds_accuracy: float
    logic_action_consistency: float
    mean_response_tokens: float
    mean_ev_bb: float | None = None


def compute_eval_summary(
    pair_mean_hand_bbs: list[float],
    records: list[TrajectoryRecord],
) -> EvaluationSummary:
    """Compute BB/100 (paired 95% CI) and Reasoning Integrity metrics."""
    n_pairs = len(pair_mean_hand_bbs)
    total_hands = n_pairs * 2
    if n_pairs > 0:
        arr = np.asarray(pair_mean_hand_bbs, dtype=np.float64)
        mean_bb = float(np.mean(arr))
        bb_per_100 = mean_bb * 100.0
        if n_pairs > 1:
            se = float(np.std(arr, ddof=1)) / math.sqrt(n_pairs)
            ci95 = 1.96 * se * 100.0
        else:
            ci95 = 0.0
    else:
        bb_per_100 = 0.0
        ci95 = 0.0

    n_dec = len(records)
    if n_dec == 0:
        return EvaluationSummary(
            total_hands=total_hands,
            total_decisions=0,
            bb_per_100=round(bb_per_100, 2),
            bb_per_100_ci95=round(ci95, 2),
            format_validity_rate=0.0,
            illegal_action_rate=0.0,
            card_hallucination_rate=0.0,
            step2_hand_eval_acc=0.0,
            pot_odds_accuracy=0.0,
            logic_action_consistency=0.0,
            mean_response_tokens=0.0,
            mean_ev_bb=None,
        )

    valid_cnt = sum(1 for r in records if r.parsed.is_valid)
    illegal_cnt = sum(1 for r in records if "ILLEGAL_ACTION" in r.parsed.error_codes)
    tokens = [r.parsed.response_tokens for r in records]

    checked = [r for r in records if r.auto_checks]
    n_chk = max(1, len(checked))
    halluc_cnt = sum(1 for r in checked if not r.auto_checks.get("step1_card_exact", True))
    s2_ok = sum(1 for r in checked if r.auto_checks.get("step2_hand_eval_ok", True))
    po_ok = sum(1 for r in checked if r.auto_checks.get("step3_pot_odds_ok", True))
    cons_ok = sum(1 for r in checked if r.auto_checks.get("logic_action_consistent", True))

    ev_vals = [r.ev_bb for r in records if r.ev_bb is not None]
    mean_ev = round(float(np.mean(ev_vals)), 3) if ev_vals else None

    return EvaluationSummary(
        total_hands=total_hands,
        total_decisions=n_dec,
        bb_per_100=round(bb_per_100, 2),
        bb_per_100_ci95=round(ci95, 2),
        format_validity_rate=round(valid_cnt / n_dec, 4),
        illegal_action_rate=round(illegal_cnt / n_dec, 4),
        card_hallucination_rate=round(halluc_cnt / n_chk, 4),
        step2_hand_eval_acc=round(s2_ok / n_chk, 4),
        pot_odds_accuracy=round(po_ok / n_chk, 4),
        logic_action_consistency=round(cons_ok / n_chk, 4),
        mean_response_tokens=round(float(np.mean(tokens)), 1),
        mean_ev_bb=mean_ev,
    )


def render_markdown_report(
    experiment_name: str,
    model_name: str,
    per_opponent: dict[str, EvaluationSummary],
    overall: EvaluationSummary,
) -> str:
    """Render a clean Markdown benchmark report."""
    lines = [
        f"# Benchmark Report — `{experiment_name}`",
        "",
        f"- **Agent / Model:** `{model_name}`",
        f"- **Total Hands:** `{overall.total_hands}` ({overall.total_decisions} Hero decision points)",
        f"- **Overall Performance:** `{overall.bb_per_100:+.2f} ± {overall.bb_per_100_ci95:.2f} BB/100` (95% CI)",
        "",
        "## Per-Opponent Results",
        "",
        "| Opponent | Hands | BB/100 (95% CI) | Format Valid | Illegal Act | Card Halluc | Pot Odds Acc | Consistency | Avg Tokens |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for opp, s in per_opponent.items():
        lines.append(
            f"| `{opp}` | {s.total_hands} | `{s.bb_per_100:+.2f} ± {s.bb_per_100_ci95:.2f}` | "
            f"{s.format_validity_rate * 100:.1f}% | {s.illegal_action_rate * 100:.2f}% | "
            f"{s.card_hallucination_rate * 100:.1f}% | {s.pot_odds_accuracy * 100:.1f}% | "
            f"{s.logic_action_consistency * 100:.1f}% | {s.mean_response_tokens:.0f} |"
        )
    lines.append(
        f"| **OVERALL** | **{overall.total_hands}** | "
        f"**`{overall.bb_per_100:+.2f} ± {overall.bb_per_100_ci95:.2f}`** | "
        f"**{overall.format_validity_rate * 100:.1f}%** | "
        f"**{overall.illegal_action_rate * 100:.2f}%** | "
        f"**{overall.card_hallucination_rate * 100:.1f}%** | "
        f"**{overall.pot_odds_accuracy * 100:.1f}%** | "
        f"**{overall.logic_action_consistency * 100:.1f}%** | "
        f"**{overall.mean_response_tokens:.0f}** |"
    )
    lines.append("")
    return "\n".join(lines)
