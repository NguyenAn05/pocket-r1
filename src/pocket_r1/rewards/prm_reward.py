"""Process Reward aggregation (R_DeBERTa) from 6 per-step probabilities."""

from __future__ import annotations

import math


def aggregate_prm_score(step_probs: list[float], mode: str = "geom_mean") -> float:
    """Aggregate 6 step probabilities into R_DeBERTa in [0.0, 1.0].

    Modes:
    - 'geom_mean': smooth geometric mean (prod p_k)^(1/T) — recommended for early GRPO.
    - 'min': strict weakest-link min(p_k) — strongly penalizes any single hallucinated step.
    """
    if not step_probs:
        return 0.0
    clamped = [max(1e-6, min(1.0, float(p))) for p in step_probs]
    if mode == "min":
        return round(min(clamped), 4)
    if mode == "geom_mean":
        val = math.exp(sum(math.log(p) for p in clamped) / len(clamped))
        return round(val, 4)
    raise ValueError(f"Unknown PRM aggregation mode: {mode}")
