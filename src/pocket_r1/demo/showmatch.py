"""Showmatch 'David vs. Goliath' runner over Duplicate Deals with latency & BB/100 tracking."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from pocket_r1.agents.base import BaseAgent
from pocket_r1.eval import DuplicateArena, generate_duplicate_suite, render_markdown_report
from pocket_r1.utils.logging import TrajectoryLogger


def run_showmatch(
    david: BaseAgent,
    goliath: BaseAgent,
    n_pairs: int = 250,
    seed: int = 202699,
    output_dir: str | Path = "reports/showmatch_david_vs_goliath",
    max_workers: int = 8,
) -> dict[str, Any]:
    """Run Duplicate Deals Showmatch between David (Pocket-R1 SLM) and Goliath (Large LLM/GTO)."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    specs = generate_duplicate_suite([goliath.name], pairs_per_opponent=n_pairs, base_seed=seed)

    traj_path = out / "showmatch_trajectories.jsonl"
    if traj_path.exists():
        traj_path.unlink()

    t0 = time.perf_counter()
    arena = DuplicateArena(compute_ev=False, trajectory_logger=TrajectoryLogger(traj_path))
    summary, _pair_bbs, _records = arena.evaluate(
        hero=david,
        opponent=goliath,
        specs=specs,
        show_progress=True,
        max_workers=max_workers,
    )
    elapsed = time.perf_counter() - t0

    md = render_markdown_report("showmatch_david_vs_goliath", david.name, {goliath.name: summary}, summary)
    (out / "summary.md").write_text(md, encoding="utf-8")

    payload = {
        "david": david.name,
        "goliath": goliath.name,
        "elapsed_sec": round(elapsed, 2),
        "sec_per_hand": round(elapsed / max(1, summary.total_hands), 3),
        "summary": summary.model_dump(),
    }
    (out / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
