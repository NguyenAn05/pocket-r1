"""Run Exp 0 Baseline benchmark on the fixed Held-out Evaluation Suite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from pocket_r1.agents import MockSLMAgent, SLMAgent, create_bot
from pocket_r1.eval import (
    DuplicateArena,
    EvaluationSummary,
    compute_eval_summary,
    load_eval_suite,
    render_markdown_report,
)
from pocket_r1.utils.logging import TrajectoryLogger, TrajectoryRecord


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Exp 0 (Base SLM) benchmark.")
    parser.add_argument("--config", type=str, default="configs/eval/exp0_baseline.yaml")
    parser.add_argument("--backend", type=str, default=None, choices=["openai_compatible", "mock"])
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--base-url", type=str, default=None)
    parser.add_argument("--pairs", type=int, default=None, help="Limit pairs per opponent")
    parser.add_argument("--workers", type=int, default=None, help="Concurrent worker threads")
    parser.add_argument("--no-ev", action="store_true", help="Skip Monte Carlo EV calculation")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    cfg: dict[str, Any] = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    slm_cfg = cfg["slm"]
    backend = args.backend or slm_cfg.get("backend", "openai_compatible")
    model_name = args.model or slm_cfg["model_name"]
    base_url = args.base_url or slm_cfg.get("base_url", "http://localhost:8000/v1")
    max_workers = args.workers if args.workers is not None else int(cfg.get("max_workers", 1))
    out_dir = Path(args.output_dir or cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    if backend == "mock":
        hero = MockSLMAgent(seed=cfg.get("seed", 42))
    else:
        hero = SLMAgent(
            model_name=model_name,
            base_url=base_url,
            api_key_env=slm_cfg.get("api_key_env", "OPENAI_API_KEY"),
            temperature=slm_cfg.get("temperature", 0.0),
            max_tokens=slm_cfg.get("max_tokens", 512),
            few_shot=slm_cfg.get("few_shot", True),
            enable_thinking=slm_cfg.get("enable_thinking", False),
        )

    all_specs = load_eval_suite(cfg["eval_suite_path"])
    traj_path = out_dir / "trajectories.jsonl"
    if traj_path.exists():
        traj_path.unlink()

    arena = DuplicateArena(
        compute_ev=(not args.no_ev) and cfg.get("compute_ev", True),
        ev_rollouts=cfg.get("ev_rollouts", 24),
        trajectory_logger=TrajectoryLogger(traj_path),
    )

    per_opp: dict[str, EvaluationSummary] = {}
    all_pair_bbs: list[float] = []
    all_records: list[TrajectoryRecord] = []

    for opp_name in cfg["opponents"]:
        opp_specs = [s for s in all_specs if s.opponent_type == opp_name]
        if args.pairs is not None:
            opp_specs = opp_specs[: args.pairs]
        opp_bot = create_bot(opp_name, seed=cfg.get("seed", 42), equity_samples=50)
        summary, pair_bbs, recs = arena.evaluate(
            hero,
            opp_bot,
            opp_specs,
            show_progress=True,
            max_workers=max_workers,
            opponent_factory=lambda s, n=opp_name: create_bot(n, seed=s, equity_samples=50),
        )
        per_opp[opp_name] = summary
        all_pair_bbs.extend(pair_bbs)
        all_records.extend(recs)

    overall = compute_eval_summary(all_pair_bbs, all_records)
    md = render_markdown_report(cfg["experiment_name"], hero.name, per_opp, overall)

    (out_dir / "summary.md").write_text(md, encoding="utf-8")
    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "experiment": cfg["experiment_name"],
                "model": hero.name,
                "overall": overall.model_dump(),
                "per_opponent": {k: v.model_dump() for k, v in per_opp.items()},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(md)


if __name__ == "__main__":
    main()
