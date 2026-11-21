"""Unit tests for Duplicate Deals Arena, Held-out Suite, and Metrics."""

from __future__ import annotations

from pathlib import Path

import pytest

from pocket_r1.agents import MockSLMAgent, create_bot
from pocket_r1.eval import (
    DuplicateArena,
    generate_duplicate_suite,
    load_eval_suite,
    render_markdown_report,
    save_eval_suite,
)
from pocket_r1.utils.logging import TrajectoryLogger


def test_duplicate_suite_roundtrip(tmp_path: Path) -> None:
    specs = generate_duplicate_suite(["calling_station", "nit"], pairs_per_opponent=5, base_seed=100)
    assert len(specs) == 10
    p = tmp_path / "suite.jsonl"
    save_eval_suite(specs, p)
    loaded = load_eval_suite(p)
    assert loaded == specs


def test_duplicate_arena_end_to_end_with_mock_slm(tmp_path: Path) -> None:
    specs = generate_duplicate_suite(["calling_station"], pairs_per_opponent=5, base_seed=55)
    traj_path = tmp_path / "traj.jsonl"
    arena = DuplicateArena(
        compute_ev=False,
        trajectory_logger=TrajectoryLogger(traj_path),
    )
    hero = MockSLMAgent(seed=42)
    opp = create_bot("calling_station", seed=42, equity_samples=25)
    summary, pair_bbs, records = arena.evaluate(hero, opp, specs, show_progress=False)

    assert summary.total_hands == 10
    assert summary.total_decisions == len(records) > 0
    assert summary.format_validity_rate == pytest.approx(1.0)
    assert summary.illegal_action_rate == pytest.approx(0.0)
    assert summary.card_hallucination_rate == pytest.approx(0.0)
    assert summary.pot_odds_accuracy == pytest.approx(1.0)

    loaded_recs = TrajectoryLogger.load(traj_path)
    assert len(loaded_recs) == len(records)

    md = render_markdown_report("test_exp", hero.name, {"calling_station": summary}, summary)
    assert "BB/100" in md and "OVERALL" in md
