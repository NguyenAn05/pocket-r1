"""Unit tests for Ablation Markdown compiler and Interactive HTML Visualizer."""

from __future__ import annotations

from pathlib import Path

from pocket_r1.demo import (
    load_ablation_summaries,
    render_ablation_markdown,
    render_interactive_html_report,
)


def test_ablation_markdown_and_html_visualizer(tmp_path: Path) -> None:
    rows = load_ablation_summaries("reports")
    assert len(rows) >= 2  # exp0_baseline and exp1b_sft_fim already exist!
    md = render_ablation_markdown(rows)
    assert "exp0_baseline" in md
    assert "exp1b_sft_fim" in md

    highlights = [
        {
            "hand_id": "demo_hand_1",
            "opponent_type": "maniac",
            "hand_outcome_bb": 24.5,
            "prompt_text": "Game: HU_NLHE | Hole Cards: As Ks | Board: Qs 7s 2h",
            "steps": ["1. ok", "2. ok", "3. ok", "4. ok", "5. ok"],
            "action": "CHECK_CALL",
            "step_probs": [0.98, 0.95, 0.99, 0.90, 0.88, 0.94],
        }
    ]
    out_html = render_interactive_html_report(rows, highlights, tmp_path / "report.html")
    content = out_html.read_text(encoding="utf-8")
    assert "Pocket-R1" in content
    assert "p1=0.98" in content
    assert "demo_hand_1" in content
