"""Generate unified Ablation Study Markdown and interactive HTML dashboard."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pocket_r1.demo import (
    load_ablation_summaries,
    render_ablation_markdown,
    render_interactive_html_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate unified Pocket-R1 Ablation & HTML report.")
    parser.add_argument("--reports-root", type=str, default="reports")
    parser.add_argument("--highlight-trajectories", type=str, default=None)
    parser.add_argument("--max-highlights", type=int, default=12)
    args = parser.parse_args()

    rows = load_ablation_summaries(args.reports_root)
    md = render_ablation_markdown(rows)
    md_path = Path(args.reports_root) / "ablation_summary.md"
    md_path.write_text(md, encoding="utf-8")
    print(md)

    highlights: list[dict] = []
    if args.highlight_trajectories and Path(args.highlight_trajectories).exists():
        with Path(args.highlight_trajectories).open("r", encoding="utf-8") as f:
            for line in f:
                if len(highlights) >= args.max_highlights:
                    break
                r = json.loads(line)
                parsed = r.get("parsed") or {}
                if len(parsed.get("steps") or []) == 5:
                    highlights.append(
                        {
                            "hand_id": r["hand_id"],
                            "opponent_type": r["opponent_type"],
                            "hand_outcome_bb": r.get("hand_outcome_bb") or 0.0,
                            "prompt_text": r["prompt_text"],
                            "steps": parsed["steps"],
                            "action": parsed.get("action"),
                            "step_probs": r.get("step_probs") or [0.95] * 6,
                        }
                    )

    html_path = render_interactive_html_report(
        rows, highlights, Path(args.reports_root) / "showmatch_report.html"
    )
    print(f"Saved Markdown -> {md_path} | Interactive HTML -> {html_path}")


if __name__ == "__main__":
    main()
