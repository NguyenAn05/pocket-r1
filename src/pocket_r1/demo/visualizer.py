"""Ablation Study Markdown compiler and interactive HTML visualizer with DeBERTa step heatmaps."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any


def load_ablation_summaries(reports_root: str | Path = "reports") -> list[dict[str, Any]]:
    """Load all available summary.json files in canonical experiment order."""
    root = Path(reports_root)
    ordered_names = [
        "exp0_baseline",
        "exp1_sft",
        "exp1b_sft_fim",
        "exp2_grpo_outcome",
        "exp3_grpo_ev",
        "exp4_grpo_hybrid",
        "exp5_coevolution",
    ]
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for name in ordered_names:
        p = root / name / "summary.json"
        if p.exists():
            rows.append(json.loads(p.read_text(encoding="utf-8")))
            seen.add(name)
    # Include any extra custom experiment folders
    if root.exists():
        for child in sorted(root.iterdir()):
            p = child / "summary.json"
            if child.is_dir() and p.exists() and child.name not in seen:
                rows.append(json.loads(p.read_text(encoding="utf-8")))
    return rows


def render_ablation_markdown(rows: list[dict[str, Any]]) -> str:
    """Compile a unified Markdown Ablation Study comparison table."""
    lines = [
        "# Pocket-R1 — Unified Ablation Study Report",
        "",
        "| Experiment | Model | Hands | BB/100 (95% CI) | Format | Illegal | Step 2 Acc | Pot Odds | Consistency | Tokens |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for r in rows:
        ov = r["overall"]
        lines.append(
            f"| `{r['experiment']}` | `{r['model']}` | {ov['total_hands']} | "
            f"`{ov['bb_per_100']:+.2f} ± {ov['bb_per_100_ci95']:.2f}` | "
            f"{ov['format_validity_rate'] * 100:.1f}% | "
            f"{ov['illegal_action_rate'] * 100:.2f}% | "
            f"{ov.get('step2_hand_eval_acc', 0.0) * 100:.1f}% | "
            f"{ov['pot_odds_accuracy'] * 100:.1f}% | "
            f"{ov['logic_action_consistency'] * 100:.1f}% | "
            f"{ov['mean_response_tokens']:.0f} |"
        )
    lines.append("")
    return "\n".join(lines)


def _prob_badge_color(p: float) -> str:
    if p >= 0.80:
        return "#16a34a"  # green
    if p >= 0.50:
        return "#d97706"  # amber
    return "#dc2626"  # red


def render_interactive_html_report(
    ablation_rows: list[dict[str, Any]],
    highlight_hands: list[dict[str, Any]],
    output_path: str | Path,
) -> Path:
    """Generate a standalone dark-mode HTML dashboard for Ablation & Highlight Hands."""
    table_rows_html = []
    for r in ablation_rows:
        ov = r["overall"]
        bb_color = "#4ade80" if ov["bb_per_100"] >= 0 else "#f87171"
        table_rows_html.append(
            f"<tr>"
            f"<td><code>{html.escape(str(r['experiment']))}</code></td>"
            f"<td><code>{html.escape(str(r['model']))}</code></td>"
            f"<td>{ov['total_hands']}</td>"
            f"<td style='color:{bb_color};font-weight:600'>{ov['bb_per_100']:+.2f} ± {ov['bb_per_100_ci95']:.2f}</td>"
            f"<td>{ov['format_validity_rate'] * 100:.1f}%</td>"
            f"<td>{ov.get('step2_hand_eval_acc', 0.0) * 100:.1f}%</td>"
            f"<td>{ov['pot_odds_accuracy'] * 100:.1f}%</td>"
            f"<td>{ov['logic_action_consistency'] * 100:.1f}%</td>"
            f"<td>{ov['mean_response_tokens']:.0f}</td>"
            f"</tr>"
        )

    hand_cards_html = []
    for h in highlight_hands[:20]:
        steps = h.get("steps") or []
        probs = h.get("step_probs") or [1.0] * 6
        step_items = []
        for idx, s in enumerate(steps):
            p = float(probs[idx]) if idx < len(probs) else 1.0
            color = _prob_badge_color(p)
            step_items.append(
                f"<div class='step-row'>"
                f"<span class='badge' style='background:{color}'>p{idx + 1}={p:.2f}</span>"
                f"<span>{html.escape(str(s))}</span>"
                f"</div>"
            )
        act_p = float(probs[5]) if len(probs) >= 6 else 1.0
        act_color = _prob_badge_color(act_p)
        hand_cards_html.append(
            f"<div class='card'>"
            f"<div class='card-header'>"
            f"<strong>{html.escape(str(h.get('hand_id', 'hand')))}</strong> "
            f"| vs <code>{html.escape(str(h.get('opponent_type', 'bot')))}</code> "
            f"| Outcome: <code>{h.get('hand_outcome_bb', 0.0):+.1f} BB</code>"
            f"</div>"
            f"<pre class='state'>{html.escape(str(h.get('prompt_text', '')))}</pre>"
            f"{''.join(step_items)}"
            f"<div class='step-row'>"
            f"<span class='badge' style='background:{act_color}'>p6={act_p:.2f}</span>"
            f"<strong>ACTION: <code>{html.escape(str(h.get('action', '')))}</code></strong>"
            f"</div>"
            f"</div>"
        )

    doc = f"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="utf-8" />
<title>Pocket-R1 — Ablation & DeBERTa PRM Visualizer</title>
<style>
  body {{ font-family: system-ui, -apple-system, sans-serif; background: #0f172a; color: #e2e8f0; margin: 0; padding: 24px; }}
  h1, h2 {{ color: #f8fafc; }}
  table {{ width: 100%; border-collapse: collapse; margin: 16px 0 32px; background: #1e293b; border-radius: 8px; overflow: hidden; }}
  th, td {{ padding: 10px 12px; border-bottom: 1px solid #334155; text-align: center; font-size: 14px; }}
  th {{ background: #0f172a; color: #94a3b8; text-transform: uppercase; font-size: 12px; }}
  .card {{ background: #1e293b; border: 1px solid #334155; border-radius: 10px; padding: 16px; margin-bottom: 16px; }}
  .card-header {{ margin-bottom: 10px; color: #cbd5e1; }}
  .state {{ background: #0f172a; padding: 10px; border-radius: 6px; font-size: 12px; overflow-x: auto; color: #93c5fd; }}
  .step-row {{ display: flex; align-items: flex-start; gap: 10px; margin: 6px 0; font-size: 14px; line-height: 1.45; }}
  .badge {{ color: #fff; font-weight: 700; padding: 2px 8px; border-radius: 999px; font-size: 12px; white-space: nowrap; }}
  code {{ background: #0f172a; padding: 2px 6px; border-radius: 4px; }}
</style>
</head>
<body>
  <h1>Pocket-R1 — Self-Evolving Poker SLM Dashboard</h1>
  <h2>1. Ablation Study Summary (Held-out 2,000 Duplicate Deals)</h2>
  <table>
    <thead>
      <tr>
        <th>Experiment</th><th>Model</th><th>Hands</th><th>BB/100 (95% CI)</th>
        <th>Format</th><th>Step 2 Acc</th><th>Pot Odds</th><th>Consistency</th><th>Tokens</th>
      </tr>
    </thead>
    <tbody>
      {''.join(table_rows_html)}
    </tbody>
  </table>
  <h2>2. Highlight Hands &amp; DeBERTa-v3 Step-level Heatmap</h2>
  {''.join(hand_cards_html) if hand_cards_html else '<p>No highlight hands provided yet.</p>'}
</body>
</html>
"""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(doc, encoding="utf-8")
    return out
