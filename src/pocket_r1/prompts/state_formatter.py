"""Concise, standardized state observation formatting for SLM and DeBERTa-v3."""

from __future__ import annotations

import math

from pocket_r1.env.base import ActionHistoryEntry, Observation


def estimate_tokens(text: str) -> int:
    """Fast, dependency-free upper-bound token count estimate (~BPE/SentencePiece)."""
    if not text:
        return 0
    word_est = int(math.ceil(len(text.split()) * 1.35))
    char_est = int(math.ceil(len(text) / 3.8))
    return max(word_est, char_est)


def format_action_history(history: list[ActionHistoryEntry], max_entries: int = 24) -> str:
    """Compress action history grouped by street into a single compact line."""
    if not history:
        return "None"
    entries = history[-max_entries:]
    grouped: list[tuple[str, list[str]]] = []
    for entry in entries:
        token = f"{entry.position} {entry.action.value}"
        if entry.amount_bb > 0:
            token += f"({entry.amount_bb:g}BB)"
        if not grouped or grouped[-1][0] != entry.street:
            grouped.append((entry.street, [token]))
        else:
            grouped[-1][1].append(token)
    return " | ".join(f"{street}: {' -> '.join(acts)}" for street, acts in grouped)


def format_observation(obs: Observation) -> str:
    """Format a canonical Observation into a compact (<250 token) state prompt.

    Pot Odds convention (locked across the entire project):
        Pot Odds = To Call / (Current Pot + To Call)
    where Current Pot already includes villain's bet currently in play.
    """
    board_str = " ".join(obs.board) if obs.board else "None"
    hole_str = " ".join(obs.hero_cards)
    legal_str = ", ".join(a.value for a in obs.legal_actions)
    history_str = format_action_history(obs.action_history)

    lines = [
        f"Game: {obs.env_name.upper()} | Street: {obs.street} | Position: {obs.position}",
        f"Hole Cards: {hole_str} | Board: {board_str}",
        (
            f"Current Pot (including villain's bet): {obs.pot_bb:g} BB | "
            f"To Call: {obs.to_call_bb:g} BB"
        ),
        (
            f"Hero Stack: {obs.hero_stack_bb:g} BB | "
            f"Villain Stack: {obs.villain_stack_bb:g} BB | Effective SPR: {obs.spr:g}"
        ),
        f"Action History: {history_str}",
    ]

    if obs.villain_stats:
        stats_parts = []
        for key in ("vpip", "pfr", "af", "fold_to_bet"):
            if key in obs.villain_stats:
                val = obs.villain_stats[key]
                if key == "af":
                    stats_parts.append(f"AF {val:.2f}")
                else:
                    stats_parts.append(f"{key.upper()} {val * 100:.0f}%")
        if stats_parts:
            lines.append("Villain Stats: " + " | ".join(stats_parts))

    lines.append(f"Legal Actions: [{legal_str}]")
    return "\n".join(lines)
