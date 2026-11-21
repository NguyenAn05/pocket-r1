"""Trajectory JSONL logging and optional experiment tracking hooks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class ParsedOutputSchema(BaseModel):
    is_valid: bool
    steps: list[str] = Field(default_factory=list)
    action: str | None = None
    error_codes: list[str] = Field(default_factory=list)
    response_tokens: int = 0


class TrajectoryRecord(BaseModel):
    """Canonical decision-point trajectory record shared across Phases 1-4."""

    hand_id: str
    env_name: str
    seed: int
    opponent_type: str
    street: str
    decision_idx: int
    state_json: dict[str, Any]
    prompt_text: str
    raw_response: str
    parsed: ParsedOutputSchema
    auto_checks: dict[str, Any] = Field(default_factory=dict)
    ev_bb: float | None = None
    hand_outcome_bb: float | None = None


class TrajectoryLogger:
    """Append-only JSONL writer for decision trajectories."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: TrajectoryRecord) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(record.model_dump_json() + "\n")

    def log_many(self, records: list[TrajectoryRecord]) -> None:
        if not records:
            return
        with self.path.open("a", encoding="utf-8") as f:
            for rec in records:
                f.write(rec.model_dump_json() + "\n")

    @staticmethod
    def load(path: str | Path) -> list[TrajectoryRecord]:
        records: list[TrajectoryRecord] = []
        with Path(path).open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(TrajectoryRecord.model_validate_json(line))
        return records
