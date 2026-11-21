"""Deterministic seeding utilities."""

from __future__ import annotations

import random

import numpy as np


def set_global_seed(seed: int) -> None:
    """Seed Python and NumPy global random generators."""
    random.seed(seed)
    np.random.seed(seed)


def make_rng(seed: int | None = None) -> random.Random:
    """Create an isolated random.Random instance."""
    return random.Random(seed)
