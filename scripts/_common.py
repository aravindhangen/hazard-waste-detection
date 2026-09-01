"""Shared bootstrap for CLI scripts under scripts/."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def bootstrap() -> Path:
    """Ensure project root is on sys.path so `hazard_detection` imports work."""
    root = str(PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    return PROJECT_ROOT
