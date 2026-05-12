"""Shared pytest fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure ``src`` is importable when the project is checked out without an
# editable install (pip install -e . is recommended but optional in CI).
SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
