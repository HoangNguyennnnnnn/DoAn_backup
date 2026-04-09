"""Script wrapper for Stage 1 trainer implementation in src/train/train_stage1.py."""

from __future__ import annotations

import sys
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.train.train_stage1 import main


if __name__ == "__main__":
    raise SystemExit(main())
