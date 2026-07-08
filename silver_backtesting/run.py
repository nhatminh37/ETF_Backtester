#!/usr/bin/env python3
"""Run the backtest when your shell is inside this package folder.

Usage::

    python run.py

This adds the parent project directory to ``sys.path`` so imports work even if
you are in ``...\\silver_backtesting\\silver_backtesting>`` instead of the root.
"""

from __future__ import annotations

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from silver_backtesting.__main__ import main

if __name__ == "__main__":
    main()
