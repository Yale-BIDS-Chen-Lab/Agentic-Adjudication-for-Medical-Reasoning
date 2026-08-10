#!/usr/bin/env python3
"""Compatibility wrapper for the HealthBench benchmark adapter."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from methods.common.benchmarks.healthbench import build_arg_parser, main, run

__all__ = ["build_arg_parser", "run", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
