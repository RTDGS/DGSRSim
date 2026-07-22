"""Compatibility entry point for the paper-aligned region evaluator.

Use ``metrics1.py`` for new commands. This module is retained so existing
workflows no longer execute the historical, invalid ``lpips(...)`` call.
"""

from metrics1 import build_parser, compute_metrics, evaluate, load_samples, main

__all__ = ["build_parser", "compute_metrics", "evaluate", "load_samples", "main"]


if __name__ == "__main__":
    main()
