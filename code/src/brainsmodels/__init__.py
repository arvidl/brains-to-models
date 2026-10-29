"""Small didactic models for the Brains to Models paper."""

from __future__ import annotations

from typing import Any

__all__ = [
    "HierarchicalPredictiveCoding",
    "PatchDataset",
    "save_paper_figure",
    "write_figure",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from brainsmodels import predictive_coding as pc

        return getattr(pc, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
