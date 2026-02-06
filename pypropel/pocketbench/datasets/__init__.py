"""
PocketBench dataset loaders.

Provides unified dataset interfaces for benchmarking binding site
prediction models on legacy and modern datasets.
"""

from .base import PBDataset
from .p2rank import COACH420Dataset, HOLO4KDataset

__all__ = [
    "PBDataset",
    "COACH420Dataset",
    "HOLO4KDataset",
]
