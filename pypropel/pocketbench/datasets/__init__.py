"""
PocketBench dataset loaders.

Provides unified dataset interfaces for benchmarking binding site
prediction models on legacy and modern datasets.
"""

from .base import PBDataset
from .p2rank import COACH420Dataset, HOLO4KDataset
from .unisite import UniSiteDSDataset, UniSiteBenchmarkDataset
from .cryptobench import CryptoBenchDataset

__all__ = [
    # Base
    "PBDataset",
    # Legacy (P2Rank)
    "COACH420Dataset",
    "HOLO4KDataset",
    # Modern (UniSite)
    "UniSiteDSDataset",
    "UniSiteBenchmarkDataset",
    # Cryptic
    "CryptoBenchDataset",
]

