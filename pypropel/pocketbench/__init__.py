"""
PocketBench: Unified benchmarking for protein-ligand binding site prediction.

This module provides:
- Core dataclasses: PBProtein, PBSite, PBPrediction
- Model interface: PBModel (abstract base class)
- Metrics: DCC, DCA, IoU, AP
- Dataset loaders: COACH420, HOLO4K, UniSite-DS, CryptoBench
- Model wrappers: P2Rank
"""

__version__ = "0.1.0"
__author__ = "pypropel team"

from .core import (
    PBProtein,
    PBSite,
    PBPrediction,
    PBModel,
)

from .metrics import (
    compute_dcc,
    compute_dca,
    compute_iou,
    compute_ap,
    expand_center_to_residues,
)

from .clustering import (
    cluster_predicted_residues,
    PocketCluster,
)

# Lazy imports for datasets and models
from . import datasets
from . import models

__all__ = [
    # Core dataclasses
    "PBProtein",
    "PBSite", 
    "PBPrediction",
    "PBModel",
    # Metrics
    "compute_dcc",
    "compute_dca",
    "compute_iou",
    "compute_ap",
    "expand_center_to_residues",
    # Clustering
    "cluster_predicted_residues",
    "PocketCluster",
    # Subpackages
    "datasets",
    "models",
]

