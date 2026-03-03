"""
PocketBench: Unified benchmarking for protein-ligand binding site prediction.

This module provides:
- Core dataclasses: PBProtein, PBSite, PBPrediction
- Model interface: PBModel (abstract base class)
- Metrics: IoU, Precision/Recall/F1, Dice, AP, DCC, DCA
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
    compute_residue_precision_recall_f1,
    compute_residue_dice,
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
    "compute_iou",
    "compute_residue_precision_recall_f1",
    "compute_residue_dice",
    "compute_ap",
    "compute_dcc",
    "compute_dca",
    "expand_center_to_residues",
    # Clustering
    "cluster_predicted_residues",
    "PocketCluster",
    # Subpackages
    "datasets",
    "models",
]

