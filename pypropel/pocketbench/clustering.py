"""
DBSCAN-based spatial clustering for predicted binding site residues.

Clusters residues that pass a probability threshold into spatially coherent
pockets, then ranks pockets by cumulative confidence.
"""

import numpy as np
from typing import List, NamedTuple
from sklearn.cluster import DBSCAN


class PocketCluster(NamedTuple):
    """A predicted pocket cluster."""
    center: np.ndarray          # (3,) probability-weighted centroid
    residue_indices: np.ndarray # indices into the original residue array
    confidence: float           # sum of contact probabilities
    n_residues: int


def cluster_predicted_residues(
    coords: np.ndarray,
    probs: np.ndarray,
    residue_indices: np.ndarray,
    eps: float = 8.0,
    min_samples: int = 3,
) -> List[PocketCluster]:
    """
    Cluster predicted binding-site residues with DBSCAN and rank by confidence.

    Args:
        coords: (N, 3) Cα coordinates for ALL residues in the protein.
        probs: (N,) contact probabilities for ALL residues.
        residue_indices: (K,) indices of residues that passed the threshold.
        eps: DBSCAN neighbourhood radius in Angstroms.
        min_samples: minimum cluster size for DBSCAN core points.

    Returns:
        List of PocketCluster sorted by confidence (sum of probs) descending.
        Falls back to a single cluster when K < min_samples or when DBSCAN
        labels everything as noise.
    """
    if len(residue_indices) == 0:
        return []

    sel_coords = coords[residue_indices]   # (K, 3)
    sel_probs = probs[residue_indices]      # (K,)

    # Fallback: too few residues for meaningful DBSCAN
    if len(residue_indices) < min_samples:
        return [_build_cluster(sel_coords, sel_probs, residue_indices)]

    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(sel_coords)

    # Fallback: everything is noise (-1)
    if np.all(labels == -1):
        return [_build_cluster(sel_coords, sel_probs, residue_indices)]

    clusters: List[PocketCluster] = []
    for label in sorted(set(labels)):
        if label == -1:
            continue
        mask = labels == label
        clusters.append(
            _build_cluster(
                sel_coords[mask],
                sel_probs[mask],
                residue_indices[mask],
            )
        )

    clusters.sort(key=lambda c: c.confidence, reverse=True)
    return clusters


def _build_cluster(
    coords: np.ndarray,
    probs: np.ndarray,
    indices: np.ndarray,
) -> PocketCluster:
    """Build a PocketCluster with probability-weighted centroid."""
    weights = probs / (probs.sum() + 1e-12)
    center = (coords * weights[:, None]).sum(axis=0)
    return PocketCluster(
        center=center,
        residue_indices=indices,
        confidence=float(probs.sum()),
        n_residues=len(indices),
    )
