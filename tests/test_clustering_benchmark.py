"""
Tests for DBSCAN clustering as used in DCC computation.

Covers single/multiple clusters, fallback behavior, and ranking.
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from pypropel.pocketbench.core import PBProtein, PBSite, PBPrediction
from pypropel.pocketbench.metrics import (
    compute_dcc,
    evaluate_predictions,
)
from conftest import make_protein, make_site, make_prediction


class TestClusteringBenchmark:
    """Tests for clustering behavior in DCC computation."""

    def test_single_cluster_dcc(self):
        """One cluster near ligand -> top1 hit."""
        protein = make_protein(10, sites=[
            make_site(center=[7.6, 0.0, 0.0], residues=[1, 2, 3]),
        ])
        preds = [make_prediction(center=[7.6, 0.0, 0.0], residues=[1, 2, 3], confidence=0.9)]

        results = evaluate_predictions([protein], [preds])

        assert results['dcc_top1_rate'] == 1.0

    def test_multiple_clusters_ranking(self):
        """Highest-confidence cluster checked first for top-1."""
        protein = make_protein(20, sites=[
            make_site(center=[57.0, 0.0, 0.0], residues=[14, 15, 16]),
        ])
        # First (highest confidence) prediction is far from GT
        # Second prediction is near GT
        preds = [
            make_prediction(center=[0.0, 0.0, 0.0], residues=[0, 1], confidence=0.9),
            make_prediction(center=[57.0, 0.0, 0.0], residues=[14, 15, 16], confidence=0.5),
        ]

        results = evaluate_predictions([protein], [preds])

        # Top-1 (highest confidence) misses, but any-hit succeeds
        assert results['dcc_top1_rate'] == 0.0
        assert results['dcc_success_rate'] == 1.0

    def test_fallback_few_residues(self):
        """Predictions with few residues still work for DCC (center-based)."""
        pred = PBPrediction(center=[10.0, 0.0, 0.0], residues=[5])
        gts = [PBSite(center=[12.0, 0.0, 0.0], residues=[5, 6])]

        is_hit, dist = compute_dcc(pred, gts, threshold=4.0)

        assert is_hit == True
        assert abs(dist - 2.0) < 1e-6

    def test_fallback_all_noise(self):
        """Predictions with no residues still work for DCC (center-based)."""
        pred = PBPrediction(center=[10.0, 0.0, 0.0], residues=[])
        gts = [PBSite(center=[12.0, 0.0, 0.0], residues=[5, 6])]

        is_hit, dist = compute_dcc(pred, gts, threshold=4.0)

        assert is_hit == True
        assert abs(dist - 2.0) < 1e-6

    def test_no_residues_above_threshold(self):
        """Prediction far from all GT sites -> DCC miss."""
        pred = PBPrediction(center=[999.0, 999.0, 999.0], residues=[])
        gts = [PBSite(center=[0.0, 0.0, 0.0], residues=[0, 1, 2])]

        is_hit, dist = compute_dcc(pred, gts, threshold=4.0)

        assert is_hit == False

    def test_cluster_ranking_across_proteins(self):
        """Multiple proteins with different cluster rankings."""
        p1 = make_protein(10, sites=[
            make_site(center=[0.0, 0.0, 0.0], residues=[0]),
        ], protein_id="p1")
        p2 = make_protein(10, sites=[
            make_site(center=[34.2, 0.0, 0.0], residues=[9]),
        ], protein_id="p2")

        # p1: top-1 hits, p2: top-1 misses
        preds1 = [
            make_prediction(center=[0.0, 0.0, 0.0], residues=[0], confidence=0.9),
        ]
        preds2 = [
            make_prediction(center=[0.0, 0.0, 0.0], residues=[0], confidence=0.9),  # miss
            make_prediction(center=[34.2, 0.0, 0.0], residues=[9], confidence=0.5),  # hit
        ]

        results = evaluate_predictions([p1, p2], [preds1, preds2])

        assert results['n_evaluated'] == 2
        assert results['dcc_top1_rate'] == 0.5  # 1 of 2 top-1 hits
        assert results['dcc_success_rate'] == 1.0  # both have at least one hit

    def test_boundary_distance_dcc(self):
        """DCC at exactly the threshold distance."""
        pred = PBPrediction(center=[0.0, 0.0, 0.0])
        gts = [PBSite(center=[4.0, 0.0, 0.0])]

        is_hit, dist = compute_dcc(pred, gts, threshold=4.0)

        # Exactly at threshold -> should be a hit (<=)
        assert is_hit == True
        assert abs(dist - 4.0) < 1e-6

    def test_just_outside_threshold_dcc(self):
        """DCC just outside the threshold distance."""
        pred = PBPrediction(center=[0.0, 0.0, 0.0])
        gts = [PBSite(center=[4.01, 0.0, 0.0])]

        is_hit, dist = compute_dcc(pred, gts, threshold=4.0)

        assert is_hit == False
