"""
Tests for CryptoBench (cryptic binding sites) scoring behavior.

Covers cryptic site DCC, center-only expansion, and empty residue fallback.
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from pypropel.pocketbench.core import PBProtein, PBSite, PBPrediction
from pypropel.pocketbench.metrics import (
    evaluate_predictions,
    compute_dcc,
    compute_ap,
    expand_center_to_residues,
    compute_iou,
)
from conftest import make_protein, make_site, make_prediction


class TestCryptoBenchBenchmark:
    """Tests specific to CryptoBench scoring."""

    def test_cryptic_site_dcc(self):
        """DCC works with cryptic site centers."""
        protein = make_protein(10, sites=[
            make_site(center=[15.2, 0.0, 0.0], residues=[3, 4, 5]),
        ])
        preds = [make_prediction(center=[15.2, 0.0, 0.0], residues=[3, 4, 5], confidence=0.9)]

        results = evaluate_predictions([protein], [preds])

        assert results['dcc_success_rate'] == 1.0
        assert results['dcc_top1_rate'] == 1.0

    def test_no_ligand_id_in_cryptic(self):
        """Cryptic sites have no ligand_id, filtering passes all predictions."""
        protein = make_protein(10, sites=[
            make_site(center=[15.2, 0.0, 0.0], residues=[3, 4, 5]),  # No ligand_id
        ])
        preds = [
            make_prediction(center=[15.2, 0.0, 0.0], residues=[3, 4, 5], confidence=0.9, ligand_id="UNK"),
            make_prediction(center=[0.0, 0.0, 0.0], residues=[0, 1], confidence=0.5, ligand_id="ATP"),
        ]

        results = evaluate_predictions([protein], [preds])

        # mean_ap_matched should equal mean_ap (no GT ligand_id -> all pass)
        assert results['mean_ap_matched'] == results['mean_ap']

    def test_multiple_cryptic_sites(self):
        """Protein with multiple cryptic sites."""
        protein = make_protein(20, sites=[
            make_site(center=[7.6, 0.0, 0.0], residues=[1, 2, 3]),
            make_site(center=[57.0, 0.0, 0.0], residues=[14, 15, 16]),
        ])
        preds = [
            make_prediction(center=[7.6, 0.0, 0.0], residues=[1, 2, 3], confidence=0.9),
            make_prediction(center=[57.0, 0.0, 0.0], residues=[14, 15, 16], confidence=0.8),
        ]

        results = evaluate_predictions([protein], [preds])

        assert results['dcc_success_rate'] == 1.0
        assert results['mean_ap'] == 1.0

    def test_center_only_predictions_with_expansion(self):
        """AP with center-only models uses 9Å radius expansion."""
        # Create protein with known coords
        protein = make_protein(10, sites=[
            make_site(center=[7.6, 0.0, 0.0], residues=[1, 2, 3]),
        ])
        # Center-only prediction (no residues) near GT
        preds = [make_prediction(center=[7.6, 0.0, 0.0], residues=[], confidence=0.9)]

        results = evaluate_predictions([protein], [preds])

        # With 9Å expansion from center [7.6, 0, 0]:
        # Residue 0 at [0, 0, 0] -> 7.6Å (within 9Å)
        # Residue 1 at [3.8, 0, 0] -> 3.8Å (within 9Å)
        # Residue 2 at [7.6, 0, 0] -> 0.0Å (within 9Å)
        # Residue 3 at [11.4, 0, 0] -> 3.8Å (within 9Å)
        # Residue 4 at [15.2, 0, 0] -> 7.6Å (within 9Å)
        # Residue 5 at [19.0, 0, 0] -> 11.4Å (outside 9Å)
        # Expanded residues: [0, 1, 2, 3, 4]
        # GT residues: [1, 2, 3]
        # IoU = |{1,2,3}| / |{0,1,2,3,4}| = 3/5 = 0.6 >= 0.5 threshold
        assert results['mean_ap'] > 0.0

    def test_empty_residue_expansion_fallback(self):
        """When expansion returns empty residues, IoU=0.0."""
        # Protein coords far from prediction center
        protein = make_protein(5, sites=[
            make_site(center=[0.0, 0.0, 0.0], residues=[0, 1]),
        ])
        # Center-only prediction very far from all residues
        preds = [make_prediction(center=[999.0, 999.0, 999.0], residues=[], confidence=0.9)]

        results = evaluate_predictions([protein], [preds])

        # Expansion finds no residues within 9Å -> empty pred residues -> IoU=0
        assert results['mean_ap'] == 0.0

    def test_dcc_independent_of_residues(self):
        """DCC only uses centers, not residues — works for center-only models."""
        pred = PBPrediction(center=[10.0, 0.0, 0.0])
        gts = [PBSite(center=[12.0, 0.0, 0.0], residues=[5, 6, 7])]

        is_hit, dist = compute_dcc(pred, gts, threshold=4.0)

        assert is_hit == True
        assert abs(dist - 2.0) < 1e-6

    def test_expansion_radius_affects_iou(self):
        """Different expansion radii produce different IoU values."""
        coords = np.array([[i * 3.8, 0.0, 0.0] for i in range(10)], dtype=np.float32)
        center = [7.6, 0.0, 0.0]

        small_expansion = expand_center_to_residues(center, coords, radius=4.0)
        large_expansion = expand_center_to_residues(center, coords, radius=12.0)

        # Larger radius captures more residues
        assert len(large_expansion) > len(small_expansion)

        gt_residues = [1, 2, 3]
        iou_small = compute_iou(small_expansion, gt_residues)
        iou_large = compute_iou(large_expansion, gt_residues)

        # Both should be valid IoU values
        assert 0.0 <= iou_small <= 1.0
        assert 0.0 <= iou_large <= 1.0
