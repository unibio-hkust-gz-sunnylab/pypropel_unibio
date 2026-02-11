"""
Tests for UniSite-DS (UniProt-centric, multi-site) scoring behavior.

Covers multi-site DCC, greedy AP matching, and ligand_id passthrough.
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
    filter_predictions_by_gt,
)
from conftest import make_protein, make_site, make_prediction


class TestUniSiteBenchmark:
    """Tests specific to UniSite-DS scoring."""

    def test_multi_site_dcc_any_match(self):
        """Prediction near ANY aggregated site is a DCC hit."""
        protein = make_protein(20, sites=[
            make_site(center=[0.0, 0.0, 0.0], residues=[0]),
            make_site(center=[38.0, 0.0, 0.0], residues=[10]),
            make_site(center=[72.2, 0.0, 0.0], residues=[19]),
        ])
        # Prediction near the third site
        preds = [make_prediction(center=[72.2, 0.0, 0.0], residues=[19], confidence=0.9)]

        results = evaluate_predictions([protein], [preds])

        assert results['dcc_success_rate'] == 1.0

    def test_multi_site_ap_greedy_matching(self):
        """AP correctly handles multiple GT sites with greedy assignment."""
        protein = make_protein(10, sites=[
            make_site(center=[0.0, 0.0, 0.0], residues=[0, 1]),
            make_site(center=[30.4, 0.0, 0.0], residues=[8, 9]),
        ])
        # Two predictions, each matching one GT site
        preds = [
            make_prediction(center=[0.0, 0.0, 0.0], residues=[0, 1], confidence=0.9),
            make_prediction(center=[30.4, 0.0, 0.0], residues=[8, 9], confidence=0.8),
        ]

        results = evaluate_predictions([protein], [preds])

        # Both predictions match their respective GT -> AP=1.0
        assert results['mean_ap'] == 1.0

    def test_no_ligand_id_passthrough(self):
        """When GT has no ligand_id, all predictions pass filter."""
        protein = make_protein(10, sites=[
            make_site(center=[1.9, 0.0, 0.0], residues=[0, 1]),  # No ligand_id
        ])
        preds = [
            make_prediction(center=[1.9, 0.0, 0.0], residues=[0, 1], confidence=0.9, ligand_id="ATP"),
            make_prediction(center=[99.0, 0.0, 0.0], residues=[8, 9], confidence=0.5, ligand_id="HEM"),
        ]

        # filter_predictions_by_gt should return all preds when GT has no ligand_id
        gts = protein.ground_truth_sites
        filtered = filter_predictions_by_gt(preds, gts)
        assert len(filtered) == 2

        # mean_ap_matched should also use all predictions
        results = evaluate_predictions([protein], [preds])
        assert results['mean_ap_matched'] == results['mean_ap']

    def test_many_sites_few_predictions(self):
        """More GT sites than predictions -> recall < 1.0, AP < 1.0."""
        protein = make_protein(20, sites=[
            make_site(center=[0.0, 0.0, 0.0], residues=[0, 1]),
            make_site(center=[19.0, 0.0, 0.0], residues=[5, 6]),
            make_site(center=[38.0, 0.0, 0.0], residues=[10, 11]),
            make_site(center=[57.0, 0.0, 0.0], residues=[15, 16]),
        ])
        # Only 1 prediction matching 1 of 4 GT sites
        preds = [make_prediction(center=[0.0, 0.0, 0.0], residues=[0, 1], confidence=0.9)]

        results = evaluate_predictions([protein], [preds])

        # AP should be < 1.0 because recall only reaches 0.25
        assert results['mean_ap'] < 1.0
        assert results['mean_ap'] > 0.0

    def test_few_sites_many_predictions(self):
        """More predictions than GT -> extra FPs degrade precision."""
        protein = make_protein(10, sites=[
            make_site(center=[1.9, 0.0, 0.0], residues=[0, 1]),
        ])
        # 1 correct + 4 wrong predictions
        preds = [
            make_prediction(center=[1.9, 0.0, 0.0], residues=[0, 1], confidence=0.9),
            make_prediction(center=[99.0, 0.0, 0.0], residues=[5, 6], confidence=0.8),
            make_prediction(center=[99.0, 0.0, 0.0], residues=[6, 7], confidence=0.7),
            make_prediction(center=[99.0, 0.0, 0.0], residues=[7, 8], confidence=0.6),
            make_prediction(center=[99.0, 0.0, 0.0], residues=[8, 9], confidence=0.5),
        ]

        results = evaluate_predictions([protein], [preds])

        # First pred is correct and highest confidence, so AP = 1.0
        # (recall reaches 1.0 at first pred, subsequent FPs don't reduce AP)
        assert results['mean_ap'] == 1.0

    def test_multi_protein_multi_site(self):
        """Multiple proteins each with multiple sites."""
        p1 = make_protein(10, sites=[
            make_site(center=[0.0, 0.0, 0.0], residues=[0, 1]),
            make_site(center=[30.4, 0.0, 0.0], residues=[8, 9]),
        ], protein_id="p1")
        p2 = make_protein(10, sites=[
            make_site(center=[0.0, 0.0, 0.0], residues=[0, 1]),
        ], protein_id="p2")

        preds1 = [
            make_prediction(center=[0.0, 0.0, 0.0], residues=[0, 1], confidence=0.9),
            make_prediction(center=[30.4, 0.0, 0.0], residues=[8, 9], confidence=0.8),
        ]
        preds2 = [
            make_prediction(center=[0.0, 0.0, 0.0], residues=[0, 1], confidence=0.9),
        ]

        results = evaluate_predictions([p1, p2], [preds1, preds2])

        assert results['n_evaluated'] == 2
        assert results['dcc_success_rate'] == 1.0
        assert results['mean_ap'] == 1.0
