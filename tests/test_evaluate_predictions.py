"""
Tests for evaluate_predictions end-to-end behavior.

Covers the main evaluation pipeline including the DCC denominator
bug fix regression test.
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from pypropel.pocketbench.core import PBProtein, PBSite, PBPrediction
from pypropel.pocketbench.metrics import evaluate_predictions
from conftest import make_protein, make_site, make_prediction


class TestEvaluatePredictions:
    """End-to-end tests for evaluate_predictions."""

    def test_perfect_predictions(self):
        """All predictions exactly match GT -> DCC=1.0, AP=1.0."""
        protein = make_protein(10, sites=[
            make_site(center=[1.9, 0.0, 0.0], residues=[0, 1]),
        ])
        preds = [make_prediction(center=[1.9, 0.0, 0.0], residues=[0, 1], confidence=0.9)]

        results = evaluate_predictions([protein], [preds])

        assert results['dcc_success_rate'] == 1.0
        assert results['dcc_top1_rate'] == 1.0
        assert results['mean_ap'] == 1.0
        assert results['n_evaluated'] == 1

    def test_no_predictions(self):
        """Empty prediction lists -> all metrics 0.0."""
        protein = make_protein(10, sites=[
            make_site(center=[1.9, 0.0, 0.0], residues=[0, 1]),
        ])

        results = evaluate_predictions([protein], [[]])

        assert results['dcc_success_rate'] == 0.0
        assert results['dcc_top1_rate'] == 0.0
        assert results['mean_ap'] == 0.0
        assert results['n_evaluated'] == 0

    def test_mixed_hits_and_misses(self):
        """Some correct, some wrong -> verify rates."""
        p1 = make_protein(10, sites=[
            make_site(center=[1.9, 0.0, 0.0], residues=[0, 1]),
        ], protein_id="p1")
        p2 = make_protein(10, sites=[
            make_site(center=[1.9, 0.0, 0.0], residues=[0, 1]),
        ], protein_id="p2")

        preds1 = [make_prediction(center=[1.9, 0.0, 0.0], residues=[0, 1], confidence=0.9)]  # hit
        preds2 = [make_prediction(center=[99.0, 99.0, 99.0], residues=[8, 9], confidence=0.9)]  # miss

        results = evaluate_predictions([p1, p2], [preds1, preds2])

        assert results['dcc_success_rate'] == 0.5
        assert results['dcc_top1_rate'] == 0.5
        assert results['n_evaluated'] == 2

    def test_denominator_with_skipped_proteins(self):
        """Proteins with no GT or no preds are excluded from denominator (Bug 1 regression)."""
        p_with_gt = make_protein(10, sites=[
            make_site(center=[1.9, 0.0, 0.0], residues=[0, 1]),
        ], protein_id="has_gt")
        p_no_gt = make_protein(10, sites=[], protein_id="no_gt")
        p_no_preds = make_protein(10, sites=[
            make_site(center=[1.9, 0.0, 0.0], residues=[0, 1]),
        ], protein_id="no_preds")

        preds_hit = [make_prediction(center=[1.9, 0.0, 0.0], residues=[0, 1], confidence=0.9)]
        preds_for_no_gt = [make_prediction(center=[5.0, 0.0, 0.0], residues=[2, 3], confidence=0.8)]
        preds_empty = []

        results = evaluate_predictions(
            [p_with_gt, p_no_gt, p_no_preds],
            [preds_hit, preds_for_no_gt, preds_empty],
        )

        # Only p_with_gt is evaluated (has both GT and preds)
        assert results['n_evaluated'] == 1
        assert results['n_proteins'] == 3
        # DCC should be 1.0 (1 hit / 1 evaluated), NOT 1/3
        assert results['dcc_success_rate'] == 1.0
        assert results['dcc_top1_rate'] == 1.0

    def test_dcc_and_ap_denominator_consistency(self):
        """DCC and AP use same effective denominator (both only count evaluated proteins)."""
        p1 = make_protein(10, sites=[
            make_site(center=[1.9, 0.0, 0.0], residues=[0, 1]),
        ], protein_id="p1")
        p_no_gt = make_protein(10, sites=[], protein_id="no_gt")

        preds1 = [make_prediction(center=[1.9, 0.0, 0.0], residues=[0, 1], confidence=0.9)]
        preds2 = [make_prediction(center=[5.0, 0.0, 0.0], residues=[2, 3], confidence=0.8)]

        results = evaluate_predictions([p1, p_no_gt], [preds1, preds2])

        # Both DCC and AP should reflect only the 1 evaluated protein
        assert results['n_evaluated'] == 1
        assert results['dcc_success_rate'] == 1.0
        assert results['mean_ap'] == 1.0

    def test_single_protein_single_site(self):
        """Simplest case: one protein, one site, one prediction."""
        protein = make_protein(5, sites=[
            make_site(center=[3.8, 0.0, 0.0], residues=[1]),
        ])
        preds = [make_prediction(center=[3.8, 0.0, 0.0], residues=[1], confidence=1.0)]

        results = evaluate_predictions([protein], [preds])

        assert results['dcc_success_rate'] == 1.0
        assert results['mean_ap'] == 1.0
        assert results['n_evaluated'] == 1

    def test_multiple_sites_per_protein(self):
        """DCC should hit if ANY GT matches."""
        protein = make_protein(10, sites=[
            make_site(center=[0.0, 0.0, 0.0], residues=[0]),
            make_site(center=[34.2, 0.0, 0.0], residues=[9]),
        ])
        # Prediction near second site only
        preds = [make_prediction(center=[34.2, 0.0, 0.0], residues=[9], confidence=0.9)]

        results = evaluate_predictions([protein], [preds])

        assert results['dcc_success_rate'] == 1.0
        assert results['dcc_top1_rate'] == 1.0

    def test_artifact_filtering_in_evaluation(self):
        """mean_ap_filtered excludes artifact predictions."""
        protein = make_protein(10, sites=[
            make_site(center=[1.9, 0.0, 0.0], residues=[0, 1], ligand_id="ATP"),
        ])
        preds = [
            make_prediction(center=[99.0, 0.0, 0.0], residues=[8, 9], confidence=0.9, ligand_id="GOL"),
            make_prediction(center=[1.9, 0.0, 0.0], residues=[0, 1], confidence=0.5, ligand_id="ATP"),
        ]

        results = evaluate_predictions([protein], [preds])

        # Unfiltered AP is degraded (correct pred has lower confidence)
        assert results['mean_ap'] < 1.0
        # Filtered AP should be 1.0 (GOL artifact removed, only ATP pred remains)
        assert results['mean_ap_filtered'] == 1.0

    def test_gt_matched_filtering(self):
        """mean_ap_matched only uses GT-matching predictions."""
        protein = make_protein(10, sites=[
            make_site(center=[1.9, 0.0, 0.0], residues=[0, 1], ligand_id="ATP"),
        ])
        preds = [
            make_prediction(center=[99.0, 0.0, 0.0], residues=[8, 9], confidence=0.9, ligand_id="HEM"),
            make_prediction(center=[1.9, 0.0, 0.0], residues=[0, 1], confidence=0.5, ligand_id="ATP"),
        ]

        results = evaluate_predictions([protein], [preds])

        # Matched AP should be 1.0 (only ATP pred kept, and it matches GT)
        assert results['mean_ap_matched'] == 1.0

    def test_n_evaluated_in_results(self):
        """n_evaluated is returned correctly."""
        proteins = [
            make_protein(5, sites=[make_site([0, 0, 0], [0])], protein_id="a"),
            make_protein(5, sites=[], protein_id="b"),
            make_protein(5, sites=[make_site([0, 0, 0], [0])], protein_id="c"),
        ]
        preds_list = [
            [make_prediction([0, 0, 0], [0], 0.9)],
            [make_prediction([0, 0, 0], [0], 0.9)],
            [],
        ]

        results = evaluate_predictions(proteins, preds_list)

        assert results['n_proteins'] == 3
        assert results['n_evaluated'] == 1  # Only "a" has both GT and preds

    def test_empty_inputs(self):
        """Empty protein/prediction lists -> all zeros."""
        results = evaluate_predictions([], [])

        assert results['dcc_success_rate'] == 0.0
        assert results['mean_ap'] == 0.0
        assert results['n_proteins'] == 0
        assert results['n_evaluated'] == 0
