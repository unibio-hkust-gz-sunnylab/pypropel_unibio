"""
Tests for P2Rank legacy dataset (COACH420/HOLO4K) scoring behavior.

Covers artifact exclusion, ligand-contact-based GT, and multi-ligand proteins.
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from pypropel.pocketbench.core import PBProtein, PBSite, PBPrediction
from pypropel.pocketbench.metrics import (
    evaluate_predictions,
    filter_artifact_predictions,
    compute_ap,
    CRYSTALLOGRAPHIC_ARTIFACTS,
)
from conftest import make_protein, make_site, make_prediction


class TestP2RankBenchmark:
    """Tests specific to P2Rank legacy dataset scoring."""

    def test_artifact_exclusion_during_scoring(self):
        """Predictions for GOL/EDO/SO4 are excluded in filtered AP."""
        protein = make_protein(10, sites=[
            make_site(center=[1.9, 0.0, 0.0], residues=[0, 1], ligand_id="ATP"),
        ])
        preds = [
            make_prediction(center=[99.0, 0.0, 0.0], residues=[8, 9], confidence=0.95, ligand_id="GOL"),
            make_prediction(center=[99.0, 0.0, 0.0], residues=[7, 8], confidence=0.90, ligand_id="EDO"),
            make_prediction(center=[99.0, 0.0, 0.0], residues=[6, 7], confidence=0.85, ligand_id="SO4"),
            make_prediction(center=[1.9, 0.0, 0.0], residues=[0, 1], confidence=0.50, ligand_id="ATP"),
        ]

        results = evaluate_predictions([protein], [preds])

        # Unfiltered: correct pred is 4th -> degraded AP
        assert results['mean_ap'] < 0.5
        # Filtered: artifacts removed, only ATP pred remains -> AP=1.0
        assert results['mean_ap_filtered'] == 1.0

    def test_ligand_contact_based_gt(self):
        """GT sites defined by 4.0Å ligand contacts produce valid residue sets."""
        # Simulate a protein where residues 0-2 are within 4Å of a ligand
        protein = make_protein(10, sites=[
            make_site(center=[3.8, 0.0, 0.0], residues=[0, 1, 2], ligand_id="ATP"),
        ])
        preds = [make_prediction(center=[3.8, 0.0, 0.0], residues=[0, 1, 2], confidence=0.9)]

        results = evaluate_predictions([protein], [preds])

        assert results['dcc_success_rate'] == 1.0
        assert results['mean_ap'] == 1.0

    def test_single_ligand_protein(self):
        """Protein with one ligand -> one GT site."""
        protein = make_protein(8, sites=[
            make_site(center=[7.6, 0.0, 0.0], residues=[1, 2, 3], ligand_id="HEM"),
        ])
        preds = [make_prediction(center=[7.6, 0.0, 0.0], residues=[1, 2, 3], confidence=0.9)]

        results = evaluate_predictions([protein], [preds])

        assert results['n_evaluated'] == 1
        assert results['dcc_success_rate'] == 1.0
        assert results['mean_ap'] == 1.0

    def test_multi_ligand_protein(self):
        """Protein with multiple ligands -> multiple GT sites, DCC checks any."""
        protein = make_protein(20, sites=[
            make_site(center=[3.8, 0.0, 0.0], residues=[0, 1, 2], ligand_id="ATP"),
            make_site(center=[57.0, 0.0, 0.0], residues=[14, 15, 16], ligand_id="HEM"),
        ])
        # Prediction near second site only
        preds = [make_prediction(center=[57.0, 0.0, 0.0], residues=[14, 15, 16], confidence=0.9)]

        results = evaluate_predictions([protein], [preds])

        # DCC hit because prediction matches second GT site
        assert results['dcc_success_rate'] == 1.0

    def test_dcc_with_artifact_predictions(self):
        """Artifact predictions still count for unfiltered DCC but not for filtered AP."""
        protein = make_protein(10, sites=[
            make_site(center=[1.9, 0.0, 0.0], residues=[0, 1], ligand_id="ATP"),
        ])
        # Only artifact prediction, near the GT site center
        preds = [make_prediction(center=[1.9, 0.0, 0.0], residues=[0, 1], confidence=0.9, ligand_id="GOL")]

        results = evaluate_predictions([protein], [preds])

        # DCC still hits (center-based, doesn't filter by ligand)
        assert results['dcc_success_rate'] == 1.0
        # But filtered AP = 0 (artifact removed, no preds left)
        assert results['mean_ap_filtered'] == 0.0

    def test_ap_degradation_from_artifacts(self):
        """mean_ap vs mean_ap_filtered shows artifact impact."""
        protein = make_protein(10, sites=[
            make_site(center=[1.9, 0.0, 0.0], residues=[0, 1], ligand_id="ATP"),
        ])
        # Artifacts ranked higher than the correct prediction
        preds = [
            make_prediction(center=[99.0, 0.0, 0.0], residues=[8, 9], confidence=0.9, ligand_id="GOL"),
            make_prediction(center=[99.0, 0.0, 0.0], residues=[7, 8], confidence=0.8, ligand_id="EDO"),
            make_prediction(center=[1.9, 0.0, 0.0], residues=[0, 1], confidence=0.5, ligand_id="ATP"),
        ]

        results = evaluate_predictions([protein], [preds])

        # Filtered should be strictly better than unfiltered
        assert results['mean_ap_filtered'] > results['mean_ap']

    def test_all_artifact_codes_are_filtered(self):
        """Verify all known artifact codes are properly filtered."""
        for code in ['GOL', 'EDO', 'SO4', 'PO4', 'DMS', 'PEG', 'ACT']:
            preds = [make_prediction(center=[0, 0, 0], residues=[0], confidence=0.9, ligand_id=code)]
            filtered = filter_artifact_predictions(preds)
            assert len(filtered) == 0, f"Artifact {code} was not filtered"
