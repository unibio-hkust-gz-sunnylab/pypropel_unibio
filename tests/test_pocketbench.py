"""
Tests for pypropel.pocketbench module.

Tests core dataclasses, metrics, and basic functionality.
"""

import pytest
import numpy as np

from pypropel.pocketbench.core import (
    PBProtein,
    PBSite,
    PBPrediction,
)
from pypropel.pocketbench.metrics import (
    compute_dcc,
    compute_dca,
    compute_iou,
    compute_ap,
    expand_center_to_residues,
    compute_site_center,
    filter_predictions_by_gt,
    filter_artifact_predictions,
    deduplicate_predictions,
)


# =============================================================================
# Core Dataclass Tests
# =============================================================================

class TestPBSite:
    """Tests for PBSite dataclass."""
    
    def test_creation(self):
        """Test basic site creation."""
        site = PBSite(
            center=[10.0, 20.0, 30.0],
            residues=[0, 1, 2, 3, 4],
            ligand_id="ATP"
        )
        assert site.num_residues == 5
        assert site.ligand_id == "ATP"
        np.testing.assert_array_almost_equal(site.center, [10.0, 20.0, 30.0])
    
    def test_center_conversion(self):
        """Test that center is converted to numpy array."""
        site = PBSite(center=[1, 2, 3])
        assert isinstance(site.center, np.ndarray)
        assert site.center.dtype == np.float32


class TestPBProtein:
    """Tests for PBProtein dataclass."""
    
    @pytest.fixture
    def sample_protein(self):
        """Create a sample protein for testing."""
        coords = np.array([
            [0.0, 0.0, 0.0],
            [3.8, 0.0, 0.0],
            [7.6, 0.0, 0.0],
            [11.4, 0.0, 0.0],
            [15.2, 0.0, 0.0],
        ])
        sites = [
            PBSite(center=[1.9, 0.0, 0.0], residues=[0, 1]),
            PBSite(center=[13.3, 0.0, 0.0], residues=[3, 4]),
        ]
        return PBProtein(
            id="1abc_A",
            sequence="ACDEF",
            coords=coords,
            ground_truth_sites=sites
        )
    
    def test_creation(self, sample_protein):
        """Test basic protein creation."""
        assert sample_protein.id == "1abc_A"
        assert sample_protein.num_residues == 5
        assert sample_protein.num_sites == 2
    
    def test_get_site_centers(self, sample_protein):
        """Test getting all site centers."""
        centers = sample_protein.get_site_centers()
        assert centers.shape == (2, 3)
        np.testing.assert_array_almost_equal(centers[0], [1.9, 0.0, 0.0])
    
    def test_get_all_site_residues(self, sample_protein):
        """Test getting union of site residues."""
        residues = sample_protein.get_all_site_residues()
        assert residues == [0, 1, 3, 4]


class TestPBPrediction:
    """Tests for PBPrediction dataclass."""
    
    def test_creation(self):
        """Test basic prediction creation."""
        pred = PBPrediction(
            center=[5.0, 5.0, 5.0],
            residues=[1, 2, 3],
            confidence=0.95,
            model_name="TestModel"
        )
        assert pred.num_residues == 3
        assert pred.confidence == 0.95
        assert pred.model_name == "TestModel"
    
    def test_confidence_clamping(self):
        """Test that confidence is clamped to [0, 1]."""
        pred_high = PBPrediction(center=[0, 0, 0], confidence=1.5)
        pred_low = PBPrediction(center=[0, 0, 0], confidence=-0.5)
        assert pred_high.confidence == 1.0
        assert pred_low.confidence == 0.0


# =============================================================================
# Metric Tests
# =============================================================================

class TestDCC:
    """Tests for DCC (Distance Center-to-Center) metric."""
    
    def test_exact_match(self):
        """Test prediction at exact ground truth location."""
        pred = PBPrediction(center=[10.0, 10.0, 10.0])
        gts = [PBSite(center=[10.0, 10.0, 10.0])]
        
        is_hit, dist = compute_dcc(pred, gts, threshold=4.0)
        assert is_hit == True
        assert dist == 0.0
    
    def test_within_threshold(self):
        """Test prediction within threshold distance."""
        pred = PBPrediction(center=[10.0, 10.0, 10.0])
        gts = [PBSite(center=[12.0, 10.0, 10.0])]  # 2Å away
        
        is_hit, dist = compute_dcc(pred, gts, threshold=4.0)
        assert is_hit == True
        assert abs(dist - 2.0) < 1e-6
    
    def test_outside_threshold(self):
        """Test prediction outside threshold distance."""
        pred = PBPrediction(center=[10.0, 10.0, 10.0])
        gts = [PBSite(center=[20.0, 10.0, 10.0])]  # 10Å away
        
        is_hit, dist = compute_dcc(pred, gts, threshold=4.0)
        assert is_hit == False
        assert abs(dist - 10.0) < 1e-6
    
    def test_multiple_ground_truths(self):
        """Test that any matching GT counts as hit."""
        pred = PBPrediction(center=[10.0, 10.0, 10.0])
        gts = [
            PBSite(center=[100.0, 100.0, 100.0]),  # Far
            PBSite(center=[12.0, 10.0, 10.0]),      # Close (2Å)
            PBSite(center=[50.0, 50.0, 50.0]),     # Far
        ]
        
        is_hit, dist = compute_dcc(pred, gts, threshold=4.0)
        assert is_hit == True
        assert abs(dist - 2.0) < 1e-6
    
    def test_empty_ground_truths(self):
        """Test with no ground truths."""
        pred = PBPrediction(center=[10.0, 10.0, 10.0])
        
        is_hit, dist = compute_dcc(pred, [], threshold=4.0)
        assert is_hit == False
        assert dist == float('inf')


class TestDCA:
    """Tests for DCA (Distance Center-to-Any-Atom) metric."""
    
    def test_close_to_atoms(self):
        """Test prediction close to ground truth atoms."""
        pred = PBPrediction(center=[10.0, 10.0, 10.0])
        gt_atoms = np.array([
            [11.0, 10.0, 10.0],  # 1Å away
            [15.0, 15.0, 15.0],
        ])
        
        is_hit, dist = compute_dca(pred, gt_atoms, threshold=4.0)
        assert is_hit == True
        assert abs(dist - 1.0) < 1e-6
    
    def test_far_from_atoms(self):
        """Test prediction far from ground truth atoms."""
        pred = PBPrediction(center=[0.0, 0.0, 0.0])
        gt_atoms = np.array([
            [10.0, 10.0, 10.0],
            [20.0, 20.0, 20.0],
        ])
        
        is_hit, dist = compute_dca(pred, gt_atoms, threshold=4.0)
        assert is_hit == False


class TestIoU:
    """Tests for IoU (Intersection over Union) metric."""
    
    def test_perfect_overlap(self):
        """Test identical residue sets."""
        iou = compute_iou([0, 1, 2, 3], [0, 1, 2, 3])
        assert iou == 1.0
    
    def test_no_overlap(self):
        """Test disjoint residue sets."""
        iou = compute_iou([0, 1, 2], [3, 4, 5])
        assert iou == 0.0
    
    def test_partial_overlap(self):
        """Test partial overlap."""
        # [0,1,2,3] vs [2,3,4,5]
        # Intersection: {2,3} = 2
        # Union: {0,1,2,3,4,5} = 6
        iou = compute_iou([0, 1, 2, 3], [2, 3, 4, 5])
        assert abs(iou - 2/6) < 1e-6
    
    def test_empty_sets(self):
        """Test edge cases with empty sets."""
        assert compute_iou([], []) == 1.0  # Both empty = perfect match
        assert compute_iou([1, 2], []) == 0.0
        assert compute_iou([], [1, 2]) == 0.0


class TestRadiusExpansion:
    """Tests for expand_center_to_residues (9Å fallback)."""
    
    def test_expansion(self):
        """Test expanding center to nearby residues."""
        coords = np.array([
            [0.0, 0.0, 0.0],    # 0Å from origin
            [5.0, 0.0, 0.0],    # 5Å from origin
            [15.0, 0.0, 0.0],   # 15Å from origin
        ])
        center = [0.0, 0.0, 0.0]
        
        residues = expand_center_to_residues(center, coords, radius=10.0)
        assert residues == [0, 1]  # Only first two within 10Å
    
    def test_no_residues_in_range(self):
        """Test when no residues are within radius."""
        coords = np.array([
            [100.0, 100.0, 100.0],
            [200.0, 200.0, 200.0],
        ])
        center = [0.0, 0.0, 0.0]
        
        residues = expand_center_to_residues(center, coords, radius=9.0)
        assert residues == []


class TestSiteCenter:
    """Tests for compute_site_center utility."""
    
    def test_center_computation(self):
        """Test computing geometric center of site."""
        coords = np.array([
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
        ])
        
        center = compute_site_center(coords, [0, 2])
        np.testing.assert_array_almost_equal(center, [10.0, 0.0, 0.0])
    
    def test_empty_residues(self):
        """Test with empty residue list."""
        coords = np.array([[1.0, 2.0, 3.0]])
        center = compute_site_center(coords, [])
        np.testing.assert_array_almost_equal(center, [0.0, 0.0, 0.0])


# =============================================================================
# AP & Filtering Tests
# =============================================================================

class TestComputeAP:
    """Tests for compute_ap (Average Precision at IoU threshold)."""
    
    def test_perfect_single_prediction(self):
        """Single correct prediction should give AP = 1.0."""
        preds = [PBPrediction(center=[5, 0, 0], residues=[0, 1, 2, 3], confidence=0.9)]
        gts = [PBSite(center=[5, 0, 0], residues=[0, 1, 2, 3])]
        ap = compute_ap(preds, gts, iou_threshold=0.5)
        assert ap == 1.0
    
    def test_wrong_prediction(self):
        """Prediction with no IoU overlap gives AP = 0."""
        preds = [PBPrediction(center=[50, 0, 0], residues=[10, 11, 12], confidence=0.9)]
        gts = [PBSite(center=[5, 0, 0], residues=[0, 1, 2, 3])]
        ap = compute_ap(preds, gts, iou_threshold=0.5)
        assert ap == 0.0
    
    def test_artifact_inflation(self):
        """Extra non-matching predictions should degrade AP."""
        # 1 correct prediction + 5 wrong ones
        preds = [
            PBPrediction(center=[5, 0, 0], residues=[0, 1, 2, 3], confidence=0.9),
            PBPrediction(center=[50, 0, 0], residues=[20, 21], confidence=0.8),
            PBPrediction(center=[60, 0, 0], residues=[30, 31], confidence=0.7),
            PBPrediction(center=[70, 0, 0], residues=[40, 41], confidence=0.6),
            PBPrediction(center=[80, 0, 0], residues=[50, 51], confidence=0.5),
            PBPrediction(center=[90, 0, 0], residues=[60, 61], confidence=0.4),
        ]
        gts = [PBSite(center=[5, 0, 0], residues=[0, 1, 2, 3])]
        ap = compute_ap(preds, gts, iou_threshold=0.5)
        # AP should be 1.0 * (1/1) = 1.0 for the first pred, no recall change after
        # But total AP = 1.0 * 1.0 = 1.0 (since first pred is TP and recall reaches 1.0)
        assert ap == 1.0  # First pred is highest confidence and correct
    
    def test_artifact_inflation_low_confidence_correct(self):
        """When the correct prediction has LOW confidence, artifacts destroy AP."""
        # 5 wrong predictions BEFORE the correct one (higher confidence)
        preds = [
            PBPrediction(center=[50, 0, 0], residues=[20, 21], confidence=0.9),
            PBPrediction(center=[60, 0, 0], residues=[30, 31], confidence=0.8),
            PBPrediction(center=[70, 0, 0], residues=[40, 41], confidence=0.7),
            PBPrediction(center=[80, 0, 0], residues=[50, 51], confidence=0.6),
            PBPrediction(center=[90, 0, 0], residues=[60, 61], confidence=0.5),
            PBPrediction(center=[5, 0, 0], residues=[0, 1, 2, 3], confidence=0.4),
        ]
        gts = [PBSite(center=[5, 0, 0], residues=[0, 1, 2, 3])]
        ap = compute_ap(preds, gts, iou_threshold=0.5)
        # TP comes at position 6: precision=1/6, recall=1.0
        # AP = (1/6) * 1.0 ≈ 0.167
        assert ap < 0.2  # Severely degraded by artifact predictions


class TestFilterByGT:
    """Tests for filter_predictions_by_gt."""
    
    def test_keeps_matching_ligands(self):
        """Predictions with GT-matching ligand_id are kept."""
        preds = [
            PBPrediction(center=[5, 0, 0], residues=[0, 1], ligand_id="ATP"),
            PBPrediction(center=[10, 0, 0], residues=[5, 6], ligand_id="GOL"),
            PBPrediction(center=[15, 0, 0], residues=[8, 9], ligand_id="ATP"),
        ]
        gts = [PBSite(center=[5, 0, 0], residues=[0, 1], ligand_id="ATP")]
        
        filtered = filter_predictions_by_gt(preds, gts)
        assert len(filtered) == 2
        assert all(p.ligand_id == "ATP" for p in filtered)
    
    def test_no_gt_ligand_info(self):
        """When GT has no ligand_id, all predictions pass through."""
        preds = [
            PBPrediction(center=[5, 0, 0], residues=[0, 1], ligand_id="GOL"),
            PBPrediction(center=[10, 0, 0], residues=[5, 6], ligand_id="ATP"),
        ]
        gts = [PBSite(center=[5, 0, 0], residues=[0, 1])]  # No ligand_id
        
        filtered = filter_predictions_by_gt(preds, gts)
        assert len(filtered) == 2  # All pass through


class TestFilterArtifacts:
    """Tests for filter_artifact_predictions."""
    
    def test_removes_artifacts(self):
        """Known artifacts (GOL, EDO, SO4) are filtered out."""
        preds = [
            PBPrediction(center=[5, 0, 0], residues=[0, 1], ligand_id="ATP"),
            PBPrediction(center=[10, 0, 0], residues=[5, 6], ligand_id="GOL"),
            PBPrediction(center=[15, 0, 0], residues=[8, 9], ligand_id="EDO"),
            PBPrediction(center=[20, 0, 0], residues=[11, 12], ligand_id="SO4"),
            PBPrediction(center=[25, 0, 0], residues=[14, 15], ligand_id="HEM"),
        ]
        
        filtered = filter_artifact_predictions(preds)
        assert len(filtered) == 2
        assert [p.ligand_id for p in filtered] == ["ATP", "HEM"]
    
    def test_keeps_all_if_no_artifacts(self):
        """Non-artifact predictions are all kept."""
        preds = [
            PBPrediction(center=[5, 0, 0], ligand_id="ATP"),
            PBPrediction(center=[10, 0, 0], ligand_id="HEM"),
        ]
        filtered = filter_artifact_predictions(preds)
        assert len(filtered) == 2


class TestDeduplication:
    """Tests for deduplicate_predictions."""
    
    def test_removes_overlapping(self):
        """Overlapping predictions are merged (keep higher confidence)."""
        preds = [
            PBPrediction(center=[5, 0, 0], residues=[0, 1, 2, 3], confidence=0.9),
            PBPrediction(center=[5, 0, 0], residues=[0, 1, 2, 3], confidence=0.7),  # Duplicate
            PBPrediction(center=[50, 0, 0], residues=[20, 21, 22, 23], confidence=0.8),
        ]
        
        deduped = deduplicate_predictions(preds, iou_threshold=0.5)
        assert len(deduped) == 2
        # Highest confidence kept
        assert deduped[0].confidence == 0.9
        assert deduped[1].confidence == 0.8
    
    def test_keeps_non_overlapping(self):
        """Non-overlapping predictions are all kept."""
        preds = [
            PBPrediction(center=[5, 0, 0], residues=[0, 1, 2], confidence=0.9),
            PBPrediction(center=[50, 0, 0], residues=[10, 11, 12], confidence=0.8),
            PBPrediction(center=[100, 0, 0], residues=[20, 21, 22], confidence=0.7),
        ]
        
        deduped = deduplicate_predictions(preds, iou_threshold=0.5)
        assert len(deduped) == 3
    
    def test_empty_input(self):
        """Empty input returns empty output."""
        assert deduplicate_predictions([]) == []
