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
    expand_center_to_residues,
    compute_site_center,
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
