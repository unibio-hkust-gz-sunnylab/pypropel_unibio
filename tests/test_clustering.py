"""
Tests for pypropel.pocketbench.clustering module.

Tests DBSCAN-based spatial clustering of predicted binding site residues.
"""

import pytest
import numpy as np

from pypropel.pocketbench.clustering import (
    cluster_predicted_residues,
    PocketCluster,
    _build_cluster,
)


# =============================================================================
# PocketCluster NamedTuple Tests
# =============================================================================

class TestPocketCluster:
    """Tests for PocketCluster NamedTuple."""

    def test_creation(self):
        """Test basic creation and field access."""
        cluster = PocketCluster(
            center=np.array([1.0, 2.0, 3.0]),
            residue_indices=np.array([0, 1, 2]),
            confidence=2.5,
            n_residues=3,
        )
        np.testing.assert_array_equal(cluster.center, [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(cluster.residue_indices, [0, 1, 2])
        assert cluster.confidence == 2.5
        assert cluster.n_residues == 3

    def test_unpacking(self):
        """Test tuple unpacking."""
        cluster = PocketCluster(
            center=np.array([0.0, 0.0, 0.0]),
            residue_indices=np.array([5]),
            confidence=0.9,
            n_residues=1,
        )
        center, indices, conf, n = cluster
        assert conf == 0.9
        assert n == 1


# =============================================================================
# _build_cluster Tests
# =============================================================================

class TestBuildCluster:
    """Tests for _build_cluster helper."""

    def test_uniform_weights(self):
        """Equal probabilities give geometric centroid."""
        coords = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
        probs = np.array([0.5, 0.5])
        indices = np.array([0, 1])

        cluster = _build_cluster(coords, probs, indices)
        np.testing.assert_array_almost_equal(cluster.center, [5.0, 0.0, 0.0])
        assert cluster.confidence == pytest.approx(1.0)
        assert cluster.n_residues == 2

    def test_weighted_centroid(self):
        """Higher probability pulls centroid toward that residue."""
        coords = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
        probs = np.array([0.9, 0.1])
        indices = np.array([3, 7])

        cluster = _build_cluster(coords, probs, indices)
        # Centroid should be at 0.9*0 + 0.1*10 = 1.0
        np.testing.assert_array_almost_equal(cluster.center, [1.0, 0.0, 0.0])
        assert cluster.confidence == pytest.approx(1.0)
        np.testing.assert_array_equal(cluster.residue_indices, [3, 7])

    def test_single_residue(self):
        """Single residue cluster center equals that residue's coords."""
        coords = np.array([[5.0, 5.0, 5.0]])
        probs = np.array([0.8])
        indices = np.array([42])

        cluster = _build_cluster(coords, probs, indices)
        np.testing.assert_array_almost_equal(cluster.center, [5.0, 5.0, 5.0])
        assert cluster.n_residues == 1


# =============================================================================
# cluster_predicted_residues Tests
# =============================================================================

class TestClusterPredictedResidues:
    """Tests for the main clustering function."""

    @pytest.fixture
    def two_pocket_protein(self):
        """Protein with two spatially separated groups of residues.

        Pocket A: residues 0-3 near origin (within 3 Angstroms of each other)
        Pocket B: residues 4-7 at ~50 Angstroms away
        """
        coords = np.array([
            # Pocket A (near origin)
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            # Pocket B (far away)
            [50.0, 50.0, 50.0],
            [51.0, 50.0, 50.0],
            [50.0, 51.0, 50.0],
            [51.0, 51.0, 50.0],
        ])
        probs = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2])
        return coords, probs

    # --- Basic clustering ---

    def test_two_distinct_clusters(self, two_pocket_protein):
        """Two well-separated groups should produce two clusters."""
        coords, probs = two_pocket_protein
        indices = np.arange(8)

        clusters = cluster_predicted_residues(coords, probs, indices)
        assert len(clusters) == 2

    def test_clusters_sorted_by_confidence(self, two_pocket_protein):
        """Clusters should be sorted by sum(probs) descending."""
        coords, probs = two_pocket_protein
        indices = np.arange(8)

        clusters = cluster_predicted_residues(coords, probs, indices)
        assert clusters[0].confidence > clusters[1].confidence

    def test_cluster_residue_partition(self, two_pocket_protein):
        """Each residue should appear in exactly one cluster."""
        coords, probs = two_pocket_protein
        indices = np.arange(8)

        clusters = cluster_predicted_residues(coords, probs, indices)
        all_indices = np.sort(np.concatenate([c.residue_indices for c in clusters]))
        np.testing.assert_array_equal(all_indices, indices)

    def test_cluster_centers_near_respective_pockets(self, two_pocket_protein):
        """Each cluster center should be near its pocket, not in between."""
        coords, probs = two_pocket_protein
        indices = np.arange(8)

        clusters = cluster_predicted_residues(coords, probs, indices)

        # Top cluster (pocket A, higher probs) center should be near origin
        top_center = clusters[0].center
        assert np.linalg.norm(top_center - np.array([0.0, 0.0, 0.0])) < 5.0

        # Second cluster (pocket B) center should be near (50, 50, 50)
        second_center = clusters[1].center
        assert np.linalg.norm(second_center - np.array([50.0, 50.0, 50.0])) < 5.0

    # --- Edge cases ---

    def test_empty_indices(self):
        """Empty residue_indices should return empty list."""
        coords = np.array([[0.0, 0.0, 0.0]])
        probs = np.array([0.9])
        indices = np.array([], dtype=int)

        clusters = cluster_predicted_residues(coords, probs, indices)
        assert clusters == []

    def test_single_residue_fallback(self):
        """Single residue should fall back to one cluster."""
        coords = np.array([[5.0, 5.0, 5.0], [100.0, 100.0, 100.0]])
        probs = np.array([0.9, 0.1])
        indices = np.array([0])

        clusters = cluster_predicted_residues(coords, probs, indices)
        assert len(clusters) == 1
        np.testing.assert_array_almost_equal(clusters[0].center, [5.0, 5.0, 5.0])

    def test_two_residues_fallback(self):
        """Fewer than min_samples residues should fall back to one cluster."""
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [100.0, 0.0, 0.0]])
        probs = np.array([0.9, 0.8, 0.3])
        indices = np.array([0, 2])  # Only 2 residues, min_samples=3

        clusters = cluster_predicted_residues(coords, probs, indices)
        assert len(clusters) == 1
        assert clusters[0].n_residues == 2

    def test_all_noise_fallback(self):
        """When all points are noise (spread out), fall back to one cluster."""
        coords = np.array([
            [0.0, 0.0, 0.0],
            [100.0, 0.0, 0.0],
            [0.0, 100.0, 0.0],
        ])
        probs = np.array([0.9, 0.8, 0.7])
        indices = np.arange(3)

        clusters = cluster_predicted_residues(coords, probs, indices)
        assert len(clusters) == 1
        assert clusters[0].n_residues == 3

    def test_single_tight_group(self):
        """All residues close together should form one cluster."""
        coords = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ])
        probs = np.array([0.9, 0.8, 0.7, 0.6])
        indices = np.arange(4)

        clusters = cluster_predicted_residues(coords, probs, indices)
        assert len(clusters) == 1
        assert clusters[0].n_residues == 4

    # --- Parameter sensitivity ---

    def test_custom_eps(self, two_pocket_protein):
        """Very small eps should fragment clusters or produce noise fallback."""
        coords, probs = two_pocket_protein
        indices = np.arange(8)

        # eps=0.5 is smaller than inter-residue distance (~1A), so all noise
        clusters = cluster_predicted_residues(coords, probs, indices, eps=0.5)
        # Should fall back to single cluster (all noise)
        assert len(clusters) == 1

    def test_custom_min_samples(self, two_pocket_protein):
        """Higher min_samples may merge small groups into noise."""
        coords, probs = two_pocket_protein
        indices = np.arange(8)

        # min_samples=5 means each pocket (4 residues) is too small
        clusters = cluster_predicted_residues(coords, probs, indices, min_samples=5)
        # All noise → fallback to single cluster
        assert len(clusters) == 1

    def test_large_eps_merges_clusters(self, two_pocket_protein):
        """Very large eps should merge everything into one cluster."""
        coords, probs = two_pocket_protein
        indices = np.arange(8)

        clusters = cluster_predicted_residues(coords, probs, indices, eps=200.0)
        assert len(clusters) == 1
        assert clusters[0].n_residues == 8

    # --- Subset of residues ---

    def test_subset_indices(self, two_pocket_protein):
        """Only indices passed in should be clustered."""
        coords, probs = two_pocket_protein
        # Only pass pocket A residues
        indices = np.array([0, 1, 2, 3])

        clusters = cluster_predicted_residues(coords, probs, indices)
        assert len(clusters) == 1
        np.testing.assert_array_equal(clusters[0].residue_indices, [0, 1, 2, 3])

    def test_non_contiguous_indices(self):
        """Non-contiguous indices should work correctly."""
        coords = np.zeros((20, 3))
        coords[2] = [0.0, 0.0, 0.0]
        coords[5] = [1.0, 0.0, 0.0]
        coords[9] = [0.0, 1.0, 0.0]
        coords[15] = [100.0, 100.0, 100.0]

        probs = np.full(20, 0.1)
        probs[2] = 0.9
        probs[5] = 0.8
        probs[9] = 0.7
        probs[15] = 0.6

        indices = np.array([2, 5, 9, 15])
        clusters = cluster_predicted_residues(coords, probs, indices)

        # Residues 2, 5, 9 are close; 15 is far → 1 cluster + noise fallback
        # or 2 clusters depending on DBSCAN
        assert len(clusters) >= 1
        all_indices = np.sort(np.concatenate([c.residue_indices for c in clusters]))
        # All passed indices should be accounted for (in clusters or via fallback)
        # Note: noise points are dropped by DBSCAN, so only clustered points appear
        # unless all are noise (fallback)
        for idx in [2, 5, 9]:
            assert idx in all_indices

    # --- Confidence values ---

    def test_confidence_is_sum_of_probs(self, two_pocket_protein):
        """Cluster confidence should equal sum of member probabilities."""
        coords, probs = two_pocket_protein
        indices = np.arange(8)

        clusters = cluster_predicted_residues(coords, probs, indices)
        for cluster in clusters:
            expected_conf = probs[cluster.residue_indices].sum()
            assert cluster.confidence == pytest.approx(expected_conf, abs=1e-6)

    def test_n_residues_matches_indices(self, two_pocket_protein):
        """n_residues should match length of residue_indices."""
        coords, probs = two_pocket_protein
        indices = np.arange(8)

        clusters = cluster_predicted_residues(coords, probs, indices)
        for cluster in clusters:
            assert cluster.n_residues == len(cluster.residue_indices)


# =============================================================================
# Package-level import Tests
# =============================================================================

class TestPackageExports:
    """Test that clustering is properly exported from pocketbench package."""

    def test_import_from_package(self):
        """cluster_predicted_residues should be importable from pocketbench."""
        from pypropel.pocketbench import cluster_predicted_residues as cpr
        assert callable(cpr)

    def test_import_pocket_cluster_from_package(self):
        """PocketCluster should be importable from pocketbench."""
        from pypropel.pocketbench import PocketCluster as PC
        assert PC is not None
