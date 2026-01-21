"""
Tests for pypropel.gvp module - GVP vector features.
"""

import pytest
import numpy as np

# Direct file import to avoid pypropel.__init__ which has mini3di dependency
import sys
import os
_pypropel_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _pypropel_dir)

# Import modules directly without going through __init__.py
import importlib.util
def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

ppgvp = _load_module('gvp', os.path.join(_pypropel_dir, 'pypropel', 'gvp.py'))


class TestCaCoords:
    """Tests for get_ca_coords function."""

    def test_ca_coords_shape(self, temp_pdb_file):
        """Test Cα coordinate extraction."""
        from Bio.PDB import PDBParser
        
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('test', temp_pdb_file)
        
        coords = ppgvp.get_ca_coords(structure)
        
        assert len(coords) > 0
        assert coords.shape[1] == 3
        assert coords.dtype == np.float32


class TestCbCoords:
    """Tests for get_cb_coords function."""

    def test_cb_coords_shape(self, temp_pdb_file):
        """Test Cβ coordinate extraction."""
        from Bio.PDB import PDBParser
        
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('test', temp_pdb_file)
        
        ca_coords = ppgvp.get_ca_coords(structure)
        cb_coords = ppgvp.get_cb_coords(structure)
        
        # Should have same number of residues
        assert len(cb_coords) == len(ca_coords)
        assert cb_coords.shape[1] == 3


class TestResidueOrientations:
    """Tests for get_residue_orientations function."""

    def test_orientation_vectors_unit(self, temp_pdb_file):
        """Test that orientation vectors are unit vectors."""
        from Bio.PDB import PDBParser
        
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('test', temp_pdb_file)
        
        orientations = ppgvp.get_residue_orientations(structure)
        
        if len(orientations) > 0:
            norms = np.linalg.norm(orientations, axis=1)
            assert np.allclose(norms, 1.0, atol=1e-5)


class TestBackboneVectors:
    """Tests for get_backbone_vectors function."""

    def test_backbone_vectors_unit(self, temp_pdb_file):
        """Test that backbone vectors are unit vectors."""
        from Bio.PDB import PDBParser
        
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('test', temp_pdb_file)
        
        backbone = ppgvp.get_backbone_vectors(structure)
        
        if len(backbone) > 0:
            norms = np.linalg.norm(backbone, axis=1)
            # Most should be unit vectors (some edge cases may be zero)
            non_zero = norms > 0.1
            assert np.allclose(norms[non_zero], 1.0, atol=1e-5)


class TestGvpNodeFeatures:
    """Tests for get_gvp_node_features function."""

    def test_gvp_features_shapes(self, temp_pdb_file):
        """Test combined GVP feature extraction."""
        from Bio.PDB import PDBParser
        
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('test', temp_pdb_file)
        
        coords, vectors = ppgvp.get_gvp_node_features(structure, k_neighbors=5)
        
        assert coords.shape[1] == 3
        if len(coords) > 0:
            assert vectors.shape == (len(coords), 3, 3)


class TestBuildKnnEdges:
    """Tests for build_knn_edges function."""

    def test_knn_edges_basic(self):
        """Test k-NN edge building."""
        # Simple test coordinates
        coords = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [2, 0, 0],
            [10, 0, 0],  # Far away
        ], dtype=np.float32)
        
        edge_index, distances = ppgvp.build_knn_edges(coords, k=2, radius=5.0)
        
        # Should have edges (nodes 0,1,2 are close, node 3 is far)
        assert edge_index.shape[0] == 2
        assert len(distances) == edge_index.shape[1]

    def test_knn_radius_cutoff(self):
        """Test that radius cutoff is respected."""
        coords = np.array([
            [0, 0, 0],
            [20, 0, 0],  # Beyond radius
        ], dtype=np.float32)
        
        edge_index, distances = ppgvp.build_knn_edges(coords, k=1, radius=10.0)
        
        # No edges should be created beyond radius
        assert edge_index.shape[1] == 0
