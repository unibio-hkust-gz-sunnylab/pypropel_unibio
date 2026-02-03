"""
Tests for multi-chain support functions in pypropel.gvp and chain_encoding.
"""

import pytest
import numpy as np
import torch

# Direct file import to avoid pypropel.__init__ issues
import sys
import os
_pypropel_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _pypropel_dir)

import importlib.util
def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

ppgvp = _load_module('gvp', os.path.join(_pypropel_dir, 'pypropel', 'gvp.py'))


class TestGetChainIds:
    """Tests for get_chain_ids function."""

    def test_chain_ids_single_chain(self, temp_pdb_file):
        """Test chain ID extraction for single chain."""
        from Bio.PDB import PDBParser
        
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('test', temp_pdb_file)
        
        chain_ids = ppgvp.get_chain_ids(structure)
        
        # All residues should have chain ID 0 (single chain)
        assert len(chain_ids) > 0
        assert np.all(chain_ids == 0)
        assert chain_ids.dtype == np.int64

    def test_chain_ids_length_matches_ca_coords(self, temp_pdb_file):
        """Test that chain_ids length matches Cα coordinates."""
        from Bio.PDB import PDBParser
        
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('test', temp_pdb_file)
        
        chain_ids = ppgvp.get_chain_ids(structure)
        ca_coords = ppgvp.get_ca_coords(structure)
        
        assert len(chain_ids) == len(ca_coords)


class TestGlobalCenteredCoords:
    """Tests for get_global_centered_coords function."""

    def test_global_centered_mean_zero(self, temp_pdb_file):
        """Test that global centered coords have mean near zero."""
        from Bio.PDB import PDBParser
        
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('test', temp_pdb_file)
        
        centered = ppgvp.get_global_centered_coords(structure)
        
        # Mean should be approximately zero
        mean = centered.mean(axis=0)
        assert np.allclose(mean, 0, atol=1e-5)

    def test_global_centered_shape(self, temp_pdb_file):
        """Test shape of global centered coords."""
        from Bio.PDB import PDBParser
        
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('test', temp_pdb_file)
        
        centered = ppgvp.get_global_centered_coords(structure)
        ca_coords = ppgvp.get_ca_coords(structure)
        
        assert centered.shape == ca_coords.shape


class TestChainCenteredCoords:
    """Tests for get_chain_centered_coords function."""

    def test_chain_centered_single_chain(self, temp_pdb_file):
        """Test that chain centered equals global centered for single chain."""
        from Bio.PDB import PDBParser
        
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('test', temp_pdb_file)
        
        global_centered = ppgvp.get_global_centered_coords(structure)
        chain_centered = ppgvp.get_chain_centered_coords(structure)
        
        # For single chain, should be identical
        assert np.allclose(global_centered, chain_centered)

    def test_chain_centered_shape(self, temp_pdb_file):
        """Test shape of chain centered coords."""
        from Bio.PDB import PDBParser
        
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('test', temp_pdb_file)
        
        chain_centered = ppgvp.get_chain_centered_coords(structure)
        ca_coords = ppgvp.get_ca_coords(structure)
        
        assert chain_centered.shape == ca_coords.shape


class TestKnnDirectionVectors:
    """Tests for get_knn_direction_vectors function."""

    def test_knn_directions_shape(self, temp_pdb_file):
        """Test shape of k-NN direction vectors."""
        from Bio.PDB import PDBParser
        
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('test', temp_pdb_file)
        
        k = 5
        directions = ppgvp.get_knn_direction_vectors(structure, k=k)
        ca_coords = ppgvp.get_ca_coords(structure)
        
        # Shape should be (N, k, 3)
        assert directions.shape == (len(ca_coords), k, 3)

    def test_knn_directions_unit_vectors(self, temp_pdb_file):
        """Test that non-zero k-NN direction vectors are unit vectors."""
        from Bio.PDB import PDBParser
        
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('test', temp_pdb_file)
        
        directions = ppgvp.get_knn_direction_vectors(structure, k=5)
        
        # Compute norms for all direction vectors
        # directions: (N, k, 3)
        norms = np.linalg.norm(directions, axis=-1)  # (N, k)
        
        # Non-zero vectors should have unit norm
        non_zero = norms > 0.1
        if non_zero.any():
            assert np.allclose(norms[non_zero], 1.0, atol=1e-5)


class TestChainEncoding:
    """Tests for chain_encoding module functions."""

    def test_cumulative_gap_single_chain(self):
        """Test cumulative gap with single chain (all zeros)."""
        # Import from src using direct path
        # _pypropel_dir = /Users/.../protein_ligand/pypropel_unibio
        # chain_encoding is at /Users/.../protein_ligand/src/models/chain_encoding.py
        project_root = os.path.dirname(_pypropel_dir)  # /Users/.../protein_ligand
        chain_encoding_path = os.path.join(project_root, 'src', 'models', 'chain_encoding.py')
        chain_encoding = _load_module('chain_encoding', chain_encoding_path)
        
        # Single chain: all zeros
        chain_ids = torch.tensor([[0, 0, 0, 0, 0]])
        pos_ids = chain_encoding.get_cumulative_gap_position_ids(chain_ids, gap_size=1000)
        
        # Should be sequential: 0, 1, 2, 3, 4
        expected = torch.tensor([[0, 1, 2, 3, 4]])
        assert torch.equal(pos_ids, expected)

    def test_cumulative_gap_two_chains(self):
        """Test cumulative gap with two chains."""
        project_root = os.path.dirname(_pypropel_dir)
        chain_encoding_path = os.path.join(project_root, 'src', 'models', 'chain_encoding.py')
        chain_encoding = _load_module('chain_encoding', chain_encoding_path)
        
        # Two chains: 0,0,0, then 1,1,1
        chain_ids = torch.tensor([[0, 0, 0, 1, 1, 1]])
        pos_ids = chain_encoding.get_cumulative_gap_position_ids(chain_ids, gap_size=1000)
        
        # Chain 0: 0, 1, 2
        # Chain 1: 3+1000, 4+1000, 5+1000 = 1003, 1004, 1005
        expected = torch.tensor([[0, 1, 2, 1003, 1004, 1005]])
        assert torch.equal(pos_ids, expected)

    def test_cumulative_gap_three_chains(self):
        """Test cumulative gap with three chains."""
        project_root = os.path.dirname(_pypropel_dir)
        chain_encoding_path = os.path.join(project_root, 'src', 'models', 'chain_encoding.py')
        chain_encoding = _load_module('chain_encoding', chain_encoding_path)
        
        # Three chains
        chain_ids = torch.tensor([[0, 0, 1, 1, 2, 2]])
        pos_ids = chain_encoding.get_cumulative_gap_position_ids(chain_ids, gap_size=1000)
        
        # 0, 1, 1002, 1003, 2004, 2005
        expected = torch.tensor([[0, 1, 1002, 1003, 2004, 2005]])
        assert torch.equal(pos_ids, expected)

    def test_is_same_chain_edge_feature(self):
        """Test is_same_chain edge feature computation."""
        project_root = os.path.dirname(_pypropel_dir)
        chain_encoding_path = os.path.join(project_root, 'src', 'models', 'chain_encoding.py')
        chain_encoding = _load_module('chain_encoding', chain_encoding_path)
        
        # 4 nodes: 2 in chain 0, 2 in chain 1
        chain_ids = torch.tensor([0, 0, 1, 1])
        # Edges: 0->1 (same), 1->2 (different), 2->3 (same)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])
        
        is_same = chain_encoding.get_is_same_chain_edge_feature(chain_ids, edge_index)
        
        expected = torch.tensor([[1.0], [0.0], [1.0]])
        assert torch.allclose(is_same, expected)
