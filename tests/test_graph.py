"""
Tests for pypropel.graph module - graph construction.
"""

import pytest
import numpy as np

# Direct file import to avoid pypropel.__init__ which has mini3di dependency
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
ppgraph = _load_module('graph', os.path.join(_pypropel_dir, 'pypropel', 'graph.py'))
ppmol = _load_module('mol', os.path.join(_pypropel_dir, 'pypropel', 'mol.py'))


class TestBuildProteinKnnGraph:
    """Tests for build_protein_knn_graph function."""

    def test_protein_graph_structure(self, temp_pdb_file):
        """Test protein graph output structure."""
        from Bio.PDB import PDBParser
        
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('test', temp_pdb_file)
        
        graph = ppgraph.build_protein_knn_graph(structure, k=5, radius=10.0)
        
        assert 'node_coords' in graph
        assert 'edge_index' in graph
        assert 'edge_attr' in graph
        
        assert graph['node_coords'].shape[1] == 3
        assert graph['edge_index'].shape[0] == 2
        if graph['edge_index'].shape[1] > 0:
            assert graph['edge_attr'].shape[1] == 4  # distance + 3D direction


class TestBuildLigandGraph:
    """Tests for build_ligand_graph function."""

    def test_ligand_graph_structure(self, temp_sdf_file):
        """Test ligand graph output structure."""
        mol = ppmol.load_sdf(temp_sdf_file)
        
        graph = ppgraph.build_ligand_graph(mol)
        
        assert 'node_features' in graph
        assert 'node_coords' in graph
        assert 'edge_index' in graph
        assert 'edge_attr' in graph
        
        if mol is not None:
            n_atoms = mol.GetNumAtoms()
            assert graph['node_features'].shape[0] == n_atoms
            assert graph['node_features'].shape[1] == 21  # Combined atom features

    def test_ligand_graph_none(self):
        """Test ligand graph with None input."""
        graph = ppgraph.build_ligand_graph(None)
        
        assert graph['node_features'].shape[0] == 0
        assert graph['edge_index'].shape[1] == 0


class TestGetBondFeatures:
    """Tests for get_bond_features function."""

    def test_bond_features_shape(self, temp_sdf_file):
        """Test bond feature extraction."""
        mol = ppmol.load_sdf(temp_sdf_file)
        
        if mol is not None:
            bond_features = ppgraph.get_bond_features(mol)
            n_bonds = mol.GetNumBonds()
            
            # Bidirectional edges
            assert bond_features.shape[0] == 2 * n_bonds
            assert bond_features.shape[1] == 4  # Bond type one-hot


class TestBuildProteinLigandBipartiteEdges:
    """Tests for build_protein_ligand_bipartite_edges function."""

    def test_bipartite_edges(self):
        """Test bipartite edge building."""
        protein_coords = np.array([
            [0, 0, 0],
            [10, 0, 0],
            [100, 0, 0],  # Far from ligand
        ], dtype=np.float32)
        
        ligand_coords = np.array([
            [1, 0, 0],
            [2, 0, 0],
        ], dtype=np.float32)
        
        edge_index, distances = ppgraph.build_protein_ligand_bipartite_edges(
            protein_coords, ligand_coords, threshold=5.0
        )
        
        assert edge_index.shape[0] == 2
        # Only protein node 0 should be within 5Å of ligand
        assert 0 in edge_index[0]
        assert 2 not in edge_index[0]  # Node 2 is too far
