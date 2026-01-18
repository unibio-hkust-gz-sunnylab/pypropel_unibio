"""
Tests for pypropel.dist module - distance calculations and contact maps.
"""

import pytest
import numpy as np


class TestAtomDistance:
    """Tests for atom-level distance functions."""

    def test_atom_distance_basic(self):
        """Test basic atom distance calculation."""
        import pypropel.dist as ppdist
        
        coord1 = np.array([0.0, 0.0, 0.0])
        coord2 = np.array([3.0, 4.0, 0.0])
        
        dist = ppdist.atom_distance(coord1, coord2)
        
        assert abs(dist - 5.0) < 1e-6

    def test_atom_distance_same_point(self):
        """Test distance between same point is zero."""
        import pypropel.dist as ppdist
        
        coord = np.array([1.0, 2.0, 3.0])
        
        dist = ppdist.atom_distance(coord, coord)
        
        assert abs(dist) < 1e-6

    def test_atom_distance_3d(self):
        """Test 3D distance calculation."""
        import pypropel.dist as ppdist
        
        coord1 = np.array([0.0, 0.0, 0.0])
        coord2 = np.array([1.0, 1.0, 1.0])
        
        dist = ppdist.atom_distance(coord1, coord2)
        expected = np.sqrt(3)
        
        assert abs(dist - expected) < 1e-6


class TestAtomDistanceMatrix:
    """Tests for pairwise distance matrix."""

    def test_matrix_shape(self, sample_coords_3d):
        """Test distance matrix has correct shape."""
        import pypropel.dist as ppdist
        
        coords1 = sample_coords_3d[:2]  # 2 points
        coords2 = sample_coords_3d[2:]  # 2 points
        
        matrix = ppdist.atom_distance_matrix(coords1, coords2)
        
        assert matrix.shape == (2, 2)

    def test_matrix_values(self):
        """Test distance matrix has correct values."""
        import pypropel.dist as ppdist
        
        coords1 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        coords2 = np.array([[3.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
        
        matrix = ppdist.atom_distance_matrix(coords1, coords2)
        
        # Distance from [0,0,0] to [3,0,0] = 3
        # Distance from [0,0,0] to [4,0,0] = 4
        # Distance from [1,0,0] to [3,0,0] = 2
        # Distance from [1,0,0] to [4,0,0] = 3
        expected = np.array([[3.0, 4.0], [2.0, 3.0]])
        
        np.testing.assert_array_almost_equal(matrix, expected)


class TestResidueLigandDistance:
    """Tests for residue-ligand distance calculation."""

    def test_residue_ligand_distance(self, sample_residue_coords, sample_ligand_coords):
        """Test minimum distance between residue and ligand."""
        import pypropel.dist as ppdist
        
        dist = ppdist.residue_ligand_distance(sample_residue_coords, sample_ligand_coords)
        
        # Closest point in residue is [2,2,2], closest in ligand is [5,5,5]
        # Distance = sqrt((5-2)^2 + (5-2)^2 + (5-2)^2) = sqrt(27) ≈ 5.196
        expected = np.sqrt(27)
        
        assert abs(dist - expected) < 1e-6

    def test_empty_residue_coords(self, sample_ligand_coords):
        """Test with empty residue coordinates."""
        import pypropel.dist as ppdist
        
        empty_coords = np.array([]).reshape(0, 3)
        
        dist = ppdist.residue_ligand_distance(empty_coords, sample_ligand_coords)
        
        assert dist == float('inf')


class TestContactMap:
    """Tests for ContactMap class."""

    def test_contact_map_class_exists(self):
        """Test ContactMap class can be imported."""
        from pypropel.prot.structure.distance.ContactMap import ContactMap
        
        cm = ContactMap()
        assert cm is not None

    def test_get_residue_coords(self, temp_pdb_file):
        """Test extracting residue coordinates from PDB."""
        from pypropel.prot.structure.distance.ContactMap import ContactMap
        from Bio.PDB import PDBParser
        
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('test', temp_pdb_file)
        
        cm = ContactMap()
        
        # Get first residue
        for model in structure:
            for chain in model:
                for residue in chain:
                    coords = cm.get_residue_coords(residue)
                    assert len(coords) > 0
                    assert coords.shape[1] == 3
                    break
                break
            break


class TestProteinDistanceMatrix:
    """Tests for protein distance matrix."""

    def test_distance_matrix_from_pdb(self, temp_pdb_file):
        """Test generating distance matrix from PDB file."""
        import pypropel.dist as ppdist
        from Bio.PDB import PDBParser
        
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('test', temp_pdb_file)
        
        dist_matrix = ppdist.protein_distance_matrix(structure, use_ca_only=True)
        
        # Should have 3 residues (ALA, GLY, SER)
        assert dist_matrix.shape[0] == 3
        assert dist_matrix.shape[1] == 3
        
        # Diagonal should be zero
        np.testing.assert_array_almost_equal(np.diag(dist_matrix.values), [0, 0, 0])
        
        # Matrix should be symmetric
        np.testing.assert_array_almost_equal(dist_matrix.values, dist_matrix.values.T)


class TestProteinContactMap:
    """Tests for binary contact map generation."""

    def test_contact_map_from_pdb(self, temp_pdb_file):
        """Test generating contact map from PDB file."""
        import pypropel.dist as ppdist
        from Bio.PDB import PDBParser
        
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('test', temp_pdb_file)
        
        contact_map = ppdist.protein_contact_map(structure, threshold=10.0, use_ca_only=True)
        
        assert contact_map.shape[0] == 3
        assert contact_map.shape[1] == 3
        
        # Diagonal should be 1 (contact with self)
        # But since we compare < threshold, and dist to self = 0, should be 1
        np.testing.assert_array_equal(np.diag(contact_map), [1, 1, 1])


class TestExtractBindingPocket:
    """Tests for binding pocket extraction."""

    def test_extract_pocket(self, temp_pdb_file):
        """Test extracting binding pocket residues."""
        import pypropel.dist as ppdist
        from Bio.PDB import PDBParser
        
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('test', temp_pdb_file)
        
        # Place ligand near residue 3
        ligand_coords = np.array([[8.0, 5.0, 0.0], [9.0, 5.0, 0.0]])
        
        pocket_df = ppdist.extract_binding_pocket(
            structure, 
            ligand_coords, 
            binding_threshold=5.0,
            non_binding_threshold=10.0
        )
        
        assert 'chain' in pocket_df.columns
        assert 'res_id' in pocket_df.columns
        assert 'res_name' in pocket_df.columns
        assert 'distance' in pocket_df.columns
        assert 'label' in pocket_df.columns
        
        # Should have 3 residues
        assert len(pocket_df) == 3
        
        # Labels should be -1, 0, or 1
        assert all(pocket_df['label'].isin([-1, 0, 1]))
