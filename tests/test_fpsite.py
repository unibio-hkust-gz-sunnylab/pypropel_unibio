"""
Tests for pypropel.fpsite module - residue features.
"""

import pytest
import numpy as np


class TestResidueOneHot:
    """Tests for residue_one_hot function."""

    def test_one_hot_basic(self):
        """Test basic one-hot encoding."""
        import pypropel.fpsite as fpsite
        
        vec = fpsite.residue_one_hot('ALA')
        
        assert vec.shape == (20,)
        assert vec[0] == 1.0  # ALA is at index 0
        assert np.sum(vec) == 1.0

    def test_one_hot_all_amino_acids(self):
        """Test one-hot for all 20 standard amino acids."""
        import pypropel.fpsite as fpsite
        
        aa_list = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 
                   'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 
                   'THR', 'TRP', 'TYR', 'VAL']
        
        for i, aa in enumerate(aa_list):
            vec = fpsite.residue_one_hot(aa)
            assert vec[i] == 1.0
            assert np.sum(vec) == 1.0

    def test_one_hot_unknown(self):
        """Test one-hot for unknown amino acid."""
        import pypropel.fpsite as fpsite
        
        vec = fpsite.residue_one_hot('XYZ')
        
        assert vec.shape == (20,)
        assert np.sum(vec) == 0.0  # All zeros for unknown


class TestResiduePhyschem:
    """Tests for residue_physchem function."""

    def test_physchem_basic(self):
        """Test basic physicochemical properties."""
        import pypropel.fpsite as fpsite
        
        props = fpsite.residue_physchem('ALA')
        
        assert props.shape == (2,)
        assert props[0] == 0.0  # Neutral charge
        assert props[1] > 0  # Hydrophobic (positive KD value)

    def test_physchem_basic_residue(self):
        """Test basic (positive) amino acid."""
        import pypropel.fpsite as fpsite
        
        props = fpsite.residue_physchem('ARG')
        
        assert props[0] == 1.0  # Positive charge

    def test_physchem_acidic_residue(self):
        """Test acidic (negative) amino acid."""
        import pypropel.fpsite as fpsite
        
        props = fpsite.residue_physchem('ASP')
        
        assert props[0] == -1.0  # Negative charge


class TestResidueCoords:
    """Tests for residue_coords function."""

    def test_residue_coords(self, temp_pdb_file):
        """Test extracting coordinates from a residue."""
        import pypropel.fpsite as fpsite
        from Bio.PDB import PDBParser
        
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('test', temp_pdb_file)
        
        for model in structure:
            for chain in model:
                for residue in chain:
                    coords = fpsite.residue_coords(residue)
                    assert len(coords) > 0
                    assert coords.shape[1] == 3
                    break
                break
            break


class TestIsStandardAA:
    """Tests for AA validation functions."""

    def test_is_standard_aa_name_three_letter(self):
        """Test with three-letter codes."""
        import pypropel.fpsite as fpsite
        
        assert fpsite.is_standard_aa_name('ALA') == True
        assert fpsite.is_standard_aa_name('GLY') == True
        assert fpsite.is_standard_aa_name('XYZ') == False

    def test_is_standard_aa_name_one_letter(self):
        """Test with one-letter codes."""
        import pypropel.fpsite as fpsite
        
        assert fpsite.is_standard_aa_name('A') == True
        assert fpsite.is_standard_aa_name('G') == True
        assert fpsite.is_standard_aa_name('X') == False

    def test_is_standard_aa_residue(self, temp_pdb_file):
        """Test with BioPython residue."""
        import pypropel.fpsite as fpsite
        from Bio.PDB import PDBParser
        
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('test', temp_pdb_file)
        
        for model in structure:
            for chain in model:
                for residue in chain:
                    assert fpsite.is_standard_aa(residue) == True
                    break
                break
            break


class TestResidueWindow:
    """Tests for sequence window functions."""

    def test_window_middle(self):
        """Test window in middle of sequence."""
        import pypropel.fpsite as fpsite
        
        indices = fpsite.get_residue_window(100, 50, k=3)
        
        assert len(indices) == 7  # 2*3+1
        assert indices == [47, 48, 49, 50, 51, 52, 53]

    def test_window_start(self):
        """Test window at start of sequence."""
        import pypropel.fpsite as fpsite
        
        indices = fpsite.get_residue_window(100, 0, k=3)
        
        assert len(indices) == 7
        assert indices[3] == 0  # Center position
        assert indices[0] == 0  # Clipped to 0

    def test_window_padded(self):
        """Test padded window at boundary."""
        import pypropel.fpsite as fpsite
        
        indices = fpsite.get_residue_window_padded(100, 0, k=3, pad_value=-1)
        
        assert len(indices) == 7
        assert indices[0] == -1  # Out of bounds
        assert indices[3] == 0   # Center


class TestSpatialNeighbors:
    """Tests for spatial neighbor functions."""

    def test_spatial_neighbor_indices(self, temp_pdb_file):
        """Test getting spatial neighbors by index."""
        import pypropel.fpsite as fpsite
        from Bio.PDB import PDBParser
        
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('test', temp_pdb_file)
        
        neighbors = fpsite.get_spatial_neighbor_indices(structure, 0, k=2)
        
        # Should have at most 2 neighbors (only 3 residues in test PDB)
        assert len(neighbors) <= 2
        # Each should be (index, distance) tuple
        if len(neighbors) > 0:
            assert len(neighbors[0]) == 2


class TestPositionalEncoding:
    """Tests for positional encoding functions."""

    def test_sinusoidal_encoding(self):
        """Test sinusoidal positional encoding."""
        import pypropel.fpsite as fpsite
        
        pe = fpsite.positional_encoding(5, 100, dim=64, mode='sinusoidal')
        
        assert pe.shape == (64,)
        assert pe.dtype == np.float32

    def test_relative_encoding(self):
        """Test relative positional encoding."""
        import pypropel.fpsite as fpsite
        
        pe = fpsite.positional_encoding(50, 100, dim=32, mode='relative')
        
        assert pe.shape == (32,)

    def test_batch_encoding(self):
        """Test batch positional encoding."""
        import pypropel.fpsite as fpsite
        
        encodings = fpsite.batch_positional_encoding(100, dim=64)
        
        assert encodings.shape == (100, 64)

    def test_relative_position(self):
        """Test relative position normalization."""
        import pypropel.fpsite as fpsite
        
        assert fpsite.relative_position(0, 100) == 0.0
        assert abs(fpsite.relative_position(99, 100) - 1.0) < 1e-6
        assert abs(fpsite.relative_position(50, 101) - 0.5) < 1e-6

