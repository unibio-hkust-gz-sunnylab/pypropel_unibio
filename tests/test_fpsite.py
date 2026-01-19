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
