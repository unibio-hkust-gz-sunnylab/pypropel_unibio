"""
Tests for pypropel.str module - structure operations.
"""

import pytest


class TestReadPdbSequence:
    """Tests for PDB sequence reading."""

    def test_read_pdb_sequence(self, temp_pdb_file):
        """Test reading sequence from PDB file."""
        import pypropel.str as ppstr
        
        seq = ppstr.read_pdb_sequence(temp_pdb_file)
        
        # Should extract ALA, GLY, SER -> A, G, S
        assert seq == "AGS"

    def test_structure_to_sequence(self, temp_pdb_file):
        """Test extracting sequence from Bio.PDB structure."""
        import pypropel.str as ppstr
        from Bio.PDB import PDBParser
        
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('test', temp_pdb_file)
        
        seq = ppstr.structure_to_sequence(structure)
        
        assert seq == "AGS"


class TestPDBSequenceClass:
    """Tests for PDBSequence class."""

    def test_pdb_sequence_from_file(self, temp_pdb_file):
        """Test PDBSequence.from_file method."""
        from pypropel.prot.sequence.PDBSequence import PDBSequence
        
        pdb_seq = PDBSequence()
        seq = pdb_seq.from_file(temp_pdb_file)
        
        assert seq == "AGS"

    def test_pdb_sequence_from_structure(self, temp_pdb_file):
        """Test PDBSequence.from_structure method."""
        from pypropel.prot.sequence.PDBSequence import PDBSequence
        from Bio.PDB import PDBParser
        
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('test', temp_pdb_file)
        
        pdb_seq = PDBSequence()
        seq = pdb_seq.from_structure(structure)
        
        assert seq == "AGS"
