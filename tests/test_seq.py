"""
Tests for pypropel.seq module - sequence operations.
"""

import pytest


class TestSequenceRead:
    """Tests for sequence reading functions."""

    def test_fasta_class_exists(self):
        """Test Fasta class can be imported."""
        from pypropel.prot.sequence.Fasta import Fasta
        
        fasta = Fasta()
        assert fasta is not None


class TestFpseqComposition:
    """Tests for fpseq composition functions."""

    def test_aac_composition(self, sample_sequence):
        """Test amino acid composition calculation."""
        import pypropel.fpseq as fpseq
        
        aac = fpseq.composition(seq=sample_sequence, mode='aac')
        
        assert isinstance(aac, dict)
        assert len(aac) == 20  # 20 standard amino acids
        
        # Each amino acid appears once in the sample sequence
        # So composition should be 1/20 = 0.05 for each
        for value in aac.values():
            assert abs(value - 0.05) < 1e-6

    def test_dac_composition(self, sample_sequence):
        """Test dipeptide composition calculation."""
        import pypropel.fpseq as fpseq
        
        dac = fpseq.composition(seq=sample_sequence, mode='dac')
        
        # DAC returns list of [pair, value]
        assert isinstance(dac, list)
        # 20^2 = 400 possible dipeptides
        assert len(dac) == 400

    def test_aveanf_composition(self, sample_sequence):
        """Test AVEANF composition calculation."""
        import pypropel.fpseq as fpseq
        
        aveanf = fpseq.composition(seq=sample_sequence, mode='aveanf')
        
        assert isinstance(aveanf, dict)
        assert len(aveanf) == 20
