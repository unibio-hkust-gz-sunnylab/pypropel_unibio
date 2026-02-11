"""
Tests for canonical residue filtering consistency across modules.

Verifies that gvp.get_canonical_residues(), fpsite.get_protein_scalar_features(),
PDBSequence.from_structure(), and ContactMap.protein_ligand_distances() all use
the same residue set: is_aa(standard=True) AND 'CA' in residue.
"""

import pytest
import numpy as np
import tempfile
import os

# Direct module imports to avoid pypropel.__init__ mini3di dependency
import sys
import importlib.util

_pypropel_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _pypropel_dir)


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ppgvp = _load_module('gvp', os.path.join(_pypropel_dir, 'pypropel', 'gvp.py'))
ppfpsite = _load_module('fpsite', os.path.join(_pypropel_dir, 'pypropel', 'fpsite.py'))


@pytest.fixture
def pdb_with_missing_ca():
    """PDB with 3 normal residues + 1 residue missing its CA atom.

    Residues:
      ALA A 1 — has N, CA, C, O, CB  (normal)
      GLY A 2 — has N, CA, C, O      (normal)
      SER A 3 — has N, C, O, CB, OG  (NO CA — should be excluded)
      VAL A 4 — has N, CA, C, O, CB  (normal)
    """
    pdb_content = """\
HEADER    TEST PROTEIN WITH MISSING CA
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C
ATOM      3  C   ALA A   1       2.009   1.420   0.000  1.00  0.00           C
ATOM      4  O   ALA A   1       1.251   2.390   0.000  1.00  0.00           O
ATOM      5  CB  ALA A   1       1.986  -0.760   1.216  1.00  0.00           C
ATOM      6  N   GLY A   2       3.310   1.540   0.000  1.00  0.00           N
ATOM      7  CA  GLY A   2       3.970   2.840   0.000  1.00  0.00           C
ATOM      8  C   GLY A   2       5.470   2.720   0.000  1.00  0.00           C
ATOM      9  O   GLY A   2       6.030   1.620   0.000  1.00  0.00           O
ATOM     10  N   SER A   3       6.110   3.880   0.000  1.00  0.00           N
ATOM     11  C   SER A   3       8.110   5.400   0.000  1.00  0.00           C
ATOM     12  O   SER A   3       7.350   6.370   0.000  1.00  0.00           O
ATOM     13  CB  SER A   3       8.090   3.230   1.216  1.00  0.00           C
ATOM     14  OG  SER A   3       7.590   1.900   1.216  1.00  0.00           O
ATOM     15  N   VAL A   4       9.400   5.500   0.000  1.00  0.00           N
ATOM     16  CA  VAL A   4      10.060   6.800   0.000  1.00  0.00           C
ATOM     17  C   VAL A   4      11.560   6.680   0.000  1.00  0.00           C
ATOM     18  O   VAL A   4      12.120   5.580   0.000  1.00  0.00           O
ATOM     19  CB  VAL A   4      10.590   7.560   1.216  1.00  0.00           C
END
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f:
        f.write(pdb_content)
        path = f.name

    yield path
    os.unlink(path)


@pytest.fixture
def normal_pdb(temp_pdb_file):
    """Re-use conftest's temp_pdb_file (ALA, GLY, SER — all have CA)."""
    return temp_pdb_file


def _parse(path):
    from Bio.PDB import PDBParser
    return PDBParser(QUIET=True).get_structure('test', path)


# ── get_canonical_residues ──────────────────────────────────────────

class TestGetCanonicalResidues:
    """Tests for gvp.get_canonical_residues()."""

    def test_excludes_residue_without_ca(self, pdb_with_missing_ca):
        structure = _parse(pdb_with_missing_ca)
        residues = ppgvp.get_canonical_residues(structure)

        names = [r.get_resname() for r in residues]
        assert names == ['ALA', 'GLY', 'VAL']
        assert len(residues) == 3  # SER A 3 excluded

    def test_includes_all_when_all_have_ca(self, temp_pdb_file):
        structure = _parse(temp_pdb_file)
        residues = ppgvp.get_canonical_residues(structure)

        assert len(residues) == 3  # ALA, GLY, SER

    def test_chain_filter(self, pdb_with_missing_ca):
        structure = _parse(pdb_with_missing_ca)

        residues_a = ppgvp.get_canonical_residues(structure, chain_id='A')
        residues_b = ppgvp.get_canonical_residues(structure, chain_id='B')

        assert len(residues_a) == 3
        assert len(residues_b) == 0

    def test_matches_get_ca_coords_length(self, pdb_with_missing_ca):
        structure = _parse(pdb_with_missing_ca)

        residues = ppgvp.get_canonical_residues(structure)
        ca_coords = ppgvp.get_ca_coords(structure)

        assert len(residues) == len(ca_coords)


# ── Cross-module consistency ────────────────────────────────────────

class TestCrossModuleConsistency:
    """All modules must produce the same residue count."""

    def test_gvp_matches_fpsite(self, pdb_with_missing_ca):
        structure = _parse(pdb_with_missing_ca)

        canonical = ppgvp.get_canonical_residues(structure)
        scalar_feats = ppfpsite.get_protein_scalar_features(structure)

        assert len(scalar_feats) == len(canonical)

    def test_gvp_matches_pdb_sequence(self, pdb_with_missing_ca):
        from pypropel.prot.sequence.PDBSequence import PDBSequence

        structure = _parse(pdb_with_missing_ca)

        canonical = ppgvp.get_canonical_residues(structure)
        seq = PDBSequence().from_structure(structure)

        assert len(seq) == len(canonical)

    def test_gvp_matches_contact_map(self, pdb_with_missing_ca):
        from pypropel.prot.structure.distance.ContactMap import ContactMap

        structure = _parse(pdb_with_missing_ca)
        ligand_coords = np.array([[5.0, 5.0, 5.0], [6.0, 6.0, 6.0]])

        canonical = ppgvp.get_canonical_residues(structure)
        cm = ContactMap()
        distances_df = cm.protein_ligand_distances(structure, ligand_coords)

        assert len(distances_df) == len(canonical)

    def test_all_four_modules_agree(self, pdb_with_missing_ca):
        """The core invariant: all modules produce the same count."""
        from pypropel.prot.sequence.PDBSequence import PDBSequence
        from pypropel.prot.structure.distance.ContactMap import ContactMap

        structure = _parse(pdb_with_missing_ca)
        ligand_coords = np.array([[5.0, 5.0, 5.0]])

        n_gvp = len(ppgvp.get_canonical_residues(structure))
        n_ca = len(ppgvp.get_ca_coords(structure))
        n_scalar = len(ppfpsite.get_protein_scalar_features(structure))
        n_seq = len(PDBSequence().from_structure(structure))
        n_dist = len(ContactMap().protein_ligand_distances(structure, ligand_coords))

        assert n_gvp == n_ca == n_scalar == n_seq == n_dist == 3

    def test_all_agree_normal_pdb(self, temp_pdb_file):
        """Same invariant on a PDB where all residues have CA."""
        from pypropel.prot.sequence.PDBSequence import PDBSequence
        from pypropel.prot.structure.distance.ContactMap import ContactMap

        structure = _parse(temp_pdb_file)
        ligand_coords = np.array([[5.0, 5.0, 5.0]])

        n_gvp = len(ppgvp.get_canonical_residues(structure))
        n_ca = len(ppgvp.get_ca_coords(structure))
        n_scalar = len(ppfpsite.get_protein_scalar_features(structure))
        n_seq = len(PDBSequence().from_structure(structure))
        n_dist = len(ContactMap().protein_ligand_distances(structure, ligand_coords))

        assert n_gvp == n_ca == n_scalar == n_seq == n_dist == 3


# ── PDBSequence canonical vs legacy ────────────────────────────────

class TestPDBSequenceCanonical:
    """Tests for PDBSequence.from_structure() using canonical filter."""

    def test_sequence_content(self, pdb_with_missing_ca):
        from pypropel.prot.sequence.PDBSequence import PDBSequence

        structure = _parse(pdb_with_missing_ca)
        seq = PDBSequence().from_structure(structure)

        # ALA=A, GLY=G, VAL=V  (SER excluded — no CA)
        assert seq == "AGV"

    def test_legacy_ppbuilder_still_works(self, temp_pdb_file):
        from pypropel.prot.sequence.PDBSequence import PDBSequence

        structure = _parse(temp_pdb_file)
        seq = PDBSequence().from_structure_ppbuilder(structure)

        assert isinstance(seq, str)
        assert len(seq) > 0


# ── fpsite scalar features CA filter ────────────────────────────────

class TestFpsiteCAFilter:
    """Tests for fpsite.get_protein_scalar_features() CA filtering."""

    def test_excludes_residue_without_ca(self, pdb_with_missing_ca):
        structure = _parse(pdb_with_missing_ca)
        feats = ppfpsite.get_protein_scalar_features(structure)

        assert feats.shape == (3, 35)  # 3 residues, not 4

    def test_feature_shape_normal(self, temp_pdb_file):
        structure = _parse(temp_pdb_file)
        feats = ppfpsite.get_protein_scalar_features(structure, use_dssp=False)

        assert feats.shape == (3, 35)
        assert feats.dtype == np.float32
