__author__ = "Jianfeng Sun"
__version__ = "v1.0"
__copyright__ = "Copyright 2024"
__license__ = "GPL v3.0"
__email__ = "jianfeng.sunmt@gmail.com"
__maintainer__ = "Jianfeng Sun"

from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Polypeptide import PPBuilder, three_to_one, is_aa


class PDBSequence:
    """
    Simplified PDB sequence extraction class.

    Provides methods to extract amino acid sequences from PDB files
    without requiring chain-specific parameters.
    """

    def __init__(self):
        self.parser = PDBParser(QUIET=True)
        self.ppb = PPBuilder()

    def from_file(self, pdb_fpn: str) -> str:
        """
        Extract amino acid sequence from a PDB file.

        Parameters
        ----------
        pdb_fpn : str
            Full path to the PDB file.

        Returns
        -------
        str
            Amino acid sequence in one-letter code.
            Concatenates sequences from all chains.

        Examples
        --------
        >>> pdb_seq = PDBSequence()
        >>> seq = pdb_seq.from_file('/path/to/protein.pdb')
        >>> print(seq)
        'MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH...'
        """
        structure = self.parser.get_structure('protein', pdb_fpn)
        return self.from_structure(structure)

    def from_structure(self, structure) -> str:
        """
        Extract amino acid sequence from a Bio.PDB structure object.

        Uses the canonical residue filter (is_aa(standard=True) AND has CA atom)
        to ensure the sequence length matches GVP, ContactMap, and fpsite outputs.

        Parameters
        ----------
        structure : Bio.PDB.Structure.Structure
            A BioPython PDB structure object.

        Returns
        -------
        str
            Amino acid sequence in one-letter code.
        """
        seq = ""
        for model in structure:
            for chain in model:
                for residue in chain:
                    if is_aa(residue, standard=True) and 'CA' in residue:
                        try:
                            seq += three_to_one(residue.get_resname())
                        except KeyError:
                            seq += 'X'
        return seq

    def from_structure_ppbuilder(self, structure) -> str:
        """
        Extract amino acid sequence using PPBuilder (legacy method).

        This uses peptide bond geometry to identify residues, which may
        produce a different residue count than the canonical filter.
        Kept for backward compatibility.

        Parameters
        ----------
        structure : Bio.PDB.Structure.Structure
            A BioPython PDB structure object.

        Returns
        -------
        str
            Amino acid sequence in one-letter code.
        """
        seq = ""
        for pp in self.ppb.build_peptides(structure):
            seq += str(pp.get_sequence())
        return seq


if __name__ == "__main__":
    # Test the class
    import os
    
    # Example usage
    pdb_seq = PDBSequence()
    
    # Test with a sample PDB file if available
    test_pdb = "/Users/zhaoj/Project/protein_ligand/database/P-L/1998-2005/1a0q/1a0q_protein.pdb"
    if os.path.exists(test_pdb):
        seq = pdb_seq.from_file(test_pdb)
        print(f"Sequence length: {len(seq)}")
        print(f"First 50 chars: {seq[:50]}")
