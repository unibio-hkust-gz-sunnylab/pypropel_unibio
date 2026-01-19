__author__ = "Jianfeng Sun"
__version__ = "v1.0"
__copyright__ = "Copyright 2024"
__license__ = "GPL v3.0"
__email__ = "jianfeng.sunmt@gmail.com"
__maintainer__ = "Jianfeng Sun"

from typing import  List, Dict

import pandas as pd

from pypropel.prot.sequence.PDB import PDB
from pypropel.prot.sequence.PDBSequence import PDBSequence
from pypropel.prot.structure.convert.ToFasta import ToFasta
from pypropel.prot.structure.chain.Format import Format
from pypropel.prot.structure.chain.Splitter import Splitter
from pypropel.prot.structure.hetatm.Remove import Remove
from pypropel.prot.structure.distance.isite.heavy.AllAgainstAll import AllAgainstAll
from pypropel.util.Console import Console


def read_pdb_sequence(pdb_fpn: str) -> str:
    """
    Extract amino acid sequence from a PDB file.
    
    A simplified function that reads a PDB file and returns
    the concatenated amino acid sequence from all chains.
    
    Parameters
    ----------
    pdb_fpn : str
        Full path to the PDB file.
        
    Returns
    -------
    str
        Amino acid sequence in one-letter code.
        
    Examples
    --------
    >>> import pypropel.str as ppstr
    >>> seq = ppstr.read_pdb_sequence('/path/to/protein.pdb')
    >>> print(seq[:20])
    'MVLSPADKTNVKAAWGKVGA'
    """
    return PDBSequence().from_file(pdb_fpn)


def structure_to_sequence(structure) -> str:
    """
    Extract amino acid sequence from a Bio.PDB structure object.
    
    Parameters
    ----------
    structure : Bio.PDB.Structure.Structure
        A BioPython PDB structure object.
        
    Returns
    -------
    str
        Amino acid sequence in one-letter code.
        
    Examples
    --------
    >>> from Bio.PDB import PDBParser
    >>> import pypropel.str as ppstr
    >>> parser = PDBParser(QUIET=True)
    >>> structure = parser.get_structure('prot', '/path/to/protein.pdb')
    >>> seq = ppstr.structure_to_sequence(structure)
    >>> print(seq[:20])
    'MVLSPADKTNVKAAWGKVGA'
    """
    return PDBSequence().from_structure(structure)


def load_pdb(pdb_path: str):
    """
    Load a PDB file and return a BioPython Structure object.
    
    A convenience function that wraps Bio.PDB.PDBParser.
    
    Parameters
    ----------
    pdb_path : str
        Full path to the PDB file.
        
    Returns
    -------
    Bio.PDB.Structure.Structure or None
        BioPython Structure object, or None if loading fails.
        
    Examples
    --------
    >>> import pypropel.str as ppstr
    >>> structure = ppstr.load_pdb('/path/to/protein.pdb')
    >>> print(structure)
    <Structure id=protein>
    """
    from Bio.PDB import PDBParser
    
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure('protein', pdb_path)
        return structure
    except Exception as e:
        print(f"Error loading PDB {pdb_path}: {e}")
        return None


def read(
        pdb_path,
        pdb_name,
        file_chain,
        seq_chain,
):
    return PDB(
        pdb_path=pdb_path,
        pdb_name=pdb_name,
        file_chain=file_chain,
        seq_chain=seq_chain,
    ).chain()


def chains(
        pdb_fp,
        pdb_name,
):
    return AllAgainstAll(
        pdb_fp=pdb_fp,
        pdb_name=pdb_name,
    ).chains()


def tofasta(
        prot_df: pd.DataFrame,
        sv_fp: str,
        pdb_path: str,
):
    return ToFasta(
        prot_df=prot_df,
        sv_fp=sv_fp,
    ).frompdb(
        pdb_path=pdb_path,
    )


def del_end(
        prot_df: pd.DataFrame,
        sv_fp: str,
        pdb_path: str,
):
    return Format(
        prot_df=prot_df,
        sv_fp=sv_fp,
    ).del_END_frompdb(
        pdb_path=pdb_path,
    )


def split_cplx_to_sgl(
        prot_df: pd.DataFrame,
        pdb_path: str,
        sv_fp: str,
):
    return Splitter(
        prot_df=prot_df,
        pdb_path=pdb_path,
        sv_fp=sv_fp,
    ).pdb_per_chain()


def remove_hetatm(
        prot_df: pd.DataFrame,
        pdb_path: str,
        sv_fp: str,
):
    return Remove(
        prot_df=prot_df,
    ).biopython(
        pdb_path=pdb_path,
        sv_fp=sv_fp,
    )


if __name__ == "__main__":
    from pypropel.path import to

    # print(read(
    #     pdb_path=to('data/pdb/pdbtm/'),
    #     pdb_name='1aij',
    #     file_chain='L',
    #     seq_chain='L',
    # ))

    print(chains(
        pdb_fp=to('data/pdb/complex/pdbtm/'),
        pdb_name='1aij',
    ))

    prot_df = pd.DataFrame({
        'prot': ['1aig', '1aij', '1xqf'],
        'chain': ['L', 'L', 'A'],
    })

    # print(tofasta(
    #     prot_df,
    #     sv_fp=to('data/'),
    #     pdb_path=to('data/pdb/pdbtm/'),
    # ))

    # print(del_end(
    #     prot_df,
    #     sv_fp=to('data/'),
    #     pdb_path=to('data/pdb/pdbtm/'),
    # ))

    # print(split_cplx_to_sgl(
    #     prot_df=prot_df,
    #     pdb_path=to('data/pdb/complex/pdbtm/'),
    #     sv_fp=to('data/'),
    # ))

    # print(remove_hetatm(
    #     prot_df=prot_df,
    #     pdb_path=to('data/pdb/complex/pdbtm/'),
    #     sv_fp=to('data/pdb/'),
    # ))