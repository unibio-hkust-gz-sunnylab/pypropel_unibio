__version__ = "v1.0"
__copyright__ = "Copyright 2024"
__license__ = "GPL v3.0"
__developer__ = "Jianfeng Sun"
__maintainer__ = "Jianfeng Sun"
__email__="jianfeng.sunmt@gmail.com"

from typing import List, Dict
import numpy as np

from pypropel.prot.feature.sequence.AminoAcidProperty import AminoAcidProperty as aaprop
from pypropel.prot.feature.sequence.AminoAcidRepresentation import AminoAcidRepresentation as aarepr
from pypropel.prot.feature.sequence.Position import Position
from pypropel.prot.feature.rsa.Reader import Reader as rsareader
from pypropel.prot.feature.ss.Reader import Reader as ssreader


# ==================== Amino Acid Mappings ====================

AA_MAP = {
    'ALA': 0, 'ARG': 1, 'ASN': 2, 'ASP': 3, 'CYS': 4,
    'GLN': 5, 'GLU': 6, 'GLY': 7, 'HIS': 8, 'ILE': 9,
    'LEU': 10, 'LYS': 11, 'MET': 12, 'PHE': 13, 'PRO': 14,
    'SER': 15, 'THR': 16, 'TRP': 17, 'TYR': 18, 'VAL': 19
}

# Charge mapping (at physiological pH ~7.4)
CHARGE_MAP = {
    'ARG': 1.0, 'LYS': 1.0, 'HIS': 0.5,  # Basic (positive)
    'ASP': -1.0, 'GLU': -1.0,             # Acidic (negative)
}

# Kyte-Doolittle Hydrophobicity scale
KD_HYDROPHOBICITY = {
    'ILE': 4.5, 'VAL': 4.2, 'LEU': 3.8, 'PHE': 2.8, 'CYS': 2.5,
    'MET': 1.9, 'ALA': 1.8, 'GLY': -0.4, 'THR': -0.7, 'SER': -0.8,
    'TRP': -0.9, 'TYR': -1.3, 'PRO': -1.6, 'HIS': -3.2, 'GLU': -3.5,
    'GLN': -3.5, 'ASP': -3.5, 'ASN': -3.5, 'LYS': -3.9, 'ARG': -4.5
}


# ==================== Residue Feature Functions ====================

def residue_one_hot(residue_name: str) -> np.ndarray:
    """
    Generate a 20-dimensional one-hot encoding for an amino acid.
    
    Parameters
    ----------
    residue_name : str
        Three-letter amino acid code (e.g., 'ALA', 'GLY').
        
    Returns
    -------
    np.ndarray
        20-dimensional one-hot vector.
        
    Examples
    --------
    >>> import pypropel.fpsite as fpsite
    >>> vec = fpsite.residue_one_hot('ALA')
    >>> print(vec.shape)  # (20,)
    >>> print(vec[0])     # 1.0 (ALA is at index 0)
    """
    vec = np.zeros(20, dtype=np.float32)
    if residue_name in AA_MAP:
        vec[AA_MAP[residue_name]] = 1.0
    return vec


def residue_physchem(residue_name: str) -> np.ndarray:
    """
    Get physicochemical properties for an amino acid.
    
    Returns [Charge, Hydrophobicity (normalized)].
    - Charge: -1 (Acidic), +1 (Basic), 0 (Neutral), 0.5 (His at pH 7)
    - Hydrophobicity: Kyte-Doolittle scale normalized to [-1, 1]
    
    Parameters
    ----------
    residue_name : str
        Three-letter amino acid code.
        
    Returns
    -------
    np.ndarray
        2-dimensional feature vector [charge, hydrophobicity].
        
    Examples
    --------
    >>> import pypropel.fpsite as fpsite
    >>> props = fpsite.residue_physchem('ARG')
    >>> print(props)  # [1.0, -1.0] (basic, hydrophilic)
    """
    charge = CHARGE_MAP.get(residue_name, 0.0)
    hydro = KD_HYDROPHOBICITY.get(residue_name, 0.0) / 4.5  # Normalize to ~[-1, 1]
    return np.array([charge, hydro], dtype=np.float32)


def residue_coords(residue) -> np.ndarray:
    """
    Extract atom coordinates from a BioPython residue object.
    
    Parameters
    ----------
    residue : Bio.PDB.Residue.Residue
        BioPython residue object.
        
    Returns
    -------
    np.ndarray
        Atom coordinates with shape (N, 3).
        
    Examples
    --------
    >>> import pypropel.fpsite as fpsite
    >>> from Bio.PDB import PDBParser
    >>> parser = PDBParser(QUIET=True)
    >>> struct = parser.get_structure('test', 'protein.pdb')
    >>> for model in struct:
    ...     for chain in model:
    ...         for residue in chain:
    ...             coords = fpsite.residue_coords(residue)
    ...             print(coords.shape)
    """
    coords = []
    for atom in residue:
        coords.append(atom.get_coord())
    return np.array(coords) if coords else np.array([]).reshape(0, 3)


# ==================== Existing Functions ====================


def property(
        prop_kind : str ='positive',
        prop_met : str ='Russell',
        standardize : bool =True,
) -> Dict:
    """
    An amino acid's property

    Parameters
    ----------
    prop_kind
        an amino acid's property kind
    prop_met
        method from which a property is derived,
    standalize
        if standardization

    Returns
    -------

    """
    return {
        "positive": aaprop().positive,
        "negative": aaprop().negative,
        "charged": aaprop().charged,
        "polar": aaprop().polar,
        "aliphatic": aaprop().aliphatic,
        "aromatic": aaprop().aromatic,
        "hydrophobic": aaprop().hydrophobic,
        "small": aaprop().small,
        "active": aaprop().active,
        "weight": aaprop().weight,
        "pI": aaprop().pI,
        "solubility": aaprop().solubility,
        "tm": aaprop().tm,
        "pka": aaprop().pka,
        "pkb": aaprop().pkb,
        "hydrophilicity": aaprop().hydrophilicity,
        "hydrophobicity": aaprop().hydrophobicity,
        "fet": aaprop().fet,
        "hydration": aaprop().hydration,
        "signal": aaprop().signal,
        "volume": aaprop().volume,
        "polarity": aaprop().polarity,
        "composition": aaprop().composition,
    }[prop_kind](standardize=standardize)


def onehot(
        arr_2d,
        arr_aa_names,
) -> List[List]:
    return aarepr().onehot(
        arr_2d=arr_2d,
        arr_aa_names=arr_aa_names,
    )


def pos_abs_val(
        pos : int,
        seq : str,
):
    return Position().absolute(
        pos=pos,
        seq=seq,
    )

def pos_rel_val(
        pos : int,
        interval : List,
):
    return Position().relative(
        pos=pos,
        interval=interval,
    )


def deepconpred():
    return Position().deepconpred()


def metapsicov():
    return Position().metapsicov()


def rsa_solvpred(
        solvpred_fp,
        prot_name,
        file_chain,
):
    return rsareader().solvpred(
        solvpred_fp=solvpred_fp,
        prot_name=prot_name,
        file_chain=file_chain,
    )


def rsa_accpro(
        accpro_fp,
        prot_name,
        file_chain,
):
    return rsareader().accpro(
        accpro_fp=accpro_fp,
        prot_name=prot_name,
        file_chain=file_chain,
    )


def rsa_accpro20(
        accpro20_fp,
        prot_name,
        file_chain,
):
    return rsareader().accpro20(
        accpro20_fp=accpro20_fp,
        prot_name=prot_name,
        file_chain=file_chain,
    )


def ss_spider3(
        spider3_path,
        prot_name,
        file_chain,
):
    return ssreader().spider3(
        spider3_path=spider3_path,
        prot_name=prot_name,
        file_chain=file_chain,
    )


def ss_spider3_ss(
        spider3_path,
        prot_name,
        file_chain,
        sv_fp,
):
    return ssreader().spider3_to_ss(
        spider3_path=spider3_path,
        prot_name=prot_name,
        file_chain=file_chain,
        sv_fp=sv_fp,
    )


def ss_psipred(
        psipred_path,
        prot_name,
        file_chain,
        kind='ss'
):
    """

    Parameters
    ----------
    psipred_path
    prot_name
    file_chain
    kind
        1. ss;
        2. ss2;
        3. horiz

    Returns
    -------

    """
    if kind == 'ss':
        return ssreader().psipred(
            psipred_ss_path=psipred_path,
            prot_name=prot_name,
            file_chain=file_chain,
        )
    if kind == 'ss2':
        return ssreader().psipred(
            psipred_ss2_path=psipred_path,
            prot_name=prot_name,
            file_chain=file_chain,
        )
    if kind == 'horiz':
        return ssreader().psipred(
            psipred_horiz_path=psipred_path,
            prot_name=prot_name,
            file_chain=file_chain,
        )
    else:
        return ssreader().psipred(
            psipred_ss_path=psipred_path,
            prot_name=prot_name,
            file_chain=file_chain,
        )


def ss_sspro(
        sspro_path,
        prot_name,
        file_chain,
):
    return ssreader().sspro(
        sspro_path=sspro_path,
        prot_name=prot_name,
        file_chain=file_chain,
    )


def ss_sspro8(
        sspro8_path,
        prot_name,
        file_chain,
):
    return ssreader().sspro8(
        sspro8_path=sspro8_path,
        prot_name=prot_name,
        file_chain=file_chain,
    )


if __name__ == "__main__":
    from pypropel.prot.sequence.Fasta import Fasta as sfasta
    from pypropel.path import to
    # import tmkit as tmk

    # print(property('positive'))
    #
    # sequence = sfasta().get(
    #     fasta_fpn=to("data/fasta/1aigL.fasta")
    # )
    # # print(sequence)
    #
    # pos_list = tmk.seq.pos_list_single(len_seq=len(sequence), seq_sep_superior=None, seq_sep_inferior=0)
    # # print(pos_list)
    #
    # positions = tmk.seq.pos_single(sequence=sequence, pos_list=pos_list)
    # # print(positions)
    #
    # win_aa_ids = tmk.seq.win_id_single(
    #     sequence=sequence,
    #     position=positions,
    #     window_size=1,
    # )
    # # print(win_aa_ids)
    #
    # win_aas = tmk.seq.win_name_single(
    #     sequence=sequence,
    #     position=positions,
    #     window_size=1,
    #     mids=win_aa_ids,
    # )
    # # print(win_aas)
    #
    # features = [[] for i in range(len(sequence))]
    # print(features)
    # print(len(features))

    # print(onehot(
    #     arr_2d=features,
    #     arr_aa_names=win_aas,
    # )[0])

    # print(pos_abs_val(
    #     pos=positions[0][0],
    #     seq=sequence,
    # ))
    #
    # print(pos_rel_val(
    #     pos=positions[0][0],
    #     interval=[4, 10],
    # ))
    #
    # print(deepconpred())
    #
    # print(metapsicov())

    # print(rsa_solvpred(
    #     solvpred_fp=to('data/accessibility/solvpred/'),
    #     prot_name='1aig',
    #     file_chain='L',
    # ))

    # print(rsa_accpro(
    #     accpro_fp=to('data/accessibility/accpro/'),
    #     prot_name='1aig',
    #     file_chain='L',
    # ))

    # print(rsa_accpro20(
    #     accpro20_fp=to('data/accessibility/accpro20/'),
    #     prot_name='1aig',
    #     file_chain='L',
    # ))

    # print(ss_psipred(
    #     psipred_path=to('data/ss/psipred/'),
    #     prot_name='1aig',
    #     file_chain='L',
    #     kind='ss', # horiz, ss, ss2
    # ))

    # print(ss_sspro(
    #     sspro_path=to('data/ss/sspro/'),
    #     prot_name='1aig',
    #     file_chain='L'
    # ))

    # print(ss_sspro8(
    #     sspro8_path=to('data/ss/sspro8/'),
    #     prot_name='1aig',
    #     file_chain='L'
    # ))

    # print(ss_spider3(
    #     spider3_path=to('data/ss/spider3/'),
    #     prot_name='E',
    #     file_chain=''
    # ))

    # print(ss_spider3_ss(
    #     spider3_path=to('data/ss/spider3/'),
    #     prot_name='E',
    #     file_chain='',
    #     sv_fp=to('data/ss/spider3/'),
    # ))


    seq = "ADGCGVGEGTGQGPMCNCMCMKWVYADEDAADLESDSFADEDASLESDSFPWSNQRVFCSFADEDAS"
    print(seq)

    feature_vector = [[] for i in range(len(seq))]
    print(feature_vector)
    print(len(feature_vector))

    print(property(
        prop_met='Hopp',
        prop_kind='hydrophilicity'
    ))

