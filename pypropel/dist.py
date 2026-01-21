__version__ = "v1.0"
__copyright__ = "Copyright 2024"
__license__ = "GPL v3.0"
__developer__ = "Jianfeng Sun"
__maintainer__ = "Jianfeng Sun"
__email__="jianfeng.sunmt@gmail.com"

from typing import List, Dict, Optional
from typing_extensions import deprecated

import numpy as np
import pandas as pd

from pypropel.prot.structure.distance.isite.heavy.AllAgainstAll import AllAgainstAll
from pypropel.prot.structure.distance.isite.heavy.OneToOne import OneToOne
from pypropel.prot.structure.distance.isite.DistanceComplexOne import DistanceComplexOne
from pypropel.prot.structure.distance.isite.check.Complex import Complex
from pypropel.prot.structure.distance.isite.check.Pair import Pair
from pypropel.prot.structure.distance.isite.Label import Label
from pypropel.prot.structure.distance.ContactMap import ContactMap

# Global instance for convenience functions
_contact_map = ContactMap()


# ==================== Atom Level ====================

def atom_distance(coord1: np.ndarray, coord2: np.ndarray) -> float:
    """
    Calculate Euclidean distance between two atom coordinates.
    
    Parameters
    ----------
    coord1 : np.ndarray
        3D coordinate of first atom, shape (3,).
    coord2 : np.ndarray
        3D coordinate of second atom, shape (3,).
        
    Returns
    -------
    float
        Euclidean distance between the atoms.
        
    Examples
    --------
    >>> import pypropel.dist as ppdist
    >>> import numpy as np
    >>> d = ppdist.atom_distance(np.array([0,0,0]), np.array([3,4,0]))
    >>> print(d)  # 5.0
    """
    return _contact_map.atom_distance(coord1, coord2)


def atom_distance_matrix(coords1: np.ndarray, coords2: np.ndarray) -> np.ndarray:
    """
    Calculate pairwise distance matrix between two sets of atom coordinates.
    
    Parameters
    ----------
    coords1 : np.ndarray
        First set of coordinates, shape (N, 3).
    coords2 : np.ndarray
        Second set of coordinates, shape (M, 3).
        
    Returns
    -------
    np.ndarray
        Distance matrix of shape (N, M).
    """
    return _contact_map.atom_distance_matrix(coords1, coords2)


# ==================== Residue Level ====================

def residue_distance(res1_coords: np.ndarray, res2_coords: np.ndarray) -> float:
    """
    Calculate minimum distance between any atoms of two residues.
    
    Parameters
    ----------
    res1_coords : np.ndarray
        Coordinates of all atoms in residue 1, shape (N, 3).
    res2_coords : np.ndarray
        Coordinates of all atoms in residue 2, shape (M, 3).
        
    Returns
    -------
    float
        Minimum distance between any atom pair.
    """
    return _contact_map.residue_distance(res1_coords, res2_coords)


def residue_ligand_distance(residue_coords: np.ndarray, ligand_coords: np.ndarray) -> float:
    """
    Calculate minimum distance from a residue to a ligand.
    
    Parameters
    ----------
    residue_coords : np.ndarray
        Coordinates of residue atoms, shape (N, 3).
    ligand_coords : np.ndarray
        Coordinates of ligand atoms, shape (M, 3).
        
    Returns
    -------
    float
        Minimum distance from any residue atom to any ligand atom.
    """
    return _contact_map.residue_ligand_distance(residue_coords, ligand_coords)


# ==================== Protein Level ====================

def protein_distance_matrix(
        structure, 
        chain_ids: Optional[List[str]] = None,
        use_ca_only: bool = False
) -> pd.DataFrame:
    """
    Calculate residue-residue distance matrix for a protein structure.
    
    Parameters
    ----------
    structure : Bio.PDB.Structure.Structure
        BioPython structure object.
    chain_ids : List[str], optional
        List of chain IDs to include. If None, includes all chains.
    use_ca_only : bool, optional
        If True, use only C-alpha atoms for distance calculation.
        
    Returns
    -------
    pd.DataFrame
        Distance matrix with residue info as index/columns.
    """
    return _contact_map.protein_distance_matrix(structure, chain_ids, use_ca_only)


def protein_ligand_distances(structure, ligand_coords: np.ndarray) -> pd.DataFrame:
    """
    Calculate distances from all protein residues to a ligand.
    
    Parameters
    ----------
    structure : Bio.PDB.Structure.Structure
        BioPython structure object.
    ligand_coords : np.ndarray
        Ligand atom coordinates, shape (M, 3).
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: chain, res_id, res_name, distance.
    """
    return _contact_map.protein_ligand_distances(structure, ligand_coords)


# ==================== Contact Maps ====================

def protein_contact_map(
        structure, 
        threshold: float = 8.0,
        chain_ids: Optional[List[str]] = None,
        use_ca_only: bool = False
) -> np.ndarray:
    """
    Generate binary contact map for a protein.
    
    Parameters
    ----------
    structure : Bio.PDB.Structure.Structure
        BioPython structure object.
    threshold : float, optional
        Distance threshold for contact. Default is 8.0 Angstroms.
    chain_ids : List[str], optional
        List of chain IDs to include.
    use_ca_only : bool, optional
        If True, use only C-alpha atoms.
        
    Returns
    -------
    np.ndarray
        Binary contact map (1 = contact, 0 = no contact).
    """
    return _contact_map.protein_contact_map(structure, threshold, chain_ids, use_ca_only)


def protein_ligand_contact_map(
        structure, 
        ligand_coords: np.ndarray,
        threshold: float = 5.0
) -> np.ndarray:
    """
    Generate contact map between protein residues and ligand atoms.
    
    Parameters
    ----------
    structure : Bio.PDB.Structure.Structure
        BioPython structure object.
    ligand_coords : np.ndarray
        Ligand atom coordinates, shape (M, 3).
    threshold : float, optional
        Distance threshold for contact. Default is 5.0 Angstroms.
        
    Returns
    -------
    np.ndarray
        Binary contact map of shape (N_residues, M_ligand_atoms).
    """
    return _contact_map.protein_ligand_contact_map(structure, ligand_coords, threshold)


# ==================== Pocket Extraction ====================

def extract_binding_pocket(
        structure, 
        ligand_coords: np.ndarray,
        binding_threshold: float = 5.0,
        non_binding_threshold: float = 8.0
) -> pd.DataFrame:
    """
    Extract binding pocket residues with classification labels.
    
    Parameters
    ----------
    structure : Bio.PDB.Structure.Structure
        BioPython structure object.
    ligand_coords : np.ndarray
        Ligand atom coordinates, shape (M, 3).
    binding_threshold : float, optional
        Distance below which residue is labeled as binding (1).
        Default is 5.0 Angstroms.
    non_binding_threshold : float, optional
        Distance above which residue is labeled as non-binding (0).
        Default is 8.0 Angstroms.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: chain, res_id, res_name, distance, label.
        label: 1 = binding, 0 = non-binding, -1 = margin.
    """
    return _contact_map.extract_binding_pocket(
        structure, ligand_coords, binding_threshold, non_binding_threshold
    )


# ==================== Distance Classification ====================

def classify_binding_distance(
    distance: float,
    thresholds: List[float] = [3.5, 6.0]
) -> int:
    """
    Classify distance into N+1 bins based on N thresholds.
    
    Parameters
    ----------
    distance : float
        Distance value to classify (Angstroms).
    thresholds : List[float]
        Sorted list of threshold values. Creates len(thresholds)+1 classes.
        
    Returns
    -------
    int
        Class label from 0 to len(thresholds).
        
    Examples
    --------
    Default 3 classes with thresholds [3.5, 6.0]:
    
    >>> import pypropel.dist as ppdist
    >>> ppdist.classify_binding_distance(2.0)  # Contact < 3.5Å
    0
    >>> ppdist.classify_binding_distance(4.5)  # Near: 3.5Å ≤ d < 6.0Å
    1
    >>> ppdist.classify_binding_distance(10.0)  # Far ≥ 6.0Å
    2
    
    Custom 4 classes with thresholds [3.5, 6.0, 10.0]:
    
    >>> ppdist.classify_binding_distance(8.0, thresholds=[3.5, 6.0, 10.0])
    2
    """
    for i, thresh in enumerate(sorted(thresholds)):
        if distance < thresh:
            return i
    return len(thresholds)


def get_distance_labels(
    structure,
    ligand_coords: np.ndarray,
    thresholds: List[float] = [3.5, 6.0]
) -> np.ndarray:
    """
    Get per-residue distance classification labels.
    
    Computes minimum distance from each residue to the ligand and 
    classifies into bins based on thresholds.
    
    Parameters
    ----------
    structure : Bio.PDB.Structure.Structure
        BioPython structure object.
    ligand_coords : np.ndarray
        Ligand atom coordinates, shape (M, 3).
    thresholds : List[float]
        Distance thresholds for classification.
        Default [3.5, 6.0] gives 3 classes: Contact, Near, Far.
        
    Returns
    -------
    np.ndarray
        Classification labels, shape (N_residues,).
        Values range from 0 to len(thresholds).
        
    Examples
    --------
    >>> import pypropel.dist as ppdist
    >>> import pypropel.str as ppstr
    >>> import pypropel.mol as ppmol
    >>> 
    >>> structure = ppstr.load_pdb('/path/to/protein.pdb')
    >>> mol = ppmol.load_sdf('/path/to/ligand.sdf')
    >>> ligand_coords = ppmol.ligand_coords(mol)
    >>> 
    >>> labels = ppdist.get_distance_labels(structure, ligand_coords)
    >>> print(labels.shape)  # (N_residues,)
    >>> print(np.unique(labels, return_counts=True))
    """
    # Get distances for all residues
    distances_df = protein_ligand_distances(structure, ligand_coords)
    distances = distances_df['distance'].values
    
    # Classify each distance
    labels = np.array([
        classify_binding_distance(d, thresholds) 
        for d in distances
    ], dtype=np.int64)
    
    return labels


def get_distance_labels_with_info(
    structure,
    ligand_coords: np.ndarray,
    thresholds: List[float] = [3.5, 6.0]
) -> pd.DataFrame:
    """
    Get per-residue distance classification labels with residue info.
    
    Parameters
    ----------
    structure : Bio.PDB.Structure.Structure
        BioPython structure object.
    ligand_coords : np.ndarray
        Ligand atom coordinates, shape (M, 3).
    thresholds : List[float]
        Distance thresholds for classification.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: chain, res_id, res_name, distance, label.
        
    Examples
    --------
    >>> df = ppdist.get_distance_labels_with_info(structure, ligand_coords)
    >>> print(df.head())
    """
    # Get distances for all residues
    distances_df = protein_ligand_distances(structure, ligand_coords)
    
    # Add classification labels
    distances_df['label'] = distances_df['distance'].apply(
        lambda d: classify_binding_distance(d, thresholds)
    )
    
    return distances_df


def get_class_weights(
    labels: np.ndarray,
    normalize: bool = True
) -> np.ndarray:
    """
    Compute class weights for imbalanced classification (for Focal Loss).
    
    Parameters
    ----------
    labels : np.ndarray
        Classification labels.
    normalize : bool
        If True, normalize weights to sum to number of classes.
        
    Returns
    -------
    np.ndarray
        Weight for each class, inverse proportional to frequency.
        
    Examples
    --------
    >>> labels = np.array([0, 0, 1, 2, 2, 2, 2, 2])  # Imbalanced
    >>> weights = ppdist.get_class_weights(labels)
    >>> print(weights)  # Higher weight for rare class 0
    """
    unique, counts = np.unique(labels, return_counts=True)
    n_classes = int(unique.max()) + 1
    
    weights = np.ones(n_classes, dtype=np.float32)
    
    for cls, count in zip(unique, counts):
        weights[int(cls)] = 1.0 / count
    
    if normalize:
        weights = weights * n_classes / weights.sum()
    
    return weights


# ==================== Existing Functions ====================


def one_vs_one(
        pdb_path1 : str,
        pdb_name1 : str,
        file_chain1 : str,
        seq_chain1 : str,
        pdb_path2 : str,
        pdb_name2 : str,
        file_chain2 : str,
        seq_chain2 : str,
) -> Dict:
    return OneToOne(
        pdb_path1=pdb_path1,
        pdb_name1=pdb_name1,
        file_chain1=file_chain1,
        seq_chain1=seq_chain1,
        pdb_path2=pdb_path2,
        pdb_name2=pdb_name2,
        file_chain2=file_chain2,
        seq_chain2=seq_chain2,
    ).calculate()


def all_vs_all(
        pdb_fp,
        pdb_name,
):
    return AllAgainstAll(
        pdb_fp=pdb_fp,
        pdb_name=pdb_name,
    ).calculate()


def check_chain_complex(
        pdb_fp : str,
        prot_name : str,
        thres : float,
        sv_fp : str,
):
    return Complex(
        pdb_fp=pdb_fp,
        prot_name=prot_name,
        thres=thres,
        sv_fp=sv_fp,
    ).run()


def check_chain_paired(
        pdb_fp1 : str,
        pdb_fp2 : str,
        prot_name1 : str,
        prot_name2 : str,
        prot_chain1 : str,
        prot_chain2 : str,
        sv_fp : str,
        thres : float,
):
    return Pair(
        pdb_fp1=pdb_fp1,
        pdb_fp2=pdb_fp2,
        prot_name1=prot_name1,
        prot_name2=prot_name2,
        prot_chain1=prot_chain1,
        prot_chain2=prot_chain2,
        sv_fp=sv_fp,
        thres=thres,
    ).run()


def complex_calc_all(
        pdb_fp : str,
        prot_name : str,
        prot_chain : str,
        method : str,
        sv_fp : str,
):
    return DistanceComplexOne(
        pdb_fp=pdb_fp,
        prot_name=prot_name,
        prot_chain=prot_chain,
        method=method,
        sv_fp=sv_fp,
    ).dist_without_aa()


def complex_calc_inter(
        pdb_fp : str,
        prot_name : str,
        prot_chain : str,
        method : str,
        sv_fp : str,
):
    return DistanceComplexOne(
        pdb_fp=pdb_fp,
        prot_name=prot_name,
        prot_chain=prot_chain,
        method=method,
        sv_fp=sv_fp,
    ).dist_with_aa()


# @deprecated
# def cloud_check(
#         order_list,
#         job_fp,
#         job_fn,
#         cpu,
#         memory,
#         method,
#         submission_method,
# ):
#     return TransmitterComplex(
#         order_list=order_list,
#         job_fp=job_fp,
#         job_fn=job_fn,
#         cpu=cpu,
#         memory=memory,
#         method=method,
#         submission_method=submission_method,
#     ).execute()


def labelling(
        dist_fp,
        prot_name,
        file_chain,
        header=0,
        cutoff=6,
) -> pd.DataFrame:
    return Label(
        dist_fp=dist_fp,
        prot_name=prot_name,
        file_chain=file_chain,
        header=header,
        cutoff=cutoff,
    ).attach()


def interation_partners(
        dist_fp,
        prot_name,
        file_chain,
        pdb_fp,
        header=0,
        cutoff=6,
) -> List:
    return Label(
        dist_fp=dist_fp,
        prot_name=prot_name,
        file_chain=file_chain,
        header=header,
        cutoff=cutoff,
    ).partner(
        pdb_fp=pdb_fp,
    )


if __name__ == "__main__":
    from pypropel.path import to

    # dist_mat = one_vs_one(
    #     pdb_path1=to('data/pdb/complex/pdbtm/'),
    #     pdb_name1='1aij',
    #     file_chain1='',
    #     seq_chain1='L',
    #     pdb_path2=to('data/pdb/complex/pdbtm/'),
    #     pdb_name2='1aij',
    #     file_chain2='',
    #     seq_chain2='M',
    # )
    # df_dist = pd.DataFrame(dist_mat)
    # df_dist = df_dist.rename(columns={
    #     0: 'res_fas_id1',
    #     1: 'res1',
    #     2: 'res_pdb_id1',
    #     3: 'res_fas_id2',
    #     4: 'res2',
    #     5: 'res_pdb_id2',
    #     6: 'dist',
    # })
    # print(df_dist)

    # df_dist = all_vs_all(
    #     pdb_fp=to('data/pdb/complex/pdbtm/'),
    #     pdb_name='1aij',
    # )
    # print(df_dist)

    # print(check_chain_complex(
    #     pdb_fp=to('data/pdb/complex/pdbtm/'),
    #     prot_name='1aij',
    #     sv_fp=to('data/pdb/complex/pdbtm/'),
    #     thres=5.5,
    # ))

    # print(check_chain_paired(
    #     pdb_fp1=to('data/pdb/pdbtm/'),
    #     pdb_fp2=to('data/pdb/pdbtm/'),
    #     prot_name1='1aij',
    #     prot_name2='1aij',
    #     prot_chain1='L',
    #     prot_chain2='M',
    #     thres=6.,
    #     sv_fp=to('data/pdb/pdbtm/'),
    # ))

    # print(complex_calc_all(
    #     pdb_fp=to('data/pdb/complex/pdbtm/'),
    #     prot_name='1aij',
    #     prot_chain='L',
    #     method='heavy',
    #     sv_fp=to('data/pdb/complex/pdbtm/'),
    # ))

    # print(complex_calc_inter(
    #     pdb_fp=to('data/pdb/complex/pdbtm/'),
    #     prot_name='1aij',
    #     prot_chain='L',
    #     method='heavy',
    #     sv_fp=to('data/pdb/complex/pdbtm/'),
    # ))

    df_dist = labelling(
        dist_fp=to('data/pdb/complex/pdbtm/'),
        prot_name='1aij',
        file_chain='L',
        cutoff=6,
    )
    print(df_dist)

    print(interation_partners(
        dist_fp=to('data/pdb/complex/pdbtm/'),
        prot_name='1aij',
        file_chain='L',
        cutoff=6,
        pdb_fp=to('data/pdb/complex/pdbtm/'),
    ))
    ## ++++++++++++++++++++++++
    # from pypropel.util.Reader import Reader as pfreader
    #
    # df = pfreader().generic(df_fpn=to('data/ex/final.txt'), header=0)
    # prots = df.prot.unique()[2000:]
    #
    # param_config = {
    #     'pdb_fp': '-fp',
    #     'pdb_fn': '-fn',
    #     'sv_fp': '-op',
    # }
    # value_config = {
    #     'tool_fp': '/path/to/python',
    #     'script_fpn': './Complex.py',
    #     'pdb_fp': '/path/to/protein complex files/',
    #     'sv_fp': '/path/to/save/results/',
    # }
    #
    # for key, prot in enumerate(prots):
    #     order_list = [
    #         value_config['tool_fp'],
    #         value_config['script_fpn'],
    #
    #         param_config['pdb_fp'], value_config['pdb_fp'],
    #         param_config['pdb_fn'], prot,
    #         param_config['sv_fp'], value_config['sv_fp'],
    #     ]
    #     print(cloud_check(
    #         order_list=order_list,
    #         job_fp='/path/to/save/job files/',
    #         job_fn=str(key),
    #         cpu=2,
    #         memory=10,
    #         method='script',
    #         submission_method='sbatch',
    #     ))