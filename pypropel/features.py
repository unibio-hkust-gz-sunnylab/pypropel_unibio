"""
Unified feature extraction utilities for pypropel.

This module provides general-purpose, configurable feature extraction functions
that combine multiple feature sources into standardized outputs. Functions are
designed to be reusable across different model architectures.

Feature Functions:
    - get_protein_features(): Configurable protein feature extraction
    - get_ligand_features(): Configurable ligand feature extraction  
    - get_binding_labels(): Distance-based binding site classification
"""

__version__ = "v1.0"
__copyright__ = "Copyright 2024"
__license__ = "GPL v3.0"

from typing import Dict, Optional, List, Tuple, Union
import numpy as np


# ==================== Feature Class Registry ====================

# Protein feature classes and their dimensions
PROTEIN_FEATURE_CLASSES = {
    'onehot': 20,        # One-hot AA encoding
    'ss': 3,             # Secondary structure (H/E/C)
    'sasa': 1,           # Solvent accessible surface area
    'charge': 1,         # Residue charge
    'hydrophobicity': 1, # Kyte-Doolittle hydrophobicity
    'aromatic': 1,       # Is aromatic residue
    'hbond_donor': 1,    # H-bond donor count
    'hbond_acceptor': 1, # H-bond acceptor count
}

# Ligand atom feature classes and their dimensions
LIGAND_FEATURE_CLASSES = {
    'atom_type': 10,      # One-hot atom type
    'hybridization': 4,   # Hybridization (SP, SP2, SP3, Aromatic)
    'aromaticity': 1,     # Is aromatic atom
    'hbond_donor': 1,     # Is H-bond donor
    'hbond_acceptor': 1,  # Is H-bond acceptor
    'partial_charge': 1,  # Gasteiger partial charge
    'ring_size': 1,       # Ring size (0 if not in ring)
    'global_tag': 45,     # Global molecular fingerprint
}


# ==================== Protein Features ====================

def get_protein_features(
    structure,
    feature_classes: List[str] = None,
    chain_id: str = None,
    use_dssp: bool = True,
    esm_embeddings: np.ndarray = None,
    include_gvp: bool = False,
    gvp_k_neighbors: int = 10
) -> Dict[str, np.ndarray]:
    """
    Extract configurable protein features from a BioPython structure.
    
    This is a general-purpose function that allows selecting which feature
    classes to extract and combine.
    
    Parameters
    ----------
    structure : Bio.PDB.Structure.Structure
        BioPython structure object.
    feature_classes : List[str], optional
        List of feature classes to include. If None, uses all available.
        Options: 'onehot', 'ss', 'sasa', 'charge', 'hydrophobicity', 
                 'aromatic', 'hbond_donor', 'hbond_acceptor'
    chain_id : str, optional
        Chain ID to extract. If None, extracts all chains.
    use_dssp : bool
        If True, use DSSP for SASA and SS. If False, use defaults.
    esm_embeddings : np.ndarray, optional
        Pre-computed ESM embeddings, shape (N, D_esm).
    include_gvp : bool
        If True, include GVP geometric features.
    gvp_k_neighbors : int
        Number of neighbors for GVP neighbor center vectors.
        
    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary containing:
        - 'scalar_features': Combined scalar features (N, D)
        - 'feature_dims': Dict mapping feature class to (start, end) indices
        - 'residue_names': List of residue names
        - 'n_residues': Number of residues
        And optionally:
        - 'esm': ESM embeddings if provided
        - 'gvp_coords': Cα coordinates if include_gvp
        - 'gvp_vectors': GVP vector features if include_gvp
        
    Examples
    --------
    >>> import pypropel.features as ppfeat
    >>> features = ppfeat.get_protein_features(structure)
    >>> print(features['scalar_features'].shape)
    
    >>> # Select specific features
    >>> features = ppfeat.get_protein_features(
    ...     structure, 
    ...     feature_classes=['onehot', 'charge', 'hydrophobicity']
    ... )
    """
    from Bio.PDB import Polypeptide
    import pypropel.fpsite as fpsite
    
    # Default to all feature classes
    if feature_classes is None:
        feature_classes = list(PROTEIN_FEATURE_CLASSES.keys())
    
    # Validate feature classes
    for fc in feature_classes:
        if fc not in PROTEIN_FEATURE_CLASSES:
            raise ValueError(f"Unknown feature class: {fc}. "
                           f"Available: {list(PROTEIN_FEATURE_CLASSES.keys())}")
    
    # Collect residue info
    residues = []
    res_names = []
    for model in structure:
        for chain in model:
            if chain_id is not None and chain.get_id() != chain_id:
                continue
            for residue in chain:
                if Polypeptide.is_aa(residue, standard=True):
                    residues.append(residue)
                    res_names.append(residue.get_resname())
    
    n = len(residues)
    if n == 0:
        return {
            'scalar_features': np.zeros((0, sum(PROTEIN_FEATURE_CLASSES[fc] for fc in feature_classes)), dtype=np.float32),
            'feature_dims': {},
            'residue_names': [],
            'n_residues': 0
        }
    
    # Calculate total dimension and feature mapping
    feature_dims = {}
    current_idx = 0
    for fc in feature_classes:
        dim = PROTEIN_FEATURE_CLASSES[fc]
        feature_dims[fc] = (current_idx, current_idx + dim)
        current_idx += dim
    
    total_dim = current_idx
    features = np.zeros((n, total_dim), dtype=np.float32)
    
    # Extract DSSP features once if needed
    dssp_data = None
    if use_dssp and ('ss' in feature_classes or 'sasa' in feature_classes):
        try:
            dssp_data = fpsite.get_residue_sasa(structure, chain_id)
        except Exception:
            dssp_data = None
    
    # Fill in each feature class
    for fc in feature_classes:
        start, end = feature_dims[fc]
        
        if fc == 'onehot':
            for i, res_name in enumerate(res_names):
                features[i, start:end] = fpsite.residue_one_hot(res_name)
                
        elif fc == 'ss':
            if dssp_data is not None:
                for i, d in enumerate(dssp_data):
                    if i < n:
                        features[i, start:end] = fpsite.residue_secondary_structure(d['ss'])
            # Else: leave as zeros (coil default)
            
        elif fc == 'sasa':
            if dssp_data is not None:
                for i, d in enumerate(dssp_data):
                    if i < n:
                        features[i, start] = d['sasa']
            # Else: leave as zeros
            
        elif fc == 'charge':
            for i, res_name in enumerate(res_names):
                physchem = fpsite.residue_physchem(res_name)
                features[i, start] = physchem[0]
                
        elif fc == 'hydrophobicity':
            for i, res_name in enumerate(res_names):
                physchem = fpsite.residue_physchem(res_name)
                # Normalize from [-1, 1] to [0, 1]
                features[i, start] = (physchem[1] + 1.0) / 2.0
                
        elif fc == 'aromatic':
            aromatic_aas = {'PHE', 'TYR', 'TRP', 'HIS'}
            for i, res_name in enumerate(res_names):
                features[i, start] = 1.0 if res_name in aromatic_aas else 0.0
                
        elif fc == 'hbond_donor':
            donor_counts = {'ARG': 3, 'ASN': 1, 'GLN': 1, 'HIS': 1, 'LYS': 1,
                           'SER': 1, 'THR': 1, 'TRP': 1, 'TYR': 1, 'CYS': 1}
            for i, res_name in enumerate(res_names):
                features[i, start] = donor_counts.get(res_name, 0)
                
        elif fc == 'hbond_acceptor':
            acceptor_counts = {'ASN': 1, 'ASP': 2, 'GLN': 1, 'GLU': 2, 'HIS': 1,
                              'SER': 1, 'THR': 1, 'TYR': 1}
            for i, res_name in enumerate(res_names):
                features[i, start] = acceptor_counts.get(res_name, 0)
    
    result = {
        'scalar_features': features.astype(np.float32),
        'feature_dims': feature_dims,
        'residue_names': res_names,
        'n_residues': n
    }
    
    # Add ESM embeddings if provided
    if esm_embeddings is not None:
        result['esm'] = esm_embeddings
    
    # Add GVP features if requested
    if include_gvp:
        import pypropel.gvp as ppgvp
        gvp_coords, gvp_vectors = ppgvp.get_gvp_node_features(
            structure, k_neighbors=gvp_k_neighbors, chain_id=chain_id
        )
        result['gvp_coords'] = gvp_coords
        result['gvp_vectors'] = gvp_vectors
    
    return result


# ==================== Ligand Features ====================

def get_ligand_features(
    mol,
    feature_classes: List[str] = None,
    include_global_tag: bool = True,
    global_tag_dim: int = 45
) -> Dict[str, np.ndarray]:
    """
    Extract configurable ligand features from an RDKit molecule.
    
    This is a general-purpose function that allows selecting which feature
    classes to extract and combine.
    
    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        RDKit molecule object.
    feature_classes : List[str], optional
        List of feature classes to include. If None, uses all available.
        Options: 'atom_type', 'hybridization', 'aromaticity', 
                 'hbond_donor', 'hbond_acceptor', 'partial_charge', 
                 'ring_size', 'global_tag'
    include_global_tag : bool
        If True, append global molecular tag to each atom.
    global_tag_dim : int
        Dimension of global tag (default 45).
        
    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary containing:
        - 'atom_features': Combined atom features (M, D)
        - 'coords': Atom coordinates (M, 3)
        - 'feature_dims': Dict mapping feature class to (start, end) indices
        - 'n_atoms': Number of atoms
        
    Examples
    --------
    >>> import pypropel.features as ppfeat
    >>> features = ppfeat.get_ligand_features(mol)
    >>> print(features['atom_features'].shape)  # (N_atoms, 64)
    """
    import pypropel.mol as ppmol
    
    # Default feature classes
    if feature_classes is None:
        feature_classes = ['atom_type', 'hybridization', 'aromaticity',
                          'hbond_donor', 'hbond_acceptor', 'partial_charge',
                          'ring_size']
        if include_global_tag:
            feature_classes.append('global_tag')
    
    if mol is None:
        total_dim = sum(LIGAND_FEATURE_CLASSES.get(fc, 0) for fc in feature_classes)
        return {
            'atom_features': np.zeros((0, total_dim), dtype=np.float32),
            'coords': np.zeros((0, 3), dtype=np.float32),
            'feature_dims': {},
            'n_atoms': 0
        }
    
    n_atoms = mol.GetNumAtoms()
    
    # Calculate total dimension and feature mapping
    feature_dims = {}
    current_idx = 0
    for fc in feature_classes:
        if fc == 'global_tag':
            dim = global_tag_dim
        else:
            dim = LIGAND_FEATURE_CLASSES.get(fc, 0)
        feature_dims[fc] = (current_idx, current_idx + dim)
        current_idx += dim
    
    total_dim = current_idx
    features = np.zeros((n_atoms, total_dim), dtype=np.float32)
    
    # Fill in each feature class
    for fc in feature_classes:
        start, end = feature_dims[fc]
        
        if fc == 'atom_type':
            features[:, start:end] = ppmol.get_atom_type_onehot(mol)
            
        elif fc == 'hybridization':
            hybrid_full = ppmol.get_atom_hybridization(mol)  # (N, 7)
            # Compress to 4 dims: SP, SP2, SP3, Aromatic
            features[:, start] = hybrid_full[:, 0]      # SP
            features[:, start+1] = hybrid_full[:, 1]    # SP2
            features[:, start+2] = hybrid_full[:, 2]    # SP3
            features[:, start+3] = hybrid_full[:, 5]    # Aromatic
            
        elif fc == 'aromaticity':
            features[:, start:end] = ppmol.get_atom_aromaticity(mol)
            
        elif fc == 'hbond_donor':
            hbond = ppmol.get_atom_hbond_features(mol)
            features[:, start] = hbond[:, 0]
            
        elif fc == 'hbond_acceptor':
            hbond = ppmol.get_atom_hbond_features(mol)
            features[:, start] = hbond[:, 1]
            
        elif fc == 'partial_charge':
            features[:, start:end] = ppmol.get_atom_partial_charges(mol)
            
        elif fc == 'ring_size':
            features[:, start:end] = ppmol.get_atom_ring_sizes(mol)
            
        elif fc == 'global_tag':
            global_tag = ppmol.get_global_tag(mol, output_dim=global_tag_dim)
            features[:, start:end] = global_tag[None, :]  # Broadcast to all atoms
    
    return {
        'atom_features': features.astype(np.float32),
        'coords': ppmol.ligand_coords(mol),
        'feature_dims': feature_dims,
        'n_atoms': n_atoms
    }


# ==================== Binding Labels ====================

def get_binding_labels(
    structure,
    ligand_coords: np.ndarray,
    thresholds: List[float] = None,
    chain_id: str = None,
    return_distances: bool = True
) -> Dict[str, np.ndarray]:
    """
    Generate binding site classification labels based on distance thresholds.
    
    This function classifies each residue based on its minimum distance to
    the ligand, using configurable distance thresholds.
    
    Parameters
    ----------
    structure : Bio.PDB.Structure.Structure
        BioPython structure object.
    ligand_coords : np.ndarray
        Ligand atom coordinates, shape (M, 3).
    thresholds : List[float], optional
        Distance thresholds for classification. Default [3.5, 6.0] gives:
        - Class 0: Contact (< 3.5 Å)
        - Class 1: Near (3.5-6.0 Å)
        - Class 2: Far (> 6.0 Å)
    chain_id : str, optional
        Chain to extract.
    return_distances : bool
        If True, include raw distances in output.
        
    Returns
    -------
    Dict[str, np.ndarray]
        - 'labels': Classification labels (N,)
        - 'n_classes': Number of classes
        - 'thresholds': Thresholds used
        And optionally:
        - 'distances': Raw min distances (N,)
        - 'class_weights': Inverse frequency weights (n_classes,)
        
    Examples
    --------
    >>> labels = ppfeat.get_binding_labels(structure, ligand_coords)
    >>> print(labels['labels'].shape)  # (N_residues,)
    >>> print(np.unique(labels['labels'], return_counts=True))
    """
    import pypropel.dist as ppdist
    
    if thresholds is None:
        thresholds = [3.5, 6.0]
    
    # Get distance-based labels
    labels = ppdist.get_distance_labels(
        structure, ligand_coords, thresholds=thresholds
    )
    
    n_classes = len(thresholds) + 1
    
    result = {
        'labels': labels,
        'n_classes': n_classes,
        'thresholds': thresholds
    }
    
    if return_distances:
        distances_df = ppdist.protein_ligand_distances(structure, ligand_coords)
        result['distances'] = distances_df['distance'].values.astype(np.float32)
        result['class_weights'] = ppdist.get_class_weights(labels)
    
    return result


# ==================== Convenience Functions ====================

def compute_feature_dim(
    feature_classes: List[str],
    feature_type: str = 'protein'
) -> int:
    """
    Compute the total feature dimension for a given set of feature classes.
    
    Parameters
    ----------
    feature_classes : List[str]
        List of feature class names.
    feature_type : str
        'protein' or 'ligand'
        
    Returns
    -------
    int
        Total feature dimension.
    """
    registry = PROTEIN_FEATURE_CLASSES if feature_type == 'protein' else LIGAND_FEATURE_CLASSES
    return sum(registry.get(fc, 0) for fc in feature_classes)


def list_feature_classes(feature_type: str = 'protein') -> Dict[str, int]:
    """
    List available feature classes and their dimensions.
    
    Parameters
    ----------
    feature_type : str
        'protein' or 'ligand'
        
    Returns
    -------
    Dict[str, int]
        Dictionary mapping feature class name to dimension.
    """
    if feature_type == 'protein':
        return PROTEIN_FEATURE_CLASSES.copy()
    else:
        return LIGAND_FEATURE_CLASSES.copy()


if __name__ == "__main__":
    print("features.py module loaded successfully")
    print(f"Protein feature classes: {list(PROTEIN_FEATURE_CLASSES.keys())}")
    print(f"Ligand feature classes: {list(LIGAND_FEATURE_CLASSES.keys())}")
