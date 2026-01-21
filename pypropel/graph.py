"""
Graph construction utilities for pypropel.

Provides functions for building PyTorch Geometric-compatible graphs
from protein structures and ligand molecules.
"""

__version__ = "v1.0"
__copyright__ = "Copyright 2024"
__license__ = "GPL v3.0"

from typing import List, Tuple, Dict, Optional
import numpy as np


def build_protein_knn_graph(
    structure,
    k: int = 20,
    radius: float = 10.0,
    chain_id: str = None
) -> Dict[str, np.ndarray]:
    """
    Build k-NN graph for protein structure with optional radius cutoff.
    
    Parameters
    ----------
    structure : Bio.PDB.Structure.Structure
        BioPython structure object.
    k : int
        Number of nearest neighbors per node.
    radius : float
        Maximum edge distance in Angstroms.
    chain_id : str, optional
        Chain ID to extract. If None, uses all chains.
        
    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary containing:
        - 'node_coords': (N, 3) - Cα coordinates
        - 'edge_index': (2, E) - source/target indices
        - 'edge_attr': (E, 4) - [distance, dx, dy, dz] normalized
        
    Examples
    --------
    >>> import pypropel.graph as ppgraph
    >>> import pypropel.str as ppstr
    >>> structure = ppstr.load_pdb('/path/to/protein.pdb')
    >>> graph = ppgraph.build_protein_knn_graph(structure, k=20, radius=10.0)
    >>> print(graph['edge_index'].shape)  # (2, E)
    """
    # Import gvp module - try relative first, then direct
    try:
        from . import gvp as ppgvp
    except ImportError:
        import os
        import importlib.util
        _gvp_path = os.path.join(os.path.dirname(__file__), 'gvp.py')
        _spec = importlib.util.spec_from_file_location('gvp', _gvp_path)
        ppgvp = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(ppgvp)
    
    # Get Cα coordinates
    coords = ppgvp.get_ca_coords(structure, chain_id)
    n = len(coords)
    
    if n == 0:
        return {
            'node_coords': np.zeros((0, 3), dtype=np.float32),
            'edge_index': np.zeros((2, 0), dtype=np.int64),
            'edge_attr': np.zeros((0, 4), dtype=np.float32),
        }
    
    # Build k-NN edges with radius cutoff
    edge_index, edge_distances = ppgvp.build_knn_edges(coords, k, radius)
    
    # Compute edge attributes (distance + direction)
    if edge_index.shape[1] > 0:
        src, tgt = edge_index
        directions = coords[tgt] - coords[src]
        # Normalize directions
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        directions = directions / norms
        
        # Edge attributes: [distance, dx, dy, dz]
        edge_attr = np.concatenate([
            edge_distances.reshape(-1, 1),
            directions
        ], axis=1).astype(np.float32)
    else:
        edge_attr = np.zeros((0, 4), dtype=np.float32)
    
    return {
        'node_coords': coords,
        'edge_index': edge_index,
        'edge_attr': edge_attr,
    }


def build_ligand_graph(mol) -> Dict[str, np.ndarray]:
    """
    Build molecular graph from RDKit molecule.
    
    Creates a graph where nodes are atoms and edges are bonds.
    
    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        RDKit molecule object.
        
    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary containing:
        - 'node_features': (N, D) - atom features
        - 'node_coords': (N, 3) - 3D coordinates (if available)
        - 'edge_index': (2, E) - bond indices (bidirectional)
        - 'edge_attr': (E, D_bond) - bond features
        
    Examples
    --------
    >>> import pypropel.graph as ppgraph
    >>> import pypropel.mol as ppmol
    >>> mol = ppmol.load_sdf('/path/to/ligand.sdf')
    >>> graph = ppgraph.build_ligand_graph(mol)
    >>> print(graph['node_features'].shape)  # (N_atoms, 21)
    """
    # Import mol module - try relative first, then direct
    try:
        from . import mol as ppmol
    except ImportError:
        import os
        import importlib.util
        _mol_path = os.path.join(os.path.dirname(__file__), 'mol.py')
        _spec = importlib.util.spec_from_file_location('mol', _mol_path)
        ppmol = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(ppmol)
    
    if mol is None:
        return {
            'node_features': np.zeros((0, 21), dtype=np.float32),
            'node_coords': np.zeros((0, 3), dtype=np.float32),
            'edge_index': np.zeros((2, 0), dtype=np.int64),
            'edge_attr': np.zeros((0, 4), dtype=np.float32),
        }
    
    # Get node features
    node_features = ppmol.get_atom_features(mol)
    
    # Get 3D coordinates
    node_coords = ppmol.ligand_coords(mol)
    
    # Build edge index from bonds (make bidirectional)
    sources = []
    targets = []
    
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        # Add both directions
        sources.extend([i, j])
        targets.extend([j, i])
    
    if len(sources) > 0:
        edge_index = np.array([sources, targets], dtype=np.int64)
        edge_attr = get_bond_features(mol)
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_attr = np.zeros((0, 4), dtype=np.float32)
    
    return {
        'node_features': node_features,
        'node_coords': node_coords,
        'edge_index': edge_index,
        'edge_attr': edge_attr,
    }


def get_bond_features(mol) -> np.ndarray:
    """
    Extract bond features for all bonds in a molecule.
    
    Features:
    - Bond type one-hot (SINGLE, DOUBLE, TRIPLE, AROMATIC)
    
    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        RDKit molecule object.
        
    Returns
    -------
    np.ndarray
        Bond features, shape (E, 4) for bidirectional edges.
        
    Examples
    --------
    >>> bond_features = ppgraph.get_bond_features(mol)
    >>> print(bond_features.shape)  # (2 * N_bonds, 4)
    """
    try:
        from rdkit import Chem
    except ImportError:
        raise ImportError("RDKit required. Install with: pip install rdkit")
    
    if mol is None:
        return np.zeros((0, 4), dtype=np.float32)
    
    bond_type_map = {
        Chem.rdchem.BondType.SINGLE: 0,
        Chem.rdchem.BondType.DOUBLE: 1,
        Chem.rdchem.BondType.TRIPLE: 2,
        Chem.rdchem.BondType.AROMATIC: 3,
    }
    
    features = []
    for bond in mol.GetBonds():
        bt = bond.GetBondType()
        feat = np.zeros(4, dtype=np.float32)
        if bt in bond_type_map:
            feat[bond_type_map[bt]] = 1.0
        else:
            feat[0] = 1.0  # Default to single
        
        # Add twice for bidirectional edges
        features.append(feat)
        features.append(feat)
    
    if len(features) > 0:
        return np.array(features, dtype=np.float32)
    return np.zeros((0, 4), dtype=np.float32)


def build_protein_ligand_bipartite_edges(
    protein_coords: np.ndarray,
    ligand_coords: np.ndarray,
    threshold: float = 8.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build bipartite edges between protein residues and ligand atoms.
    
    Creates edges from protein nodes to nearby ligand atoms.
    Useful for cross-attention or message passing between modalities.
    
    Parameters
    ----------
    protein_coords : np.ndarray
        Protein Cα coordinates, shape (N_res, 3).
    ligand_coords : np.ndarray
        Ligand atom coordinates, shape (N_atoms, 3).
    threshold : float
        Maximum distance for creating an edge.
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        - edge_index: (2, E) - [protein_idx, ligand_idx]
        - edge_distances: (E,) - distances
        
    Examples
    --------
    >>> edges, dists = ppgraph.build_protein_ligand_bipartite_edges(
    ...     protein_coords, ligand_coords, threshold=8.0
    ... )
    """
    if len(protein_coords) == 0 or len(ligand_coords) == 0:
        return np.zeros((2, 0), dtype=np.int64), np.zeros(0, dtype=np.float32)
    
    # Compute pairwise distances
    diff = protein_coords[:, None, :] - ligand_coords[None, :, :]  # (N_res, N_atoms, 3)
    dist_matrix = np.linalg.norm(diff, axis=2)  # (N_res, N_atoms)
    
    # Find edges within threshold
    prot_idx, lig_idx = np.where(dist_matrix < threshold)
    distances = dist_matrix[prot_idx, lig_idx]
    
    edge_index = np.array([prot_idx, lig_idx], dtype=np.int64)
    
    return edge_index, distances.astype(np.float32)


if __name__ == "__main__":
    print("graph.py module loaded successfully")
