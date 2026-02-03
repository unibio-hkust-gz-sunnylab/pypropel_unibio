"""
GVP (Geometric Vector Perceptron) feature extraction for pypropel.

Provides functions for extracting geometric vector features from protein structures
for use with GVP-based neural networks.

Features:
- Cα coordinates extraction
- Cβ coordinates extraction  
- Sidechain orientation vectors (Cα→Cβ)
- Backbone flow vectors (Cα[i]→Cα[i+1])
- Neighbor center vectors (Cα→center of k-NN)
"""

__version__ = "v1.0"
__copyright__ = "Copyright 2024"
__license__ = "GPL v3.0"

from typing import List, Tuple, Dict, Optional
import numpy as np


def get_ca_coords(structure, chain_id: str = None) -> np.ndarray:
    """
    Extract Cα (alpha carbon) coordinates for all residues.
    
    Parameters
    ----------
    structure : Bio.PDB.Structure.Structure
        BioPython structure object.
    chain_id : str, optional
        Chain ID to extract. If None, extracts all chains.
        
    Returns
    -------
    np.ndarray
        Cα coordinates with shape (N_residues, 3).
        
    Examples
    --------
    >>> import pypropel.gvp as ppgvp
    >>> import pypropel.str as ppstr
    >>> structure = ppstr.load_pdb('/path/to/protein.pdb')
    >>> ca_coords = ppgvp.get_ca_coords(structure)
    >>> print(ca_coords.shape)  # (N_residues, 3)
    """
    from Bio.PDB import Polypeptide
    
    coords = []
    for model in structure:
        for chain in model:
            if chain_id is not None and chain.get_id() != chain_id:
                continue
            for residue in chain:
                if not Polypeptide.is_aa(residue, standard=True):
                    continue
                if 'CA' in residue:
                    coords.append(residue['CA'].get_coord())
    
    return np.array(coords, dtype=np.float32) if coords else np.zeros((0, 3), dtype=np.float32)


def get_cb_coords(structure, chain_id: str = None) -> np.ndarray:
    """
    Extract Cβ (beta carbon) coordinates for all residues.
    
    For glycine (no Cβ), uses a virtual Cβ position computed from backbone.
    
    Parameters
    ----------
    structure : Bio.PDB.Structure.Structure
        BioPython structure object.
    chain_id : str, optional
        Chain ID to extract. If None, extracts all chains.
        
    Returns
    -------
    np.ndarray
        Cβ coordinates with shape (N_residues, 3).
        
    Examples
    --------
    >>> cb_coords = ppgvp.get_cb_coords(structure)
    >>> print(cb_coords.shape)  # (N_residues, 3)
    """
    from Bio.PDB import Polypeptide
    
    coords = []
    for model in structure:
        for chain in model:
            if chain_id is not None and chain.get_id() != chain_id:
                continue
            for residue in chain:
                if not Polypeptide.is_aa(residue, standard=True):
                    continue
                    
                if 'CB' in residue:
                    coords.append(residue['CB'].get_coord())
                elif 'CA' in residue:
                    # For GLY or missing CB, compute virtual CB from backbone
                    cb_virtual = _compute_virtual_cb(residue)
                    coords.append(cb_virtual)
    
    return np.array(coords, dtype=np.float32) if coords else np.zeros((0, 3), dtype=np.float32)


def _compute_virtual_cb(residue) -> np.ndarray:
    """
    Compute virtual Cβ position for glycine or residues with missing Cβ.
    
    Uses backbone geometry: N, CA, C atoms to place Cβ in tetrahedral geometry.
    If atoms are missing, returns CA position as fallback.
    """
    try:
        n = residue['N'].get_coord()
        ca = residue['CA'].get_coord()
        c = residue['C'].get_coord()
        
        # Vectors from CA
        n_ca = n - ca
        c_ca = c - ca
        
        # Normalize
        n_ca = n_ca / (np.linalg.norm(n_ca) + 1e-8)
        c_ca = c_ca / (np.linalg.norm(c_ca) + 1e-8)
        
        # CB direction is roughly opposite to the N-C bisector
        cb_direction = -(n_ca + c_ca)
        cb_direction = cb_direction / (np.linalg.norm(cb_direction) + 1e-8)
        
        # CB is approximately 1.52 Å from CA
        cb_virtual = ca + 1.52 * cb_direction
        return cb_virtual
        
    except KeyError:
        # Missing backbone atoms, return CA position
        if 'CA' in residue:
            return residue['CA'].get_coord()
        return np.zeros(3, dtype=np.float32)


def get_residue_orientations(structure, chain_id: str = None) -> np.ndarray:
    """
    Extract sidechain orientation vectors (Cα→Cβ unit vectors).
    
    These vectors point from alpha carbon toward the sidechain,
    representing the local sidechain orientation.
    
    Parameters
    ----------
    structure : Bio.PDB.Structure.Structure
        BioPython structure object.
    chain_id : str, optional
        Chain ID to extract. If None, extracts all chains.
        
    Returns
    -------
    np.ndarray
        Unit vectors with shape (N_residues, 3).
        
    Examples
    --------
    >>> orientations = ppgvp.get_residue_orientations(structure)
    >>> print(orientations.shape)  # (N_residues, 3)
    """
    ca_coords = get_ca_coords(structure, chain_id)
    cb_coords = get_cb_coords(structure, chain_id)
    
    if len(ca_coords) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    
    # Compute Cα→Cβ vectors
    vectors = cb_coords - ca_coords
    
    # Normalize to unit vectors
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)  # Avoid division by zero
    
    return (vectors / norms).astype(np.float32)


def get_backbone_vectors(structure, chain_id: str = None) -> np.ndarray:
    """
    Extract backbone flow vectors (Cα[i]→Cα[i+1] unit vectors).
    
    These vectors represent the direction of the backbone chain.
    The last residue uses the vector from the previous residue.
    
    Parameters
    ----------
    structure : Bio.PDB.Structure.Structure
        BioPython structure object.
    chain_id : str, optional
        Chain ID to extract. If None, extracts all chains.
        
    Returns
    -------
    np.ndarray
        Unit vectors with shape (N_residues, 3).
        
    Examples
    --------
    >>> backbone = ppgvp.get_backbone_vectors(structure)
    >>> print(backbone.shape)  # (N_residues, 3)
    """
    ca_coords = get_ca_coords(structure, chain_id)
    n = len(ca_coords)
    
    if n == 0:
        return np.zeros((0, 3), dtype=np.float32)
    
    if n == 1:
        return np.zeros((1, 3), dtype=np.float32)
    
    # Compute Cα[i]→Cα[i+1] vectors
    vectors = np.zeros((n, 3), dtype=np.float32)
    vectors[:-1] = ca_coords[1:] - ca_coords[:-1]
    vectors[-1] = vectors[-2]  # Last residue uses previous vector
    
    # Normalize to unit vectors
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    
    return (vectors / norms).astype(np.float32)


def get_backward_vectors(structure, chain_id: str = None) -> np.ndarray:
    """
    Extract backward backbone vectors (Cα[i]→Cα[i-1] unit vectors).
    
    These vectors point backward along the backbone chain.
    The first residue uses the vector from the next residue.
    
    Parameters
    ----------
    structure : Bio.PDB.Structure.Structure
        BioPython structure object.
    chain_id : str, optional
        Chain ID to extract. If None, extracts all chains.
        
    Returns
    -------
    np.ndarray
        Unit vectors with shape (N_residues, 3).
        
    Examples
    --------
    >>> backward = ppgvp.get_backward_vectors(structure)
    >>> print(backward.shape)  # (N_residues, 3)
    """
    ca_coords = get_ca_coords(structure, chain_id)
    n = len(ca_coords)
    
    if n == 0:
        return np.zeros((0, 3), dtype=np.float32)
    
    if n == 1:
        return np.zeros((1, 3), dtype=np.float32)
    
    # Compute Cα[i]→Cα[i-1] vectors (points backward)
    vectors = np.zeros((n, 3), dtype=np.float32)
    vectors[1:] = ca_coords[:-1] - ca_coords[1:]
    vectors[0] = vectors[1]  # First residue uses next vector
    
    # Normalize to unit vectors
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    
    return (vectors / norms).astype(np.float32)


def get_neighbor_center_vectors(
    structure, 
    k: int = 10, 
    chain_id: str = None
) -> np.ndarray:
    """
    Extract vectors from each Cα to the center of its k-nearest neighbors.
    
    These vectors point toward the local neighborhood center,
    capturing the local geometric environment.
    
    Parameters
    ----------
    structure : Bio.PDB.Structure.Structure
        BioPython structure object.
    k : int
        Number of nearest neighbors to consider.
    chain_id : str, optional
        Chain ID to extract. If None, extracts all chains.
        
    Returns
    -------
    np.ndarray
        Unit vectors with shape (N_residues, 3).
        
    Examples
    --------
    >>> neighbor_vecs = ppgvp.get_neighbor_center_vectors(structure, k=10)
    >>> print(neighbor_vecs.shape)  # (N_residues, 3)
    """
    ca_coords = get_ca_coords(structure, chain_id)
    n = len(ca_coords)
    
    if n == 0:
        return np.zeros((0, 3), dtype=np.float32)
    
    if n == 1:
        return np.zeros((1, 3), dtype=np.float32)
    
    vectors = np.zeros((n, 3), dtype=np.float32)
    
    for i in range(n):
        # Compute distances to all other residues
        dists = np.linalg.norm(ca_coords - ca_coords[i], axis=1)
        
        # Get k nearest neighbors (excluding self)
        k_actual = min(k, n - 1)
        neighbor_indices = np.argsort(dists)[1:k_actual + 1]
        
        if len(neighbor_indices) > 0:
            # Compute center of neighbors
            center = ca_coords[neighbor_indices].mean(axis=0)
            vec = center - ca_coords[i]
            
            # Normalize
            norm = np.linalg.norm(vec)
            if norm > 1e-8:
                vectors[i] = vec / norm
    
    return vectors.astype(np.float32)


# ==================== Multi-Chain Support Functions ====================

def get_chain_ids(structure, chain_id: str = None) -> np.ndarray:
    """
    Extract chain ID for each residue as integer indices.
    
    Maps chain IDs (e.g., 'A', 'B', 'C') to sequential integers (0, 1, 2).
    
    Parameters
    ----------
    structure : Bio.PDB.Structure.Structure
        BioPython structure object.
    chain_id : str, optional
        Chain ID to extract. If None, extracts all chains.
        
    Returns
    -------
    np.ndarray
        Chain indices with shape (N_residues,), dtype int64.
        
    Examples
    --------
    >>> chain_ids = ppgvp.get_chain_ids(structure)
    >>> print(chain_ids)  # e.g., [0, 0, 0, 1, 1, 1, 2, 2]
    """
    from Bio.PDB import Polypeptide
    
    chain_indices = []
    chain_map = {}  # Maps chain ID strings to integers
    
    for model in structure:
        for chain in model:
            if chain_id is not None and chain.get_id() != chain_id:
                continue
            cid = chain.get_id()
            if cid not in chain_map:
                chain_map[cid] = len(chain_map)
            
            for residue in chain:
                if Polypeptide.is_aa(residue, standard=True):
                    if 'CA' in residue:  # Only count residues with CA
                        chain_indices.append(chain_map[cid])
    
    return np.array(chain_indices, dtype=np.int64) if chain_indices else np.zeros(0, dtype=np.int64)


def get_global_centered_coords(structure, chain_id: str = None) -> np.ndarray:
    """
    Extract Cα coordinates centered at the protein's mean position.
    
    Subtracts the global centroid from all Cα coordinates, providing
    translation-invariant coordinates with the protein center as origin.
    
    Parameters
    ----------
    structure : Bio.PDB.Structure.Structure
        BioPython structure object.
    chain_id : str, optional
        Chain ID to extract. If None, extracts all chains.
        
    Returns
    -------
    np.ndarray
        Centered Cα coordinates with shape (N_residues, 3).
        
    Examples
    --------
    >>> centered = ppgvp.get_global_centered_coords(structure)
    >>> print(centered.mean(axis=0))  # Should be ~[0, 0, 0]
    """
    ca_coords = get_ca_coords(structure, chain_id)
    
    if len(ca_coords) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    
    # Compute global centroid
    centroid = ca_coords.mean(axis=0)
    
    # Center coordinates
    centered = ca_coords - centroid
    
    return centered.astype(np.float32)


def get_chain_centered_coords(structure, chain_id: str = None) -> np.ndarray:
    """
    Extract Cα coordinates centered at each chain's mean position.
    
    Each residue is centered relative to its own chain's centroid.
    For single-chain proteins, this is identical to global centering.
    
    Parameters
    ----------
    structure : Bio.PDB.Structure.Structure
        BioPython structure object.
    chain_id : str, optional
        Chain ID to extract. If None, extracts all chains.
        
    Returns
    -------
    np.ndarray
        Chain-centered Cα coordinates with shape (N_residues, 3).
        
    Examples
    --------
    >>> chain_centered = ppgvp.get_chain_centered_coords(structure)
    """
    ca_coords = get_ca_coords(structure, chain_id)
    chain_ids = get_chain_ids(structure, chain_id)
    
    if len(ca_coords) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    
    centered = np.zeros_like(ca_coords)
    
    # Center each chain independently
    unique_chains = np.unique(chain_ids)
    for cid in unique_chains:
        mask = chain_ids == cid
        chain_coords = ca_coords[mask]
        chain_centroid = chain_coords.mean(axis=0)
        centered[mask] = chain_coords - chain_centroid
    
    return centered.astype(np.float32)


def get_knn_direction_vectors(
    structure, 
    k: int = 10, 
    chain_id: str = None
) -> np.ndarray:
    """
    Extract direction vectors from each Cα to its k-nearest neighbors.
    
    Returns unit vectors pointing from each residue toward each of its
    k nearest neighbors, providing rich local geometric context.
    
    Parameters
    ----------
    structure : Bio.PDB.Structure.Structure
        BioPython structure object.
    k : int
        Number of nearest neighbors.
    chain_id : str, optional
        Chain ID to extract. If None, extracts all chains.
        
    Returns
    -------
    np.ndarray
        Direction vectors with shape (N_residues, k, 3).
        If fewer than k neighbors exist, remaining vectors are zero.
        
    Examples
    --------
    >>> knn_dirs = ppgvp.get_knn_direction_vectors(structure, k=10)
    >>> print(knn_dirs.shape)  # (N_residues, 10, 3)
    """
    ca_coords = get_ca_coords(structure, chain_id)
    n = len(ca_coords)
    
    if n == 0:
        return np.zeros((0, k, 3), dtype=np.float32)
    
    # Initialize output: (N, k, 3)
    directions = np.zeros((n, k, 3), dtype=np.float32)
    
    # Compute pairwise distance matrix
    diff = ca_coords[:, None, :] - ca_coords[None, :, :]  # (N, N, 3)
    dist_matrix = np.linalg.norm(diff, axis=2)  # (N, N)
    
    for i in range(n):
        dists = dist_matrix[i]
        
        # Get k nearest neighbors (excluding self)
        k_actual = min(k, n - 1)
        sorted_indices = np.argsort(dists)[1:k_actual + 1]  # Skip index 0 (self)
        
        for j_idx, neighbor_idx in enumerate(sorted_indices):
            # Vector from i to neighbor
            vec = ca_coords[neighbor_idx] - ca_coords[i]
            norm = np.linalg.norm(vec)
            if norm > 1e-8:
                directions[i, j_idx] = vec / norm
    
    return directions


def get_gvp_node_features(
    structure, 
    k_neighbors: int = 10,
    chain_id: str = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract complete GVP node features (scalar and vector features).
    
    Returns both scalar features and 3D vector features for each residue,
    suitable for input to a GVP-GNN encoder.
    
    Parameters
    ----------
    structure : Bio.PDB.Structure.Structure
        BioPython structure object.
    k_neighbors : int
        Number of neighbors for neighbor center vectors.
    chain_id : str, optional
        Chain ID to extract. If None, extracts all chains.
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        - node_coords: Shape (N, 3) - Cα coordinates
        - vector_features: Shape (N, 4, 3) - 4 vector features per residue:
          [sidechain_orientation, backbone_forward, backbone_backward, neighbor_center]
        
    Examples
    --------
    >>> coords, vectors = ppgvp.get_gvp_node_features(structure, k_neighbors=10)
    >>> print(coords.shape)    # (N_residues, 3)
    >>> print(vectors.shape)   # (N_residues, 4, 3)
    """
    ca_coords = get_ca_coords(structure, chain_id)
    
    if len(ca_coords) == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 4, 3), dtype=np.float32)
    
    # Get all vector features
    sidechain_orient = get_residue_orientations(structure, chain_id)
    backbone_forward = get_backbone_vectors(structure, chain_id)
    backbone_backward = get_backward_vectors(structure, chain_id)
    neighbor_center = get_neighbor_center_vectors(structure, k_neighbors, chain_id)
    
    # Stack into (N, 4, 3) - 4 vectors of dimension 3 each
    vector_features = np.stack([
        sidechain_orient,     # Cα→Cβ
        backbone_forward,     # Cα→Cα_next
        backbone_backward,    # Cα→Cα_prev (NEW)
        neighbor_center       # Cα→neighbors_mean
    ], axis=1)
    
    return ca_coords, vector_features


def build_knn_edges(
    coords: np.ndarray, 
    k: int = 20, 
    radius: float = 10.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build k-NN graph edges with optional radius cutoff.
    
    Parameters
    ----------
    coords : np.ndarray
        Node coordinates, shape (N, 3).
    k : int
        Number of nearest neighbors per node.
    radius : float
        Maximum distance for edges (Angstroms).
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        - edge_index: Shape (2, E) - source and target node indices
        - edge_distances: Shape (E,) - edge distances
        
    Examples
    --------
    >>> edge_index, distances = ppgvp.build_knn_edges(ca_coords, k=20, radius=10.0)
    >>> print(edge_index.shape)  # (2, E)
    """
    n = len(coords)
    if n == 0:
        return np.zeros((2, 0), dtype=np.int64), np.zeros(0, dtype=np.float32)
    
    # Compute pairwise distances
    diff = coords[:, None, :] - coords[None, :, :]  # (N, N, 3)
    dist_matrix = np.linalg.norm(diff, axis=2)  # (N, N)
    
    sources = []
    targets = []
    distances = []
    
    for i in range(n):
        dists = dist_matrix[i]
        
        # Get k nearest (excluding self)
        k_actual = min(k, n - 1)
        sorted_indices = np.argsort(dists)
        
        for j in sorted_indices[1:k_actual + 1]:
            d = dists[j]
            if d <= radius:
                sources.append(i)
                targets.append(j)
                distances.append(d)
    
    if len(sources) == 0:
        return np.zeros((2, 0), dtype=np.int64), np.zeros(0, dtype=np.float32)
    
    edge_index = np.array([sources, targets], dtype=np.int64)
    edge_distances = np.array(distances, dtype=np.float32)
    
    return edge_index, edge_distances


def get_edge_vectors(
    coords: np.ndarray, 
    edge_index: np.ndarray
) -> np.ndarray:
    """
    Compute edge direction vectors for graph edges.
    
    Parameters
    ----------
    coords : np.ndarray
        Node coordinates, shape (N, 3).
    edge_index : np.ndarray
        Edge indices, shape (2, E).
        
    Returns
    -------
    np.ndarray
        Unit direction vectors, shape (E, 3).
    """
    if edge_index.shape[1] == 0:
        return np.zeros((0, 3), dtype=np.float32)
    
    src, tgt = edge_index
    vectors = coords[tgt] - coords[src]
    
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    
    return (vectors / norms).astype(np.float32)


if __name__ == "__main__":
    print("gvp.py module loaded successfully")
