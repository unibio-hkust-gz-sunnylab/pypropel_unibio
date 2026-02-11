__author__ = "Jianfeng Sun"
__version__ = "v1.0"
__copyright__ = "Copyright 2024"
__license__ = "GPL v3.0"
__email__ = "jianfeng.sunmt@gmail.com"
__maintainer__ = "Jianfeng Sun"

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Union
from Bio.PDB.Polypeptide import is_aa


class ContactMap:
    """
    Contact map and distance matrix generation for proteins and ligands.
    
    Provides methods for calculating distances at atom and residue levels,
    generating pairwise distance matrices, and creating contact maps.
    
    Examples
    --------
    >>> from pypropel.prot.structure.distance.ContactMap import ContactMap
    >>> cm = ContactMap()
    >>> 
    >>> # Atom distance
    >>> d = cm.atom_distance(np.array([0,0,0]), np.array([3,4,0]))
    >>> print(d)  # 5.0
    >>>
    >>> # Distance matrix
    >>> coords1 = np.array([[0,0,0], [1,0,0]])
    >>> coords2 = np.array([[3,0,0], [4,0,0]])
    >>> matrix = cm.atom_distance_matrix(coords1, coords2)
    >>> print(matrix.shape)  # (2, 2)
    """

    def __init__(self):
        pass

    # ==================== Atom Level ====================
    
    def atom_distance(self, coord1: np.ndarray, coord2: np.ndarray) -> float:
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
        """
        return float(np.sqrt(np.sum((coord1 - coord2) ** 2)))

    def atom_distance_matrix(self, coords1: np.ndarray, coords2: np.ndarray) -> np.ndarray:
        """
        Calculate pairwise distance matrix between two sets of atom coordinates.
        
        Uses vectorized numpy operations for efficient computation.
        
        Parameters
        ----------
        coords1 : np.ndarray
            First set of coordinates, shape (N, 3).
        coords2 : np.ndarray
            Second set of coordinates, shape (M, 3).
            
        Returns
        -------
        np.ndarray
            Distance matrix of shape (N, M) where entry [i,j] is the
            distance between coords1[i] and coords2[j].
        """
        # Vectorized pairwise distance calculation
        # coords1: (N, 3), coords2: (M, 3)
        # Expand dims: (N, 1, 3) - (1, M, 3) -> (N, M, 3)
        diff = coords1[:, np.newaxis, :] - coords2[np.newaxis, :, :]
        dist_sq = np.sum(diff ** 2, axis=-1)
        return np.sqrt(dist_sq)

    # ==================== Residue Level ====================

    def residue_distance(self, res1_coords: np.ndarray, res2_coords: np.ndarray) -> float:
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
        if len(res1_coords) == 0 or len(res2_coords) == 0:
            return float('inf')
        
        dist_matrix = self.atom_distance_matrix(res1_coords, res2_coords)
        return float(np.min(dist_matrix))

    def residue_ligand_distance(self, residue_coords: np.ndarray, ligand_coords: np.ndarray) -> float:
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
        return self.residue_distance(residue_coords, ligand_coords)

    # ==================== Protein Level ====================

    def get_residue_coords(self, residue) -> np.ndarray:
        """
        Extract atom coordinates from a BioPython residue object.
        
        Parameters
        ----------
        residue : Bio.PDB.Residue.Residue
            BioPython residue object.
            
        Returns
        -------
        np.ndarray
            Atom coordinates, shape (N, 3).
        """
        coords = []
        for atom in residue:
            coords.append(atom.get_coord())
        return np.array(coords) if coords else np.array([]).reshape(0, 3)

    def protein_distance_matrix(
            self, 
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
            Default is False (use minimum distance across all atoms).
            
        Returns
        -------
        pd.DataFrame
            Distance matrix with residue info as index/columns.
        """
        residues = []
        residue_info = []
        
        for model in structure:
            for chain in model:
                if chain_ids is not None and chain.get_id() not in chain_ids:
                    continue
                for residue in chain:
                    if not is_aa(residue, standard=True):
                        continue
                    if use_ca_only:
                        if 'CA' in residue:
                            coords = np.array([residue['CA'].get_coord()])
                        else:
                            continue
                    else:
                        coords = self.get_residue_coords(residue)
                    
                    if len(coords) > 0:
                        residues.append(coords)
                        residue_info.append({
                            'chain': chain.get_id(),
                            'res_id': residue.get_id()[1],
                            'res_name': residue.get_resname()
                        })
        
        n = len(residues)
        dist_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                dist = self.residue_distance(residues[i], residues[j])
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist
        
        # Create labels
        labels = [f"{r['chain']}:{r['res_name']}{r['res_id']}" for r in residue_info]
        
        return pd.DataFrame(dist_matrix, index=labels, columns=labels)

    def protein_ligand_distances(
            self, 
            structure, 
            ligand_coords: np.ndarray
    ) -> pd.DataFrame:
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
        results = []
        
        for model in structure:
            for chain in model:
                for residue in chain:
                    if not is_aa(residue, standard=True):
                        continue
                    if 'CA' not in residue:
                        continue

                    res_coords = self.get_residue_coords(residue)
                    if len(res_coords) == 0:
                        continue

                    dist = self.residue_ligand_distance(res_coords, ligand_coords)

                    results.append({
                        'chain': chain.get_id(),
                        'res_id': residue.get_id()[1],
                        'res_name': residue.get_resname(),
                        'distance': dist
                    })
        
        return pd.DataFrame(results)

    # ==================== Contact Maps ====================

    def protein_contact_map(
            self, 
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
        dist_df = self.protein_distance_matrix(
            structure, 
            chain_ids=chain_ids, 
            use_ca_only=use_ca_only
        )
        contact_map = (dist_df.values < threshold).astype(int)
        return contact_map

    def protein_ligand_contact_map(
            self, 
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
        contact_rows = []
        
        for model in structure:
            for chain in model:
                for residue in chain:
                    if not is_aa(residue, standard=True):
                        continue
                    
                    res_coords = self.get_residue_coords(residue)
                    if len(res_coords) == 0:
                        contact_rows.append(np.zeros(len(ligand_coords)))
                        continue
                    
                    # Distance from each residue atom to each ligand atom
                    dist_matrix = self.atom_distance_matrix(res_coords, ligand_coords)
                    # For each ligand atom, take min distance from any residue atom
                    min_dist_per_ligand_atom = np.min(dist_matrix, axis=0)
                    # Contact if distance < threshold
                    contact_row = (min_dist_per_ligand_atom < threshold).astype(int)
                    contact_rows.append(contact_row)
        
        return np.array(contact_rows)

    # ==================== Pocket Extraction ====================

    def extract_binding_pocket(
            self, 
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
            Residues between thresholds are marked as 'margin' (-1).
            
        Returns
        -------
        pd.DataFrame
            DataFrame with columns: chain, res_id, res_name, distance, label.
            label: 1 = binding, 0 = non-binding, -1 = margin (ambiguous).
        """
        df = self.protein_ligand_distances(structure, ligand_coords)
        
        def assign_label(dist):
            if dist < binding_threshold:
                return 1
            elif dist > non_binding_threshold:
                return 0
            else:
                return -1  # Margin
        
        df['label'] = df['distance'].apply(assign_label)
        return df


if __name__ == "__main__":
    # Quick test
    cm = ContactMap()
    
    # Test atom distance
    d = cm.atom_distance(np.array([0, 0, 0]), np.array([3, 4, 0]))
    print(f"Atom distance: {d}")  # Expected: 5.0
    
    # Test distance matrix
    coords1 = np.array([[0, 0, 0], [1, 0, 0]])
    coords2 = np.array([[3, 0, 0], [4, 0, 0]])
    matrix = cm.atom_distance_matrix(coords1, coords2)
    print(f"Distance matrix shape: {matrix.shape}")
    print(f"Distance matrix:\n{matrix}")
