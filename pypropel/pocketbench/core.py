"""
Core dataclasses and abstractions for PocketBench.

This module provides the unified data objects for benchmarking
protein-ligand binding site prediction models.
"""

__author__ = "pypropel team"
__version__ = "0.1.0"

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Any, Union
import numpy as np


@dataclass
class PBSite:
    """
    Represents a single binding site (ground truth or predicted).
    
    Attributes
    ----------
    center : np.ndarray
        Geometric center [x, y, z] of the binding site.
    residues : List[int]
        0-indexed residue indices belonging to this pocket.
    ligand_id : str, optional
        Source ligand identifier (e.g., "ATP", "HEM").
    pdb_id : str, optional
        Source PDB ID if site comes from a specific structure.
    """
    center: np.ndarray
    residues: List[int] = field(default_factory=list)
    ligand_id: Optional[str] = None
    pdb_id: Optional[str] = None
    
    def __post_init__(self):
        """Ensure center is a numpy array."""
        if not isinstance(self.center, np.ndarray):
            self.center = np.array(self.center, dtype=np.float32)
    
    @property
    def num_residues(self) -> int:
        """Number of residues in this site."""
        return len(self.residues)
    
    def __repr__(self) -> str:
        lig_str = f", ligand={self.ligand_id}" if self.ligand_id else ""
        return f"PBSite(center={self.center.tolist()}, n_res={self.num_residues}{lig_str})"


@dataclass
class PBProtein:
    """
    Unified protein data object for benchmarking.
    
    Raw PDB files are inconsistent; all loaders must convert input
    into this standardized dataclass.
    
    Attributes
    ----------
    id : str
        Unique identifier (e.g., "1a2b_A" or UniProt ID).
    sequence : str
        Amino acid sequence (one-letter codes).
    coords : np.ndarray
        C-alpha coordinates, shape (N, 3).
    full_atoms : Any, optional
        Lazy-loaded full atom representation (e.g., BioPython Structure).
    ground_truth_sites : List[PBSite], optional
        Multiple valid binding sites (crucial for UniProt-centric benchmarking).
    metadata : dict, optional
        Additional metadata (source dataset, resolution, etc.).
    """
    id: str
    sequence: str
    coords: np.ndarray
    full_atoms: Any = None
    ground_truth_sites: List[PBSite] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    
    def __post_init__(self):
        """Ensure coords is a numpy array."""
        if not isinstance(self.coords, np.ndarray):
            self.coords = np.array(self.coords, dtype=np.float32)
    
    @property
    def num_residues(self) -> int:
        """Number of residues in the sequence."""
        return len(self.sequence)
    
    @property
    def num_sites(self) -> int:
        """Number of ground truth binding sites."""
        return len(self.ground_truth_sites)
    
    def get_site_centers(self) -> np.ndarray:
        """
        Get all ground truth site centers as an array.
        
        Returns
        -------
        np.ndarray
            Shape (K, 3) where K is number of sites.
        """
        if not self.ground_truth_sites:
            return np.empty((0, 3), dtype=np.float32)
        return np.stack([s.center for s in self.ground_truth_sites])
    
    def get_all_site_residues(self) -> List[int]:
        """
        Get union of all binding site residues.
        
        Returns
        -------
        List[int]
            Sorted unique residue indices from all sites.
        """
        all_res = set()
        for site in self.ground_truth_sites:
            all_res.update(site.residues)
        return sorted(all_res)
    
    def __repr__(self) -> str:
        return (f"PBProtein(id={self.id}, len={self.num_residues}, "
                f"n_sites={self.num_sites})")


@dataclass
class PBPrediction:
    """
    Normalized prediction output from any model.
    
    Models have vastly different outputs (virtual nodes, voxel grids,
    residue lists). This dataclass normalizes them to a common format.
    
    Attributes
    ----------
    center : np.ndarray
        Geometric center [x, y, z] of the predicted pocket.
    residues : List[int]
        0-indexed residue indices belonging to the predicted pocket.
    confidence : float
        Confidence score, range [0.0, 1.0].
    model_name : str
        Name of the model that made this prediction.
    """
    center: np.ndarray
    residues: List[int] = field(default_factory=list)
    confidence: float = 1.0
    model_name: str = ""
    ligand_id: Optional[str] = None
    
    def __post_init__(self):
        """Ensure center is a numpy array and confidence is valid."""
        if not isinstance(self.center, np.ndarray):
            self.center = np.array(self.center, dtype=np.float32)
        self.confidence = max(0.0, min(1.0, self.confidence))
    
    @property
    def num_residues(self) -> int:
        """Number of predicted residues."""
        return len(self.residues)
    
    def __repr__(self) -> str:
        return (f"PBPrediction(center={self.center.tolist()}, "
                f"n_res={self.num_residues}, conf={self.confidence:.3f})")


class PBModel(ABC):
    """
    Abstract base class for binding site prediction models.
    
    All model wrappers must implement this interface to be
    compatible with the PocketBench evaluation pipeline.
    
    Subclasses handle the internal logic of converting PBProtein
    to model-specific input (Graph, Voxel, Surface, etc.).
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Model name for logging and reporting."""
        pass
    
    @abstractmethod
    def predict(self, protein: PBProtein) -> List[PBPrediction]:
        """
        Predict binding sites for a protein.
        
        Parameters
        ----------
        protein : PBProtein
            Input protein in standardized format.
            
        Returns
        -------
        List[PBPrediction]
            List of predicted binding sites, sorted by confidence.
        """
        pass
    
    def predict_batch(
        self, 
        proteins: List[PBProtein]
    ) -> List[List[PBPrediction]]:
        """
        Predict binding sites for multiple proteins.
        
        Default implementation calls predict() sequentially.
        Override for batch-optimized inference.
        
        Parameters
        ----------
        proteins : List[PBProtein]
            List of input proteins.
            
        Returns
        -------
        List[List[PBPrediction]]
            Predictions for each protein.
        """
        return [self.predict(p) for p in proteins]


# Convenience type alias
ProteinInput = Union[PBProtein, str, Any]  # PBProtein, PDB path, or Structure
