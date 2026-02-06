"""
Base dataset class for PocketBench.

Provides abstract interfaces for dataset loading and iteration.
"""

__author__ = "pypropel team"
__version__ = "0.1.0"

from abc import ABC, abstractmethod
from typing import List, Optional, Iterator, Union
from pathlib import Path
import os

from ..core import PBProtein


class PBDataset(ABC):
    """
    Abstract base class for PocketBench datasets.
    
    All dataset loaders must implement this interface for unified
    benchmarking across different data sources.
    
    Attributes
    ----------
    name : str
        Dataset name for logging and reporting.
    root : Path
        Root directory for dataset storage.
    """
    
    def __init__(
        self, 
        root: Optional[Union[str, Path]] = None,
        download: bool = False
    ):
        """
        Initialize dataset.
        
        Parameters
        ----------
        root : str or Path, optional
            Root directory for dataset. Defaults to ~/.pocketbench/datasets.
        download : bool, optional
            Whether to download the dataset if not found. Default False.
        """
        if root is None:
            root = Path.home() / ".pocketbench" / "datasets"
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        
        self._proteins: List[PBProtein] = []
        self._loaded = False
        
        if download and not self._check_exists():
            self.download()
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Dataset name."""
        pass
    
    @abstractmethod
    def _load(self) -> List[PBProtein]:
        """
        Load all proteins from dataset.
        
        Returns
        -------
        List[PBProtein]
            List of proteins with ground truth sites.
        """
        pass
    
    @abstractmethod
    def _check_exists(self) -> bool:
        """Check if dataset files exist."""
        pass
    
    def download(self):
        """
        Download dataset files.
        
        Should be overridden by subclasses that support automatic download.
        """
        raise NotImplementedError(
            f"Automatic download not implemented for {self.name}. "
            f"Please download manually to {self.root}"
        )
    
    def load(self) -> "PBDataset":
        """
        Load dataset into memory.
        
        Returns
        -------
        PBDataset
            Self for method chaining.
        """
        if not self._loaded:
            self._proteins = self._load()
            self._loaded = True
        return self
    
    def __len__(self) -> int:
        """Number of proteins in dataset."""
        if not self._loaded:
            self.load()
        return len(self._proteins)
    
    def __getitem__(self, idx: int) -> PBProtein:
        """Get protein by index."""
        if not self._loaded:
            self.load()
        return self._proteins[idx]
    
    def __iter__(self) -> Iterator[PBProtein]:
        """Iterate over proteins."""
        if not self._loaded:
            self.load()
        return iter(self._proteins)
    
    def get_by_id(self, protein_id: str) -> Optional[PBProtein]:
        """
        Get protein by ID.
        
        Parameters
        ----------
        protein_id : str
            Protein identifier.
            
        Returns
        -------
        PBProtein or None
            Protein if found, else None.
        """
        if not self._loaded:
            self.load()
        for p in self._proteins:
            if p.id == protein_id:
                return p
        return None
    
    def split(
        self, 
        train_ratio: float = 0.8,
        seed: int = 42
    ) -> tuple:
        """
        Split dataset into train and test sets.
        
        Parameters
        ----------
        train_ratio : float
            Fraction for training set.
        seed : int
            Random seed for reproducibility.
            
        Returns
        -------
        tuple
            (train_proteins, test_proteins)
        """
        import random
        if not self._loaded:
            self.load()
        
        proteins = list(self._proteins)
        random.Random(seed).shuffle(proteins)
        
        n_train = int(len(proteins) * train_ratio)
        return proteins[:n_train], proteins[n_train:]
    
    def __repr__(self) -> str:
        n = len(self) if self._loaded else "?"
        return f"{self.__class__.__name__}(n={n}, root={self.root})"
