"""
Dataset loaders for P2Rank benchmark datasets.

Implements loaders for legacy PDB-centric benchmarks:
- COACH420: 420 proteins from COACH
- HOLO4K: 4000+ holo structures

Source: https://github.com/rdk/p2rank-datasets
"""

__author__ = "pypropel team"
__version__ = "0.1.0"

from typing import List, Optional, Union
from pathlib import Path
import os
import urllib.request
import zipfile
import shutil

import numpy as np

from ..core import PBProtein, PBSite
from .base import PBDataset


# P2Rank datasets GitHub repository
P2RANK_REPO_URL = "https://github.com/rdk/p2rank-datasets"
P2RANK_RELEASE_URL = "https://github.com/rdk/p2rank-datasets/archive/refs/heads/master.zip"


class P2RankDataset(PBDataset):
    """
    Base class for P2Rank benchmark datasets.
    
    Handles downloading and parsing of .ds files and PDB structures
    from the p2rank-datasets repository.
    """
    
    # Subclasses must set these
    _ds_file: str = ""
    _pdb_subdir: str = ""
    
    def __init__(
        self,
        root: Optional[Union[str, Path]] = None,
        download: bool = False
    ):
        if root is None:
            root = Path.home() / ".pocketbench" / "datasets"
        
        self.root = Path(root)
        self._dataset_dir = self.root / "p2rank-datasets-master"
        
        # Now call super, which handles directory creation and download
        super().__init__(root, download)
    
    def _check_exists(self) -> bool:
        """Check if dataset files exist."""
        ds_path = self._dataset_dir / self._ds_file
        return ds_path.exists()
    
    def download(self):
        """Download p2rank-datasets from GitHub."""
        zip_path = self.root / "p2rank-datasets.zip"
        
        # Clean up corrupted file if exists
        if zip_path.exists():
            zip_path.unlink()
        
        print(f"Downloading p2rank-datasets to {self.root}...")
        
        # Use requests for better redirect handling
        try:
            import requests
            response = requests.get(P2RANK_RELEASE_URL, stream=True, timeout=300)
            response.raise_for_status()
            
            with open(zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        except ImportError:
            # Fallback to urllib with SSL bypass
            import ssl
            ssl._create_default_https_context = ssl._create_unverified_context
            urllib.request.urlretrieve(P2RANK_RELEASE_URL, zip_path)
        
        # Validate zip file
        if not zipfile.is_zipfile(zip_path):
            zip_path.unlink()
            raise RuntimeError(
                f"Downloaded file is not a valid ZIP. "
                f"Please download manually from {P2RANK_REPO_URL}"
            )
        
        # Extract
        print("Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(self.root)
        
        # Cleanup
        zip_path.unlink()
        print(f"Done. Dataset available at {self._dataset_dir}")
    
    def _load(self) -> List[PBProtein]:
        """Load proteins from .ds file."""
        if not self._check_exists():
            raise FileNotFoundError(
                f"Dataset not found at {self._dataset_dir}. "
                f"Set download=True or download manually from {P2RANK_REPO_URL}"
            )
        
        ds_path = self._dataset_dir / self._ds_file
        # Note: .ds file contains paths like "coach420/148lE.pdb" relative to dataset root
        pdb_base_dir = self._dataset_dir
        
        proteins = []
        
        # Parse .ds file
        with open(ds_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                # Format: "PROTEIN_PDB  LIGAND_CODES"
                parts = line.split()
                if not parts:
                    continue
                
                pdb_file = parts[0]
                ligand_codes = parts[1:] if len(parts) > 1 else []
                
                # Load protein
                pdb_path = pdb_base_dir / pdb_file
                if not pdb_path.exists():
                    print(f"Warning: PDB not found: {pdb_path}")
                    continue
                
                try:
                    protein = self._load_protein(pdb_path, ligand_codes)
                    if protein is not None:
                        proteins.append(protein)
                except Exception as e:
                    print(f"Warning: Failed to load {pdb_path}: {e}")
        
        print(f"Loaded {len(proteins)} proteins from {self.name}")
        return proteins
    
    def _load_protein(
        self, 
        pdb_path: Path, 
        ligand_codes: List[str]
    ) -> Optional[PBProtein]:
        """
        Load a single protein from PDB file.
        
        Parameters
        ----------
        pdb_path : Path
            Path to PDB file.
        ligand_codes : List[str]
            Ligand residue codes for ground truth.
            
        Returns
        -------
        PBProtein or None
            Loaded protein or None if failed.
        """
        from Bio.PDB import PDBParser
        from Bio.PDB.Polypeptide import is_aa
        # Handle BioPython version differences (three_to_one moved in 1.84+)
        try:
            from Bio.PDB.Polypeptide import three_to_one
        except ImportError:
            from Bio.Data.IUPACData import protein_letters_3to1
            def three_to_one(res_name):
                return protein_letters_3to1.get(res_name.capitalize(), 'X')
        
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure(pdb_path.stem, pdb_path)
        
        # Extract sequence and C-alpha coordinates
        sequence = []
        ca_coords = []
        residue_list = []
        
        for model in structure:
            for chain in model:
                for residue in chain:
                    if is_aa(residue, standard=True):
                        try:
                            one_letter = three_to_one(residue.get_resname())
                            ca = residue['CA']
                            sequence.append(one_letter)
                            ca_coords.append(ca.get_coord())
                            residue_list.append(residue)
                        except (KeyError, ValueError):
                            continue
            break  # Only first model
        
        if not sequence:
            return None
        
        coords = np.array(ca_coords, dtype=np.float32)
        seq_str = ''.join(sequence)
        
        # Extract ground truth sites from ligands
        ground_truth_sites = self._extract_sites_from_ligands(
            structure, residue_list, ligand_codes
        )
        
        return PBProtein(
            id=pdb_path.stem,
            sequence=seq_str,
            coords=coords,
            full_atoms=structure,
            ground_truth_sites=ground_truth_sites,
            metadata={'source': self.name, 'pdb_path': str(pdb_path)}
        )
    
    def _extract_sites_from_ligands(
        self,
        structure,
        residue_list: list,
        ligand_codes: List[str],
        contact_threshold: float = 4.0
    ) -> List[PBSite]:
        """
        Extract binding sites based on ligand contacts.
        
        A residue is part of the binding site if any of its atoms
        is within contact_threshold of any ligand atom.
        """
        from Bio.PDB import NeighborSearch
        
        # Get all protein atoms
        protein_atoms = []
        for res in residue_list:
            protein_atoms.extend(res.get_atoms())
        
        if not protein_atoms:
            return []
        
        ns = NeighborSearch(protein_atoms)
        sites = []
        
        # Find ligand residues
        for model in structure:
            for chain in model:
                for residue in chain:
                    res_name = residue.get_resname()
                    if res_name in ligand_codes or (not ligand_codes and residue.id[0] != ' '):
                        # This is a ligand
                        ligand_atoms = list(residue.get_atoms())
                        if not ligand_atoms:
                            continue
                        
                        # Find contacting protein residues
                        contacting_residues = set()
                        ligand_coords = []
                        
                        for atom in ligand_atoms:
                            ligand_coords.append(atom.get_coord())
                            nearby = ns.search(atom.get_coord(), contact_threshold)
                            for nearby_atom in nearby:
                                parent = nearby_atom.get_parent()
                                if parent in residue_list:
                                    contacting_residues.add(
                                        residue_list.index(parent)
                                    )
                        
                        if contacting_residues:
                            # Compute ligand center
                            center = np.mean(ligand_coords, axis=0)
                            sites.append(PBSite(
                                center=center,
                                residues=sorted(contacting_residues),
                                ligand_id=res_name,
                                pdb_id=structure.id
                            ))
            break  # Only first model
        
        return sites


class COACH420Dataset(P2RankDataset):
    """
    COACH420 benchmark dataset.
    
    Contains 420 proteins from the COACH benchmark for binding
    site prediction evaluation.
    
    Examples
    --------
    >>> from pypropel.pocketbench.datasets import COACH420Dataset
    >>> ds = COACH420Dataset(download=True)
    >>> print(f"Loaded {len(ds)} proteins")
    >>> protein = ds[0]
    >>> print(f"{protein.id}: {protein.num_sites} binding sites")
    """
    
    _ds_file = "coach420.ds"
    _pdb_subdir = "coach420"
    
    @property
    def name(self) -> str:
        return "COACH420"


class HOLO4KDataset(P2RankDataset):
    """
    HOLO4K benchmark dataset.
    
    Contains 4000+ holo structures for large-scale binding
    site prediction benchmarking.
    
    Examples
    --------
    >>> from pypropel.pocketbench.datasets import HOLO4KDataset
    >>> ds = HOLO4KDataset(download=True)
    >>> print(f"Loaded {len(ds)} proteins")
    """
    
    _ds_file = "holo4k.ds"
    _pdb_subdir = "holo4k"
    
    @property
    def name(self) -> str:
        return "HOLO4K"
