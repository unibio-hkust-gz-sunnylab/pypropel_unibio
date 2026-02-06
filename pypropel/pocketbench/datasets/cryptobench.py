"""
Dataset loader for CryptoBench dataset.

CryptoBench is a comprehensive dataset of cryptic protein-ligand binding sites.
Cryptic sites are spatially malformed or inaccessible in unbound (apo) state
but become visible upon ligand binding (holo state).

Source: https://github.com/skrhakv/CryptoBench
Data: https://osf.io/pz4a9/
Paper: Bioinformatics 2025 (btae745)
"""

__author__ = "pypropel team"
__version__ = "0.1.0"

from typing import List, Optional, Union, Dict
from pathlib import Path
import os

import numpy as np

from ..core import PBProtein, PBSite
from .base import PBDataset


# OSF data repository
CRYPTOBENCH_OSF_URL = "https://osf.io/pz4a9/"
CRYPTOBENCH_GITHUB_URL = "https://github.com/skrhakv/CryptoBench"


class CryptoBenchDataset(PBDataset):
    """
    CryptoBench dataset for cryptic binding site evaluation.
    
    Cryptic binding sites are sites that are spatially malformed or
    inaccessible in their unbound (apo) state but become visible
    through ligand binding. This dataset provides paired apo/holo
    structures for evaluating cryptic pocket prediction.
    
    Key features:
    - 1000+ protein structures
    - Paired apo/holo structures
    - Train/test splits available
    - Focus on cryptic (hidden) binding sites
    
    The evaluation paradigm:
    - Input: Apo (unbound) structure
    - Ground truth: Binding sites from Holo (bound) structure
    
    Examples
    --------
    >>> from pypropel.pocketbench.datasets import CryptoBenchDataset
    >>> ds = CryptoBenchDataset(root="/path/to/cryptobench", split="test")
    >>> print(f"Loaded {len(ds)} apo structures")
    >>> protein = ds[0]
    >>> # protein.coords are from APO structure
    >>> # protein.ground_truth_sites are from HOLO structure
    
    Notes
    -----
    The dataset must be downloaded manually from OSF:
    https://osf.io/pz4a9/
    """
    
    def __init__(
        self,
        root: Optional[Union[str, Path]] = None,
        split: str = "test",
        use_apo_structure: bool = True,
        download: bool = False
    ):
        """
        Initialize CryptoBench dataset.
        
        Parameters
        ----------
        root : str or Path, optional
            Root directory containing the extracted dataset.
            Defaults to ~/.pocketbench/datasets/cryptobench.
        split : str, optional
            Data split: "train" or "test". Default "test".
        use_apo_structure : bool, optional
            If True (default), load apo structure as input.
            If False, load holo structure for non-cryptic evaluation.
        download : bool, optional
            Whether to attempt download. Currently requires manual download.
        """
        if root is None:
            root = Path.home() / ".pocketbench" / "datasets" / "cryptobench"
        self.split = split
        self.use_apo_structure = use_apo_structure
        super().__init__(root, download)
    
    @property
    def name(self) -> str:
        struct_type = "apo" if self.use_apo_structure else "holo"
        return f"CryptoBench ({self.split}, {struct_type})"
    
    def _check_exists(self) -> bool:
        """Check if dataset files exist."""
        split_file = self.root / f"{self.split}_set.csv"
        return split_file.exists() or (self.root / "data").exists()
    
    def download(self):
        """Download dataset from OSF."""
        import urllib.request
        import zipfile
        import shutil
        
        print(f"Downloading CryptoBench from OSF...")
        print(f"This may take a while...")
        
        # Create root directory
        self.root.mkdir(parents=True, exist_ok=True)
        
        # OSF direct download URL for the dataset archive
        # The dataset is available as a zip file
        osf_download_url = "https://osf.io/pz4a9/download"
        
        zip_path = self.root / "cryptobench.zip"
        
        try:
            # Download the zip file
            print(f"Downloading from {osf_download_url}...")
            urllib.request.urlretrieve(osf_download_url, zip_path)
            
            # Extract the zip file
            print(f"Extracting to {self.root}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.root)
            
            # Cleanup zip file
            zip_path.unlink()
            
            print(f"Download complete! Dataset saved to: {self.root}")
            
        except Exception as e:
            # Cleanup on failure
            if zip_path.exists():
                zip_path.unlink()
            
            raise RuntimeError(
                f"Failed to download CryptoBench: {e}\n"
                f"You can manually download from: {CRYPTOBENCH_OSF_URL}"
            )
    
    def _load(self) -> List[PBProtein]:
        """Load proteins from CryptoBench directory."""
        if not self._check_exists():
            raise FileNotFoundError(
                f"Dataset not found at {self.root}. "
                f"Please download from {CRYPTOBENCH_OSF_URL}"
            )
        
        # Try different possible directory structures
        proteins = []
        
        # Check for split CSV files
        split_file = self.root / f"{self.split}_set.csv"
        if split_file.exists():
            proteins = self._load_from_split_file(split_file)
        else:
            # Try loading from data directory
            data_dir = self.root / "data"
            if data_dir.exists():
                proteins = self._load_from_data_dir(data_dir)
        
        print(f"Loaded {len(proteins)} proteins from {self.name}")
        return proteins
    
    def _load_from_split_file(self, split_file: Path) -> List[PBProtein]:
        """Load proteins from split CSV file."""
        import csv
        
        proteins = []
        with open(split_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    protein = self._load_protein_from_row(row)
                    if protein is not None:
                        proteins.append(protein)
                except Exception as e:
                    print(f"Warning: Failed to load {row}: {e}")
        
        return proteins
    
    def _load_from_data_dir(self, data_dir: Path) -> List[PBProtein]:
        """Load proteins from data directory structure."""
        proteins = []
        
        # Look for CIF/PDB files
        for cif_file in data_dir.glob("**/*.cif"):
            try:
                protein = self._load_protein_from_file(cif_file)
                if protein is not None:
                    proteins.append(protein)
            except Exception as e:
                print(f"Warning: Failed to load {cif_file}: {e}")
        
        return proteins
    
    def _load_protein_from_row(self, row: Dict) -> Optional[PBProtein]:
        """Load protein from a row in split CSV."""
        # Expected columns: apo_pdb, holo_pdb, binding_site_residues, etc.
        apo_id = row.get('apo_pdb', row.get('apo_id', ''))
        holo_id = row.get('holo_pdb', row.get('holo_id', ''))
        
        # Determine which structure to load
        if self.use_apo_structure:
            struct_id = apo_id
            struct_type = "apo"
        else:
            struct_id = holo_id
            struct_type = "holo"
        
        if not struct_id:
            return None
        
        # Find the structure file
        struct_path = self._find_structure_file(struct_id)
        if struct_path is None:
            return None
        
        # Load structure
        sequence, coords, structure = self._parse_structure(struct_path)
        if not sequence:
            return None
        
        # Parse binding site residues from holo
        binding_residues_str = row.get('binding_site_residues', 
                                       row.get('site_residues', ''))
        ground_truth_sites = self._parse_binding_sites(
            binding_residues_str, coords
        )
        
        return PBProtein(
            id=f"{apo_id}_{holo_id}" if apo_id and holo_id else struct_id,
            sequence=sequence,
            coords=coords,
            full_atoms=structure,
            ground_truth_sites=ground_truth_sites,
            metadata={
                'source': 'CryptoBench',
                'split': self.split,
                'apo_id': apo_id,
                'holo_id': holo_id,
                'struct_type': struct_type
            }
        )
    
    def _load_protein_from_file(self, file_path: Path) -> Optional[PBProtein]:
        """Load protein from structure file."""
        sequence, coords, structure = self._parse_structure(file_path)
        if not sequence:
            return None
        
        return PBProtein(
            id=file_path.stem,
            sequence=sequence,
            coords=coords,
            full_atoms=structure,
            ground_truth_sites=[],  # Would need separate annotation
            metadata={
                'source': 'CryptoBench',
                'file_path': str(file_path)
            }
        )
    
    def _find_structure_file(self, struct_id: str) -> Optional[Path]:
        """Find structure file by ID."""
        # Try various locations and extensions
        extensions = ['.cif', '.pdb', '.pdb.gz', '.cif.gz']
        locations = [
            self.root,
            self.root / "data",
            self.root / "structures",
            self.root / "apo" if self.use_apo_structure else self.root / "holo",
        ]
        
        for loc in locations:
            if not loc.exists():
                continue
            for ext in extensions:
                path = loc / f"{struct_id}{ext}"
                if path.exists():
                    return path
                # Also try subdirectory
                subdir_path = loc / struct_id[:4] / f"{struct_id}{ext}"
                if subdir_path.exists():
                    return subdir_path
        
        return None
    
    def _parse_structure(self, file_path: Path) -> tuple:
        """Parse structure file and extract sequence and coordinates."""
        try:
            from Bio.PDB import PDBParser, MMCIFParser
            from Bio.PDB.Polypeptide import is_aa, three_to_one
            
            # Choose parser based on file extension
            if '.cif' in file_path.name.lower():
                parser = MMCIFParser(QUIET=True)
            else:
                parser = PDBParser(QUIET=True)
            
            structure = parser.get_structure(file_path.stem, file_path)
            
            sequence = []
            ca_coords = []
            
            for model in structure:
                for chain in model:
                    for residue in chain:
                        if is_aa(residue, standard=True):
                            try:
                                one_letter = three_to_one(residue.get_resname())
                                ca = residue['CA']
                                sequence.append(one_letter)
                                ca_coords.append(ca.get_coord())
                            except (KeyError, ValueError):
                                continue
                break  # Only first model
            
            if not sequence:
                return "", np.empty((0, 3)), None
            
            return (
                ''.join(sequence),
                np.array(ca_coords, dtype=np.float32),
                structure
            )
        except Exception as e:
            print(f"Warning: Failed to parse {file_path}: {e}")
            return "", np.empty((0, 3)), None
    
    def _parse_binding_sites(
        self, 
        residues_str: str,
        coords: np.ndarray
    ) -> List[PBSite]:
        """Parse binding site residues from string representation."""
        if not residues_str:
            return []
        
        sites = []
        
        # Try different formats: comma-separated, semicolon-separated sites
        if ';' in residues_str:
            # Multiple sites separated by semicolons
            site_strs = residues_str.split(';')
        else:
            site_strs = [residues_str]
        
        for i, site_str in enumerate(site_strs):
            site_str = site_str.strip()
            if not site_str:
                continue
            
            # Parse residue indices
            residue_indices = []
            for part in site_str.split(','):
                part = part.strip()
                if part.isdigit():
                    # Convert to 0-indexed
                    idx = int(part) - 1
                    if 0 <= idx < len(coords):
                        residue_indices.append(idx)
                elif '-' in part:
                    # Range: "10-20"
                    try:
                        start, end = map(int, part.split('-'))
                        for idx in range(start - 1, end):
                            if 0 <= idx < len(coords):
                                residue_indices.append(idx)
                    except ValueError:
                        continue
            
            if residue_indices:
                # Compute center from residue coordinates
                site_coords = coords[residue_indices]
                center = np.mean(site_coords, axis=0)
                
                sites.append(PBSite(
                    center=center,
                    residues=sorted(set(residue_indices)),
                    ligand_id=f"cryptic_site{i + 1}"
                ))
        
        return sites
