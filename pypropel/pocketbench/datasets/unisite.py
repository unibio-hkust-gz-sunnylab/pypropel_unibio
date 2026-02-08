"""
Dataset loaders for UniSite-DS benchmark.

UniSite-DS is the first UniProt-centric ligand binding site dataset,
systematically integrating all binding sites for a protein across
multiple PDB structures.

Source: https://github.com/quanlin-wu/unisite
Data: https://huggingface.co/datasets/quanlin-wu/unisite-ds_v1
"""

__author__ = "pypropel team"
__version__ = "0.1.0"

from typing import List, Optional, Union
from pathlib import Path
import os
import pickle

import numpy as np

from ..core import PBProtein, PBSite
from .base import PBDataset


# HuggingFace dataset URL
UNISITE_HF_URL = "https://huggingface.co/datasets/quanlin-wu/unisite-ds_v1"


class UniSiteDSDataset(PBDataset):
    """
    UniSite-DS benchmark dataset loader.
    
    UniSite-DS is a UniProt-centric dataset that aggregates all ligand
    binding sites for a given protein sequence across multiple PDB
    structures. This handles the "union of ground truths" logic to
    prevent missing label bias.
    
    Key features:
    - 11,510 unique proteins
    - 3,670 multi-site proteins
    - UniProt sequence as canonical reference
    - Multiple binding sites per protein
    
    Data format:
    - PKL files with pocket_masks (binary masks for each site)
    - metadata.csv with protein info
    - Split files for train/test
    
    Examples
    --------
    >>> from pypropel.pocketbench.datasets import UniSiteDSDataset
    >>> ds = UniSiteDSDataset(root="/path/to/unisite-ds", split="test")
    >>> print(f"Loaded {len(ds)} proteins")
    >>> protein = ds[0]
    >>> print(f"{protein.id}: {protein.num_sites} binding sites")
    
    Notes
    -----
    The dataset must be downloaded manually from HuggingFace:
    https://huggingface.co/datasets/quanlin-wu/unisite-ds_v1
    """
    
    def __init__(
        self,
        root: Optional[Union[str, Path]] = None,
        split: str = "test",
        sim_threshold: float = 0.9,
        download: bool = False
    ):
        """
        Initialize UniSite-DS dataset.
        
        Parameters
        ----------
        root : str or Path, optional
            Root directory containing the extracted dataset.
            Defaults to ~/.pocketbench/datasets/unisite-ds.
        split : str, optional
            Data split: "train" or "test". Default "test".
        sim_threshold : float, optional
            Sequence similarity threshold for train/test split.
            Options: 0.3, 0.5, 0.7, 0.9. Default 0.9.
        download : bool, optional
            Whether to attempt download. Currently requires manual download.
        """
        if root is None:
            root = Path.home() / ".pocketbench" / "datasets" / "unisite-ds"
        self.split = split
        self.sim_threshold = sim_threshold
        super().__init__(root, download)
    
    @property
    def name(self) -> str:
        return f"UniSite-DS ({self.split})"
    
    def _check_exists(self) -> bool:
        """Check if dataset files exist, extracting archives if needed."""
        # Check both possible structures:
        # 1. HuggingFace download: files directly in root
        # 2. Manual extraction: files in pkl_files subdirectory
        
        # First, try to extract archives if they exist but aren't extracted
        main_archive = self.root / "unisite-ds-v1.tar.gz"
        pkl_dir = self.root / "pkl_files"
        if main_archive.exists() and not pkl_dir.exists():
            self._extract_archives()
        
        # Try pkl_files subdirectory first (manual extract)
        if not pkl_dir.exists():
            # Fallback: check if pkl_files is at root level (some HF downloads)
            pkl_dir = self.root
        
        split_file = pkl_dir / f"{self.split}_{self.sim_threshold}.csv"
        
        # Store the actual pkl_dir for _load to use
        self._pkl_dir = pkl_dir if split_file.exists() else None
        
        return split_file.exists()
    
    def download(self):
        """Download dataset from HuggingFace and extract archives."""
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            raise ImportError(
                "huggingface_hub is required for auto-download. "
                "Install with: pip install huggingface_hub"
            )
        
        print(f"Downloading UniSite-DS from HuggingFace...")
        print(f"This may take a while (dataset is ~10GB)")
        
        # Create root directory
        self.root.mkdir(parents=True, exist_ok=True)
        
        # Download dataset snapshot
        try:
            snapshot_download(
                repo_id="quanlin-wu/unisite-ds_v1",
                repo_type="dataset",
                local_dir=str(self.root),
                local_dir_use_symlinks=False,
            )
            print(f"Download complete! Dataset saved to: {self.root}")
        except Exception as e:
            raise RuntimeError(
                f"Failed to download UniSite-DS: {e}\n"
                f"You can manually download from: {UNISITE_HF_URL}"
            )
        
        # Extract archives
        self._extract_archives()
    
    def _extract_archives(self):
        """Extract tar.gz archives if they exist."""
        import tarfile
        
        # Main dataset archive
        main_archive = self.root / "unisite-ds-v1.tar.gz"
        if main_archive.exists():
            pkl_dir = self.root / "pkl_files"
            if not pkl_dir.exists():
                print(f"Extracting {main_archive.name}...")
                with tarfile.open(main_archive, 'r:gz') as tar:
                    tar.extractall(path=self.root)
                print(f"Extracted to {self.root}")
        
        # Benchmark datasets archive
        bench_archive = self.root / "benchmark_datasets.tar.gz"
        if bench_archive.exists():
            bench_dir = self.root / "benchmark_datasets"
            if not bench_dir.exists():
                print(f"Extracting {bench_archive.name}...")
                with tarfile.open(bench_archive, 'r:gz') as tar:
                    tar.extractall(path=self.root)
                print(f"Extracted benchmark datasets")
    
    def _load(self) -> List[PBProtein]:
        """Load proteins from PKL files."""
        if not self._check_exists():
            raise FileNotFoundError(
                f"Dataset not found at {self.root}. "
                f"Please download from {UNISITE_HF_URL}"
            )
        
        # Use the pkl_dir detected by _check_exists
        pkl_dir = getattr(self, '_pkl_dir', None) or self.root / "pkl_files"
        split_file = pkl_dir / f"{self.split}_{self.sim_threshold}.csv"
        
        # Read split file to get protein IDs
        protein_ids = []
        with open(split_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    protein_ids.append(line)
        
        proteins = []
        for unp_id in protein_ids:
            pkl_path = pkl_dir / f"{unp_id}.pkl"
            if not pkl_path.exists():
                continue
            
            try:
                protein = self._load_protein_from_pkl(pkl_path, unp_id)
                if protein is not None:
                    proteins.append(protein)
            except Exception as e:
                print(f"Warning: Failed to load {unp_id}: {e}")
        
        print(f"Loaded {len(proteins)} proteins from {self.name}")
        return proteins
    
    def _load_protein_from_pkl(
        self, 
        pkl_path: Path, 
        unp_id: str
    ) -> Optional[PBProtein]:
        """
        Load a single protein from PKL file.
        
        PKL format:
        - "label": UniProt ID
        - "sequence": amino acid sequence
        - "pdb_file_path": path to representative structure
        - "map_dict": residue mapping PDB -> UniProt
        - "target": {
            "pocket_masks": (num_sites, seq_len) binary masks
            "res_mask": (seq_len,) valid residue mask
          }
        """
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        sequence = data.get("sequence", "")
        if not sequence:
            return None
        
        # Extract binding site masks
        target = data.get("target", {})
        pocket_masks = target.get("pocket_masks", None)
        
        ground_truth_sites = []
        if pocket_masks is not None:
            pocket_masks = np.array(pocket_masks)
            for site_idx in range(pocket_masks.shape[0]):
                mask = pocket_masks[site_idx]
                residue_indices = np.where(mask > 0)[0].tolist()
                if residue_indices:
                    # Compute approximate center (centroid of residue positions)
                    # Note: Without 3D coords, we use sequence position as proxy
                    center_idx = np.mean(residue_indices)
                    center = np.array([center_idx, 0.0, 0.0])  # Placeholder
                    
                    ground_truth_sites.append(PBSite(
                        center=center,
                        residues=residue_indices,
                        ligand_id=f"site{site_idx + 1}"
                    ))
        
        # Try to load PDB structure for actual coordinates
        coords = self._load_coords_if_available(data, unp_id)
        
        return PBProtein(
            id=unp_id,
            sequence=sequence,
            coords=coords,
            full_atoms=None,
            ground_truth_sites=ground_truth_sites,
            metadata={
                'source': 'UniSite-DS',
                'split': self.split,
                'pdb_path': data.get("pdb_file_path", "")
            }
        )
    
    def _load_coords_if_available(
        self, 
        data: dict, 
        unp_id: str
    ) -> np.ndarray:
        """Load C-alpha coordinates if PDB file is available."""
        pdb_rel_path = data.get("pdb_file_path", "")
        if not pdb_rel_path:
            # Return placeholder coords
            seq_len = len(data.get("sequence", ""))
            return np.zeros((seq_len, 3), dtype=np.float32)
        
        # Try to find PDB file
        pdb_path = self.root / pdb_rel_path
        if not pdb_path.exists():
            # Also try direct path from unp_id
            pdb_path = self.root / unp_id / f"{unp_id}.pdb"
        
        if pdb_path.exists():
            try:
                from Bio.PDB import PDBParser
                from Bio.PDB.Polypeptide import is_aa
                
                parser = PDBParser(QUIET=True)
                structure = parser.get_structure(unp_id, pdb_path)
                
                ca_coords = []
                for model in structure:
                    for chain in model:
                        for residue in chain:
                            if is_aa(residue, standard=True):
                                try:
                                    ca = residue['CA']
                                    ca_coords.append(ca.get_coord())
                                except KeyError:
                                    pass
                    break
                
                if ca_coords:
                    return np.array(ca_coords, dtype=np.float32)
            except Exception:
                pass
        
        # Fallback: placeholder coords
        seq_len = len(data.get("sequence", ""))
        return np.zeros((seq_len, 3), dtype=np.float32)


class UniSiteBenchmarkDataset(PBDataset):
    """
    UniSite benchmark dataset loaders (HOLO4K-sc, COACH420).
    
    These are processed versions of the legacy benchmarks,
    prepared in the UniSite format with PKL files.
    
    Examples
    --------
    >>> ds = UniSiteBenchmarkDataset(root="/path/to/unisite-ds", 
    ...                              benchmark="coach420")
    >>> print(f"Loaded {len(ds)} proteins")
    """
    
    def __init__(
        self,
        root: Optional[Union[str, Path]] = None,
        benchmark: str = "coach420",
        download: bool = False
    ):
        """
        Initialize benchmark dataset.
        
        Parameters
        ----------
        root : str or Path, optional
            Root directory containing the extracted dataset.
        benchmark : str, optional
            Benchmark name: "coach420" or "holo4k-sc". Default "coach420".
        """
        if root is None:
            root = Path.home() / ".pocketbench" / "datasets" / "unisite-ds"
        self.benchmark = benchmark
        super().__init__(root, download)
    
    @property
    def name(self) -> str:
        return f"UniSite-{self.benchmark.upper()}"
    
    def _check_exists(self) -> bool:
        """Check if benchmark files exist."""
        bench_dir = self.root / "benchmark_datasets" / self.benchmark
        return bench_dir.exists()
    
    def _load(self) -> List[PBProtein]:
        """Load proteins from benchmark directory."""
        bench_dir = self.root / "benchmark_datasets" / self.benchmark
        
        if not bench_dir.exists():
            raise FileNotFoundError(
                f"Benchmark {self.benchmark} not found at {bench_dir}"
            )
        
        pkl_dir = bench_dir / "target_pkl"
        proteins = []
        
        for pkl_file in pkl_dir.glob("*.pkl"):
            try:
                protein = self._load_benchmark_protein(pkl_file)
                if protein is not None:
                    proteins.append(protein)
            except Exception as e:
                print(f"Warning: Failed to load {pkl_file.stem}: {e}")
        
        print(f"Loaded {len(proteins)} proteins from {self.name}")
        return proteins
    
    def _load_benchmark_protein(self, pkl_path: Path) -> Optional[PBProtein]:
        """Load a single protein from benchmark PKL file."""
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        protein_id = pkl_path.stem
        
        # Extract targets
        target = data.get("target", {})
        pocket_masks = target.get("pocket_masks", None)
        centers = data.get("centers", None)
        
        ground_truth_sites = []
        if pocket_masks is not None:
            pocket_masks = np.array(pocket_masks)
            for site_idx in range(pocket_masks.shape[0]):
                mask = pocket_masks[site_idx]
                residue_indices = np.where(mask > 0)[0].tolist()
                
                # Get center if available
                if centers is not None and site_idx < len(centers):
                    center = np.array(centers[site_idx])
                else:
                    center = np.zeros(3)
                
                if residue_indices:
                    ground_truth_sites.append(PBSite(
                        center=center,
                        residues=residue_indices,
                        ligand_id=f"pocket{site_idx + 1}"
                    ))
        
        # Try to load sequence from CSV
        seq = self._get_sequence(protein_id)
        
        return PBProtein(
            id=protein_id,
            sequence=seq,
            coords=np.zeros((len(seq) if seq else 1, 3), dtype=np.float32),
            ground_truth_sites=ground_truth_sites,
            metadata={'source': self.name}
        )
    
    def _get_sequence(self, protein_id: str) -> str:
        """Load sequence from CSV file."""
        bench_dir = self.root / "benchmark_datasets" / self.benchmark
        seq_file = bench_dir / f"{self.benchmark}_seq.csv"
        
        if not seq_file.exists():
            return ""
        
        import csv
        with open(seq_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get('name') == protein_id:
                    return row.get('sequence', '')
        return ""
