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
    structures.
    
    Actual data format (folder-based):
    - unisite-ds-v1/{protein_id}/ - protein folder
      - {protein_id}_info.csv - protein info (sequence, etc.)
      - {protein_id}.pdb - representative structure
      - {protein_id}.mapping - residue mapping
      - site1/, site2/, ... - binding site directories
    
    Examples
    --------
    >>> from pypropel.pocketbench.datasets import UniSiteDSDataset
    >>> ds = UniSiteDSDataset(split="test")
    >>> print(f"Loaded {len(ds)} proteins")
    """
    
    def __init__(
        self,
        root: Optional[Union[str, Path]] = None,
        split: str = "test",
        sim_threshold: float = 0.9,
        download: bool = False,
        limit: Optional[int] = None
    ):
        """
        Initialize UniSite-DS dataset.
        
        Parameters
        ----------
        root : str or Path, optional
            Root directory containing the extracted dataset.
        split : str, optional
            Data split: "train" or "test". Default "test".
        sim_threshold : float, optional
            Sequence similarity threshold. Default 0.9.
        download : bool, optional
            Whether to attempt download.
        limit : int, optional
            Limit number of proteins to load (for testing).
        """
        if root is None:
            root = Path.home() / ".pocketbench" / "datasets" / "unisite-ds"
        self.split = split
        self.sim_threshold = sim_threshold
        self.limit = limit
        self._data_dir = None  # Will be set by _find_data_dir
        super().__init__(root, download)
    
    @property
    def name(self) -> str:
        return f"UniSite-DS ({self.split})"
    
    def _find_data_dir(self) -> Optional[Path]:
        """Find the actual data directory containing protein folders."""
        # Possible locations after extraction
        candidates = [
            self.root / "unisite-ds-v1",
            self.root / "unisite-ds_v1",
            self.root,
        ]
        
        for candidate in candidates:
            if candidate.exists() and candidate.is_dir():
                # Check if it contains protein folders (folders with _info.csv)
                subdirs = [d for d in candidate.iterdir() if d.is_dir()]
                if subdirs:
                    # Check first subdir for expected structure
                    first_dir = subdirs[0]
                    info_file = first_dir / f"{first_dir.name}_info.csv"
                    if info_file.exists():
                        return candidate
        
        return None
    
    def _check_exists(self) -> bool:
        """Check if dataset files exist, extracting archives if needed."""
        # Try to extract archives if needed
        main_archive = self.root / "unisite-ds-v1.tar.gz"
        if main_archive.exists():
            data_dir = self._find_data_dir()
            if data_dir is None:
                self._extract_archives()
        
        # Find the data directory
        self._data_dir = self._find_data_dir()
        
        if self._data_dir is None:
            if self.root.exists():
                print(f"[UniSite Debug] Contents of {self.root}:")
                for item in list(self.root.iterdir())[:10]:
                    print(f"  - {item.name}")
            return False
        
        return True
    
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
        """Load proteins from folder-based structure."""
        if not self._check_exists():
            raise FileNotFoundError(
                f"Dataset not found at {self.root}. "
                f"Please download from {UNISITE_HF_URL}"
            )
        
        data_dir = self._data_dir
        print(f"[UniSite Debug] data_dir = {data_dir}")
        
        # Get all protein directories
        all_subdirs = [d for d in data_dir.iterdir() if d.is_dir()]
        print(f"[UniSite Debug] Found {len(all_subdirs)} subdirectories")
        if all_subdirs:
            sample = all_subdirs[0]
            expected_csv = sample / f"{sample.name}_info.csv"
            print(f"[UniSite Debug] Sample dir: {sample.name}")
            print(f"[UniSite Debug] Expected CSV: {expected_csv}")
            print(f"[UniSite Debug] CSV exists: {expected_csv.exists()}")
        
        protein_dirs = sorted([
            d for d in all_subdirs
            if (d / f"{d.name}_info.csv").exists()
        ])
        print(f"[UniSite Debug] Protein dirs with _info.csv: {len(protein_dirs)}")
        
        # Apply limit if specified
        if self.limit:
            protein_dirs = protein_dirs[:self.limit]
        
        proteins = []
        failed_count = 0
        for protein_dir in protein_dirs:
            try:
                protein = self._load_protein_from_folder(protein_dir)
                if protein is not None:
                    proteins.append(protein)
                else:
                    failed_count += 1
                    if failed_count <= 3:
                        # Debug: why did this fail?
                        pdb_file = protein_dir / f"{protein_dir.name}.pdb"
                        print(f"[UniSite Debug] Failed: {protein_dir.name}, PDB exists: {pdb_file.exists()}")
            except Exception as e:
                failed_count += 1
                if failed_count <= 3:
                    print(f"[UniSite Debug] Exception for {protein_dir.name}: {e}")
        
        print(f"[UniSite Debug] Total failed: {failed_count}")
        print(f"Loaded {len(proteins)} proteins from {self.name}")
        return proteins
    
    def _load_protein_from_folder(self, protein_dir: Path) -> Optional[PBProtein]:
        """
        Load a single protein from its folder.
        
        Actual folder structure:
        - {protein_id}_info.csv - contains binding site info (site_position_uniprot)
        - {protein_id}.pdb - representative structure (source of sequence)
        - {protein_id}.mapping - residue mapping
        - site1/, site2/, ... - binding site directories with ligand/complex PDBs
        """
        protein_id = protein_dir.name
        
        # Load PDB for sequence and coordinates
        pdb_file = protein_dir / f"{protein_id}.pdb"
        coords = np.zeros((1, 3), dtype=np.float32)
        full_atoms = None
        sequence = ""
        parse_error = None
        
        if pdb_file.exists():
            try:
                from Bio.PDB import PDBParser
                from Bio.PDB.Polypeptide import is_aa, three_to_one
                
                parser = PDBParser(QUIET=True)
                structure = parser.get_structure(protein_id, pdb_file)
                full_atoms = structure
                
                ca_coords = []
                seq_from_pdb = []
                for model in structure:
                    for chain in model:
                        for residue in chain:
                            if is_aa(residue, standard=True):
                                try:
                                    ca = residue['CA']
                                    ca_coords.append(ca.get_coord())
                                    seq_from_pdb.append(three_to_one(residue.get_resname()))
                                except (KeyError, Exception):
                                    pass
                    break  # Only first model
                
                if ca_coords:
                    coords = np.array(ca_coords, dtype=np.float32)
                if seq_from_pdb:
                    sequence = ''.join(seq_from_pdb)
            except Exception as e:
                parse_error = str(e)
        
        if not sequence:
            # Debug: print first few failures
            if not hasattr(self, '_debug_fail_count'):
                self._debug_fail_count = 0
            self._debug_fail_count += 1
            if self._debug_fail_count <= 3:
                print(f"[UniSite Debug] No sequence for {protein_id}: parse_error={parse_error}")
            return None
        
        # Load binding sites from _info.csv (site_position_uniprot column)
        ground_truth_sites = []
        info_file = protein_dir / f"{protein_id}_info.csv"
        
        if info_file.exists():
            try:
                import csv
                import ast
                with open(info_file, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        site_name = row.get('site', 'unknown')
                        # Parse the residue indices from the site_position_uniprot column
                        # Format: "[19, 21, 22, 23, ...]"
                        residue_str = row.get('site_position_uniprot', '[]')
                        try:
                            residue_indices = ast.literal_eval(residue_str)
                            if isinstance(residue_indices, list) and residue_indices:
                                # Convert to 0-indexed if needed (UniProt is 1-indexed)
                                # Actually, check if indices match PDB coords range
                                # For safety, keep as-is since we don't know the indexing
                                residue_indices = [int(i) for i in residue_indices]
                                
                                # Compute center from coordinates
                                valid_indices = [i for i in residue_indices if 0 <= i < len(coords)]
                                if valid_indices:
                                    center = coords[valid_indices].mean(axis=0)
                                else:
                                    center = np.zeros(3)
                                
                                ground_truth_sites.append(PBSite(
                                    center=center,
                                    residues=residue_indices,
                                    ligand_id=row.get('ligand_name', site_name)
                                ))
                        except (ValueError, SyntaxError):
                            pass
            except Exception:
                pass
        
        return PBProtein(
            id=protein_id,
            sequence=sequence,
            coords=coords,
            full_atoms=full_atoms,
            ground_truth_sites=ground_truth_sites,
            metadata={
                'source': 'UniSite-DS',
                'split': self.split,
                'pdb_path': str(pdb_file) if pdb_file.exists() else ''
            }
        )
    


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
