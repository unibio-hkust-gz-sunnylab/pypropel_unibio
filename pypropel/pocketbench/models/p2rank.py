"""
P2Rank model wrapper for PocketBench.

P2Rank is a machine learning-based tool for prediction of ligand binding
sites from protein structure. It uses Random Forest on local features
extracted from SAS (Solvent Accessible Surface) points.

Source: https://github.com/rdk/p2rank
"""

__author__ = "pypropel team"
__version__ = "0.1.0"

from typing import List, Optional, Union
from pathlib import Path
import subprocess
import tempfile
import csv
import re

import numpy as np

from ..core import PBProtein, PBPrediction
from .base import PBModelWrapper


# Default P2Rank download URL
P2RANK_RELEASE_URL = "https://github.com/rdk/p2rank/releases"


class P2RankWrapper(PBModelWrapper):
    """
    Wrapper for P2Rank binding site prediction tool.
    
    P2Rank is an external Java tool that must be downloaded separately.
    This wrapper handles:
    - Writing input PDB files
    - Running the P2Rank JAR via subprocess
    - Parsing CSV output to PBPrediction format
    
    Examples
    --------
    >>> from pypropel.pocketbench.models import P2RankWrapper
    >>> p2rank = P2RankWrapper(p2rank_home="/path/to/p2rank")
    >>> predictions = p2rank.predict(protein)
    >>> for pred in predictions:
    ...     print(f"Pocket at {pred.center}, score={pred.confidence:.3f}")
    
    Notes
    -----
    Requires Java 11+ to be installed and in PATH.
    Download P2Rank from: https://github.com/rdk/p2rank/releases
    """
    
    def __init__(
        self,
        p2rank_home: Optional[Union[str, Path]] = None,
        config: str = "default",
        temp_dir: Optional[Union[str, Path]] = None,
        cleanup: bool = True,
        java_path: str = "java",
        threads: int = 1
    ):
        """
        Initialize P2Rank wrapper.
        
        Parameters
        ----------
        p2rank_home : str or Path, optional
            Path to P2Rank installation directory containing prank.sh/prank.bat.
            If None, looks for P2RANK_HOME environment variable.
        config : str, optional
            P2Rank configuration profile. Options:
            - "default": Standard prediction
            - "alphafold": For AlphaFold/NMR/cryo-EM structures
        temp_dir : str or Path, optional
            Directory for temporary files.
        cleanup : bool, optional
            Whether to cleanup temporary files. Default True.
        java_path : str, optional
            Path to Java executable. Default "java".
        threads : int, optional
            Number of threads for prediction. Default 1.
        """
        super().__init__(model_path=p2rank_home, temp_dir=temp_dir, cleanup=cleanup)
        
        # Find P2Rank home
        if p2rank_home:
            self.p2rank_home = Path(p2rank_home)
        elif os.environ.get("P2RANK_HOME"):
            self.p2rank_home = Path(os.environ["P2RANK_HOME"])
        else:
            self.p2rank_home = None
        
        self.config = config
        self.java_path = java_path
        self.threads = threads
    
    @property
    def name(self) -> str:
        return "P2Rank"
    
    def _find_prank_script(self) -> Optional[Path]:
        """Find the prank executable script."""
        if not self.p2rank_home:
            return None
        
        # Try different possible script names
        scripts = ["prank", "prank.sh", "prank.bat"]
        for script in scripts:
            path = self.p2rank_home / script
            if path.exists():
                return path
        
        # Try bin subdirectory
        for script in scripts:
            path = self.p2rank_home / "bin" / script
            if path.exists():
                return path
        
        return None
    
    def _check_available(self) -> bool:
        """Check if P2Rank is available and configured."""
        prank = self._find_prank_script()
        if prank is None:
            return False
        
        # Check Java
        try:
            result = subprocess.run(
                [self.java_path, "-version"],
                capture_output=True,
                timeout=10
            )
            return result.returncode == 0
        except Exception:
            return False
    
    def predict(self, protein: PBProtein) -> List[PBPrediction]:
        """
        Predict binding sites using P2Rank.
        
        Parameters
        ----------
        protein : PBProtein
            Input protein.
            
        Returns
        -------
        List[PBPrediction]
            Predicted binding sites sorted by confidence.
        """
        if not self._check_available():
            raise RuntimeError(
                f"P2Rank not available. Please set p2rank_home or P2RANK_HOME "
                f"environment variable. Download from: {P2RANK_RELEASE_URL}"
            )
        
        # Create temp directory for this prediction
        with tempfile.TemporaryDirectory(prefix="p2rank_") as temp_dir:
            temp_path = Path(temp_dir)
            
            # Write input PDB
            input_pdb = temp_path / f"{protein.id}.pdb"
            self._write_pdb(protein, input_pdb)
            
            # Run P2Rank
            output_dir = temp_path / "output"
            self._run_p2rank(input_pdb, output_dir)
            
            # Parse results
            predictions = self._parse_predictions(
                output_dir, protein.id, protein
            )
        
        return predictions
    
    def _run_p2rank(self, input_pdb: Path, output_dir: Path):
        """Run P2Rank on input PDB file."""
        prank = self._find_prank_script()
        
        # Build command
        cmd = [
            str(prank),
            "predict",
            "-f", str(input_pdb),
            "-o", str(output_dir),
            "-threads", str(self.threads),
            "-visualizations", "0",  # Disable visualizations for speed
        ]
        
        if self.config != "default":
            cmd.extend(["-c", self.config])
        
        # Run
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode != 0:
            raise RuntimeError(
                f"P2Rank failed with exit code {result.returncode}:\n"
                f"stdout: {result.stdout}\n"
                f"stderr: {result.stderr}"
            )
    
    def _parse_predictions(
        self, 
        output_dir: Path, 
        protein_id: str,
        protein: PBProtein
    ) -> List[PBPrediction]:
        """Parse P2Rank output CSV files."""
        predictions = []
        
        # Find predictions CSV
        predictions_file = output_dir / f"{protein_id}.pdb_predictions.csv"
        if not predictions_file.exists():
            # Try without .pdb extension
            predictions_file = output_dir / f"{protein_id}_predictions.csv"
        
        if not predictions_file.exists():
            # Search for any predictions file
            for f in output_dir.glob("*_predictions.csv"):
                predictions_file = f
                break
        
        if not predictions_file.exists():
            return predictions
        
        # Parse predictions CSV
        with open(predictions_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    pred = self._parse_prediction_row(row, protein)
                    if pred is not None:
                        predictions.append(pred)
                except Exception as e:
                    print(f"Warning: Failed to parse prediction row: {e}")
        
        # Sort by confidence (descending)
        predictions.sort(key=lambda p: p.confidence, reverse=True)
        
        return predictions
    
    def _parse_prediction_row(
        self, 
        row: dict,
        protein: PBProtein
    ) -> Optional[PBPrediction]:
        """Parse a single prediction row from P2Rank output."""
        # P2Rank CSV columns:
        # name, rank, score, probability, sas_points, surf_atoms,
        # center_x, center_y, center_z, residue_ids, surf_atom_ids
        
        # Get center coordinates
        try:
            center_x = float(row.get('center_x', 0))
            center_y = float(row.get('center_y', 0))
            center_z = float(row.get('center_z', 0))
            center = np.array([center_x, center_y, center_z])
        except (ValueError, TypeError):
            return None
        
        # Get score/probability as confidence
        try:
            # Prefer probability if available (calibrated)
            confidence = float(row.get('probability', row.get('score', 0)))
            # Normalize score if needed (P2Rank scores can be > 1)
            if confidence > 1:
                confidence = confidence / 100 if confidence > 100 else confidence / 10
            confidence = max(0.0, min(1.0, confidence))
        except (ValueError, TypeError):
            confidence = 0.5
        
        # Parse residue IDs
        residues = []
        residue_ids_str = row.get('residue_ids', '')
        if residue_ids_str:
            # Format: "A_1 A_2 A_3" or "1 2 3"
            for res_str in residue_ids_str.split():
                try:
                    # Try to extract number from various formats
                    match = re.search(r'(\d+)', res_str)
                    if match:
                        res_idx = int(match.group(1)) - 1  # Convert to 0-indexed
                        if 0 <= res_idx < len(protein.sequence):
                            residues.append(res_idx)
                except ValueError:
                    continue
        
        return PBPrediction(
            center=center,
            residues=sorted(set(residues)),
            confidence=confidence,
            model_name=self.name
        )
    
    def predict_batch(
        self, 
        proteins: List[PBProtein],
        show_progress: bool = True
    ) -> List[List[PBPrediction]]:
        """
        Predict binding sites for multiple proteins.
        
        Uses P2Rank's batch processing capability for efficiency.
        
        Parameters
        ----------
        proteins : List[PBProtein]
            List of proteins.
        show_progress : bool, optional
            Whether to show progress. Default True.
            
        Returns
        -------
        List[List[PBPrediction]]
            Predictions for each protein.
        """
        if not self._check_available():
            raise RuntimeError(f"P2Rank not available")
        
        results = []
        
        with tempfile.TemporaryDirectory(prefix="p2rank_batch_") as temp_dir:
            temp_path = Path(temp_dir)
            
            # Write all PDB files and create dataset file
            ds_file = temp_path / "batch.ds"
            pdb_dir = temp_path / "pdbs"
            pdb_dir.mkdir()
            
            with open(ds_file, 'w') as f:
                for protein in proteins:
                    pdb_file = pdb_dir / f"{protein.id}.pdb"
                    self._write_pdb(protein, pdb_file)
                    f.write(f"{protein.id}.pdb\n")
            
            # Run P2Rank on dataset
            output_dir = temp_path / "output"
            self._run_p2rank_batch(ds_file, pdb_dir, output_dir)
            
            # Parse results for each protein
            for protein in proteins:
                try:
                    preds = self._parse_predictions(output_dir, protein.id, protein)
                    results.append(preds)
                except Exception as e:
                    print(f"Warning: Failed to get predictions for {protein.id}: {e}")
                    results.append([])
        
        return results
    
    def _run_p2rank_batch(
        self, 
        ds_file: Path, 
        pdb_dir: Path,
        output_dir: Path
    ):
        """Run P2Rank on a dataset file."""
        prank = self._find_prank_script()
        
        cmd = [
            str(prank),
            "predict",
            str(ds_file),
            "-o", str(output_dir),
            "-threads", str(self.threads),
            "-visualizations", "0",
        ]
        
        if self.config != "default":
            cmd.extend(["-c", self.config])
        
        # Set working directory to pdb_dir
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(pdb_dir),
            timeout=3600  # 1 hour timeout for batch
        )
        
        if result.returncode != 0:
            raise RuntimeError(
                f"P2Rank batch failed: {result.stderr}"
            )


# Import for type alias
import os
