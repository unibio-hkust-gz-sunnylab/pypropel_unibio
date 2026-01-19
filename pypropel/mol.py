__author__ = "Jianfeng Sun"
__version__ = "v1.0"
__copyright__ = "Copyright 2024"
__license__ = "GPL v3.0"
__email__ = "jianfeng.sunmt@gmail.com"
__maintainer__ = "Jianfeng Sun"

"""
Small molecule (ligand) utilities for pypropel.

Provides functions for loading and processing small molecule files
such as SDF and MOL2 formats using RDKit.
"""

from typing import Optional
import numpy as np

try:
    from rdkit import Chem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False


def load_sdf(sdf_path: str):
    """
    Load a ligand from an SDF file.
    
    Parameters
    ----------
    sdf_path : str
        Path to the SDF file.
        
    Returns
    -------
    rdkit.Chem.rdchem.Mol or None
        RDKit molecule object, or None if loading fails.
        
    Examples
    --------
    >>> import pypropel.mol as ppmol
    >>> mol = ppmol.load_sdf('/path/to/ligand.sdf')
    """
    if not RDKIT_AVAILABLE:
        raise ImportError("RDKit is required for ligand loading. Install with: pip install rdkit")
    
    suppl = Chem.SDMolSupplier(sdf_path)
    for mol in suppl:
        if mol is not None:
            return mol
    return None


def load_mol2(mol2_path: str):
    """
    Load a ligand from a MOL2 file.
    
    Parameters
    ----------
    mol2_path : str
        Path to the MOL2 file.
        
    Returns
    -------
    rdkit.Chem.rdchem.Mol or None
        RDKit molecule object, or None if loading fails.
        
    Examples
    --------
    >>> import pypropel.mol as ppmol
    >>> mol = ppmol.load_mol2('/path/to/ligand.mol2')
    """
    if not RDKIT_AVAILABLE:
        raise ImportError("RDKit is required for ligand loading. Install with: pip install rdkit")
    
    return Chem.MolFromMol2File(mol2_path)


def ligand_coords(mol) -> np.ndarray:
    """
    Extract 3D coordinates from an RDKit molecule.
    
    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        RDKit molecule object with 3D conformer.
        
    Returns
    -------
    np.ndarray
        Atom coordinates with shape (N, 3).
        
    Examples
    --------
    >>> import pypropel.mol as ppmol
    >>> mol = ppmol.load_sdf('/path/to/ligand.sdf')
    >>> coords = ppmol.ligand_coords(mol)
    >>> print(coords.shape)  # (N_atoms, 3)
    """
    if mol is None:
        return np.array([]).reshape(0, 3)
    
    conf = mol.GetConformer()
    coords = []
    for i in range(mol.GetNumAtoms()):
        pos = conf.GetAtomPosition(i)
        coords.append([pos.x, pos.y, pos.z])
    return np.array(coords)


def get_ligand_features(mol) -> np.ndarray:
    """
    Extract global ligand features using RDKit descriptors.
    
    Returns [LogP, MolWt (normalized), H-Donors, H-Acceptors].
    
    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        RDKit molecule object.
        
    Returns
    -------
    np.ndarray
        4-dimensional feature vector.
        
    Examples
    --------
    >>> import pypropel.mol as ppmol
    >>> mol = ppmol.load_sdf('/path/to/ligand.sdf')
    >>> features = ppmol.get_ligand_features(mol)
    >>> print(features.shape)  # (4,)
    """
    if not RDKIT_AVAILABLE:
        raise ImportError("RDKit is required for ligand features. Install with: pip install rdkit")
    
    from rdkit.Chem import Descriptors, Crippen
    
    if mol is None:
        return np.zeros(4, dtype=np.float32)
    
    logp = Crippen.MolLogP(mol)
    mw = Descriptors.MolWt(mol) / 500.0  # Normalize roughly
    h_donors = Descriptors.NumHDonors(mol)
    h_acceptors = Descriptors.NumHAcceptors(mol)
    
    return np.array([logp, mw, h_donors, h_acceptors], dtype=np.float32)


if __name__ == "__main__":
    # Quick test
    print("mol.py module loaded successfully")
    print(f"RDKit available: {RDKIT_AVAILABLE}")
