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


# ==================== Atom-Level Features ====================

# Atom type vocabulary for one-hot encoding
ATOM_TYPES = ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I', 'Other']
ATOM_TYPE_MAP = {atom: i for i, atom in enumerate(ATOM_TYPES[:-1])}

# Hybridization vocabulary
HYBRIDIZATION_TYPES = ['SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'AROMATIC', 'OTHER']


def get_atom_type_onehot(mol) -> np.ndarray:
    """
    Get one-hot encoded atom types for all atoms in a molecule.
    
    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        RDKit molecule object.
        
    Returns
    -------
    np.ndarray
        One-hot encoded atom types, shape (N_atoms, 10).
        Types: C, N, O, S, F, P, Cl, Br, I, Other
        
    Examples
    --------
    >>> import pypropel.mol as ppmol
    >>> mol = ppmol.load_sdf('/path/to/ligand.sdf')
    >>> atom_types = ppmol.get_atom_type_onehot(mol)
    >>> print(atom_types.shape)  # (N_atoms, 10)
    """
    if mol is None:
        return np.zeros((0, len(ATOM_TYPES)), dtype=np.float32)
    
    n_atoms = mol.GetNumAtoms()
    features = np.zeros((n_atoms, len(ATOM_TYPES)), dtype=np.float32)
    
    for i, atom in enumerate(mol.GetAtoms()):
        symbol = atom.GetSymbol()
        if symbol in ATOM_TYPE_MAP:
            features[i, ATOM_TYPE_MAP[symbol]] = 1.0
        else:
            features[i, -1] = 1.0  # Other
    
    return features


def get_atom_hybridization(mol) -> np.ndarray:
    """
    Get one-hot encoded hybridization states for all atoms.
    
    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        RDKit molecule object.
        
    Returns
    -------
    np.ndarray
        One-hot hybridization, shape (N_atoms, 7).
        Types: SP, SP2, SP3, SP3D, SP3D2, AROMATIC, OTHER
        
    Examples
    --------
    >>> hybridization = ppmol.get_atom_hybridization(mol)
    >>> print(hybridization.shape)  # (N_atoms, 7)
    """
    if not RDKIT_AVAILABLE:
        raise ImportError("RDKit is required. Install with: pip install rdkit")
    
    from rdkit.Chem import HybridizationType
    
    if mol is None:
        return np.zeros((0, len(HYBRIDIZATION_TYPES)), dtype=np.float32)
    
    hybridization_map = {
        HybridizationType.SP: 0,
        HybridizationType.SP2: 1,
        HybridizationType.SP3: 2,
        HybridizationType.SP3D: 3,
        HybridizationType.SP3D2: 4,
    }
    
    n_atoms = mol.GetNumAtoms()
    features = np.zeros((n_atoms, len(HYBRIDIZATION_TYPES)), dtype=np.float32)
    
    for i, atom in enumerate(mol.GetAtoms()):
        hyb = atom.GetHybridization()
        if atom.GetIsAromatic():
            features[i, 5] = 1.0  # AROMATIC
        elif hyb in hybridization_map:
            features[i, hybridization_map[hyb]] = 1.0
        else:
            features[i, -1] = 1.0  # OTHER
    
    return features


def get_atom_hbond_features(mol) -> np.ndarray:
    """
    Get hydrogen bond donor/acceptor features for all atoms.
    
    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        RDKit molecule object.
        
    Returns
    -------
    np.ndarray
        H-bond features, shape (N_atoms, 2).
        Columns: [is_donor, is_acceptor]
        
    Examples
    --------
    >>> hbond = ppmol.get_atom_hbond_features(mol)
    >>> print(hbond.shape)  # (N_atoms, 2)
    """
    if not RDKIT_AVAILABLE:
        raise ImportError("RDKit is required. Install with: pip install rdkit")
    
    from rdkit.Chem import Lipinski
    
    if mol is None:
        return np.zeros((0, 2), dtype=np.float32)
    
    n_atoms = mol.GetNumAtoms()
    features = np.zeros((n_atoms, 2), dtype=np.float32)
    
    # Get donor and acceptor atom indices
    donor_smarts = Chem.MolFromSmarts('[!$([#6,H0,-,-2,-3])]')
    acceptor_smarts = Chem.MolFromSmarts('[!$([#6,F,Cl,Br,I,o,s,nX3,#7v5,#15v5,#16v4,#16v6,*+1,*+2,*+3])]')
    
    if donor_smarts:
        donor_matches = set(idx for match in mol.GetSubstructMatches(donor_smarts) for idx in match)
    else:
        donor_matches = set()
    
    if acceptor_smarts:
        acceptor_matches = set(idx for match in mol.GetSubstructMatches(acceptor_smarts) for idx in match)
    else:
        acceptor_matches = set()
    
    # Alternative: Use simple rules based on atom type and neighbors
    for i, atom in enumerate(mol.GetAtoms()):
        symbol = atom.GetSymbol()
        total_h = atom.GetTotalNumHs()
        
        # Simple H-bond donor: N or O with attached H
        if symbol in ['N', 'O'] and total_h > 0:
            features[i, 0] = 1.0
        
        # Simple H-bond acceptor: N, O, or F
        if symbol in ['N', 'O', 'F']:
            features[i, 1] = 1.0
    
    return features


def get_atom_aromaticity(mol) -> np.ndarray:
    """
    Get aromaticity flag for all atoms.
    
    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        RDKit molecule object.
        
    Returns
    -------
    np.ndarray
        Aromaticity flags, shape (N_atoms, 1).
        
    Examples
    --------
    >>> aromatic = ppmol.get_atom_aromaticity(mol)
    >>> print(aromatic.shape)  # (N_atoms, 1)
    """
    if mol is None:
        return np.zeros((0, 1), dtype=np.float32)
    
    n_atoms = mol.GetNumAtoms()
    features = np.zeros((n_atoms, 1), dtype=np.float32)
    
    for i, atom in enumerate(mol.GetAtoms()):
        features[i, 0] = 1.0 if atom.GetIsAromatic() else 0.0
    
    return features


def get_atom_partial_charges(mol) -> np.ndarray:
    """
    Get Gasteiger partial charges for all atoms.
    
    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        RDKit molecule object.
        
    Returns
    -------
    np.ndarray
        Partial charges, shape (N_atoms, 1).
        
    Examples
    --------
    >>> charges = ppmol.get_atom_partial_charges(mol)
    >>> print(charges.shape)  # (N_atoms, 1)
    """
    if not RDKIT_AVAILABLE:
        raise ImportError("RDKit is required. Install with: pip install rdkit")
    
    from rdkit.Chem import AllChem
    
    if mol is None:
        return np.zeros((0, 1), dtype=np.float32)
    
    # Compute Gasteiger charges
    try:
        AllChem.ComputeGasteigerCharges(mol)
    except Exception:
        # Return zeros if charge computation fails
        return np.zeros((mol.GetNumAtoms(), 1), dtype=np.float32)
    
    n_atoms = mol.GetNumAtoms()
    features = np.zeros((n_atoms, 1), dtype=np.float32)
    
    for i, atom in enumerate(mol.GetAtoms()):
        charge = atom.GetDoubleProp('_GasteigerCharge')
        # Handle NaN values
        if np.isnan(charge):
            charge = 0.0
        features[i, 0] = charge
    
    return features


def get_atom_features(mol) -> np.ndarray:
    """
    Get combined atom-level features for GIN encoder.
    
    Concatenates all atom features into a single feature matrix.
    
    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        RDKit molecule object.
        
    Returns
    -------
    np.ndarray
        Combined features, shape (N_atoms, D_lig).
        D_lig = 10 (type) + 7 (hybrid) + 2 (hbond) + 1 (aromatic) + 1 (charge) = 21
        
    Examples
    --------
    >>> atom_features = ppmol.get_atom_features(mol)
    >>> print(atom_features.shape)  # (N_atoms, 21)
    """
    if mol is None:
        return np.zeros((0, 21), dtype=np.float32)
    
    type_feat = get_atom_type_onehot(mol)      # (N, 10)
    hybrid_feat = get_atom_hybridization(mol)  # (N, 7)
    hbond_feat = get_atom_hbond_features(mol)  # (N, 2)
    arom_feat = get_atom_aromaticity(mol)      # (N, 1)
    charge_feat = get_atom_partial_charges(mol) # (N, 1)
    
    return np.concatenate([
        type_feat,
        hybrid_feat,
        hbond_feat,
        arom_feat,
        charge_feat
    ], axis=1).astype(np.float32)


# ==================== Morgan Fingerprints ====================

def get_morgan_fingerprint(
    mol, 
    radius: int = 2, 
    n_bits: int = 2048
) -> np.ndarray:
    """
    Get Morgan fingerprint (ECFP) for a molecule.
    
    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        RDKit molecule object.
    radius : int
        Fingerprint radius. radius=2 gives ECFP4.
    n_bits : int
        Number of bits in fingerprint.
        
    Returns
    -------
    np.ndarray
        Binary fingerprint, shape (n_bits,).
        
    Examples
    --------
    >>> fp = ppmol.get_morgan_fingerprint(mol, radius=2, n_bits=2048)
    >>> print(fp.shape)  # (2048,)
    """
    if not RDKIT_AVAILABLE:
        raise ImportError("RDKit is required. Install with: pip install rdkit")
    
    from rdkit.Chem import AllChem
    
    if mol is None:
        return np.zeros(n_bits, dtype=np.float32)
    
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return np.array(fp, dtype=np.float32)


def get_morgan_fingerprint_compressed(
    mol, 
    radius: int = 2, 
    output_dim: int = 128, 
    seed: int = 42
) -> np.ndarray:
    """
    Get compressed Morgan fingerprint using random projection.
    
    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        RDKit molecule object.
    radius : int
        Fingerprint radius.
    output_dim : int
        Dimension of compressed fingerprint.
    seed : int
        Random seed for reproducibility.
        
    Returns
    -------
    np.ndarray
        Compressed fingerprint, shape (output_dim,).
        
    Examples
    --------
    >>> fp_compressed = ppmol.get_morgan_fingerprint_compressed(mol, output_dim=128)
    >>> print(fp_compressed.shape)  # (128,)
    """
    if not RDKIT_AVAILABLE:
        raise ImportError("RDKit is required. Install with: pip install rdkit")
    
    from rdkit.Chem import AllChem
    
    if mol is None:
        return np.zeros(output_dim, dtype=np.float32)
    
    # Get full fingerprint as a count vector
    fp = AllChem.GetMorganFingerprint(mol, radius)
    fp_dict = fp.GetNonzeroElements()
    
    # Use deterministic random projection for compression
    np.random.seed(seed)
    result = np.zeros(output_dim, dtype=np.float32)
    
    for idx, count in fp_dict.items():
        # Hash each feature to output dimensions
        np.random.seed((seed + idx) % (2**31))
        projection = np.random.randn(output_dim)
        result += count * projection / np.sqrt(output_dim)
    
    return result.astype(np.float32)


def get_ligand_global_features(mol) -> np.ndarray:
    """
    Get extended global ligand features including Morgan fingerprint.
    
    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        RDKit molecule object.
        
    Returns
    -------
    np.ndarray
        Global features, shape (132,).
        [LogP, MolWt, HBD, HBA] (4) + Morgan FP compressed (128) = 132
        
    Examples
    --------
    >>> global_feat = ppmol.get_ligand_global_features(mol)
    >>> print(global_feat.shape)  # (132,)
    """
    basic_feat = get_ligand_features(mol)  # (4,)
    morgan_feat = get_morgan_fingerprint_compressed(mol, output_dim=128)  # (128,)
    
    return np.concatenate([basic_feat, morgan_feat]).astype(np.float32)


if __name__ == "__main__":
    # Quick test
    print("mol.py module loaded successfully")
    print(f"RDKit available: {RDKIT_AVAILABLE}")

