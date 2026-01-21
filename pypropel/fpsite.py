__version__ = "v1.0"
__copyright__ = "Copyright 2024"
__license__ = "GPL v3.0"
__developer__ = "Jianfeng Sun"
__maintainer__ = "Jianfeng Sun"
__email__="jianfeng.sunmt@gmail.com"

from typing import List, Dict
import numpy as np

from pypropel.prot.feature.sequence.AminoAcidProperty import AminoAcidProperty as aaprop
from pypropel.prot.feature.sequence.AminoAcidRepresentation import AminoAcidRepresentation as aarepr
from pypropel.prot.feature.sequence.Position import Position
from pypropel.prot.feature.rsa.Reader import Reader as rsareader
from pypropel.prot.feature.ss.Reader import Reader as ssreader


# ==================== Amino Acid Mappings ====================

AA_MAP = {
    'ALA': 0, 'ARG': 1, 'ASN': 2, 'ASP': 3, 'CYS': 4,
    'GLN': 5, 'GLU': 6, 'GLY': 7, 'HIS': 8, 'ILE': 9,
    'LEU': 10, 'LYS': 11, 'MET': 12, 'PHE': 13, 'PRO': 14,
    'SER': 15, 'THR': 16, 'TRP': 17, 'TYR': 18, 'VAL': 19
}

# Charge mapping (at physiological pH ~7.4)
CHARGE_MAP = {
    'ARG': 1.0, 'LYS': 1.0, 'HIS': 0.5,  # Basic (positive)
    'ASP': -1.0, 'GLU': -1.0,             # Acidic (negative)
}

# Kyte-Doolittle Hydrophobicity scale
KD_HYDROPHOBICITY = {
    'ILE': 4.5, 'VAL': 4.2, 'LEU': 3.8, 'PHE': 2.8, 'CYS': 2.5,
    'MET': 1.9, 'ALA': 1.8, 'GLY': -0.4, 'THR': -0.7, 'SER': -0.8,
    'TRP': -0.9, 'TYR': -1.3, 'PRO': -1.6, 'HIS': -3.2, 'GLU': -3.5,
    'GLN': -3.5, 'ASP': -3.5, 'ASN': -3.5, 'LYS': -3.9, 'ARG': -4.5
}


# ==================== Residue Feature Functions ====================

def residue_one_hot(residue_name: str) -> np.ndarray:
    """
    Generate a 20-dimensional one-hot encoding for an amino acid.
    
    Parameters
    ----------
    residue_name : str
        Three-letter amino acid code (e.g., 'ALA', 'GLY').
        
    Returns
    -------
    np.ndarray
        20-dimensional one-hot vector.
        
    Examples
    --------
    >>> import pypropel.fpsite as fpsite
    >>> vec = fpsite.residue_one_hot('ALA')
    >>> print(vec.shape)  # (20,)
    >>> print(vec[0])     # 1.0 (ALA is at index 0)
    """
    vec = np.zeros(20, dtype=np.float32)
    if residue_name in AA_MAP:
        vec[AA_MAP[residue_name]] = 1.0
    return vec


def residue_physchem(residue_name: str) -> np.ndarray:
    """
    Get physicochemical properties for an amino acid.
    
    Returns [Charge, Hydrophobicity (normalized)].
    - Charge: -1 (Acidic), +1 (Basic), 0 (Neutral), 0.5 (His at pH 7)
    - Hydrophobicity: Kyte-Doolittle scale normalized to [-1, 1]
    
    Parameters
    ----------
    residue_name : str
        Three-letter amino acid code.
        
    Returns
    -------
    np.ndarray
        2-dimensional feature vector [charge, hydrophobicity].
        
    Examples
    --------
    >>> import pypropel.fpsite as fpsite
    >>> props = fpsite.residue_physchem('ARG')
    >>> print(props)  # [1.0, -1.0] (basic, hydrophilic)
    """
    charge = CHARGE_MAP.get(residue_name, 0.0)
    hydro = KD_HYDROPHOBICITY.get(residue_name, 0.0) / 4.5  # Normalize to ~[-1, 1]
    return np.array([charge, hydro], dtype=np.float32)


def residue_coords(residue) -> np.ndarray:
    """
    Extract atom coordinates from a BioPython residue object.
    
    Parameters
    ----------
    residue : Bio.PDB.Residue.Residue
        BioPython residue object.
        
    Returns
    -------
    np.ndarray
        Atom coordinates with shape (N, 3).
        
    Examples
    --------
    >>> import pypropel.fpsite as fpsite
    >>> from Bio.PDB import PDBParser
    >>> parser = PDBParser(QUIET=True)
    >>> struct = parser.get_structure('test', 'protein.pdb')
    >>> for model in struct:
    ...     for chain in model:
    ...         for residue in chain:
    ...             coords = fpsite.residue_coords(residue)
    ...             print(coords.shape)
    """
    coords = []
    for atom in residue:
        coords.append(atom.get_coord())
    return np.array(coords) if coords else np.array([]).reshape(0, 3)


# ==================== AA Validation ====================

STANDARD_AA_NAMES = set(AA_MAP.keys())
ONE_LETTER_CODES = set('ACDEFGHIKLMNPQRSTVWY')


def is_standard_aa(residue, standard: bool = True) -> bool:
    """
    Check if a BioPython residue is a standard amino acid.
    
    Wraps Bio.PDB.Polypeptide.is_aa() for full pypropel encapsulation.
    
    Parameters
    ----------
    residue : Bio.PDB.Residue.Residue
        BioPython residue object.
    standard : bool
        If True, only accept 20 standard amino acids.
        
    Returns
    -------
    bool
        True if residue is a (standard) amino acid.
    """
    from Bio.PDB import Polypeptide
    return Polypeptide.is_aa(residue, standard=standard)


def is_standard_aa_name(name: str) -> bool:
    """
    Check if a residue name is a standard amino acid.
    
    Parameters
    ----------
    name : str
        Three-letter amino acid code (e.g., 'ALA') or one-letter code (e.g., 'A').
        
    Returns
    -------
    bool
        True if name is a standard amino acid.
    """
    if len(name) == 1:
        return name.upper() in ONE_LETTER_CODES
    return name.upper() in STANDARD_AA_NAMES


# ==================== Sequence Window Functions ====================

def get_residue_window(seq_length: int, pos: int, k: int = 5) -> List[int]:
    """
    Get indices of k-nearest neighbors in sequence (1D window).
    
    Parameters
    ----------
    seq_length : int
        Total length of the sequence.
    pos : int
        Current position (0-indexed).
    k : int
        Window size (returns 2*k+1 positions centered at pos).
        
    Returns
    -------
    List[int]
        List of indices [pos-k, ..., pos, ..., pos+k].
        Boundary positions are clipped to valid range.
        
    Examples
    --------
    >>> fpsite.get_residue_window(100, 5, k=3)
    [2, 3, 4, 5, 6, 7, 8]
    >>> fpsite.get_residue_window(100, 0, k=3)  # At start
    [0, 0, 0, 0, 1, 2, 3]  # Padded
    """
    indices = []
    for offset in range(-k, k + 1):
        idx = pos + offset
        idx = max(0, min(seq_length - 1, idx))  # Clip to valid range
        indices.append(idx)
    return indices


def get_residue_window_padded(seq_length: int, pos: int, k: int = 5, 
                               pad_value: int = -1) -> List[int]:
    """
    Get indices of k-nearest neighbors with padding for boundaries.
    
    Parameters
    ----------
    seq_length : int
        Total length of the sequence.
    pos : int
        Current position (0-indexed).
    k : int
        Window size.
    pad_value : int
        Value to use for out-of-bounds positions.
        
    Returns
    -------
    List[int]
        List of indices with pad_value for out-of-bounds.
    """
    indices = []
    for offset in range(-k, k + 1):
        idx = pos + offset
        if 0 <= idx < seq_length:
            indices.append(idx)
        else:
            indices.append(pad_value)
    return indices


# ==================== Spatial Neighbor Functions ====================

def get_spatial_neighbors(structure, target_residue, k: int = 10, 
                          use_ca: bool = True) -> List[tuple]:
    """
    Find k spatially nearest residues to a target residue (3D neighbors).
    
    Parameters
    ----------
    structure : Bio.PDB.Structure.Structure
        BioPython structure object.
    target_residue : Bio.PDB.Residue.Residue
        The residue to find neighbors for.
    k : int
        Number of nearest neighbors to return.
    use_ca : bool
        If True, use C-alpha distances. Otherwise, use minimum atom distance.
        
    Returns
    -------
    List[tuple]
        List of (residue, distance) tuples sorted by distance.
        
    Examples
    --------
    >>> neighbors = fpsite.get_spatial_neighbors(structure, residue, k=5)
    >>> for res, dist in neighbors:
    ...     print(f"{res.get_resname()}: {dist:.2f} Å")
    """
    from Bio.PDB import Polypeptide
    
    # Get target coordinates
    if use_ca and 'CA' in target_residue:
        target_coord = target_residue['CA'].get_coord()
    else:
        target_coord = residue_coords(target_residue).mean(axis=0)
    
    # Collect all residues with distances
    distances = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if not Polypeptide.is_aa(residue, standard=True):
                    continue
                if residue == target_residue:
                    continue
                    
                if use_ca and 'CA' in residue:
                    res_coord = residue['CA'].get_coord()
                else:
                    coords = residue_coords(residue)
                    if len(coords) == 0:
                        continue
                    res_coord = coords.mean(axis=0)
                
                dist = np.linalg.norm(target_coord - res_coord)
                distances.append((residue, dist))
    
    # Sort by distance and return top k
    distances.sort(key=lambda x: x[1])
    return distances[:k]


def get_spatial_neighbor_indices(structure, target_idx: int, k: int = 10,
                                  use_ca: bool = True) -> List[tuple]:
    """
    Find k spatially nearest residue indices (3D neighbors by index).
    
    Parameters
    ----------
    structure : Bio.PDB.Structure.Structure
        BioPython structure object.
    target_idx : int
        Index of the target residue (0-indexed in residue list).
    k : int
        Number of nearest neighbors.
    use_ca : bool
        Use C-alpha distances.
        
    Returns
    -------
    List[tuple]
        List of (residue_index, distance) tuples sorted by distance.
    """
    from Bio.PDB import Polypeptide
    
    # Build residue list
    residues = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if Polypeptide.is_aa(residue, standard=True):
                    residues.append(residue)
    
    if target_idx >= len(residues):
        return []
    
    target_residue = residues[target_idx]
    
    if use_ca and 'CA' in target_residue:
        target_coord = target_residue['CA'].get_coord()
    else:
        target_coord = residue_coords(target_residue).mean(axis=0)
    
    distances = []
    for idx, residue in enumerate(residues):
        if idx == target_idx:
            continue
            
        if use_ca and 'CA' in residue:
            res_coord = residue['CA'].get_coord()
        else:
            coords = residue_coords(residue)
            if len(coords) == 0:
                continue
            res_coord = coords.mean(axis=0)
        
        dist = np.linalg.norm(target_coord - res_coord)
        distances.append((idx, dist))
    
    distances.sort(key=lambda x: x[1])
    return distances[:k]


# ==================== Positional Encoding ====================

def positional_encoding(pos: int, max_len: int, dim: int = 64, 
                        mode: str = 'sinusoidal') -> np.ndarray:
    """
    Generate positional encoding for transformer-style models.
    
    Parameters
    ----------
    pos : int
        Position in sequence (0-indexed).
    max_len : int
        Maximum sequence length.
    dim : int
        Dimension of the encoding.
    mode : str
        Encoding mode: 'sinusoidal', 'absolute', 'relative'.
        
    Returns
    -------
    np.ndarray
        Positional encoding vector.
        
    Examples
    --------
    >>> pe = fpsite.positional_encoding(5, 100, dim=64, mode='sinusoidal')
    >>> print(pe.shape)  # (64,)
    """
    if mode == 'sinusoidal':
        return _sinusoidal_encoding(pos, dim)
    elif mode == 'absolute':
        return _absolute_encoding(pos, max_len, dim)
    elif mode == 'relative':
        return _relative_encoding(pos, max_len, dim)
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'sinusoidal', 'absolute', or 'relative'")


def _sinusoidal_encoding(pos: int, dim: int) -> np.ndarray:
    """Standard sinusoidal positional encoding (Vaswani et al.)."""
    pe = np.zeros(dim, dtype=np.float32)
    for i in range(0, dim, 2):
        div_term = np.exp(i * (-np.log(10000.0) / dim))
        pe[i] = np.sin(pos * div_term)
        if i + 1 < dim:
            pe[i + 1] = np.cos(pos * div_term)
    return pe


def _absolute_encoding(pos: int, max_len: int, dim: int) -> np.ndarray:
    """Absolute position as normalized value + learnable features placeholder."""
    pe = np.zeros(dim, dtype=np.float32)
    pe[0] = pos / max_len  # Normalized position
    # Can be extended with learned embeddings
    return pe


def _relative_encoding(pos: int, max_len: int, dim: int) -> np.ndarray:
    """Relative position encoding with multiple scales."""
    pe = np.zeros(dim, dtype=np.float32)
    rel_pos = pos / max_len  # [0, 1]
    
    # Multiple scales of relative position
    for i in range(dim):
        scale = 2 ** (i // 2)
        if i % 2 == 0:
            pe[i] = np.sin(rel_pos * np.pi * scale)
        else:
            pe[i] = np.cos(rel_pos * np.pi * scale)
    return pe


def relative_position(pos: int, total_len: int) -> float:
    """
    Get normalized relative position in sequence [0, 1].
    
    Parameters
    ----------
    pos : int
        Position (0-indexed).
    total_len : int
        Total sequence length.
        
    Returns
    -------
    float
        Normalized position.
    """
    if total_len <= 1:
        return 0.0
    return pos / (total_len - 1)


def batch_positional_encoding(seq_length: int, dim: int = 64,
                               mode: str = 'sinusoidal') -> np.ndarray:
    """
    Generate positional encodings for an entire sequence.
    
    Parameters
    ----------
    seq_length : int
        Length of the sequence.
    dim : int
        Dimension of each encoding.
    mode : str
        Encoding mode.
        
    Returns
    -------
    np.ndarray
        Shape (seq_length, dim) positional encodings.
    """
    encodings = np.zeros((seq_length, dim), dtype=np.float32)
    for i in range(seq_length):
        encodings[i] = positional_encoding(i, seq_length, dim, mode)
    return encodings


# ==================== SASA and Secondary Structure ====================

# Secondary structure mapping (DSSP codes to categories)
SS_HELIX_CODES = {'H', 'G', 'I'}  # Alpha-helix, 3-10 helix, Pi-helix
SS_SHEET_CODES = {'E', 'B'}       # Beta-sheet, Beta-bridge
SS_COIL_CODES = {'T', 'S', '-', ' ', 'C'}  # Turn, Bend, Coil/Loop


def residue_secondary_structure(ss_code: str) -> np.ndarray:
    """
    One-hot encode secondary structure into 3 categories.
    
    Parameters
    ----------
    ss_code : str
        Single-letter DSSP secondary structure code.
        H/G/I = Helix, E/B = Sheet, T/S/-/C = Coil
        
    Returns
    -------
    np.ndarray
        3-dimensional one-hot vector [Helix, Sheet, Coil].
        
    Examples
    --------
    >>> import pypropel.fpsite as fpsite
    >>> fpsite.residue_secondary_structure('H')
    array([1., 0., 0.], dtype=float32)
    >>> fpsite.residue_secondary_structure('E')
    array([0., 1., 0.], dtype=float32)
    >>> fpsite.residue_secondary_structure('-')
    array([0., 0., 1.], dtype=float32)
    """
    vec = np.zeros(3, dtype=np.float32)
    ss_code = ss_code.upper() if ss_code else '-'
    
    if ss_code in SS_HELIX_CODES:
        vec[0] = 1.0  # Helix
    elif ss_code in SS_SHEET_CODES:
        vec[1] = 1.0  # Sheet
    else:
        vec[2] = 1.0  # Coil (default)
    
    return vec


def get_residue_sasa(structure, chain_id: str = None) -> List[Dict]:
    """
    Extract per-residue SASA (Solvent Accessible Surface Area) using DSSP.
    
    Parameters
    ----------
    structure : Bio.PDB.Structure.Structure
        BioPython structure object.
    chain_id : str, optional
        Chain ID to extract. If None, extracts all chains.
        
    Returns
    -------
    List[Dict]
        List of dicts with keys: 'chain', 'res_id', 'res_name', 'sasa', 'ss'
        
    Examples
    --------
    >>> import pypropel.fpsite as fpsite
    >>> import pypropel.str as ppstr
    >>> structure = ppstr.load_pdb('/path/to/protein.pdb')
    >>> sasa_data = fpsite.get_residue_sasa(structure)
    >>> for res in sasa_data[:3]:
    ...     print(f"{res['res_name']}: SASA={res['sasa']:.2f}, SS={res['ss']}")
    """
    from Bio.PDB import Polypeptide
    
    try:
        from Bio.PDB.DSSP import DSSP
    except ImportError:
        raise ImportError("BioPython DSSP module required. Install with: pip install biopython")
    
    # Get model (first one)
    model = structure[0]
    
    # Run DSSP - requires dssp/mkdssp installed
    try:
        dssp = DSSP(model, structure.get_id(), dssp='mkdssp')
    except Exception as e:
        # Try alternative dssp binary name
        try:
            dssp = DSSP(model, structure.get_id(), dssp='dssp')
        except Exception:
            raise RuntimeError(f"DSSP execution failed: {e}. Ensure mkdssp/dssp is installed.")
    
    results = []
    for key in dssp.keys():
        chain, res_id_tuple = key
        
        # Filter by chain if specified
        if chain_id is not None and chain != chain_id:
            continue
            
        res_data = dssp[key]
        # DSSP returns: (index, aa, ss, rsa, phi, psi, ...)
        # rsa = relative solvent accessibility
        
        results.append({
            'chain': chain,
            'res_id': res_id_tuple[1],  # Residue number
            'res_name': res_data[0] if len(res_data[0]) == 1 else res_data[1],
            'sasa': float(res_data[3]),  # Relative solvent accessibility [0, 1]
            'ss': res_data[2],  # Secondary structure code
        })
    
    return results


def get_structure_sasa(structure, chain_id: str = None) -> np.ndarray:
    """
    Extract per-residue SASA as a numpy array.
    
    Parameters
    ----------
    structure : Bio.PDB.Structure.Structure
        BioPython structure object.
    chain_id : str, optional
        Chain ID to extract.
        
    Returns
    -------
    np.ndarray
        SASA values for each residue. Shape: (N_residues,)
        
    Examples
    --------
    >>> sasa_array = fpsite.get_structure_sasa(structure)
    >>> print(sasa_array.shape)  # (N_residues,)
    """
    data = get_residue_sasa(structure, chain_id)
    return np.array([d['sasa'] for d in data], dtype=np.float32)


def get_structure_ss(structure, chain_id: str = None) -> np.ndarray:
    """
    Extract per-residue secondary structure as one-hot encoded array.
    
    Parameters
    ----------
    structure : Bio.PDB.Structure.Structure
        BioPython structure object.
    chain_id : str, optional
        Chain ID to extract.
        
    Returns
    -------
    np.ndarray
        One-hot encoded secondary structure. Shape: (N_residues, 3)
        Columns: [Helix, Sheet, Coil]
        
    Examples
    --------
    >>> ss_array = fpsite.get_structure_ss(structure)
    >>> print(ss_array.shape)  # (N_residues, 3)
    """
    data = get_residue_sasa(structure, chain_id)
    ss_codes = [d['ss'] for d in data]
    return np.array([residue_secondary_structure(ss) for ss in ss_codes], dtype=np.float32)


def get_structure_features_dssp(structure, chain_id: str = None) -> Dict[str, np.ndarray]:
    """
    Extract all DSSP-based features (SASA + Secondary Structure) in one call.
    
    Parameters
    ----------
    structure : Bio.PDB.Structure.Structure
        BioPython structure object.
    chain_id : str, optional
        Chain ID to extract.
        
    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary with keys:
        - 'sasa': Shape (N, 1) - Relative solvent accessibility
        - 'ss_onehot': Shape (N, 3) - Secondary structure one-hot [Helix, Sheet, Coil]
        - 'res_ids': Shape (N,) - Residue IDs
        
    Examples
    --------
    >>> features = fpsite.get_structure_features_dssp(structure)
    >>> print(features['sasa'].shape, features['ss_onehot'].shape)
    """
    data = get_residue_sasa(structure, chain_id)
    
    sasa = np.array([d['sasa'] for d in data], dtype=np.float32).reshape(-1, 1)
    ss_onehot = np.array([residue_secondary_structure(d['ss']) for d in data], dtype=np.float32)
    res_ids = np.array([d['res_id'] for d in data], dtype=np.int32)
    
    return {
        'sasa': sasa,
        'ss_onehot': ss_onehot,
        'res_ids': res_ids,
    }


# ==================== Existing Functions ====================


def property(
        prop_kind : str ='positive',
        prop_met : str ='Russell',
        standardize : bool =True,
) -> Dict:
    """
    An amino acid's property

    Parameters
    ----------
    prop_kind
        an amino acid's property kind
    prop_met
        method from which a property is derived,
    standalize
        if standardization

    Returns
    -------

    """
    return {
        "positive": aaprop().positive,
        "negative": aaprop().negative,
        "charged": aaprop().charged,
        "polar": aaprop().polar,
        "aliphatic": aaprop().aliphatic,
        "aromatic": aaprop().aromatic,
        "hydrophobic": aaprop().hydrophobic,
        "small": aaprop().small,
        "active": aaprop().active,
        "weight": aaprop().weight,
        "pI": aaprop().pI,
        "solubility": aaprop().solubility,
        "tm": aaprop().tm,
        "pka": aaprop().pka,
        "pkb": aaprop().pkb,
        "hydrophilicity": aaprop().hydrophilicity,
        "hydrophobicity": aaprop().hydrophobicity,
        "fet": aaprop().fet,
        "hydration": aaprop().hydration,
        "signal": aaprop().signal,
        "volume": aaprop().volume,
        "polarity": aaprop().polarity,
        "composition": aaprop().composition,
    }[prop_kind](standardize=standardize)


def onehot(
        arr_2d,
        arr_aa_names,
) -> List[List]:
    return aarepr().onehot(
        arr_2d=arr_2d,
        arr_aa_names=arr_aa_names,
    )


def pos_abs_val(
        pos : int,
        seq : str,
):
    return Position().absolute(
        pos=pos,
        seq=seq,
    )

def pos_rel_val(
        pos : int,
        interval : List,
):
    return Position().relative(
        pos=pos,
        interval=interval,
    )


def deepconpred():
    return Position().deepconpred()


def metapsicov():
    return Position().metapsicov()


def rsa_solvpred(
        solvpred_fp,
        prot_name,
        file_chain,
):
    return rsareader().solvpred(
        solvpred_fp=solvpred_fp,
        prot_name=prot_name,
        file_chain=file_chain,
    )


def rsa_accpro(
        accpro_fp,
        prot_name,
        file_chain,
):
    return rsareader().accpro(
        accpro_fp=accpro_fp,
        prot_name=prot_name,
        file_chain=file_chain,
    )


def rsa_accpro20(
        accpro20_fp,
        prot_name,
        file_chain,
):
    return rsareader().accpro20(
        accpro20_fp=accpro20_fp,
        prot_name=prot_name,
        file_chain=file_chain,
    )


def ss_spider3(
        spider3_path,
        prot_name,
        file_chain,
):
    return ssreader().spider3(
        spider3_path=spider3_path,
        prot_name=prot_name,
        file_chain=file_chain,
    )


def ss_spider3_ss(
        spider3_path,
        prot_name,
        file_chain,
        sv_fp,
):
    return ssreader().spider3_to_ss(
        spider3_path=spider3_path,
        prot_name=prot_name,
        file_chain=file_chain,
        sv_fp=sv_fp,
    )


def ss_psipred(
        psipred_path,
        prot_name,
        file_chain,
        kind='ss'
):
    """

    Parameters
    ----------
    psipred_path
    prot_name
    file_chain
    kind
        1. ss;
        2. ss2;
        3. horiz

    Returns
    -------

    """
    if kind == 'ss':
        return ssreader().psipred(
            psipred_ss_path=psipred_path,
            prot_name=prot_name,
            file_chain=file_chain,
        )
    if kind == 'ss2':
        return ssreader().psipred(
            psipred_ss2_path=psipred_path,
            prot_name=prot_name,
            file_chain=file_chain,
        )
    if kind == 'horiz':
        return ssreader().psipred(
            psipred_horiz_path=psipred_path,
            prot_name=prot_name,
            file_chain=file_chain,
        )
    else:
        return ssreader().psipred(
            psipred_ss_path=psipred_path,
            prot_name=prot_name,
            file_chain=file_chain,
        )


def ss_sspro(
        sspro_path,
        prot_name,
        file_chain,
):
    return ssreader().sspro(
        sspro_path=sspro_path,
        prot_name=prot_name,
        file_chain=file_chain,
    )


def ss_sspro8(
        sspro8_path,
        prot_name,
        file_chain,
):
    return ssreader().sspro8(
        sspro8_path=sspro8_path,
        prot_name=prot_name,
        file_chain=file_chain,
    )


if __name__ == "__main__":
    from pypropel.prot.sequence.Fasta import Fasta as sfasta
    from pypropel.path import to
    # import tmkit as tmk

    # print(property('positive'))
    #
    # sequence = sfasta().get(
    #     fasta_fpn=to("data/fasta/1aigL.fasta")
    # )
    # # print(sequence)
    #
    # pos_list = tmk.seq.pos_list_single(len_seq=len(sequence), seq_sep_superior=None, seq_sep_inferior=0)
    # # print(pos_list)
    #
    # positions = tmk.seq.pos_single(sequence=sequence, pos_list=pos_list)
    # # print(positions)
    #
    # win_aa_ids = tmk.seq.win_id_single(
    #     sequence=sequence,
    #     position=positions,
    #     window_size=1,
    # )
    # # print(win_aa_ids)
    #
    # win_aas = tmk.seq.win_name_single(
    #     sequence=sequence,
    #     position=positions,
    #     window_size=1,
    #     mids=win_aa_ids,
    # )
    # # print(win_aas)
    #
    # features = [[] for i in range(len(sequence))]
    # print(features)
    # print(len(features))

    # print(onehot(
    #     arr_2d=features,
    #     arr_aa_names=win_aas,
    # )[0])

    # print(pos_abs_val(
    #     pos=positions[0][0],
    #     seq=sequence,
    # ))
    #
    # print(pos_rel_val(
    #     pos=positions[0][0],
    #     interval=[4, 10],
    # ))
    #
    # print(deepconpred())
    #
    # print(metapsicov())

    # print(rsa_solvpred(
    #     solvpred_fp=to('data/accessibility/solvpred/'),
    #     prot_name='1aig',
    #     file_chain='L',
    # ))

    # print(rsa_accpro(
    #     accpro_fp=to('data/accessibility/accpro/'),
    #     prot_name='1aig',
    #     file_chain='L',
    # ))

    # print(rsa_accpro20(
    #     accpro20_fp=to('data/accessibility/accpro20/'),
    #     prot_name='1aig',
    #     file_chain='L',
    # ))

    # print(ss_psipred(
    #     psipred_path=to('data/ss/psipred/'),
    #     prot_name='1aig',
    #     file_chain='L',
    #     kind='ss', # horiz, ss, ss2
    # ))

    # print(ss_sspro(
    #     sspro_path=to('data/ss/sspro/'),
    #     prot_name='1aig',
    #     file_chain='L'
    # ))

    # print(ss_sspro8(
    #     sspro8_path=to('data/ss/sspro8/'),
    #     prot_name='1aig',
    #     file_chain='L'
    # ))

    # print(ss_spider3(
    #     spider3_path=to('data/ss/spider3/'),
    #     prot_name='E',
    #     file_chain=''
    # ))

    # print(ss_spider3_ss(
    #     spider3_path=to('data/ss/spider3/'),
    #     prot_name='E',
    #     file_chain='',
    #     sv_fp=to('data/ss/spider3/'),
    # ))


    seq = "ADGCGVGEGTGQGPMCNCMCMKWVYADEDAADLESDSFADEDASLESDSFPWSNQRVFCSFADEDAS"
    print(seq)

    feature_vector = [[] for i in range(len(seq))]
    print(feature_vector)
    print(len(feature_vector))

    print(property(
        prop_met='Hopp',
        prop_kind='hydrophilicity'
    ))

