"""
ESM-2 embedding extraction for pypropel.

Provides wrapper functions for extracting per-residue embeddings
from the ESM-2 protein language model.

Note: Requires `fair-esm` package. Install with:
    pip install fair-esm
    
Embeddings should be pre-computed and cached for training efficiency.
"""

__version__ = "v1.0"
__copyright__ = "Copyright 2024"
__license__ = "GPL v3.0"

from typing import List, Tuple, Dict, Optional, Union
import numpy as np


# Check if ESM is available
try:
    import torch
    import esm
    ESM_AVAILABLE = True
except ImportError:
    ESM_AVAILABLE = False

# Global model cache
_esm_model = None
_esm_batch_converter = None
_esm_alphabet = None


def is_esm_available() -> bool:
    """Check if ESM library is installed."""
    return ESM_AVAILABLE


def load_esm_model(
    model_name: str = "esm2_t33_650M_UR50D",
    device: str = 'cpu'
) -> Tuple:
    """
    Load ESM-2 model and batch converter.
    
    Parameters
    ----------
    model_name : str
        Name of ESM model. Options:
        - 'esm2_t6_8M_UR50D' (8M params, fastest)
        - 'esm2_t12_35M_UR50D' (35M params)
        - 'esm2_t30_150M_UR50D' (150M params)
        - 'esm2_t33_650M_UR50D' (650M params, default)
        - 'esm2_t36_3B_UR50D' (3B params, best quality)
    device : str
        Device to load model on ('cpu', 'cuda', 'cuda:0', etc.)
        
    Returns
    -------
    Tuple
        (model, alphabet, batch_converter)
        
    Examples
    --------
    >>> import pypropel.esm as ppesm
    >>> model, alphabet, batch_converter = ppesm.load_esm_model()
    """
    global _esm_model, _esm_batch_converter, _esm_alphabet
    
    if not ESM_AVAILABLE:
        raise ImportError(
            "ESM library not available. Install with: pip install fair-esm"
        )
    
    # Load model if not cached
    if _esm_model is None:
        model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
        model.eval()
        model = model.to(device)
        
        _esm_model = model
        _esm_alphabet = alphabet
        _esm_batch_converter = alphabet.get_batch_converter()
    
    return _esm_model, _esm_alphabet, _esm_batch_converter


def get_esm_embeddings(
    sequence: str,
    model=None,
    device: str = 'cpu',
    layer: int = -1
) -> np.ndarray:
    """
    Extract per-residue ESM-2 embeddings for a sequence.
    
    Parameters
    ----------
    sequence : str
        Amino acid sequence (one-letter codes).
    model : optional
        Pre-loaded ESM model. If None, loads default model.
    device : str
        Device for computation.
    layer : int
        Which layer to extract. -1 means last layer.
        
    Returns
    -------
    np.ndarray
        Per-residue embeddings, shape (L, 1280) for ESM-2 650M.
        
    Examples
    --------
    >>> import pypropel.esm as ppesm
    >>> sequence = "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH"
    >>> embeddings = ppesm.get_esm_embeddings(sequence)
    >>> print(embeddings.shape)  # (51, 1280)
    """
    if not ESM_AVAILABLE:
        raise ImportError(
            "ESM library not available. Install with: pip install fair-esm"
        )
    
    if model is None:
        model, _, batch_converter = load_esm_model(device=device)
    else:
        _, _, batch_converter = load_esm_model(device=device)
    
    # Prepare data
    data = [("protein", sequence)]
    _, _, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.to(device)
    
    # Get representations
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[layer if layer >= 0 else model.num_layers])
    
    # Extract embeddings (exclude BOS and EOS tokens)
    layer_idx = layer if layer >= 0 else model.num_layers
    embeddings = results["representations"][layer_idx][0, 1:-1].cpu().numpy()
    
    return embeddings.astype(np.float32)


def get_esm_embeddings_batch(
    sequences: List[str],
    model=None,
    device: str = 'cpu',
    layer: int = -1,
    batch_size: int = 8
) -> List[np.ndarray]:
    """
    Extract ESM-2 embeddings for multiple sequences.
    
    Parameters
    ----------
    sequences : List[str]
        List of amino acid sequences.
    model : optional
        Pre-loaded ESM model.
    device : str
        Device for computation.
    layer : int
        Which layer to extract.
    batch_size : int
        Batch size for processing.
        
    Returns
    -------
    List[np.ndarray]
        List of embedding arrays, each with shape (L_i, 1280).
        
    Examples
    --------
    >>> sequences = ["MVLSPADKTNVK", "AAWGKVGAHAGE"]
    >>> embeddings = ppesm.get_esm_embeddings_batch(sequences)
    >>> print(len(embeddings), embeddings[0].shape)
    """
    if not ESM_AVAILABLE:
        raise ImportError(
            "ESM library not available. Install with: pip install fair-esm"
        )
    
    if model is None:
        model, _, batch_converter = load_esm_model(device=device)
    else:
        _, _, batch_converter = load_esm_model(device=device)
    
    all_embeddings = []
    
    # Process in batches
    for i in range(0, len(sequences), batch_size):
        batch_seqs = sequences[i:i + batch_size]
        data = [(f"protein_{j}", seq) for j, seq in enumerate(batch_seqs)]
        
        _, _, batch_tokens = batch_converter(data)
        batch_lengths = [len(seq) for seq in batch_seqs]
        batch_tokens = batch_tokens.to(device)
        
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[layer if layer >= 0 else model.num_layers])
        
        layer_idx = layer if layer >= 0 else model.num_layers
        representations = results["representations"][layer_idx]
        
        # Extract each sequence (excluding BOS/EOS tokens)
        for j, length in enumerate(batch_lengths):
            emb = representations[j, 1:length + 1].cpu().numpy()
            all_embeddings.append(emb.astype(np.float32))
    
    return all_embeddings


def save_embeddings(
    embeddings: Union[np.ndarray, Dict[str, np.ndarray]],
    filepath: str
) -> None:
    """
    Save pre-computed embeddings to disk.
    
    Parameters
    ----------
    embeddings : np.ndarray or Dict[str, np.ndarray]
        Embeddings to save. Can be single array or dict of arrays.
    filepath : str
        Path to save file (.npz format).
        
    Examples
    --------
    >>> ppesm.save_embeddings(embeddings, '/path/to/embeddings.npz')
    """
    if isinstance(embeddings, np.ndarray):
        np.savez_compressed(filepath, embeddings=embeddings)
    else:
        np.savez_compressed(filepath, **embeddings)


def load_embeddings(filepath: str) -> Dict[str, np.ndarray]:
    """
    Load pre-computed embeddings from disk.
    
    Parameters
    ----------
    filepath : str
        Path to saved embeddings (.npz format).
        
    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary of loaded embeddings.
        
    Examples
    --------
    >>> embeddings = ppesm.load_embeddings('/path/to/embeddings.npz')
    >>> print(embeddings['embeddings'].shape)
    """
    data = np.load(filepath, allow_pickle=True)
    return dict(data)


if __name__ == "__main__":
    print(f"esm.py module loaded successfully")
    print(f"ESM available: {ESM_AVAILABLE}")
