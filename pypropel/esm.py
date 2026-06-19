from __future__ import annotations

__version__ = "v1.0"
__copyright__ = "Copyright 2024"
__license__ = "GPL v3.0"
__developer__ = "Jianfeng Sun"
__maintainer__ = "Jianfeng Sun"
__email__ = "jianfeng.sunmt@gmail.com"

import hashlib
import importlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

DEFAULT_MODEL_NAME = "esm2_t33_650M_UR50D"
DEFAULT_LAYER = -1
VALID_SEQUENCE_SYMBOLS = frozenset("ACDEFGHIKLMNPQRSTVWYBXZUO")
METADATA_KEY = "__metadata_json__"

_MODEL_CACHE: Dict[Tuple[str, str], Tuple[Any, Any, Any]] = {}


def is_available() -> bool:
    """Return True when the optional torch and fair-esm dependencies import."""
    try:
        _import_optional_dependencies()
    except ImportError:
        return False
    return True


def clear_model_cache() -> None:
    """Clear cached ESM model instances."""
    _MODEL_CACHE.clear()


def load_model(
    model_name: str = DEFAULT_MODEL_NAME,
    device: str = "cpu",
    use_cache: bool = True,
) -> Tuple[Any, Any, Any]:
    """
    Load an ESM model, alphabet, and batch converter.

    Parameters
    ----------
    model_name
        Name understood by ``esm.pretrained.load_model_and_alphabet``.
    device
        Torch device string, for example ``cpu`` or ``cuda:0``.
    use_cache
        If True, reuse a model cached by ``(model_name, device)``.
    """
    _, esm_lib = _import_optional_dependencies()
    cache_key = (model_name, str(device))
    if use_cache and cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    model, alphabet = esm_lib.pretrained.load_model_and_alphabet(model_name)
    if hasattr(model, "eval"):
        model.eval()
    if device is not None and hasattr(model, "to"):
        model = model.to(device)
    batch_converter = alphabet.get_batch_converter()
    loaded = (model, alphabet, batch_converter)
    if use_cache:
        _MODEL_CACHE[cache_key] = loaded
    return loaded


def embed_sequence(
    sequence: str,
    model_name: str = DEFAULT_MODEL_NAME,
    device: str = "cpu",
    layer: int = DEFAULT_LAYER,
    model: Any = None,
    alphabet: Any = None,
    batch_converter: Any = None,
) -> np.ndarray:
    """
    Extract per-residue ESM embeddings for one protein sequence.

    The returned array has one row per residue in ``sequence``. The embedding
    dimension depends on the selected ESM model.
    """
    return embed_batch(
        [sequence],
        model_name=model_name,
        device=device,
        layer=layer,
        model=model,
        alphabet=alphabet,
        batch_converter=batch_converter,
    )[0]


def embed_batch(
    sequences: Sequence[str],
    model_name: str = DEFAULT_MODEL_NAME,
    device: str = "cpu",
    layer: int = DEFAULT_LAYER,
    batch_size: int = 8,
    model: Any = None,
    alphabet: Any = None,
    batch_converter: Any = None,
) -> List[np.ndarray]:
    """
    Extract per-residue ESM embeddings for multiple protein sequences.

    Parameters
    ----------
    sequences
        Protein sequences using one-letter amino acid symbols.
    model_name
        ESM model name used when ``model`` is not supplied.
    device
        Torch device string.
    layer
        Representation layer. Negative values follow Python-style indexing
        over model layers, so ``-1`` means the last layer.
    batch_size
        Number of sequences per model call.
    model, alphabet, batch_converter
        Optional preloaded ESM runtime objects. If ``model`` is supplied,
        provide either ``batch_converter`` or ``alphabet`` so no implicit
        model load is needed.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    normalized = [normalize_sequence(seq) for seq in sequences]
    if not normalized:
        return []

    torch, _ = _import_optional_dependencies()
    model, alphabet, batch_converter = _resolve_runtime(
        model_name=model_name,
        device=device,
        model=model,
        alphabet=alphabet,
        batch_converter=batch_converter,
    )
    layer_idx = resolve_layer(model, layer)

    embeddings: List[np.ndarray] = []
    for batch_start in range(0, len(normalized), batch_size):
        batch_sequences = normalized[batch_start : batch_start + batch_size]
        batch_data = [
            (f"sequence_{batch_start + offset}", seq)
            for offset, seq in enumerate(batch_sequences)
        ]
        _, _, batch_tokens = batch_converter(batch_data)
        if device is not None and hasattr(batch_tokens, "to"):
            batch_tokens = batch_tokens.to(device)

        with torch.no_grad():
            result = model(batch_tokens, repr_layers=[layer_idx])

        representations = result["representations"][layer_idx]
        for batch_index, seq in enumerate(batch_sequences):
            arr = _to_numpy(representations[batch_index, 1 : len(seq) + 1])
            arr = np.asarray(arr, dtype=np.float32)
            validate_embedding_alignment(arr, seq)
            embeddings.append(arr)
    return embeddings


def save_embeddings(
    embeddings: Union[np.ndarray, Mapping[str, np.ndarray]],
    filepath: Union[str, Path],
    metadata: Optional[Mapping[str, Any]] = None,
) -> None:
    """
    Save embeddings to a compressed ``.npz`` file with optional JSON metadata.
    """
    payload: Dict[str, Any]
    if isinstance(embeddings, np.ndarray):
        payload = {"embeddings": embeddings}
    else:
        payload = {str(key): np.asarray(value) for key, value in embeddings.items()}
    if metadata is not None:
        payload[METADATA_KEY] = np.asarray(json.dumps(dict(metadata), sort_keys=True))
    np.savez_compressed(filepath, **payload)


def load_embeddings(
    filepath: Union[str, Path],
    include_metadata: bool = False,
) -> Union[Dict[str, np.ndarray], Tuple[Dict[str, np.ndarray], Dict[str, Any]]]:
    """
    Load embeddings from a compressed ``.npz`` file.
    """
    with np.load(filepath, allow_pickle=False) as data:
        embeddings = {
            key: data[key]
            for key in data.files
            if key != METADATA_KEY
        }
        metadata: Dict[str, Any] = {}
        if METADATA_KEY in data.files:
            metadata = json.loads(str(data[METADATA_KEY].item()))
    if include_metadata:
        return embeddings, metadata
    return embeddings


def build_metadata(
    sequence: str,
    embedding: Optional[np.ndarray] = None,
    model_name: str = DEFAULT_MODEL_NAME,
    layer: int = DEFAULT_LAYER,
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Build reproducibility metadata for a sequence embedding artifact."""
    normalized = normalize_sequence(sequence)
    metadata: Dict[str, Any] = {
        "model_name": model_name,
        "layer": int(layer),
        "sequence_length": len(normalized),
        "sequence_sha256": hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
    }
    if embedding is not None:
        arr = np.asarray(embedding)
        metadata["embedding_shape"] = list(arr.shape)
        metadata["embedding_dtype"] = str(arr.dtype)
    if extra:
        metadata.update(dict(extra))
    return metadata


def normalize_sequence(sequence: str) -> str:
    """Normalize and validate a one-letter protein sequence."""
    if not isinstance(sequence, str):
        raise TypeError("sequence must be a string")
    normalized = "".join(sequence.split()).upper()
    if not normalized:
        raise ValueError("sequence must not be empty")
    invalid = sorted(set(normalized) - VALID_SEQUENCE_SYMBOLS)
    if invalid:
        raise ValueError(f"sequence contains unsupported symbols: {''.join(invalid)}")
    return normalized


def validate_embedding_alignment(embedding: np.ndarray, sequence: str) -> None:
    """Raise if an embedding array does not have one row per residue."""
    normalized = normalize_sequence(sequence)
    arr = np.asarray(embedding)
    if arr.ndim != 2:
        raise ValueError("embedding must be a 2D array")
    if arr.shape[0] != len(normalized):
        raise ValueError(
            f"embedding row count {arr.shape[0]} does not match sequence length {len(normalized)}"
        )


def resolve_layer(model: Any, layer: int) -> int:
    """Resolve a requested representation layer against an ESM model."""
    if not isinstance(layer, int):
        raise TypeError("layer must be an integer")
    num_layers = getattr(model, "num_layers", None)
    if num_layers is None:
        if layer < 0:
            raise ValueError("negative layer values require model.num_layers")
        return layer
    resolved = num_layers + 1 + layer if layer < 0 else layer
    if resolved < 0 or resolved > num_layers:
        raise ValueError(f"layer {layer} resolves outside available layers 0..{num_layers}")
    return int(resolved)


def _resolve_runtime(
    model_name: str,
    device: str,
    model: Any = None,
    alphabet: Any = None,
    batch_converter: Any = None,
) -> Tuple[Any, Any, Any]:
    if model is None:
        return load_model(model_name=model_name, device=device)
    if batch_converter is None:
        if alphabet is None:
            raise ValueError("alphabet or batch_converter is required when model is supplied")
        batch_converter = alphabet.get_batch_converter()
    return model, alphabet, batch_converter


def _import_optional_dependencies() -> Tuple[Any, Any]:
    try:
        torch = importlib.import_module("torch")
        esm_lib = importlib.import_module("esm")
    except ImportError as exc:
        raise ImportError(
            "ESM embedding support requires optional dependencies. "
            "Install them with: pip install 'pypropel[esm]'"
        ) from exc
    return torch, esm_lib


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


# Backward-compatible aliases matching common ESM wrapper naming.
load_esm_model = load_model
get_esm_embeddings = embed_sequence
get_esm_embeddings_batch = embed_batch
