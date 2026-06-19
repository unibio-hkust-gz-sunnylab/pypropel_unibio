ESM2 embeddings are protein language model representations for amino acid sequences. PyPropel exposes them as an optional feature extraction module because the runtime depends on large model weights and PyTorch.

Install the optional dependencies before extracting embeddings.

:material-console: Shell
``` shell
pip install "pypropel[esm]"
```

### Availability

:material-language-python: Python
``` py linenums="1"
import pypropel as pp

print(pp.esm.is_available())
```

### Single sequence

:material-language-python: Python
``` py linenums="1"
import pypropel as pp

sequence = "MVLSPADKTNVKAAW"
embedding = pp.esm.embed_sequence(
    sequence=sequence,
    model_name="esm2_t33_650M_UR50D",
    device="cuda:0",
    layer=-1,
)
print(embedding.shape)
```

The returned array has one row per residue. Its column count depends on the selected ESM2 model.

### Batch extraction

:material-language-python: Python
``` py linenums="1"
import pypropel as pp

sequences = ["MVLSPADKTNVKAAW", "ACDEFGHIKLMNPQ"]
embeddings = pp.esm.embed_batch(
    sequences=sequences,
    model_name="esm2_t12_35M_UR50D",
    batch_size=4,
)
```

### Cache embeddings

:material-language-python: Python
``` py linenums="1"
import pypropel as pp

sequence = "MVLSPADKTNVKAAW"
embedding = pp.esm.embed_sequence(sequence)
metadata = pp.esm.build_metadata(
    sequence=sequence,
    embedding=embedding,
    model_name="esm2_t33_650M_UR50D",
    layer=-1,
)

pp.esm.save_embeddings(
    embeddings=embedding,
    filepath="protein_esm2.npz",
    metadata=metadata,
)

loaded, loaded_metadata = pp.esm.load_embeddings(
    "protein_esm2.npz",
    include_metadata=True,
)
```

Record the model name, layer, sequence hash, cache path, and checksum in downstream experiment run records when embeddings are used as model features.
