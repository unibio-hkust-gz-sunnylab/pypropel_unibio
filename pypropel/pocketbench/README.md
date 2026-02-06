# PocketBench

**Unified benchmarking library for protein-ligand binding site prediction.**

PocketBench provides standardized datasets, metrics, and model wrappers for evaluating binding site prediction methods.

## Features

- **Core dataclasses**: `PBProtein`, `PBSite`, `PBPrediction` for unified data representation
- **5 benchmark datasets** with auto-download support
- **Standard metrics**: DCC, DCA, IoU, AP
- **Model wrappers**: P2Rank (more coming soon)

## Installation

PocketBench is included in pypropel:

```python
from pypropel import pocketbench
```

### Optional Dependencies

```bash
# For UniSite-DS auto-download
pip install huggingface_hub

# For PDB parsing (recommended)
pip install biopython
```

---

## Datasets

### Quick Start

```python
from pypropel.pocketbench.datasets import (
    COACH420Dataset,       # Legacy benchmark (420 proteins)
    HOLO4KDataset,         # Legacy benchmark (4000+ proteins)
    UniSiteDSDataset,      # Modern UniProt-centric (11,510 proteins)
    CryptoBenchDataset,    # Cryptic binding sites (1000+ pairs)
)

# All datasets support auto-download
ds = COACH420Dataset(download=True)
print(f"Loaded {len(ds)} proteins")

for protein in ds:
    print(f"{protein.id}: {protein.num_sites} binding sites")
```

### Dataset Summary

| Dataset | Class | Size | Ground Truth | Auto-Download |
|---------|-------|------|--------------|---------------|
| COACH420 | `COACH420Dataset` | 420 | Ligand contacts | ✅ GitHub |
| HOLO4K | `HOLO4KDataset` | 4,000+ | Ligand contacts | ✅ GitHub |
| UniSite-DS | `UniSiteDSDataset` | 11,510 | UniProt-centric | ✅ HuggingFace |
| UniSite Benchmarks | `UniSiteBenchmarkDataset` | varies | PKL masks | ✅ HuggingFace |
| CryptoBench | `CryptoBenchDataset` | 1,000+ | Apo→Holo | ✅ OSF |

### Legacy Datasets (P2Rank)

**COACH420** and **HOLO4K** are classic benchmarks from [p2rank-datasets](https://github.com/rdk/p2rank-datasets).

```python
from pypropel.pocketbench.datasets import COACH420Dataset, HOLO4KDataset

# Downloads ~50MB from GitHub on first use
coach = COACH420Dataset(download=True)
holo = HOLO4KDataset(download=True)

# Access proteins
protein = coach[0]
print(f"ID: {protein.id}")
print(f"Sequence length: {len(protein.sequence)}")
print(f"C-alpha coords shape: {protein.coords.shape}")
print(f"Ground truth sites: {protein.num_sites}")
```

### UniSite-DS (Modern)

[UniSite-DS](https://github.com/quanlin-wu/unisite) is a **UniProt-centric** dataset that aggregates all binding sites for a protein across multiple PDB structures. This addresses the "missing label" problem in traditional PDB-centric evaluation.

```python
from pypropel.pocketbench.datasets import UniSiteDSDataset

# Downloads ~10GB from HuggingFace (requires huggingface_hub)
ds = UniSiteDSDataset(
    split="test",           # "train" or "test"
    sim_threshold=0.9,      # Sequence identity cutoff (0.3, 0.5, 0.7, 0.9)
    download=True
)

# UniSite-DS protein has multiple consolidated sites
protein = ds[0]
print(f"UniProt ID: {protein.id}")
print(f"Binding sites: {protein.num_sites}")

for site in protein.ground_truth_sites:
    print(f"  Site {site.ligand_id}: {len(site.residues)} residues")
```

### CryptoBench (Cryptic Pockets)

[CryptoBench](https://github.com/skrhakv/CryptoBench) evaluates detection of **cryptic binding sites**—sites that are hidden in the apo (unbound) structure but become visible upon ligand binding.

```python
from pypropel.pocketbench.datasets import CryptoBenchDataset

# Downloads from OSF
ds = CryptoBenchDataset(
    split="test",
    use_apo_structure=True,  # Use unbound structure as input
    download=True
)

# Protein has APO coords, HOLO ground truth
protein = ds[0]
print(f"Apo structure: {protein.metadata['apo_id']}")
print(f"Holo reference: {protein.metadata['holo_id']}")
```

---

## Metrics

### Distance-Based (Localization)

```python
from pypropel.pocketbench import compute_dcc, compute_dca

# DCC: Distance Center-to-Center
is_hit, distance = compute_dcc(
    prediction,                    # PBPrediction
    protein.ground_truth_sites,    # List[PBSite]
    threshold=4.0                  # Angstroms
)

# DCA: Distance Center-to-Any-Atom (requires ground truth atoms)
is_hit, distance = compute_dca(
    prediction,
    ground_truth_atoms,  # (N, 3) array of atom coords
    threshold=4.0
)
```

### Segmentation (IoU-Based)

```python
from pypropel.pocketbench import compute_iou, compute_ap

# IoU: Intersection over Union (residue-based)
iou = compute_iou(
    prediction.residues,           # List[int] predicted residue indices
    ground_truth_site.residues     # List[int] ground truth indices
)

# AP: Average Precision at IoU thresholds
ap = compute_ap(
    predictions,           # List[PBPrediction]
    ground_truth_sites,    # List[PBSite]
    iou_threshold=0.5      # AP@50
)
```

### 9Å Radius Fallback

For models that only predict centers (no residue list), expand to residues:

```python
from pypropel.pocketbench import expand_center_to_residues

# Convert center to residue list
residues = expand_center_to_residues(
    center=prediction.center,   # [x, y, z]
    coords=protein.coords,      # (N, 3) C-alpha coords
    radius=9.0                  # Angstroms
)
```

---

## Model Wrappers

### P2Rank

Wrapper for [P2Rank](https://github.com/rdk/p2rank), a machine learning tool for binding site prediction.

**Prerequisites:**
1. Download P2Rank from [GitHub releases](https://github.com/rdk/p2rank/releases)
2. Install Java 11+

```python
from pypropel.pocketbench.models import P2RankWrapper

# Initialize wrapper
p2rank = P2RankWrapper(
    p2rank_home="/path/to/p2rank",  # Or set P2RANK_HOME env var
    config="default",                # "alphafold" for AF/NMR/cryo-EM
    threads=4
)

# Run prediction
predictions = p2rank.predict(protein)

for pred in predictions:
    print(f"Center: {pred.center}")
    print(f"Score: {pred.confidence:.3f}")
    print(f"Residues: {pred.residues[:5]}...")

# Batch prediction
all_predictions = p2rank.predict_batch(proteins)
```

---

## Core Data Structures

### PBProtein

```python
from pypropel.pocketbench import PBProtein, PBSite

protein = PBProtein(
    id="1abc_A",
    sequence="ACDEFGHIKLMNPQRSTVWY",
    coords=ca_coordinates,              # (N, 3) C-alpha coords
    full_atoms=biopython_structure,     # Optional: full structure
    ground_truth_sites=[                # List of binding sites
        PBSite(
            center=[10.0, 20.0, 30.0],
            residues=[5, 6, 7, 8, 9],
            ligand_id="ATP"
        )
    ],
    metadata={'source': 'PDBbind'}
)

# Properties
print(f"Residues: {protein.num_residues}")
print(f"Sites: {protein.num_sites}")
print(f"All site residues: {protein.get_all_site_residues()}")
```

### PBPrediction

```python
from pypropel.pocketbench import PBPrediction

pred = PBPrediction(
    center=[11.0, 20.5, 29.8],
    residues=[5, 6, 7, 8],
    confidence=0.95,
    model_name="MyModel"
)
```

### PBModel (Abstract Base)

Implement this interface to wrap your own model:

```python
from pypropel.pocketbench import PBModel, PBProtein, PBPrediction
from typing import List

class MyModelWrapper(PBModel):
    @property
    def name(self) -> str:
        return "MyModel"
    
    def predict(self, protein: PBProtein) -> List[PBPrediction]:
        # Your prediction logic here
        return [PBPrediction(center=[...], residues=[...], confidence=0.9)]
```

---

## Full Evaluation Example

```python
from pypropel.pocketbench import compute_dcc, compute_iou
from pypropel.pocketbench.datasets import COACH420Dataset
from pypropel.pocketbench.models import P2RankWrapper

# Load dataset
dataset = COACH420Dataset(download=True)

# Initialize model
model = P2RankWrapper(p2rank_home="/path/to/p2rank")

# Evaluate
dcc_hits = 0
total_iou = 0.0
n_predictions = 0

for protein in dataset[:100]:  # First 100 proteins
    predictions = model.predict(protein)
    
    if predictions and protein.ground_truth_sites:
        # Top-1 prediction vs any ground truth
        is_hit, dist = compute_dcc(predictions[0], protein.ground_truth_sites)
        dcc_hits += int(is_hit)
        
        # IoU with best-matching ground truth
        best_iou = max(
            compute_iou(predictions[0].residues, gt.residues)
            for gt in protein.ground_truth_sites
        )
        total_iou += best_iou
        n_predictions += 1

print(f"DCC Success Rate: {dcc_hits / n_predictions:.2%}")
print(f"Mean IoU: {total_iou / n_predictions:.3f}")
```

---

## References

- **P2Rank**: Krivák & Hoksza (2018). P2Rank: machine learning based tool for rapid and accurate prediction of ligand binding sites from protein structure. *J Cheminform*
- **UniSite**: Wu et al. (2025). UniSite: The First Cross-Structure Dataset and Learning Framework for End-to-End Ligand Binding Site Detection. *NeurIPS 2025*
- **CryptoBench**: Škrhák et al. (2025). CryptoBench: cryptic protein-ligand binding sites dataset and benchmark. *Bioinformatics*
