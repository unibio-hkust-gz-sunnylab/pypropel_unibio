# UniO Modifications

This fork (`pypropel_unibio`) extends the original pypropel with additional features for protein-ligand interaction analysis.

## New Modules

### `pypropel.mol` - Small Molecule Utilities

```python
import pypropel.mol as ppmol

# Load ligands
mol = ppmol.load_sdf("/path/to/ligand.sdf")
mol = ppmol.load_mol2("/path/to/ligand.mol2")

# Extract coordinates
coords = ppmol.ligand_coords(mol)  # Shape: (N_atoms, 3)

# Get ligand features [LogP, MolWt, H-Donors, H-Acceptors]
features = ppmol.get_ligand_features(mol)  # Shape: (4,)
```

---

## Extended `pypropel.str` - Structure Utilities

```python
import pypropel.str as ppstr

# Load PDB structure
structure = ppstr.load_pdb("/path/to/protein.pdb")

# Extract sequence from PDB file
seq = ppstr.read_pdb_sequence("/path/to/protein.pdb")

# Extract sequence from structure object
seq = ppstr.structure_to_sequence(structure)
```

---

## Extended `pypropel.fpsite` - Residue Features

### AA Validation

```python
import pypropel.fpsite as fpsite

# Check if residue is standard AA (BioPython residue object)
fpsite.is_standard_aa(residue)  # -> bool

# Check by name (3-letter or 1-letter code)
fpsite.is_standard_aa_name("ALA")  # -> True
fpsite.is_standard_aa_name("X")    # -> False
```

### Residue Features

```python
# One-hot encoding (20 dimensions)
one_hot = fpsite.residue_one_hot("ALA")  # Shape: (20,)

# Physicochemical properties [charge, hydrophobicity]
physchem = fpsite.residue_physchem("ARG")  # [1.0, -1.0]

# Extract atom coordinates from a residue
coords = fpsite.residue_coords(residue)  # Shape: (N_atoms, 3)
```

### Sequence Window (1D Neighbors)

```python
# Get k-nearest neighbors in sequence
indices = fpsite.get_residue_window(seq_length=100, pos=50, k=3)
# Returns: [47, 48, 49, 50, 51, 52, 53]

# With padding for boundaries
indices = fpsite.get_residue_window_padded(100, 0, k=3, pad_value=-1)
# Returns: [-1, -1, -1, 0, 1, 2, 3]
```

### Spatial Neighbors (3D Neighbors)

```python
# Find k-nearest residues in 3D space (returns residue objects)
neighbors = fpsite.get_spatial_neighbors(structure, target_residue, k=5)
for res, dist in neighbors:
    print(f"{res.get_resname()}: {dist:.2f} Å")

# Find k-nearest by index
neighbor_indices = fpsite.get_spatial_neighbor_indices(structure, target_idx=10, k=5)
# Returns: [(idx, distance), ...]
```

### Positional Encoding

```python
# Single position encoding (for transformer models)
pe = fpsite.positional_encoding(pos=5, max_len=100, dim=64, mode='sinusoidal')
# Modes: 'sinusoidal' (default), 'absolute', 'relative'

# Batch encoding for entire sequence
encodings = fpsite.batch_positional_encoding(seq_length=100, dim=64)
# Shape: (100, 64)

# Normalized relative position [0, 1]
rel_pos = fpsite.relative_position(pos=50, total_len=100)  # 0.505
```

---

## Extended `pypropel.dist` - Distance & Contact Maps

```python
import pypropel.dist as ppdist

# Atom-level distances
dist = ppdist.atom_distance(coord1, coord2)
matrix = ppdist.atom_distance_matrix(coords1, coords2)

# Residue-level distances
dist = ppdist.residue_distance(res1_coords, res2_coords)
dist = ppdist.residue_ligand_distance(residue_coords, ligand_coords)

# Protein distance matrix (residue-residue)
dist_matrix = ppdist.protein_distance_matrix(structure, use_ca_only=True)

# Protein-ligand distances (all residues to ligand)
distances = ppdist.protein_ligand_distances(structure, ligand_coords)

# Binary contact maps
contact_map = ppdist.protein_contact_map(structure, threshold=8.0)
pl_contact = ppdist.protein_ligand_contact_map(structure, ligand_coords, threshold=5.0)

# Binding pocket extraction
pocket_df = ppdist.extract_binding_pocket(
    structure, 
    ligand_coords,
    binding_threshold=5.0,
    non_binding_threshold=8.0
)
# Returns DataFrame with columns: chain, res_id, res_name, distance, label
# label: 1 (binding), 0 (non-binding), -1 (margin)
```

---

## Installation

```bash
pip install git+https://github.com/unibio-hkust-gz-sunnylab/pypropel_unibio.git
```

## Requirements

- Python >= 3.9
- BioPython >= 1.81
- RDKit >= 2023.0.0
- NumPy >= 1.24.0
- Pandas >= 2.0.0
