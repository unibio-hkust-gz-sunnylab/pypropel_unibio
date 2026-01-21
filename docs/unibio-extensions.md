# UniO Modifications

This fork (`pypropel_unibio`) extends the original pypropel with additional features for protein-ligand interaction analysis and GVP-Fusion model support.

## New Modules

### `pypropel.mol` - Small Molecule Utilities

```python
import pypropel.mol as ppmol

# Load ligands
mol = ppmol.load_sdf("/path/to/ligand.sdf")
mol = ppmol.load_mol2("/path/to/ligand.mol2")

# Extract coordinates
coords = ppmol.ligand_coords(mol)  # Shape: (N_atoms, 3)

# Get global ligand features [LogP, MolWt, H-Donors, H-Acceptors]
features = ppmol.get_ligand_features(mol)  # Shape: (4,)

# Atom-level features for GIN encoder
atom_types = ppmol.get_atom_type_onehot(mol)      # Shape: (N, 10)
hybrid = ppmol.get_atom_hybridization(mol)        # Shape: (N, 7)
hbond = ppmol.get_atom_hbond_features(mol)        # Shape: (N, 2)
aromatic = ppmol.get_atom_aromaticity(mol)        # Shape: (N, 1)
charges = ppmol.get_atom_partial_charges(mol)     # Shape: (N, 1) Gasteiger
atom_features = ppmol.get_atom_features(mol)      # Shape: (N, 21) Combined

# Morgan fingerprints
fp = ppmol.get_morgan_fingerprint(mol, radius=2, n_bits=2048)
fp_compressed = ppmol.get_morgan_fingerprint_compressed(mol, output_dim=128)

# Extended global features (basic + compressed Morgan FP)
global_features = ppmol.get_ligand_global_features(mol)  # Shape: (132,)
```

---

### `pypropel.gvp` - GVP Vector Features

```python
import pypropel.gvp as ppgvp

# Extract coordinates
ca_coords = ppgvp.get_ca_coords(structure)    # Shape: (N, 3)
cb_coords = ppgvp.get_cb_coords(structure)    # Shape: (N, 3) (virtual for GLY)

# Vector features for GVP encoder
sidechain = ppgvp.get_residue_orientations(structure)      # Cα→Cβ unit vectors
backbone = ppgvp.get_backbone_vectors(structure)           # Cα[i]→Cα[i+1]
neighbors = ppgvp.get_neighbor_center_vectors(structure, k=10)  # Cα→neighbor center

# Combined GVP features
coords, vectors = ppgvp.get_gvp_node_features(structure, k_neighbors=10)
# coords: (N, 3), vectors: (N, 3, 3) - 3 unit vectors per residue

# k-NN graph edges
edge_index, distances = ppgvp.build_knn_edges(coords, k=20, radius=10.0)
```

---

### `pypropel.graph` - Graph Construction

```python
import pypropel.graph as ppgraph

# Build protein k-NN graph
graph = ppgraph.build_protein_knn_graph(structure, k=20, radius=10.0)
# Returns: {'node_coords': (N,3), 'edge_index': (2,E), 'edge_attr': (E,4)}

# Build ligand molecular graph
ligand_graph = ppgraph.build_ligand_graph(mol)
# Returns: {'node_features': (N,21), 'node_coords': (N,3), 
#           'edge_index': (2,E), 'edge_attr': (E,4)}

# Bond features for edges
bond_features = ppgraph.get_bond_features(mol)  # One-hot bond types

# Cross-modal bipartite edges (for attention)
edges, dists = ppgraph.build_protein_ligand_bipartite_edges(
    protein_coords, ligand_coords, threshold=8.0
)
```

---

### `pypropel.esm` - ESM-2 Embeddings (Requires `fair-esm`)

```python
import pypropel.esm as ppesm

# Load model
model, alphabet, converter = ppesm.load_esm_model("esm2_t33_650M_UR50D")

# Extract embeddings
embeddings = ppesm.get_esm_embeddings(sequence)  # Shape: (L, 1280)

# Batch extraction
embeddings_list = ppesm.get_esm_embeddings_batch(sequences, batch_size=8)

# Save/load cached embeddings
ppesm.save_embeddings(embeddings, "/path/to/cache.npz")
embeddings = ppesm.load_embeddings("/path/to/cache.npz")
```

---

## Extended `pypropel.str` - Structure Utilities

```python
import pypropel.str as ppstr

# Load PDB structure
structure = ppstr.load_pdb("/path/to/protein.pdb")

# Extract sequence
seq = ppstr.read_pdb_sequence("/path/to/protein.pdb")
seq = ppstr.structure_to_sequence(structure)
```

---

## Extended `pypropel.fpsite` - Residue Features

### SASA & Secondary Structure (via DSSP)

```python
import pypropel.fpsite as fpsite

# Secondary structure one-hot (Helix/Sheet/Coil)
ss_onehot = fpsite.residue_secondary_structure('H')  # [1, 0, 0]

# SASA per residue
sasa_data = fpsite.get_residue_sasa(structure)  # List of dicts
sasa_array = fpsite.get_structure_sasa(structure)  # Shape: (N,)

# Secondary structure per residue
ss_array = fpsite.get_structure_ss(structure)  # Shape: (N, 3)

# Combined DSSP features
features = fpsite.get_structure_features_dssp(structure)
# Returns: {'sasa': (N,1), 'ss_onehot': (N,3), 'res_ids': (N,)}
```

### Other Features

```python
# One-hot encoding (20 dimensions)
one_hot = fpsite.residue_one_hot("ALA")  # Shape: (20,)

# Physicochemical properties [charge, hydrophobicity]
physchem = fpsite.residue_physchem("ARG")  # [1.0, -1.0]

# Spatial k-NN neighbors
neighbors = fpsite.get_spatial_neighbors(structure, target_residue, k=5)
indices = fpsite.get_spatial_neighbor_indices(structure, target_idx=10, k=5)

# Positional encoding for transformers
pe = fpsite.positional_encoding(pos=5, max_len=100, dim=64)
encodings = fpsite.batch_positional_encoding(seq_length=100, dim=64)
```

---

## Extended `pypropel.dist` - Distance & Classification

```python
import pypropel.dist as ppdist

# Existing distance functions
dist = ppdist.residue_ligand_distance(residue_coords, ligand_coords)
distances = ppdist.protein_ligand_distances(structure, ligand_coords)
contact_map = ppdist.protein_contact_map(structure, threshold=8.0)

# NEW: Distance classification for training labels
label = ppdist.classify_binding_distance(distance, thresholds=[3.5, 6.0])
# Returns 0 (Contact <3.5Å), 1 (Near <6.0Å), 2 (Far)

# Per-residue labels
labels = ppdist.get_distance_labels(structure, ligand_coords, thresholds=[3.5, 6.0])
# Shape: (N_residues,)

# With residue info
df = ppdist.get_distance_labels_with_info(structure, ligand_coords)
# DataFrame: chain, res_id, res_name, distance, label

# Class weights for Focal Loss
weights = ppdist.get_class_weights(labels)  # Inverse frequency weights

# Binding pocket extraction
pocket_df = ppdist.extract_binding_pocket(
    structure, ligand_coords,
    binding_threshold=5.0, non_binding_threshold=8.0
)
```

---

## Installation

```bash
pip install git+https://github.com/unibio-hkust-gz-sunnylab/pypropel_unibio.git

# For ESM embeddings (optional)
pip install fair-esm
```

## Requirements

- Python >= 3.9
- BioPython >= 1.81
- RDKit >= 2023.0.0
- NumPy >= 1.24.0
- Pandas >= 2.0.0
- fair-esm (optional, for ESM-2 embeddings)
