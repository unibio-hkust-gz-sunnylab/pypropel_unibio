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

# Vector features for GVP encoder (4 vectors)
sidechain = ppgvp.get_residue_orientations(structure)      # Cα→Cβ unit vectors
forward = ppgvp.get_backbone_vectors(structure)            # Cα[i]→Cα[i+1]
backward = ppgvp.get_backward_vectors(structure)           # Cα[i]→Cα[i-1]
neighbors = ppgvp.get_neighbor_center_vectors(structure, k=10)  # Cα→neighbor center

# Combined GVP features (4 vectors per residue)
coords, vectors = ppgvp.get_gvp_node_features(structure, k_neighbors=10)
# coords: (N, 3), vectors: (N, 4, 3) - 4 unit vectors per residue

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

### `pypropel.features` - Unified Feature Extraction (NEW)

Configurable, general-purpose feature extraction functions:

```python
import pypropel.features as ppfeat

# List available feature classes
print(ppfeat.list_feature_classes('protein'))
# {'onehot': 20, 'ss': 3, 'sasa': 1, 'charge': 1, ...}

print(ppfeat.list_feature_classes('ligand'))
# {'atom_type': 10, 'hybridization': 4, 'aromaticity': 1, ...}

# Protein features - configurable
features = ppfeat.get_protein_features(
    structure,
    feature_classes=['onehot', 'ss', 'sasa', 'charge', 'hydrophobicity'],
    include_gvp=True,      # Include GVP vector features
    use_dssp=True          # Use DSSP for SS and SASA
)
# Returns: {
#   'scalar_features': (N, 26) - Combined scalar features
#   'gvp_coords': (N, 3) - Cα coordinates  
#   'gvp_vectors': (N, 4, 3) - 4 GVP vectors
#   'feature_dims': {'onehot': (0, 20), 'ss': (20, 23), ...}
# }

# Ligand features - configurable
features = ppfeat.get_ligand_features(
    mol,
    include_global_tag=True,  # Append global FP to each atom
    global_tag_dim=45
)
# Returns: {
#   'atom_features': (M, 64) - Combined atom features
#   'coords': (M, 3) - Atom coordinates
#   'feature_dims': {'atom_type': (0, 10), ...}
# }

# Binding labels - configurable thresholds
labels = ppfeat.get_binding_labels(
    structure, ligand_coords,
    thresholds=[3.5, 6.0]  # 3-class: Contact/Near/Far
)
# Returns: {'labels': (N,), 'distances': (N,), 'class_weights': (3,)}
```

---

### Extended Atom Features (64-dim)

```python
import pypropel.mol as ppmol

# Extended atom features for GVP-Fusion v2 (64 dimensions)
atom_features = ppmol.get_atom_features_extended(mol, global_tag_dim=45)
# Shape: (N_atoms, 64)
# Breakdown:
#   [0-9]   Atom type one-hot (10): C, N, O, S, F, P, Cl, Br, I, Other
#   [10-13] Hybridization (4): SP, SP2, SP3, Aromatic
#   [14]    Aromaticity (1): Binary
#   [15]    Is_Donor (1): Binary
#   [16]    Is_Acceptor (1): Binary
#   [17]    Gasteiger Charge (1): Float
#   [18]    Ring Size (1): Integer (0 if not in ring)
#   [19-63] Global Tag (45): Morgan FP + LogP + TPSA

# Ring size per atom
ring_sizes = ppmol.get_atom_ring_sizes(mol)  # Shape: (N, 1)

# Global tag (molecular fingerprint for each atom)
global_tag = ppmol.get_global_tag(mol, output_dim=45)  # Shape: (45,)
```

---

### Protein Scalar Features (35-dim)

```python
import pypropel.fpsite as fpsite

# Full scalar features for GVP-Fusion v2 (35 dimensions)
features = fpsite.get_protein_scalar_features(structure, use_dssp=True)
# Shape: (N_residues, 35)
# Breakdown:
#   [0-19]  AA one-hot (20)
#   [20-22] Secondary structure one-hot (3): Helix/Sheet/Coil
#   [23]    SASA normalized (1)
#   [24]    Charge (1)
#   [25]    Hydrophobicity normalized (1)
#   [26]    Is_Aromatic (1): Phe/Tyr/Trp/His
#   [27]    H-Bond Donor count (1)
#   [28]    H-Bond Acceptor count (1)
#   [29-34] Reserved/padding (6)
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
