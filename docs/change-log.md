# Change Log

## UniO Fork Extensions

[Jan. 19th 2026: pypropel_unibio fork]

- **pypropel.mol**: New module for ligand loading (SDF/MOL2), coordinates, features
- **pypropel.str**: Added `load_pdb()`, `read_pdb_sequence()`, `structure_to_sequence()`
- **pypropel.fpsite**: Added residue features (`residue_one_hot`, `residue_physchem`, `residue_coords`)
- **pypropel.fpsite**: Added AA validation (`is_standard_aa`, `is_standard_aa_name`)
- **pypropel.fpsite**: Added window functions (`get_residue_window`, `get_residue_window_padded`)
- **pypropel.fpsite**: Added spatial neighbors (`get_spatial_neighbors`, `get_spatial_neighbor_indices`)
- **pypropel.fpsite**: Added positional encoding (`positional_encoding`, `batch_positional_encoding`, `relative_position`)
- **pypropel.dist**: Added `ContactMap` class with distance and contact map functions
- **pypropel.dist**: Added `extract_binding_pocket()` for binding site classification
- **CI/CD**: Added `tests.yml` and `lint.yml` workflows (43 tests passing)

---

## Original PyPropel

[Dec. 8th 2024: version = "0.1.3"] adding extracting from UniProt fasta sequences to single sequences according to species.
[Jul. 3th 2025: version = "0.1.5"] making function pypropel_struct_check_cplx
[Jul. 7th 2025: version = "0.1.6"] making function pypropel_struct_dist_cplx1
[Jul. 20th 2025: version = "0.1.7"] harmonising label header protocols 2021 and 2025
    
    1. tutorial/protein/convert/