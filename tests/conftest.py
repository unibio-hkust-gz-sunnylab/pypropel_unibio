"""
Pytest configuration and fixtures for pypropel tests.
"""

import pytest
import numpy as np
import tempfile
import os


@pytest.fixture
def sample_coords_3d():
    """Sample 3D coordinates for testing."""
    return np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])


@pytest.fixture
def sample_residue_coords():
    """Sample residue atom coordinates."""
    return np.array([
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [2.0, 2.0, 2.0],
    ])


@pytest.fixture
def sample_ligand_coords():
    """Sample ligand atom coordinates."""
    return np.array([
        [5.0, 5.0, 5.0],
        [6.0, 6.0, 6.0],
        [7.0, 7.0, 7.0],
    ])


@pytest.fixture
def sample_sequence():
    """Sample amino acid sequence."""
    return "ACDEFGHIKLMNPQRSTVWY"


@pytest.fixture
def temp_pdb_file():
    """Create a temporary minimal PDB file for testing."""
    pdb_content = """HEADER    TEST PROTEIN
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C
ATOM      3  C   ALA A   1       2.009   1.420   0.000  1.00  0.00           C
ATOM      4  O   ALA A   1       1.251   2.390   0.000  1.00  0.00           O
ATOM      5  CB  ALA A   1       1.986  -0.760   1.216  1.00  0.00           C
ATOM      6  N   GLY A   2       3.310   1.540   0.000  1.00  0.00           N
ATOM      7  CA  GLY A   2       3.970   2.840   0.000  1.00  0.00           C
ATOM      8  C   GLY A   2       5.470   2.720   0.000  1.00  0.00           C
ATOM      9  O   GLY A   2       6.030   1.620   0.000  1.00  0.00           O
ATOM     10  N   SER A   3       6.110   3.880   0.000  1.00  0.00           N
ATOM     11  CA  SER A   3       7.560   3.990   0.000  1.00  0.00           C
ATOM     12  C   SER A   3       8.110   5.400   0.000  1.00  0.00           C
ATOM     13  O   SER A   3       7.350   6.370   0.000  1.00  0.00           O
ATOM     14  CB  SER A   3       8.090   3.230   1.216  1.00  0.00           C
ATOM     15  OG  SER A   3       7.590   1.900   1.216  1.00  0.00           O
END
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f:
        f.write(pdb_content)
        pdb_path = f.name
    
    yield pdb_path
    
    # Cleanup
    os.unlink(pdb_path)


@pytest.fixture
def temp_sdf_file():
    """Create a temporary minimal SDF file for testing."""
    sdf_content = """
     RDKit          3D

  6  6  0  0  0  0  0  0  0  0999 V2000
    1.2124    0.7000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    1.2124   -0.7000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.0000   -1.4000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
   -1.2124   -0.7000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
   -1.2124    0.7000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.0000    1.4000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  2  0
  2  3  1  0
  3  4  2  0
  4  5  1  0
  5  6  2  0
  6  1  1  0
M  END
$$$$
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.sdf', delete=False) as f:
        f.write(sdf_content)
        sdf_path = f.name
    
    yield sdf_path
    
    # Cleanup
    os.unlink(sdf_path)
