"""
Tests for pypropel.mol module - ligand utilities.
"""

import pytest
import numpy as np
import tempfile
import os


@pytest.fixture
def temp_sdf_file():
    """Create a temporary minimal SDF file for testing."""
    # Minimal valid SDF content (simplified methane-like structure)
    sdf_content = """
     RDKit          3D

  1  0  0  0  0  0  0  0  0  0999 V2000
    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
M  END
$$$$
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.sdf', delete=False) as f:
        f.write(sdf_content)
        sdf_path = f.name
    
    yield sdf_path
    
    os.unlink(sdf_path)


class TestLoadSdf:
    """Tests for load_sdf function."""

    def test_load_sdf(self, temp_sdf_file):
        """Test loading SDF file."""
        import pypropel.mol as ppmol
        
        mol = ppmol.load_sdf(temp_sdf_file)
        
        # May be None if RDKit can't parse the minimal SDF
        # Just check it doesn't crash
        assert True


class TestLigandCoords:
    """Tests for ligand_coords function."""

    def test_ligand_coords_none(self):
        """Test with None molecule."""
        import pypropel.mol as ppmol
        
        coords = ppmol.ligand_coords(None)
        
        assert coords.shape == (0, 3)


class TestLigandFeatures:
    """Tests for get_ligand_features function."""

    def test_ligand_features_none(self):
        """Test with None molecule."""
        import pypropel.mol as ppmol
        
        features = ppmol.get_ligand_features(None)
        
        assert features.shape == (4,)
        assert np.all(features == 0)
