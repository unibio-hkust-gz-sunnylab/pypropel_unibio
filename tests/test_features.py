"""
Tests for pypropel.features module - unified feature extraction.
"""

import pytest
import numpy as np


class TestProteinFeatureClasses:
    """Tests for protein feature class registry."""
    
    def test_list_protein_features(self):
        """Test listing available protein feature classes."""
        from pypropel.features import list_feature_classes, PROTEIN_FEATURE_CLASSES
        
        classes = list_feature_classes('protein')
        assert 'onehot' in classes
        assert 'ss' in classes
        assert 'sasa' in classes
        assert 'charge' in classes
        assert classes == PROTEIN_FEATURE_CLASSES
    
    def test_compute_feature_dim(self):
        """Test computing feature dimensions."""
        from pypropel.features import compute_feature_dim
        
        dim = compute_feature_dim(['onehot', 'charge'], 'protein')
        assert dim == 21  # 20 + 1
        
        dim = compute_feature_dim(['onehot', 'ss', 'sasa', 'charge', 'hydrophobicity'], 'protein')
        assert dim == 26  # 20 + 3 + 1 + 1 + 1


class TestGetProteinFeatures:
    """Tests for get_protein_features function."""
    
    def test_protein_features_shape(self, temp_pdb_file):
        """Test that protein features have correct shapes."""
        from pypropel.features import get_protein_features
        from Bio.PDB import PDBParser
        
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('test', temp_pdb_file)
        
        features = get_protein_features(structure, use_dssp=False)
        
        n_res = features['n_residues']
        assert n_res > 0
        
        # Default should include all feature classes = 29 dims
        # onehot(20) + ss(3) + sasa(1) + charge(1) + hydro(1) + aromatic(1) + hbd(1) + hba(1)
        expected_dim = 20 + 3 + 1 + 1 + 1 + 1 + 1 + 1
        assert features['scalar_features'].shape == (n_res, expected_dim)
    
    def test_protein_features_selected_classes(self, temp_pdb_file):
        """Test selecting specific feature classes."""
        from pypropel.features import get_protein_features
        from Bio.PDB import PDBParser
        
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('test', temp_pdb_file)
        
        features = get_protein_features(
            structure, 
            feature_classes=['onehot', 'charge'],
            use_dssp=False
        )
        
        n_res = features['n_residues']
        # onehot(20) + charge(1) = 21
        assert features['scalar_features'].shape == (n_res, 21)
        assert 'onehot' in features['feature_dims']
        assert 'charge' in features['feature_dims']
        assert 'ss' not in features['feature_dims']
    
    def test_protein_features_with_gvp(self, temp_pdb_file):
        """Test including GVP features."""
        from pypropel.features import get_protein_features
        from Bio.PDB import PDBParser
        
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('test', temp_pdb_file)
        
        features = get_protein_features(
            structure,
            feature_classes=['onehot'],
            include_gvp=True,
            use_dssp=False
        )
        
        n_res = features['n_residues']
        assert 'gvp_coords' in features
        assert 'gvp_vectors' in features
        assert features['gvp_coords'].shape == (n_res, 3)
        assert features['gvp_vectors'].shape == (n_res, 4, 3)  # 4 vectors
    
    def test_protein_features_with_esm(self, temp_pdb_file):
        """Test including ESM embeddings."""
        from pypropel.features import get_protein_features
        from Bio.PDB import PDBParser
        
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('test', temp_pdb_file)
        
        # Create mock ESM embeddings
        n_res = 3  # temp_pdb_file has 3 residues
        mock_esm = np.random.randn(n_res, 1280).astype(np.float32)
        
        features = get_protein_features(
            structure,
            feature_classes=['onehot'],
            esm_embeddings=mock_esm,
            use_dssp=False
        )
        
        assert 'esm' in features
        assert features['esm'].shape == (n_res, 1280)
    
    def test_invalid_feature_class(self, temp_pdb_file):
        """Test that invalid feature class raises error."""
        from pypropel.features import get_protein_features
        from Bio.PDB import PDBParser
        
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('test', temp_pdb_file)
        
        with pytest.raises(ValueError, match="Unknown feature class"):
            get_protein_features(structure, feature_classes=['invalid_class'])


class TestLigandFeatureClasses:
    """Tests for ligand feature class registry."""
    
    def test_list_ligand_features(self):
        """Test listing available ligand feature classes."""
        from pypropel.features import list_feature_classes, LIGAND_FEATURE_CLASSES
        
        classes = list_feature_classes('ligand')
        assert 'atom_type' in classes
        assert 'hybridization' in classes
        assert 'partial_charge' in classes
        assert classes == LIGAND_FEATURE_CLASSES


class TestGetLigandFeatures:
    """Tests for get_ligand_features function."""
    
    def test_ligand_features_shape(self, temp_sdf_file):
        """Test that ligand features have correct shapes."""
        from pypropel.features import get_ligand_features
        import pypropel.mol as ppmol
        
        mol = ppmol.load_sdf(temp_sdf_file)
        features = get_ligand_features(mol)
        
        n_atoms = features['n_atoms']
        assert n_atoms > 0
        
        # Default with global_tag = 64 dims
        # atom_type(10) + hybrid(4) + aromatic(1) + donor(1) + acceptor(1) + charge(1) + ring(1) + global(45)
        expected_dim = 10 + 4 + 1 + 1 + 1 + 1 + 1 + 45
        assert features['atom_features'].shape == (n_atoms, expected_dim)
        assert features['coords'].shape == (n_atoms, 3)
    
    def test_ligand_features_no_global_tag(self, temp_sdf_file):
        """Test excluding global tag."""
        from pypropel.features import get_ligand_features
        import pypropel.mol as ppmol
        
        mol = ppmol.load_sdf(temp_sdf_file)
        features = get_ligand_features(mol, include_global_tag=False)
        
        n_atoms = features['n_atoms']
        # Without global_tag = 19 dims
        expected_dim = 10 + 4 + 1 + 1 + 1 + 1 + 1
        assert features['atom_features'].shape == (n_atoms, expected_dim)
        assert 'global_tag' not in features['feature_dims']
    
    def test_ligand_features_none_mol(self):
        """Test handling None molecule."""
        from pypropel.features import get_ligand_features
        
        features = get_ligand_features(None)
        assert features['n_atoms'] == 0
        assert features['atom_features'].shape[0] == 0


class TestGetBindingLabels:
    """Tests for get_binding_labels function."""
    
    def test_binding_labels_shape(self, temp_pdb_file):
        """Test that binding labels have correct shape."""
        from pypropel.features import get_binding_labels
        from Bio.PDB import PDBParser
        
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('test', temp_pdb_file)
        
        # Create mock ligand coords
        ligand_coords = np.array([[10, 10, 10]], dtype=np.float32)
        
        result = get_binding_labels(structure, ligand_coords)
        
        assert 'labels' in result
        assert 'n_classes' in result
        assert result['n_classes'] == 3  # Default 2 thresholds = 3 classes
        assert 'distances' in result
        assert 'class_weights' in result
    
    def test_binding_labels_custom_thresholds(self, temp_pdb_file):
        """Test custom thresholds."""
        from pypropel.features import get_binding_labels
        from Bio.PDB import PDBParser
        
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('test', temp_pdb_file)
        
        ligand_coords = np.array([[10, 10, 10]], dtype=np.float32)
        
        result = get_binding_labels(
            structure, ligand_coords, 
            thresholds=[3.0, 5.0, 8.0]
        )
        
        assert result['n_classes'] == 4  # 3 thresholds = 4 classes
        assert result['thresholds'] == [3.0, 5.0, 8.0]
