"""
Tests for checkpoint and resume functionality in label_dataset.py.
"""

import sys
import os
import numpy as np
import tempfile
import shutil
import pytest

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.label_dataset import (
    _save_checkpoint,
    _load_checkpoint,
    _compute_config_hash,
    generate_dataset,
)


@pytest.fixture
def temp_output_dir():
    """Create a temporary directory for test outputs."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


def test_checkpoint_save_and_load(temp_output_dir):
    """Test that checkpoints save and load correctly."""
    methods = ["WeakIDENT"]
    config_hash = "abc123"
    
    # Create test data
    X_features = [np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])]
    Y_dict = {"WeakIDENT": [np.array([0.5, 0.1, 0.2]), np.array([0.6, 0.2, 0.3])]}
    processed_windows = [0, 1]
    
    # Save checkpoint
    _save_checkpoint(
        temp_output_dir, X_features, Y_dict,
        processed_windows, config_hash, methods
    )
    
    # Verify files exist
    assert os.path.exists(os.path.join(temp_output_dir, ".checkpoint.npz"))
    assert os.path.exists(os.path.join(temp_output_dir, ".checkpoint_meta.json"))
    
    # Load checkpoint
    loaded_X, loaded_Y, loaded_windows = _load_checkpoint(
        temp_output_dir, methods, config_hash
    )
    
    # Verify data matches
    assert len(loaded_X) == 2
    assert len(loaded_Y["WeakIDENT"]) == 2
    assert loaded_windows == {0, 1}
    np.testing.assert_array_almost_equal(loaded_X[0], X_features[0])
    np.testing.assert_array_almost_equal(loaded_Y["WeakIDENT"][0], Y_dict["WeakIDENT"][0])


def test_checkpoint_config_mismatch(temp_output_dir):
    """Test that mismatched config hash returns empty data."""
    methods = ["WeakIDENT"]
    
    # Save with one config hash
    X_features = [np.array([1.0, 2.0, 3.0])]
    Y_dict = {"WeakIDENT": [np.array([0.5, 0.1, 0.2])]}
    _save_checkpoint(
        temp_output_dir, X_features, Y_dict,
        [0], "config_hash_1", methods
    )
    
    # Load with different config hash
    loaded_X, loaded_Y, loaded_windows = _load_checkpoint(
        temp_output_dir, methods, "different_hash"
    )
    
    # Should return empty
    assert len(loaded_X) == 0
    assert len(loaded_Y["WeakIDENT"]) == 0
    assert len(loaded_windows) == 0


def test_checkpoint_no_file(temp_output_dir):
    """Test loading when no checkpoint exists."""
    methods = ["WeakIDENT"]
    
    loaded_X, loaded_Y, loaded_windows = _load_checkpoint(
        temp_output_dir, methods, "any_hash"
    )
    
    assert len(loaded_X) == 0
    assert len(loaded_Y["WeakIDENT"]) == 0
    assert len(loaded_windows) == 0


def test_config_hash_deterministic():
    """Test that config hash is deterministic."""
    config = {"pdes": ["burgers"], "noise_levels": [0.0, 0.01]}
    methods = ["WeakIDENT"]
    
    hash1 = _compute_config_hash(config, methods)
    hash2 = _compute_config_hash(config, methods)
    
    assert hash1 == hash2
    assert len(hash1) == 16  # MD5 truncated


def test_config_hash_changes_with_config():
    """Test that different configs produce different hashes."""
    config1 = {"pdes": ["burgers"]}
    config2 = {"pdes": ["kdv"]}
    methods = ["WeakIDENT"]
    
    hash1 = _compute_config_hash(config1, methods)
    hash2 = _compute_config_hash(config2, methods)
    
    assert hash1 != hash2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
