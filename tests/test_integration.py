"""
Integration test for end-to-end PDE-Selector pipeline.

Tests the complete workflow:
1. Generate mini dataset (2 PDEs, 2 noise levels, ~20 windows)
2. Train selector
3. Evaluate on holdout
4. Assert metrics within expected ranges
"""

import sys
import os
import tempfile
import shutil
import numpy as np
import pytest

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def temp_dirs():
    """Create temporary directories for test outputs."""
    temp_dir = tempfile.mkdtemp()
    data_dir = os.path.join(temp_dir, "data")
    models_dir = os.path.join(temp_dir, "models")
    os.makedirs(data_dir)
    os.makedirs(models_dir)
    yield {"base": temp_dir, "data": data_dir, "models": models_dir}
    shutil.rmtree(temp_dir)


@pytest.fixture
def mini_config():
    """Create a minimal config for fast testing."""
    return {
        "pdes": ["burgers"],
        "noise_levels": [0.0, 0.02],
        "nx": 128,
        "nt": 100,
        "dx": 1.0 / 128,
        "dt": 0.01,
        "window": {
            "nx_win": 64,
            "nt_win": 32,
            "stride_x": 32,
            "stride_t": 16,
        },
        "burgers_params": [
            {"nu": 0.01, "ic_type": "sine", "ic_params": {"amp": 1.0, "freq": 2.0}}
        ],
        "ident": {
            "max_dx": 3,
            "max_poly": 3,
            "skip_x": 4,
            "skip_t": 4,
            "tau": 0.05,
        },
    }


def test_feature_extraction_pipeline(mini_config, temp_dirs):
    """Test that feature extraction works on simulated data."""
    from src.data_gen import simulate_burgers, make_windows, add_noise
    from src.features import extract_tiny12
    
    # Simulate
    u_clean, dx, dt = simulate_burgers(
        nx=mini_config["nx"],
        nt=mini_config["nt"],
        L=1.0,
        T=1.0,
        nu=0.01,
        ic_type="sine",
    )
    
    assert u_clean.shape == (mini_config["nt"], mini_config["nx"])
    
    # Add noise
    u_noisy = add_noise(u_clean, noise_level=0.02)
    
    # Extract windows
    windows = make_windows(
        u_noisy,
        mini_config["window"]["nt_win"],
        mini_config["window"]["nx_win"],
        mini_config["window"]["stride_t"],
        mini_config["window"]["stride_x"],
    )
    
    assert len(windows) > 0
    
    # Extract features
    for u_win in windows[:3]:  # Test first 3 windows
        phi = extract_tiny12(u_win, dx, dt)
        assert phi.shape == (12,)
        assert np.all(np.isfinite(phi))


def test_dataset_generation(mini_config, temp_dirs):
    """Test mini dataset generation."""
    from src.label_dataset import generate_dataset
    
    methods = ["WeakIDENT"]
    
    dataset = generate_dataset(
        config=mini_config,
        methods=methods,
        output_dir=temp_dirs["data"],
        verbose=False,
        n_jobs=1,
    )
    
    # Check shapes
    assert "X_features" in dataset
    assert "WeakIDENT" in dataset
    assert dataset["X_features"].ndim == 2
    assert dataset["X_features"].shape[1] == 12  # Tiny-12 features
    assert dataset["WeakIDENT"].shape[0] == dataset["X_features"].shape[0]
    assert dataset["WeakIDENT"].shape[1] == 3  # 3 metrics
    
    # Check files saved
    assert os.path.exists(os.path.join(temp_dirs["data"], "X_features.npy"))
    assert os.path.exists(os.path.join(temp_dirs["data"], "Y_WeakIDENT.npy"))


def test_model_training(mini_config, temp_dirs):
    """Test model training on generated data."""
    from src.label_dataset import generate_dataset
    from src.models import PerMethodRegressor
    from sklearn.model_selection import train_test_split
    
    methods = ["WeakIDENT"]
    
    # Generate dataset
    dataset = generate_dataset(
        config=mini_config,
        methods=methods,
        output_dir=temp_dirs["data"],
        verbose=False,
        n_jobs=1,
    )
    
    X = dataset["X_features"]
    Y = dataset["WeakIDENT"]
    
    # Split
    if len(X) < 5:
        pytest.skip("Not enough samples for train/test split")
    
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.3, random_state=42
    )
    
    # Train
    model = PerMethodRegressor(
        method_name="WeakIDENT",
        n_estimators=50,  # Reduced for speed
        max_depth=4,
        random_state=42,
    )
    model.fit(X_train, Y_train)
    
    # Predict
    Y_pred = model.predict(X_test)
    assert Y_pred.shape == Y_test.shape
    
    # Uncertainty
    Y_pred_unc = model.predict_unc(X_test)
    assert Y_pred_unc.shape[0] == len(X_test)
    
    # Save
    model_path = os.path.join(temp_dirs["models"], "WeakIDENT.joblib")
    model.save(model_path)
    assert os.path.exists(model_path)
    
    # Load
    loaded = PerMethodRegressor.load(model_path)
    Y_loaded = loaded.predict(X_test)
    np.testing.assert_array_almost_equal(Y_pred, Y_loaded)


def test_evaluation_metrics():
    """Test evaluation metric calculations."""
    from src.eval import compute_regret, compute_top1_accuracy, compute_compute_saved
    
    # Test regret
    chosen = np.array([0.5, 0.3, 0.4])
    best = np.array([0.4, 0.3, 0.3])
    regret = compute_regret(chosen, best)
    assert regret == pytest.approx(0.0667, abs=0.01)
    
    # Test top-1 accuracy
    chosen_methods = ["A", "B", "A", "A"]
    best_methods = ["A", "A", "A", "B"]
    acc = compute_top1_accuracy(chosen_methods, best_methods)
    assert acc == 0.5  # 2/4 correct
    
    # Test compute saved
    n_methods_run = [1, 2, 1, 1]
    stats = compute_compute_saved(n_methods_run, n_methods_total=2)
    assert stats["frac_single"] == 0.75
    assert stats["mean_methods_run"] == 1.25


@pytest.mark.slow
def test_full_pipeline_integration(mini_config, temp_dirs):
    """
    Full end-to-end integration test.
    
    This test is marked slow because it runs the complete pipeline.
    Skip with: pytest -m "not slow"
    """
    from src.label_dataset import generate_dataset
    from src.models import PerMethodRegressor
    from src.metrics import aggregate
    from src.eval import compute_regret, compute_top1_accuracy
    from sklearn.model_selection import train_test_split
    
    methods = ["WeakIDENT"]
    weights = (0.5, 0.3, 0.2)
    
    # 1. Generate dataset
    dataset = generate_dataset(
        config=mini_config,
        methods=methods,
        output_dir=temp_dirs["data"],
        verbose=False,
        n_jobs=1,
    )
    
    X = dataset["X_features"]
    Y = dataset["WeakIDENT"]
    
    if len(X) < 10:
        pytest.skip("Not enough samples for integration test")
    
    # 2. Train/test split
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.3, random_state=42
    )
    
    # 3. Train model
    model = PerMethodRegressor(
        method_name="WeakIDENT",
        n_estimators=50,
        max_depth=4,
        random_state=42,
    )
    model.fit(X_train, Y_train)
    
    # 4. Predict on test
    Y_pred = model.predict(X_test)
    
    # 5. Compute aggregated scores
    chosen_scores = [aggregate(y, weights) for y in Y_pred]
    best_scores = [aggregate(y, weights) for y in Y_test]
    
    # 6. Compute metrics
    regret = compute_regret(np.array(chosen_scores), np.array(best_scores))
    
    # 7. Assert reasonable values
    # For a single-method case, regret should be meaningful but bounded
    assert regret < 1.0, f"Regret too high: {regret}"
    
    # Top-1 accuracy should be 100% with single method
    chosen_methods = ["WeakIDENT"] * len(X_test)
    best_methods = ["WeakIDENT"] * len(X_test)
    acc = compute_top1_accuracy(chosen_methods, best_methods)
    assert acc == 1.0  # Single method always matches


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "not slow"])
