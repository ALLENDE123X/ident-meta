"""
Tests for PerMethodRegressor with all 5 model types.

Reference: pde-selector-implementation-plan.md §13
"""

import sys
import os
import numpy as np
import pytest

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import PerMethodRegressor


# generate synthetic dataset
@pytest.fixture
def synthetic_data():
    """generate synthetic X (n=64, p=8) and Y via known nonlinear map with noise."""
    np.random.seed(0)
    n_samples = 64
    n_features = 12  # tiny-12 features
    
    # generate X
    X = np.random.randn(n_samples, n_features)
    
    # generate Y via nonlinear map: y = f(X) + noise
    # use a simple nonlinear function: y_i = sum(x_j^2) + noise
    Y = np.zeros((n_samples, 3))
    for i in range(3):
        Y[:, i] = np.sum(X[:, :4] ** 2, axis=1) + 0.1 * np.random.randn(n_samples)
        Y[:, i] = np.abs(Y[:, i])  # ensure non-negative
    
    return X, Y


@pytest.mark.parametrize("model_name", [
    "linear_ols",
    "ridge_multi",
    "regressor_chain_ridge",
    "rf_multi",
    "catboost_multi",
])
def test_all_models_fit_predict(model_name, synthetic_data):
    """test that all 5 models can fit and predict."""
    X, Y = synthetic_data
    
    # skip catboost if not installed
    if model_name == "catboost_multi":
        try:
            import catboost
        except ImportError:
            pytest.skip("catboost not installed")
    
    # create model with small params for speed
    if model_name == "rf_multi":
        model = PerMethodRegressor(model_name=model_name, n_estimators=10, max_depth=3)
    elif model_name == "catboost_multi":
        model = PerMethodRegressor(model_name=model_name, iterations=10, depth=3, verbose=False)
    elif model_name == "ridge_multi":
        model = PerMethodRegressor(model_name=model_name, alpha=1.0)
    else:
        model = PerMethodRegressor(model_name=model_name)
    
    # fit
    model.fit(X, Y)
    
    # predict
    Y_pred = model.predict(X)
    
    # check shape
    assert Y_pred.shape == (64, 3), f"Expected shape (64, 3), got {Y_pred.shape}"
    
    # check non-negative
    assert np.all(Y_pred >= 0), "Predictions should be non-negative"
    
    # check finite
    assert np.all(np.isfinite(Y_pred)), "Predictions should be finite"
    
    # check no crashes
    assert True  # if we got here, no crash occurred


def test_rf_multi_variance(synthetic_data):
    """test that rf_multi returns nonnegative variance."""
    X, Y = synthetic_data
    
    model = PerMethodRegressor(model_name="rf_multi", n_estimators=10, max_depth=3)
    model.fit(X, Y)
    
    # predict uncertainty
    var_y = model.predict_unc(X)
    
    # check shape
    assert var_y.shape == (64, 3), f"Expected shape (64, 3), got {var_y.shape}"
    
    # check non-negative
    assert np.all(var_y >= 0), "Variance should be non-negative"
    
    # check finite
    assert np.all(np.isfinite(var_y)), "Variance should be finite"


@pytest.mark.parametrize("model_name", [
    "linear_ols",
    "ridge_multi",
    "regressor_chain_ridge",
    "catboost_multi",
])
def test_non_rf_models_return_nan_variance(model_name, synthetic_data):
    """test that non-RF models return NaN for variance."""
    X, Y = synthetic_data
    
    # skip catboost if not installed
    if model_name == "catboost_multi":
        try:
            import catboost
        except ImportError:
            pytest.skip("catboost not installed")
    
    # create model
    if model_name == "catboost_multi":
        model = PerMethodRegressor(model_name=model_name, iterations=10, depth=3, verbose=False)
    elif model_name == "ridge_multi":
        model = PerMethodRegressor(model_name=model_name, alpha=1.0)
    else:
        model = PerMethodRegressor(model_name=model_name)
    
    model.fit(X, Y)
    
    # predict uncertainty
    var_y = model.predict_unc(X)
    
    # check shape
    assert var_y.shape == (64, 3), f"Expected shape (64, 3), got {var_y.shape}"
    
    # check all NaN
    assert np.all(np.isnan(var_y)), f"{model_name} should return NaN for variance"


def test_save_load(synthetic_data, tmp_path):
    """test saving and loading model."""
    X, Y = synthetic_data
    
    # test with rf_multi (most common)
    model = PerMethodRegressor(model_name="rf_multi", n_estimators=10, max_depth=3)
    model.fit(X, Y)
    
    # save
    model_path = tmp_path / "test_model.joblib"
    model.save(str(model_path))
    
    # load
    model_loaded = PerMethodRegressor.load(str(model_path))
    
    # check that predictions match
    Y_pred_orig = model.predict(X[:5])
    Y_pred_loaded = model_loaded.predict(X[:5])
    
    assert np.allclose(
        Y_pred_orig, Y_pred_loaded, rtol=1e-5
    ), "Loaded model predictions don't match original"


def test_feature_importances(synthetic_data):
    """test feature importance extraction."""
    X, Y = synthetic_data
    
    model = PerMethodRegressor(model_name="rf_multi", n_estimators=10, max_depth=3)
    model.fit(X, Y)
    
    # get importances
    importances = model.get_feature_importances()
    
    # check shape
    assert importances.shape == (12,), f"Importances shape: {importances.shape}"
    
    # check non-negative
    assert np.all(importances >= 0), "Importances should be non-negative"


def test_catboost_import_error():
    """test that catboost raises clear ImportError if not installed."""
    # this test assumes catboost is not installed in test env
    # if it is installed, we skip
    try:
        import catboost
        pytest.skip("catboost is installed, skipping import error test")
    except ImportError:
        # test that factory raises clear error
        from src.models.factory import create_model
        with pytest.raises(ImportError, match="catboost is not installed"):
            create_model("catboost_multi")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
