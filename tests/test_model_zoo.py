"""
Tests for model zoo.
"""

import sys
import os
import numpy as np
import pytest

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.model_zoo import (
    get_available_models,
    create_model,
    train_model,
    evaluate_model,
    compare_all_models,
    print_comparison_table,
    get_feature_importance,
    MODEL_CONFIGS,
)


class TestModelZoo:
    """Tests for model zoo functionality."""
    
    def test_available_models(self):
        """Test that models are available."""
        available = get_available_models()
        assert len(available) >= 5  # At least 5 sklearn models
        assert "ridge" in available
        assert "random_forest" in available
        assert "svr" in available
        assert "knn" in available
        assert "mlp" in available
    
    def test_create_model_ridge(self):
        """Test creating Ridge model."""
        pipeline = create_model("ridge")
        assert pipeline is not None
        assert "scaler" in pipeline.named_steps
        assert "model" in pipeline.named_steps
    
    def test_create_model_random_forest(self):
        """Test creating Random Forest model."""
        pipeline = create_model("random_forest")
        assert pipeline is not None
        assert "model" in pipeline.named_steps
    
    def test_create_model_invalid(self):
        """Test that invalid model name raises error."""
        with pytest.raises(ValueError):
            create_model("invalid_model")
    
    def test_train_model(self):
        """Test training a model."""
        np.random.seed(42)
        X = np.random.randn(100, 12)
        y = np.random.randn(100)
        
        pipeline, train_time = train_model("random_forest", X, y)
        
        assert pipeline is not None
        assert train_time > 0
        
        # Should be able to predict
        y_pred = pipeline.predict(X[:5])
        assert y_pred.shape == (5,)
    
    def test_evaluate_model(self):
        """Test evaluating a model."""
        np.random.seed(42)
        X_train = np.random.randn(100, 12)
        y_train = np.random.randn(100)
        X_test = np.random.randn(20, 12)
        y_test = np.random.randn(20)
        
        pipeline, _ = train_model("ridge", X_train, y_train)
        metrics = evaluate_model(pipeline, X_test, y_test)
        
        assert "mse" in metrics
        assert "mae" in metrics
        assert "r2" in metrics
        assert metrics["mse"] >= 0
        assert metrics["mae"] >= 0
    
    def test_compare_all_models(self):
        """Test comparing all available models."""
        np.random.seed(42)
        X_train = np.random.randn(100, 12)
        y_train = np.random.randn(100)
        X_test = np.random.randn(20, 12)
        y_test = np.random.randn(20)
        
        results = compare_all_models(
            X_train, y_train, X_test, y_test, verbose=False
        )
        
        assert len(results) >= 5
        
        for model_key, data in results.items():
            if "error" not in data:
                assert "metrics" in data
                assert "train_time" in data
                assert "pipeline" in data
    
    def test_print_comparison_table(self):
        """Test generating comparison table."""
        np.random.seed(42)
        X_train = np.random.randn(50, 12)
        y_train = np.random.randn(50)
        X_test = np.random.randn(10, 12)
        y_test = np.random.randn(10)
        
        results = compare_all_models(
            X_train, y_train, X_test, y_test, verbose=False
        )
        
        table = print_comparison_table(results)
        
        assert "| Model |" in table
        assert "MSE" in table
        assert "R²" in table
    
    def test_feature_importance_rf(self):
        """Test getting feature importance from Random Forest."""
        np.random.seed(42)
        X = np.random.randn(100, 12)
        y = np.random.randn(100)
        
        pipeline, _ = train_model("random_forest", X, y)
        importance = get_feature_importance(pipeline)
        
        assert importance is not None
        assert len(importance) == 12


class TestIndividualModels:
    """Test each model type individually."""
    
    @pytest.fixture
    def data(self):
        """Generate test data."""
        np.random.seed(42)
        X_train = np.random.randn(100, 12)
        y_train = np.random.randn(100)
        X_test = np.random.randn(20, 12)
        y_test = np.random.randn(20)
        return X_train, y_train, X_test, y_test
    
    def test_ridge(self, data):
        """Test Ridge regression."""
        X_train, y_train, X_test, y_test = data
        pipeline, _ = train_model("ridge", X_train, y_train)
        metrics = evaluate_model(pipeline, X_test, y_test)
        assert np.isfinite(metrics["mse"])
    
    def test_svr(self, data):
        """Test SVR."""
        X_train, y_train, X_test, y_test = data
        pipeline, _ = train_model("svr", X_train, y_train)
        metrics = evaluate_model(pipeline, X_test, y_test)
        assert np.isfinite(metrics["mse"])
    
    def test_knn(self, data):
        """Test KNN."""
        X_train, y_train, X_test, y_test = data
        pipeline, _ = train_model("knn", X_train, y_train)
        metrics = evaluate_model(pipeline, X_test, y_test)
        assert np.isfinite(metrics["mse"])
    
    def test_mlp(self, data):
        """Test MLP."""
        X_train, y_train, X_test, y_test = data
        pipeline, _ = train_model("mlp", X_train, y_train)
        metrics = evaluate_model(pipeline, X_test, y_test)
        assert np.isfinite(metrics["mse"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
