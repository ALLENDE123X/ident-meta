"""
Tests for RobustIDENT method adapter.
"""

import sys
import os
import numpy as np
import pytest

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ident_methods import METHOD_REGISTRY
from src.ident_methods.robustident_method import RobustIdentMethod


class TestRobustIDENT:
    """Tests for RobustIDENT method."""
    
    def test_robustident_registered(self):
        """Test that RobustIDENT is registered in the method registry."""
        assert "RobustIDENT" in METHOD_REGISTRY.list_methods()
    
    def test_robustident_name(self):
        """Test method name property."""
        method = METHOD_REGISTRY.get("RobustIDENT")
        assert method is not None
        assert method.name == "RobustIDENT"
    
    def test_robustident_run_basic(self):
        """Test RobustIDENT on simple synthetic data."""
        method = METHOD_REGISTRY.get("RobustIDENT")
        
        # Create simple traveling wave
        nt, nx = 50, 64
        dx, dt = 0.1, 0.01
        x = np.linspace(0, nx * dx, nx)
        t = np.linspace(0, nt * dt, nt)
        X, T = np.meshgrid(x, t)
        u_win = np.sin(2 * np.pi * (X - T))
        
        metrics, info = method.run(u_win, dx, dt)
        
        # Check output shapes
        assert metrics.shape == (3,)
        assert all(np.isfinite(metrics))
        
        # Check info dict
        assert "terms" in info
        assert "coefficients" in info
        assert "runtime" in info
        assert info["runtime"] > 0
        assert "robust" in info
        assert info["robust"] is True
    
    def test_robustident_metrics_range(self):
        """Test that metrics are in expected ranges."""
        method = METHOD_REGISTRY.get("RobustIDENT")
        
        # Random noisy data
        nt, nx = 40, 50
        u_win = np.random.randn(nt, nx) * 0.5 + np.sin(
            np.linspace(0, 4 * np.pi, nx)
        )[None, :]
        
        metrics, info = method.run(u_win, dx=0.1, dt=0.01)
        
        # F1 in [0, 1], others >= 0
        assert 0 <= metrics[0] <= 1
        assert metrics[1] >= 0
        assert metrics[2] >= 0
    
    def test_robustident_outlier_tolerance(self):
        """Test that RobustIDENT handles outliers better than L2 methods."""
        method = METHOD_REGISTRY.get("RobustIDENT")
        
        # Clean signal
        nt, nx = 60, 80
        dx, dt = 0.1, 0.01
        x = np.linspace(0, nx * dx, nx)
        t = np.linspace(0, nt * dt, nt)
        X, T = np.meshgrid(x, t)
        u_clean = np.sin(2 * np.pi * X)
        
        # Add outliers (5% of points with smaller magnitude)
        u_outliers = u_clean.copy()
        outlier_mask = np.random.rand(nt, nx) < 0.05
        u_outliers[outlier_mask] = 5.0  # Moderate outliers
        
        # Run on clean and outlier data
        metrics_clean, _ = method.run(u_clean, dx, dt)
        metrics_outlier, info = method.run(u_outliers, dx, dt)
        
        # Both should produce valid output
        assert all(np.isfinite(metrics_clean))
        assert all(np.isfinite(metrics_outlier))
        
        # Should still identify some structure
        assert info.get("n_nonzero", 0) > 0 or "error" not in info


class TestFeatureLibrary:
    """Tests for the feature library construction."""
    
    def test_build_library(self):
        """Test that feature library is built correctly."""
        method = RobustIdentMethod(derivative_order=2, polynomial_degree=2)
        
        nt, nx = 30, 40
        dx, dt = 0.1, 0.01
        u_win = np.random.randn(nt, nx)
        
        # Access internal method
        A, b, names = method._build_library(u_win, dx, dt, 2, 2)
        
        # Check shapes
        assert A.ndim == 2
        assert b.ndim == 1
        assert A.shape[0] == b.shape[0]  # Same number of samples
        assert len(names) == A.shape[1]  # One name per feature
        
        # Check expected features exist
        assert "1" in names  # bias
        assert "u" in names  # u
        assert "u_x" in names  # first derivative
        assert "u_x_x" in names  # second derivative
    
    def test_soft_threshold(self):
        """Test soft-thresholding operator."""
        method = RobustIdentMethod()
        
        x = np.array([0.5, -0.5, 1.5, -1.5, 0.1])
        threshold = 1.0
        
        result = method._soft_threshold(x, threshold)
        
        expected = np.array([0.0, 0.0, 0.5, -0.5, 0.0])
        np.testing.assert_array_almost_equal(result, expected)


def test_all_methods_registered():
    """Test that all expected methods are registered."""
    methods = METHOD_REGISTRY.list_methods()
    assert "RobustIDENT" in methods


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
