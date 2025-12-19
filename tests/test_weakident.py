"""
Tests for WeakIDENT method adapter.
"""

import sys
import os
import numpy as np
import pytest

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ident_methods import METHOD_REGISTRY


class TestWeakIDENT:
    """Tests for WeakIDENT method."""
    
    def test_weakident_registered(self):
        """Test that WeakIDENT is registered in the method registry."""
        assert "WeakIDENT" in METHOD_REGISTRY.list_methods()
    
    def test_weakident_name(self):
        """Test method name property."""
        method = METHOD_REGISTRY.get("WeakIDENT")
        assert method is not None
        assert method.name == "WeakIDENT"
    
    def test_weakident_run_basic(self):
        """Test WeakIDENT on simple synthetic data."""
        method = METHOD_REGISTRY.get("WeakIDENT")
        
        # Create simple sine wave
        nt, nx = 50, 64
        dx, dt = 0.1, 0.01
        x = np.linspace(0, nx * dx, nx)
        t = np.linspace(0, nt * dt, nt)
        X, T = np.meshgrid(x, t)
        u_win = np.sin(2 * np.pi * X)
        
        metrics, info = method.run(u_win, dx, dt)
        
        # Check output shapes
        assert metrics.shape == (3,)
        assert all(np.isfinite(metrics))
        
        # Check info dict has expected keys
        assert "runtime" in info
        assert info["runtime"] > 0 or "error" in info
    
    def test_weakident_metrics_range(self):
        """Test that metrics are in expected ranges."""
        method = METHOD_REGISTRY.get("WeakIDENT")
        
        # Random noisy data
        nt, nx = 40, 50
        u_win = np.random.randn(nt, nx) * 0.3 + np.sin(
            np.linspace(0, 4 * np.pi, nx)
        )[None, :]
        
        metrics, info = method.run(u_win, dx=0.1, dt=0.01)
        
        # F1 in [0, 1], others >= 0
        assert 0 <= metrics[0] <= 1
        assert metrics[1] >= 0
        assert metrics[2] >= 0


def test_all_four_methods_registered():
    """Test that all 4 methods are registered."""
    methods = METHOD_REGISTRY.list_methods()
    assert "PySINDy" in methods
    assert "WSINDy" in methods
    assert "RobustIDENT" in methods
    assert "WeakIDENT" in methods
    assert len(set(methods)) >= 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
