"""
Tests for PySINDy and WSINDy method adapters.
"""

import sys
import os
import numpy as np
import pytest

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ident_methods import METHOD_REGISTRY


# Check if pysindy is available
try:
    import pysindy
    PYSINDY_AVAILABLE = True
except ImportError:
    PYSINDY_AVAILABLE = False


@pytest.mark.skipif(not PYSINDY_AVAILABLE, reason="pysindy not installed")
class TestPySINDy:
    """Tests for PySINDy method."""
    
    def test_pysindy_registered(self):
        """Test that PySINDy is registered in the method registry."""
        assert "PySINDy" in METHOD_REGISTRY.list_methods()
    
    def test_pysindy_name(self):
        """Test method name property."""
        method = METHOD_REGISTRY.get("PySINDy")
        assert method is not None
        assert method.name == "PySINDy"
    
    def test_pysindy_run_basic(self):
        """Test PySINDy on simple synthetic data."""
        method = METHOD_REGISTRY.get("PySINDy")
        
        # Create simple traveling wave
        nt, nx = 50, 64
        dx, dt = 0.1, 0.01
        x = np.linspace(0, nx * dx, nx)
        t = np.linspace(0, nt * dt, nt)
        X, T = np.meshgrid(x, t)
        u_win = np.sin(2 * np.pi * (X - T))  # traveling wave
        
        metrics, info = method.run(u_win, dx, dt)
        
        # Check output shapes
        assert metrics.shape == (3,)
        assert all(np.isfinite(metrics))
        
        # Check info dict
        assert "terms" in info
        assert "coefficients" in info
        assert "runtime" in info
        assert info["runtime"] > 0
    
    def test_pysindy_metrics_range(self):
        """Test that metrics are in expected ranges."""
        method = METHOD_REGISTRY.get("PySINDy")
        
        # Random noisy data
        nt, nx = 40, 50
        u_win = np.random.randn(nt, nx)
        
        metrics, info = method.run(u_win, dx=0.1, dt=0.01)
        
        # F1 in [0, 1], others >= 0
        assert 0 <= metrics[0] <= 1
        assert metrics[1] >= 0
        assert metrics[2] >= 0


@pytest.mark.skipif(not PYSINDY_AVAILABLE, reason="pysindy not installed")
class TestWSINDy:
    """Tests for WSINDy method."""
    
    def test_wsindy_registered(self):
        """Test that WSINDy is registered in the method registry."""
        assert "WSINDy" in METHOD_REGISTRY.list_methods()
    
    def test_wsindy_name(self):
        """Test method name property."""
        method = METHOD_REGISTRY.get("WSINDy")
        assert method is not None
        assert method.name == "WSINDy"
    
    def test_wsindy_run_basic(self):
        """Test WSINDy on simple synthetic data."""
        method = METHOD_REGISTRY.get("WSINDy")
        
        # Create simple sine wave with larger dimensions for WSINDy
        nt, nx = 80, 100  # Larger to give WSINDy enough data
        dx, dt = 0.1, 0.01
        x = np.linspace(0, nx * dx, nx)
        t = np.linspace(0, nt * dt, nt)
        X, T = np.meshgrid(x, t)
        u_win = np.sin(2 * np.pi * X) * np.exp(-0.1 * T)
        
        metrics, info = method.run(u_win, dx, dt, K=20)
        
        # Check output shapes - should work even if method fails
        assert metrics.shape == (3,)
        assert all(np.isfinite(metrics))
        
        # WSINDy may fail on some data, check we got valid structure
        assert "terms" in info or "error" in info
        assert "coefficients" in info or "error" in info
    
    def test_wsindy_noise_robust(self):
        """Test that WSINDy handles noisy data."""
        method = METHOD_REGISTRY.get("WSINDy")
        
        # Clean signal with noise
        nt, nx = 50, 64
        dx, dt = 0.1, 0.01
        x = np.linspace(0, nx * dx, nx)
        t = np.linspace(0, nt * dt, nt)
        X, T = np.meshgrid(x, t)
        u_clean = np.sin(2 * np.pi * X)
        noise = np.random.randn(nt, nx) * 0.1
        u_noisy = u_clean + noise
        
        metrics, info = method.run(u_noisy, dx, dt, K=20)
        
        # Should still produce valid output
        assert metrics.shape == (3,)
        assert all(np.isfinite(metrics))


@pytest.mark.skipif(not PYSINDY_AVAILABLE, reason="pysindy not installed")
def test_method_comparison():
    """Test that both methods run on the same data."""
    pysindy_method = METHOD_REGISTRY.get("PySINDy")
    wsindy_method = METHOD_REGISTRY.get("WSINDy")
    
    # Create test data
    nt, nx = 50, 64
    dx, dt = 0.1, 0.01
    u_win = np.random.randn(nt, nx) * 0.5 + np.sin(
        np.linspace(0, 4 * np.pi, nx)
    )[None, :]
    
    # Run both methods
    metrics_pysindy, _ = pysindy_method.run(u_win, dx, dt)
    metrics_wsindy, _ = wsindy_method.run(u_win, dx, dt, K=20)
    
    # Both should produce valid metrics
    assert metrics_pysindy.shape == (3,)
    assert metrics_wsindy.shape == (3,)
    assert all(np.isfinite(metrics_pysindy))
    assert all(np.isfinite(metrics_wsindy))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
