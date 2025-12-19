"""
Tests for method selection logic.

Reference: pde-selector-implementation-plan.md §13
"""

import sys
import os
import numpy as np
import pytest

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import PerMethodRegressor
from src.select_and_run import choose_method


def test_choose_single_method():
    """Test that selector chooses only 1 method when confident."""
    # Create mock models that always predict low scores and low uncertainty
    np.random.seed(42)

    class MockModel:
        def predict(self, X):
            # Method A: low error
            if self.name == "A":
                return np.array([[0.9, 0.05, 0.05]])  # high F1, low errors
            else:
                return np.array([[0.5, 0.2, 0.2]])  # lower F1, higher errors

        def predict_unc(self, X):
            return np.array([[0.01, 0.01, 0.01]])  # low uncertainty

        def __init__(self, name):
            self.name = name
            self.fitted = True

    models = {"A": MockModel("A"), "B": MockModel("B")}

    phi = np.random.randn(1, 12)

    # Choose with low tau (confident)
    chosen = choose_method(phi, models, w=(0.5, 0.3, 0.2), tau=0.5, k_fallback=2)

    # Should choose only 1 method (A)
    assert len(chosen) == 1, f"Expected 1 method, got {len(chosen)}"
    assert chosen[0] == "A", f"Expected method A, got {chosen[0]}"


def test_choose_fallback():
    """Test that selector runs top-2 when score is high."""
    np.random.seed(42)

    class MockModel:
        def predict(self, X):
            # Both methods have high errors
            if self.name == "A":
                return np.array([[0.5, 0.8, 0.8]])  # low F1, high errors
            else:
                return np.array([[0.4, 0.9, 0.9]])

        def predict_unc(self, X):
            return np.array([[0.05, 0.05, 0.05]])  # moderate uncertainty

        def __init__(self, name):
            self.name = name
            self.fitted = True

    models = {"A": MockModel("A"), "B": MockModel("B")}

    phi = np.random.randn(1, 12)

    # Choose with low tau (should trigger fallback due to high score)
    chosen = choose_method(phi, models, w=(0.5, 0.3, 0.2), tau=0.3, k_fallback=2)

    # Should choose 2 methods (fallback)
    assert len(chosen) == 2, f"Expected 2 methods (fallback), got {len(chosen)}"


def test_choose_high_uncertainty():
    """Test that selector runs top-2 when uncertainty is high."""
    np.random.seed(42)

    class MockModel:
        def predict(self, X):
            if self.name == "A":
                return np.array([[0.9, 0.05, 0.05]])  # low errors
            else:
                return np.array([[0.8, 0.1, 0.1]])

        def predict_unc(self, X):
            # Method A has high uncertainty
            if self.name == "A":
                return np.array([[0.5, 0.5, 0.5]])
            else:
                return np.array([[0.05, 0.05, 0.05]])

        def __init__(self, name):
            self.name = name
            self.fitted = True

    models = {"A": MockModel("A"), "B": MockModel("B")}

    phi = np.random.randn(1, 12)

    # Choose (should trigger fallback due to high uncertainty of best method)
    chosen = choose_method(phi, models, w=(0.5, 0.3, 0.2), tau=0.5, k_fallback=2)

    # Should choose 2 methods (fallback due to high uncertainty)
    assert len(chosen) == 2, f"Expected 2 methods (fallback), got {len(chosen)}"


def test_choose_always_best():
    """Test that selector always chooses the best method when obvious."""
    np.random.seed(42)

    class MockModel:
        def predict(self, X):
            # Method A is always much better
            if self.name == "A":
                return np.array([[0.95, 0.01, 0.01]])  # very low errors
            else:
                return np.array([[0.5, 0.5, 0.5]])  # high errors

        def predict_unc(self, X):
            return np.array([[0.01, 0.01, 0.01]])  # low uncertainty

        def __init__(self, name):
            self.name = name
            self.fitted = True

    models = {"A": MockModel("A"), "B": MockModel("B")}

    phi = np.random.randn(1, 12)

    # Run multiple times
    for _ in range(10):
        chosen = choose_method(phi, models, w=(0.5, 0.3, 0.2), tau=0.5, k_fallback=2)
        # Should always choose A (and possibly skip fallback)
        assert chosen[0] == "A", f"Expected method A, got {chosen[0]}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

