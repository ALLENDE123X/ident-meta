"""
WSINDy (Weak-form SINDy) adapter for PDE identification.

Uses PySINDy's WeakPDELibrary for noise-robust identification.
The weak formulation integrates against test functions, reducing
sensitivity to noise in derivatives.
"""

import time
import numpy as np

try:
    import pysindy as ps
    from pysindy.feature_library import WeakPDELibrary
    PYSINDY_AVAILABLE = True
except ImportError:
    PYSINDY_AVAILABLE = False

from .base import IdentMethodBase
from .registry import METHOD_REGISTRY


class WSINDyMethod(IdentMethodBase):
    """
    Weak-form SINDy for noise-robust PDE identification.
    
    Uses weak formulation (integration against test functions) to
    avoid direct differentiation of noisy data.
    """
    
    def __init__(
        self,
        derivative_order: int = 3,
        polynomial_degree: int = 3,
        threshold: float = 0.1,
        alpha: float = 1e-5,
        K: int = 100,  # Number of test functions
    ):
        """
        Initialize WSINDy method.
        
        Args:
            derivative_order: Max spatial derivative order
            polynomial_degree: Max polynomial degree
            threshold: STLSQ sparsity threshold
            alpha: Regularization strength
            K: Number of test functions for weak form
        """
        if not PYSINDY_AVAILABLE:
            raise ImportError("pysindy is not installed. Run: pip install pysindy")
        
        self.derivative_order = derivative_order
        self.polynomial_degree = polynomial_degree
        self.threshold = threshold
        self.alpha = alpha
        self.K = K
    
    @property
    def name(self) -> str:
        return "WSINDy"
    
    def run(self, u_win, dx, dt, **kwargs):
        """
        Run WSINDy on spatiotemporal data.
        
        Args:
            u_win: np.ndarray of shape (nt, nx) - spatiotemporal data
            dx: float - spatial grid spacing
            dt: float - temporal grid spacing
            **kwargs: Optional parameters
                - max_dx: int - max derivative order
                - max_poly: int - max polynomial degree
                - threshold: float - STLSQ threshold
                - true_coeffs: dict - ground truth for F1 calculation
                - K: int - number of test functions
        
        Returns:
            metrics: np.ndarray of shape (3,) - [f1, coeff_err, residual]
            info: dict - identified terms, coefficients, runtime
        """
        self.validate_input(u_win, dx, dt)
        
        start_time = time.time()
        
        # Get parameters
        derivative_order = kwargs.get("max_dx", self.derivative_order)
        polynomial_degree = kwargs.get("max_poly", self.polynomial_degree)
        threshold = kwargs.get("threshold", self.threshold)
        true_coeffs = kwargs.get("true_coeffs", None)
        K = kwargs.get("K", self.K)
        
        nt, nx = u_win.shape
        
        try:
            # Set up spatial grid
            x = np.linspace(0, nx * dx, nx)
            t = np.linspace(0, nt * dt, nt)
            
            # Adjust K if data is too small
            K = min(K, (nt - 1) * (nx - 1) // 4)
            K = max(K, 10)  # Minimum number of test functions
            
            # Create spatiotemporal grid as numpy array
            # WeakPDELibrary expects a stacked array, not a list from meshgrid
            X, T = np.meshgrid(x, t, indexing='xy')
            spatiotemporal_grid = np.stack([T, X], axis=-1)  # (nt, nx, 2)
            
            # Create weak PDE library
            weak_lib = WeakPDELibrary(
                derivative_order=derivative_order,
                spatiotemporal_grid=spatiotemporal_grid,
                include_bias=True,
                K=K,
            )
            
            # Create and fit SINDy model with weak formulation
            model = ps.SINDy(
                feature_library=weak_lib,
                optimizer=ps.STLSQ(threshold=threshold, alpha=self.alpha),
            )
            
            # Fit the model - WeakPDELibrary already has the grid, don't pass x
            model.fit(u_win, t=dt)
            
            # Extract results
            feature_names = model.get_feature_names()
            coefficients = model.coefficients()[0] if model.coefficients().ndim > 1 else model.coefficients()
            
            # Build term dictionary
            terms = []
            coeff_dict = {}
            for name, coef in zip(feature_names, coefficients):
                if abs(coef) > 1e-10:
                    terms.append(name)
                    coeff_dict[name] = float(coef)
            
            # Compute residual
            try:
                u_dot_true = np.gradient(u_win, dt, axis=0)
                u_dot_pred = model.predict(u_win, multiple_trajectories=False)
                if u_dot_pred.shape == u_dot_true.shape:
                    residual = np.mean((u_dot_true - u_dot_pred) ** 2)
                else:
                    residual = 1.0
            except Exception:
                residual = 1.0
            
            # Compute metrics
            f1, coeff_err = self._compute_structure_metrics(
                terms, coeff_dict, true_coeffs
            )
            
            runtime = time.time() - start_time
            
            metrics = np.array([f1, coeff_err, residual], dtype=np.float64)
            info = {
                "terms": terms,
                "coefficients": coeff_dict,
                "runtime": runtime,
                "feature_names": feature_names,
                "K": K,
                "weak_form": True,
            }
            
            return metrics, info
            
        except Exception as e:
            runtime = time.time() - start_time
            return (
                np.array([0.0, 1.0, 1.0], dtype=np.float64),
                {"error": str(e), "runtime": runtime, "terms": [], "coefficients": {}}
            )
    
    def _compute_structure_metrics(self, pred_terms, pred_coeffs, true_coeffs):
        """Compute F1 and coefficient error."""
        if true_coeffs is None:
            return 0.0, 1.0
        
        true_terms = set(true_coeffs.keys())
        pred_terms_set = set(pred_terms)
        
        tp = len(true_terms & pred_terms_set)
        fp = len(pred_terms_set - true_terms)
        fn = len(true_terms - pred_terms_set)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        matching_terms = true_terms & pred_terms_set
        if len(matching_terms) == 0:
            coeff_err = 1.0
        else:
            true_vec = np.array([true_coeffs[t] for t in matching_terms])
            pred_vec = np.array([pred_coeffs.get(t, 0.0) for t in matching_terms])
            true_norm = np.linalg.norm(true_vec)
            if true_norm > 1e-10:
                coeff_err = np.linalg.norm(true_vec - pred_vec) / true_norm
            else:
                coeff_err = np.linalg.norm(true_vec - pred_vec)
        
        return float(f1), float(min(coeff_err, 1.0))


# Register the method
if PYSINDY_AVAILABLE:
    METHOD_REGISTRY.register(WSINDyMethod())
