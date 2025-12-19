"""
PySINDy adapter for PDE identification.

Uses PySINDy's PDELibrary for sparse identification of PDEs.
"""

import time
import numpy as np

try:
    import pysindy as ps
    from pysindy.feature_library import PDELibrary, PolynomialLibrary
    from pysindy.differentiation import FiniteDifference
    PYSINDY_AVAILABLE = True
except ImportError:
    PYSINDY_AVAILABLE = False

from .base import IdentMethodBase
from .registry import METHOD_REGISTRY


class PySINDyMethod(IdentMethodBase):
    """
    PySINDy-based PDE identification.
    
    Uses sparse regression (STLSQ) with PDE library containing
    polynomial and derivative terms.
    """
    
    def __init__(
        self,
        derivative_order: int = 3,
        polynomial_degree: int = 3,
        threshold: float = 0.1,
        alpha: float = 1e-5,
    ):
        """
        Initialize PySINDy method.
        
        Args:
            derivative_order: Max spatial derivative order
            polynomial_degree: Max polynomial degree
            threshold: STLSQ sparsity threshold
            alpha: Regularization strength
        """
        if not PYSINDY_AVAILABLE:
            raise ImportError("pysindy is not installed. Run: pip install pysindy")
        
        self.derivative_order = derivative_order
        self.polynomial_degree = polynomial_degree
        self.threshold = threshold
        self.alpha = alpha
    
    @property
    def name(self) -> str:
        return "PySINDy"
    
    def run(self, u_win, dx, dt, **kwargs):
        """
        Run PySINDy on spatiotemporal data.
        
        Args:
            u_win: np.ndarray of shape (nt, nx) - spatiotemporal data
            dx: float - spatial grid spacing
            dt: float - temporal grid spacing
            **kwargs: Optional parameters
                - max_dx: int - max derivative order (overrides init)
                - max_poly: int - max polynomial degree (overrides init)
                - threshold: float - STLSQ threshold
                - true_coeffs: dict - ground truth for F1 calculation
        
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
        
        nt, nx = u_win.shape
        
        try:
            # Set up spatial grid
            x = np.linspace(0, nx * dx, nx)
            t = np.linspace(0, nt * dt, nt)
            
            # Reshape for PySINDy: (n_samples, n_features)
            # For PDEs, we treat each spatial point as a feature
            u_flat = u_win.T  # (nx, nt) - spatial points x time samples
            
            # Create PDE library (API changed in newer versions)
            try:
                # Try newer API first
                pde_lib = PDELibrary(
                    derivative_order=derivative_order,
                    spatial_grid=x,
                    include_bias=True,
                    is_uniform=True,
                )
            except TypeError:
                # Fall back to older API
                pde_lib = PDELibrary(
                    function_library=PolynomialLibrary(degree=polynomial_degree),
                    derivative_order=derivative_order,
                    spatial_grid=x,
                    include_bias=True,
                    is_uniform=True,
                )
            
            # Create differentiator
            differentiator = FiniteDifference(order=2)
            
            # Create and fit SINDy model
            model = ps.SINDy(
                differentiation_method=differentiator,
                feature_library=pde_lib,
                optimizer=ps.STLSQ(threshold=threshold, alpha=self.alpha),
            )
            
            # PySINDy expects (n_samples, n_features) 
            # For PDEs: u_data shape (n_time, n_space)
            model.fit(u_win, t=dt, x=dx, multiple_trajectories=False)
            
            # Extract results
            feature_names = model.get_feature_names()
            coefficients = model.coefficients()[0] if model.coefficients().ndim > 1 else model.coefficients()
            
            # Build term dictionary (non-zero terms only)
            terms = []
            coeff_dict = {}
            for name, coef in zip(feature_names, coefficients):
                if abs(coef) > 1e-10:
                    terms.append(name)
                    coeff_dict[name] = float(coef)
            
            # Compute residual (reconstruction error)
            try:
                u_dot_true = np.gradient(u_win, dt, axis=0)
                u_dot_pred = model.predict(u_win, multiple_trajectories=False)
                if u_dot_pred.shape == u_dot_true.shape:
                    residual = np.mean((u_dot_true - u_dot_pred) ** 2)
                    # Normalize residual
                    residual = residual / (np.var(u_dot_true) + 1e-10)
                else:
                    residual = 1.0
            except Exception:
                residual = 1.0
            
            # Compute metrics
            f1, coeff_err = self._compute_structure_metrics(
                terms, coeff_dict, true_coeffs, residual
            )
            
            runtime = time.time() - start_time
            
            metrics = np.array([f1, coeff_err, residual], dtype=np.float64)
            info = {
                "terms": terms,
                "coefficients": coeff_dict,
                "runtime": runtime,
                "feature_names": feature_names,
            }
            
            return metrics, info
            
        except Exception as e:
            runtime = time.time() - start_time
            # Return worst-case metrics on failure
            return (
                np.array([0.0, 1.0, 1.0], dtype=np.float64),
                {"error": str(e), "runtime": runtime, "terms": [], "coefficients": {}}
            )
    
    def _compute_structure_metrics(self, pred_terms, pred_coeffs, true_coeffs, residual=None):
        """
        Compute F1 score and coefficient error.
        
        Args:
            pred_terms: list of identified term names
            pred_coeffs: dict of {term: coeff}
            true_coeffs: dict of {term: coeff} or None
            residual: optional residual to use as e2 if no term match
        
        Returns:
            f1: float in [0, 1]
            coeff_err: float >= 0
        """
        if true_coeffs is None:
            # No ground truth - use residual as e2 if available
            return 0.0, min(residual, 1.0) if residual is not None else 1.0
        
        true_terms = set(true_coeffs.keys())
        pred_terms_set = set(pred_terms)
        
        # F1 = 2 * (precision * recall) / (precision + recall)
        tp = len(true_terms & pred_terms_set)
        fp = len(pred_terms_set - true_terms)
        fn = len(true_terms - pred_terms_set)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Coefficient error (relative L2 on matching terms)
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
    METHOD_REGISTRY.register(PySINDyMethod())
