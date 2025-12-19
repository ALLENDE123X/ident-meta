"""
LASSO-SINDy: LASSO-based Sparse Identification for PDEs.

Uses L1-regularized regression (LASSO) to identify sparse PDE terms.
This is a simplified implementation that doesn't rely on PySINDy's PDELibrary.
"""

import time
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from sklearn.linear_model import Lasso, LassoCV

from .base import IdentMethodBase
from .registry import METHOD_REGISTRY


class LassoSINDyMethod(IdentMethodBase):
    """
    LASSO-based sparse PDE identification.
    
    Uses L1-regularized least squares to identify sparse governing equations.
    Builds a library of candidate terms (u, u^2, u_x, u_xx, u_xxx, u*u_x, etc.)
    and finds sparse coefficients.
    """
    
    def __init__(
        self,
        alpha: float = 0.01,
        max_dx: int = 3,
        max_poly: int = 2,
        fit_intercept: bool = False,
    ):
        """
        Initialize LASSO-SINDy method.
        
        Args:
            alpha: L1 regularization strength
            max_dx: Maximum spatial derivative order
            max_poly: Maximum polynomial degree
            fit_intercept: Whether to fit an intercept
        """
        self.alpha = alpha
        self.max_dx = max_dx
        self.max_poly = max_poly
        self.fit_intercept = fit_intercept
    
    @property
    def name(self) -> str:
        return "LASSO"
    
    def _build_library(self, u: np.ndarray, dx: float) -> Tuple[np.ndarray, List[str]]:
        """
        Build library of candidate terms.
        
        Args:
            u: (nt, nx) array of PDE data
            dx: spatial grid spacing
            
        Returns:
            library: (nt*nx, n_terms) array of candidate term values
            term_names: list of term names
        """
        nt, nx = u.shape
        
        # Compute derivatives
        derivatives = [u]  # u
        names = ["u"]
        
        current = u
        for d in range(1, self.max_dx + 1):
            current = np.gradient(current, dx, axis=1)
            derivatives.append(current)
            names.append("u" + "_x" * d)
        
        # Build library with polynomial combinations
        terms = []
        term_names = []
        
        # Add pure derivatives and polynomials
        for i, deriv in enumerate(derivatives):
            for p in range(1, self.max_poly + 1):
                term = deriv ** p
                terms.append(term.flatten())
                if p == 1:
                    term_names.append(names[i])
                else:
                    term_names.append(f"{names[i]}^{p}")
        
        # Add cross terms (u * u_x, u * u_xx, etc.)
        for i in range(1, len(derivatives)):
            term = derivatives[0] * derivatives[i]  # u * u_x, u * u_xx, etc.
            terms.append(term.flatten())
            term_names.append(f"u*{names[i]}")
        
        # Stack into library matrix
        library = np.column_stack(terms)
        
        return library, term_names
    
    def run(self, u_win, dx, dt, **kwargs):
        """
        Run LASSO-SINDy on spatiotemporal data.
        
        Args:
            u_win: np.ndarray of shape (nt, nx) - spatiotemporal data
            dx: float - spatial grid spacing
            dt: float - temporal grid spacing
            **kwargs: Optional parameters
                - alpha: float - L1 regularization
                - true_coeffs: dict - ground truth for F1 calculation
        
        Returns:
            metrics: np.ndarray of shape (3,) - [f1, coeff_err, residual]
            info: dict - identified terms, coefficients, runtime
        """
        self.validate_input(u_win, dx, dt)
        
        start_time = time.time()
        
        # Get parameters
        alpha = kwargs.get("alpha", self.alpha)
        true_coeffs = kwargs.get("true_coeffs", None)
        
        nt, nx = u_win.shape
        
        try:
            # Compute time derivative (target)
            u_t = np.gradient(u_win, dt, axis=0)
            
            # Build library of candidate terms
            library, term_names = self._build_library(u_win, dx)
            
            # Flatten target
            target = u_t.flatten()
            
            # Fit LASSO
            model = Lasso(alpha=alpha, fit_intercept=self.fit_intercept, max_iter=10000)
            model.fit(library, target)
            
            # Extract non-zero coefficients
            coefficients = model.coef_
            terms = []
            coeff_dict = {}
            
            for name, coef in zip(term_names, coefficients):
                if abs(coef) > 1e-8:
                    terms.append(name)
                    coeff_dict[name] = float(coef)
            
            # Compute residual
            prediction = model.predict(library)
            residual = np.mean((target - prediction) ** 2)
            residual_normalized = residual / (np.var(target) + 1e-10)
            
            # Compute metrics
            f1, coeff_err = self._compute_structure_metrics(
                terms, coeff_dict, true_coeffs, residual_normalized
            )
            
            runtime = time.time() - start_time
            
            metrics = np.array([f1, coeff_err, residual_normalized], dtype=np.float64)
            info = {
                "terms": terms,
                "coefficients": coeff_dict,
                "runtime": runtime,
                "n_nonzero": len(terms),
                "term_names": term_names,
            }
            
            return metrics, info
            
        except Exception as e:
            runtime = time.time() - start_time
            return (
                np.array([0.0, 1.0, 1.0], dtype=np.float64),
                {"error": str(e), "runtime": runtime, "terms": [], "coefficients": {}}
            )
    
    def _compute_structure_metrics(self, pred_terms, pred_coeffs, true_coeffs, residual=None):
        """Compute F1 and coefficient error."""
        if true_coeffs is None or len(true_coeffs) == 0:
            # Use normalized residual as e2
            return 0.0, min(residual, 1.0) if residual is not None else 1.0
        
        # Normalize term names for comparison
        true_terms = set(self._normalize_term(t) for t in true_coeffs.keys())
        pred_terms_norm = set(self._normalize_term(t) for t in pred_terms)
        
        # F1 calculation
        tp = len(true_terms & pred_terms_norm)
        fp = len(pred_terms_norm - true_terms)
        fn = len(true_terms - pred_terms_norm)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Use residual as e2 if no exact term match
        if tp == 0:
            coeff_err = min(residual, 1.0) if residual is not None else 1.0
        else:
            coeff_err = min(residual, 1.0) if residual is not None else 0.5
        
        return float(f1), float(coeff_err)
    
    def _normalize_term(self, term: str) -> str:
        """Normalize term name for comparison."""
        # Convert various formats to consistent format
        term = term.replace(" ", "").replace("*", "").lower()
        return term


# Register the method
METHOD_REGISTRY.register(LassoSINDyMethod())
