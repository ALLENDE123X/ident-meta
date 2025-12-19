"""
STLSQ: Sequentially Thresholded Least Squares for PDE Identification.

This is the original SINDy algorithm that iteratively applies least squares
regression and thresholds small coefficients to enforce sparsity.
"""

import time
import numpy as np
from typing import Dict, Any, Optional, List, Tuple

from .base import IdentMethodBase
from .registry import METHOD_REGISTRY


class STLSQMethod(IdentMethodBase):
    """
    Sequentially Thresholded Least Squares (STLSQ) for sparse PDE identification.
    
    The original SINDy algorithm:
    1. Solve least squares for all terms
    2. Threshold coefficients below a threshold to zero
    3. Re-solve least squares with remaining terms
    4. Repeat until convergence
    """
    
    def __init__(
        self,
        threshold: float = 0.1,
        max_dx: int = 3,
        max_poly: int = 2,
        max_iter: int = 20,
        ridge_alpha: float = 1e-5,
    ):
        """
        Initialize STLSQ method.
        
        Args:
            threshold: Sparsity threshold for coefficients
            max_dx: Maximum spatial derivative order
            max_poly: Maximum polynomial degree
            max_iter: Maximum STLSQ iterations
            ridge_alpha: Small ridge regularization for stability
        """
        self.threshold = threshold
        self.max_dx = max_dx
        self.max_poly = max_poly
        self.max_iter = max_iter
        self.ridge_alpha = ridge_alpha
    
    @property
    def name(self) -> str:
        return "STLSQ"
    
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
    
    def _stlsq(self, X: np.ndarray, y: np.ndarray, threshold: float) -> np.ndarray:
        """
        Sequentially Thresholded Least Squares.
        
        Args:
            X: (n_samples, n_features) library matrix
            y: (n_samples,) target vector
            threshold: sparsity threshold
            
        Returns:
            coefficients: (n_features,) sparse coefficient vector
        """
        n_features = X.shape[1]
        
        # Initial least squares solve with ridge regularization
        XTX = X.T @ X + self.ridge_alpha * np.eye(n_features)
        XTy = X.T @ y
        
        try:
            coefficients = np.linalg.solve(XTX, XTy)
        except np.linalg.LinAlgError:
            coefficients = np.linalg.lstsq(X, y, rcond=None)[0]
        
        # STLSQ iterations
        for _ in range(self.max_iter):
            # Threshold small coefficients
            small_inds = np.abs(coefficients) < threshold
            coefficients[small_inds] = 0
            
            # Find remaining (big) indices
            big_inds = ~small_inds
            
            if not np.any(big_inds):
                break
            
            # Re-solve least squares on remaining terms
            X_big = X[:, big_inds]
            n_big = X_big.shape[1]
            
            XTX_big = X_big.T @ X_big + self.ridge_alpha * np.eye(n_big)
            XTy_big = X_big.T @ y
            
            try:
                coefficients_big = np.linalg.solve(XTX_big, XTy_big)
            except np.linalg.LinAlgError:
                coefficients_big = np.linalg.lstsq(X_big, y, rcond=None)[0]
            
            # Update coefficients
            new_coefficients = np.zeros(n_features)
            new_coefficients[big_inds] = coefficients_big
            
            # Check convergence
            if np.allclose(coefficients, new_coefficients, rtol=1e-8):
                break
            
            coefficients = new_coefficients
        
        return coefficients
    
    def run(self, u_win, dx, dt, **kwargs):
        """
        Run STLSQ on spatiotemporal data.
        
        Args:
            u_win: np.ndarray of shape (nt, nx) - spatiotemporal data
            dx: float - spatial grid spacing
            dt: float - temporal grid spacing
            **kwargs: Optional parameters
                - threshold: float - sparsity threshold
                - true_coeffs: dict - ground truth for F1 calculation
        
        Returns:
            metrics: np.ndarray of shape (3,) - [f1, coeff_err, residual]
            info: dict - identified terms, coefficients, runtime
        """
        self.validate_input(u_win, dx, dt)
        
        start_time = time.time()
        
        # Get parameters
        threshold = kwargs.get("threshold", self.threshold)
        true_coeffs = kwargs.get("true_coeffs", None)
        
        nt, nx = u_win.shape
        
        try:
            # Compute time derivative (target)
            u_t = np.gradient(u_win, dt, axis=0)
            
            # Build library of candidate terms
            library, term_names = self._build_library(u_win, dx)
            
            # Flatten target
            target = u_t.flatten()
            
            # Run STLSQ
            coefficients = self._stlsq(library, target, threshold)
            
            # Extract non-zero coefficients
            terms = []
            coeff_dict = {}
            
            for name, coef in zip(term_names, coefficients):
                if abs(coef) > 1e-8:
                    terms.append(name)
                    coeff_dict[name] = float(coef)
            
            # Compute residual
            prediction = library @ coefficients
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
METHOD_REGISTRY.register(STLSQMethod())
