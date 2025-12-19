"""
RobustIDENT (RLAD-SID) adapter for PDE identification.

Uses L1-regularized sparse regression (Least Absolute Deviation) for
robust identification in the presence of outliers. Implements the
RLAD-SID algorithm via ADMM optimization.

Reference:
    "Robust Identification of Differential Equations" - uses L1 loss
    instead of L2 to handle outliers in the data.
"""

import time
import numpy as np
from typing import Dict, Tuple, Any, Optional

from .base import IdentMethodBase
from .registry import METHOD_REGISTRY


class RobustIdentMethod(IdentMethodBase):
    """
    RobustIDENT using L1-regularized sparse regression (RLAD-SID).
    
    Uses Least Absolute Deviation loss instead of Least Squares,
    making it more robust to outliers in the data. Solves:
    
        min_w ||Aw - b||_1 + lambda * ||w||_1
    
    via ADMM (Alternating Direction Method of Multipliers).
    """
    
    def __init__(
        self,
        derivative_order: int = 4,
        polynomial_degree: int = 4,
        lambda_reg: float = 0.1,
        threshold: float = 0.05,
        max_iter: int = 500,
        tol: float = 1e-6,
    ):
        """
        Initialize RobustIDENT method.
        
        Args:
            derivative_order: Max spatial derivative order
            polynomial_degree: Max polynomial degree  
            lambda_reg: L1 regularization strength
            threshold: Sparsity threshold for final coefficients
            max_iter: Max ADMM iterations
            tol: Convergence tolerance
        """
        self.derivative_order = derivative_order
        self.polynomial_degree = polynomial_degree
        self.lambda_reg = lambda_reg
        self.threshold = threshold
        self.max_iter = max_iter
        self.tol = tol
    
    @property
    def name(self) -> str:
        return "RobustIDENT"
    
    def run(self, u_win, dx, dt, **kwargs):
        """
        Run RobustIDENT on spatiotemporal data.
        
        Args:
            u_win: np.ndarray of shape (nt, nx) - spatiotemporal data
            dx: float - spatial grid spacing
            dt: float - temporal grid spacing
            **kwargs: Optional parameters
                - max_dx: int - max derivative order
                - max_poly: int - max polynomial degree
                - lambda_reg: float - L1 regularization
                - threshold: float - sparsity threshold
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
        lambda_reg = kwargs.get("lambda_reg", self.lambda_reg)
        threshold = kwargs.get("threshold", self.threshold)
        true_coeffs = kwargs.get("true_coeffs", None)
        
        nt, nx = u_win.shape
        
        try:
            # Build feature library and target
            A, b, feature_names = self._build_library(
                u_win, dx, dt, derivative_order, polynomial_degree
            )
            
            # Solve L1-regularized regression via ADMM
            coefficients = self._solve_lad_admm(
                A, b, lambda_reg, self.max_iter, self.tol
            )
            
            # Apply sparsity threshold
            coefficients[np.abs(coefficients) < threshold] = 0.0
            
            # Build term dictionary
            terms = []
            coeff_dict = {}
            for name, coef in zip(feature_names, coefficients):
                if abs(coef) > 1e-10:
                    terms.append(name)
                    coeff_dict[name] = float(coef)
            
            # Compute residual (L1 residual for consistency)
            u_dot_pred = A @ coefficients
            residual = np.mean(np.abs(b - u_dot_pred))
            
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
                "feature_names": list(feature_names),
                "n_features": len(feature_names),
                "n_nonzero": len(terms),
                "robust": True,
            }
            
            return metrics, info
            
        except Exception as e:
            runtime = time.time() - start_time
            return (
                np.array([0.0, 1.0, 1.0], dtype=np.float64),
                {"error": str(e), "runtime": runtime, "terms": [], "coefficients": {}}
            )
    
    def _build_library(
        self, u_win, dx, dt, derivative_order, polynomial_degree
    ) -> Tuple[np.ndarray, np.ndarray, list]:
        """
        Build PDE feature library.
        
        Creates columns for:
        - Polynomial terms: 1, u, u^2, ..., u^p
        - Derivative terms: u_x, u_xx, ..., u_{x^d}
        - Mixed terms: u*u_x, u^2*u_x, etc.
        
        Args:
            u_win: (nt, nx) array
            dx, dt: grid spacings
            derivative_order: max derivative
            polynomial_degree: max polynomial
            
        Returns:
            A: (n_samples, n_features) library matrix
            b: (n_samples,) time derivative target
            feature_names: list of feature names
        """
        nt, nx = u_win.shape
        
        # Trim edges for finite difference stencils
        trim = max(2, derivative_order)
        interior_t = slice(trim, nt - trim)
        interior_x = slice(trim, nx - trim)
        
        # Compute time derivative (target)
        u_t = (u_win[2:, :] - u_win[:-2, :]) / (2 * dt)
        u_t = u_t[interior_t.start-1:interior_t.stop-1, interior_x]
        
        # Get interior u values
        u_interior = u_win[interior_t, interior_x]
        
        # Flatten
        n_samples = u_interior.size
        b = u_t.flatten()
        
        # Initialize feature library
        features = []
        feature_names = []
        
        # Bias term
        features.append(np.ones(n_samples))
        feature_names.append("1")
        
        # Polynomial terms: u, u^2, ..., u^p
        u_flat = u_interior.flatten()
        for p in range(1, polynomial_degree + 1):
            features.append(u_flat ** p)
            if p == 1:
                feature_names.append("u")
            else:
                feature_names.append(f"u^{p}")
        
        # Compute spatial derivatives
        derivatives = self._compute_derivatives(u_win, dx, derivative_order)
        
        # Derivative terms and mixed terms
        for d in range(1, derivative_order + 1):
            # Get derivative
            du = derivatives[d][interior_t, interior_x].flatten()
            
            # Pure derivative
            d_name = "u" + "_x" * d
            features.append(du)
            feature_names.append(d_name)
            
            # Mixed polynomial-derivative terms
            for p in range(1, min(polynomial_degree, 3) + 1):
                if p == 1:
                    features.append(u_flat * du)
                    feature_names.append(f"u*{d_name}")
                else:
                    features.append((u_flat ** p) * du)
                    feature_names.append(f"u^{p}*{d_name}")
        
        # Stack into library matrix
        A = np.column_stack(features)
        
        return A, b, feature_names
    
    def _compute_derivatives(self, u, dx, max_order) -> Dict[int, np.ndarray]:
        """
        Compute spatial derivatives using finite differences.
        
        Uses central differences with appropriate stencils:
        - 1st derivative: (-1, 0, 1) / 2dx
        - 2nd derivative: (1, -2, 1) / dx^2
        - Higher orders: convolution of lower orders
        """
        derivatives = {0: u}
        
        for order in range(1, max_order + 1):
            if order == 1:
                # Central difference for 1st derivative
                du = np.zeros_like(u)
                du[:, 1:-1] = (u[:, 2:] - u[:, :-2]) / (2 * dx)
                derivatives[1] = du
            elif order == 2:
                # Central difference for 2nd derivative
                ddu = np.zeros_like(u)
                ddu[:, 1:-1] = (u[:, 2:] - 2*u[:, 1:-1] + u[:, :-2]) / (dx**2)
                derivatives[2] = ddu
            else:
                # Higher orders via repeated application
                d_prev = derivatives[order - 2]
                d_curr = np.zeros_like(u)
                d_curr[:, 1:-1] = (d_prev[:, 2:] - 2*d_prev[:, 1:-1] + d_prev[:, :-2]) / (dx**2)
                derivatives[order] = d_curr
        
        return derivatives
    
    def _solve_lad_admm(
        self, A, b, lambda_reg, max_iter, tol
    ) -> np.ndarray:
        """
        Solve L1-regularized LAD regression via ADMM.
        
        Solves: min_w ||Aw - b||_1 + lambda * ||w||_1
        
        Uses ADMM reformulation:
            min ||z||_1 + lambda||w||_1
            s.t. z = Aw - b
            
        Args:
            A: (m, n) feature matrix
            b: (m,) target vector
            lambda_reg: L1 regularization strength
            max_iter: maximum iterations
            tol: convergence tolerance
            
        Returns:
            w: (n,) coefficient vector
        """
        m, n = A.shape
        
        # ADMM parameters
        rho = 1.0
        
        # Initialize variables
        w = np.zeros(n)
        z = np.zeros(m)  # z = Aw - b
        u = np.zeros(m)  # dual variable
        
        # Precompute matrix factorization for efficiency
        # (A'A + rho*I)^(-1) A'
        AtA = A.T @ A
        AtA_inv = np.linalg.inv(AtA + rho * np.eye(n))
        
        for iteration in range(max_iter):
            w_old = w.copy()
            
            # w-update: minimize (rho/2)||Aw - b - z + u||^2 + lambda||w||_1
            # This is a LASSO problem, use soft-thresholding
            v = z - u + b
            w = AtA_inv @ (A.T @ v)
            w = self._soft_threshold(w, lambda_reg / rho)
            
            # z-update: minimize ||z||_1 + (rho/2)||z - (Aw - b + u)||^2
            # Soft-thresholding
            Aw_minus_b = A @ w - b
            z = self._soft_threshold(Aw_minus_b + u, 1.0 / rho)
            
            # u-update (dual)
            u = u + Aw_minus_b - z
            
            # Check convergence
            primal_residual = np.linalg.norm(Aw_minus_b - z)
            dual_residual = np.linalg.norm(rho * (w - w_old))
            
            if primal_residual < tol and dual_residual < tol:
                break
        
        return w
    
    def _soft_threshold(self, x, threshold):
        """Soft-thresholding operator for L1 optimization."""
        return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)
    
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
METHOD_REGISTRY.register(RobustIdentMethod())
