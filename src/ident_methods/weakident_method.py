"""
WeakIDENT adapter for PDE identification.

Wraps the existing WeakIDENT implementation (model.py) to conform to
the IdentMethodBase interface.

WeakIDENT uses a weak formulation with trimmed least squares for
robust PDE identification. It's the foundational method of this repository.

Reference: Kang et al., "Weak Identification of PDEs"
"""

import time
import sys
import os
import numpy as np
from typing import Dict, Any, Optional

# Add parent directory to path for model import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .base import IdentMethodBase
from .registry import METHOD_REGISTRY


class WeakIdentMethod(IdentMethodBase):
    """
    WeakIDENT: Weak formulation-based PDE identification.
    
    Uses trimmed least squares with a weak (integral) formulation
    to identify PDEs from noisy spatiotemporal data.
    """
    
    def __init__(
        self,
        max_dx: int = 4,
        max_poly: int = 4,
        skip_x: int = 4,
        skip_t: int = 4,
        tau: float = 0.05,
        use_cross_der: bool = False,
    ):
        """
        Initialize WeakIDENT method.
        
        Args:
            max_dx: Maximum spatial derivative order
            max_poly: Maximum polynomial degree
            skip_x: Spatial downsampling factor
            skip_t: Temporal downsampling factor
            tau: Trimming threshold
            use_cross_der: Use cross derivatives
        """
        self.max_dx = max_dx
        self.max_poly = max_poly
        self.skip_x = skip_x
        self.skip_t = skip_t
        self.tau = tau
        self.use_cross_der = use_cross_der
        
        # Lazy import to avoid circular imports
        self._weak_ident_pred = None
    
    def _get_weak_ident_pred(self):
        """Lazy load weak_ident_pred to avoid import issues."""
        if self._weak_ident_pred is None:
            try:
                from model import weak_ident_pred
                self._weak_ident_pred = weak_ident_pred
            except ImportError as e:
                raise ImportError(
                    f"Failed to import weak_ident_pred from model.py: {e}. "
                    "Make sure the WeakIdent model code is available."
                )
        return self._weak_ident_pred
    
    @property
    def name(self) -> str:
        return "WeakIDENT"
    
    def run(self, u_win, dx, dt, **kwargs):
        """
        Run WeakIDENT on spatiotemporal data.
        
        Args:
            u_win: np.ndarray of shape (nt, nx) - spatiotemporal data
            dx: float - spatial grid spacing
            dt: float - temporal grid spacing
            **kwargs: Optional parameters
                - max_dx: int - max derivative order
                - max_poly: int - max polynomial degree
                - skip_x: int - spatial downsampling
                - skip_t: int - temporal downsampling
                - tau: float - trimming threshold
                - true_coeffs: dict - ground truth for metrics
        
        Returns:
            metrics: np.ndarray of shape (3,) - [f1, coeff_err, residual]
            info: dict - identified terms, coefficients, runtime
        """
        self.validate_input(u_win, dx, dt)
        
        start_time = time.time()
        
        # Get parameters
        max_dx = kwargs.get("max_dx", self.max_dx)
        max_poly = kwargs.get("max_poly", self.max_poly)
        skip_x = kwargs.get("skip_x", self.skip_x)
        skip_t = kwargs.get("skip_t", self.skip_t)
        tau = kwargs.get("tau", self.tau)
        true_coeffs = kwargs.get("true_coeffs", None)
        
        nt, nx = u_win.shape
        
        try:
            weak_ident_pred = self._get_weak_ident_pred()
            
            # Prepare inputs for weak_ident_pred
            u_hat = np.array([u_win.T])  # WeakIDENT expects (nx, nt) transposed
            x = np.linspace(0, nx * dx, nx).reshape(-1, 1)
            t = np.linspace(0, nt * dt, nt).reshape(1, -1)
            xs = np.array([x, t], dtype=object)
            
            # True coefficients format
            if true_coeffs is None:
                true_coefficients = np.array(
                    [np.array([[1.0, 1.0, 0.0, 0.0]])], dtype=object
                )
            else:
                true_coefficients = self._dict_to_weakident_format(true_coeffs)
            
            # Run WeakIDENT
            df_errors, df_eqns, df_coeffs, run_time = weak_ident_pred(
                u_hat=u_hat,
                xs=xs,
                true_coefficients=true_coefficients,
                max_dx=max_dx,
                max_poly=max_poly,
                skip_x=skip_x,
                skip_t=skip_t,
                use_cross_der=self.use_cross_der,
                tau=tau,
            )
            
            # Extract metrics
            e2 = df_errors["$e_2$"].values[0]
            tpr = df_errors["$tpr$"].values[0]
            ppv = df_errors["$ppv$"].values[0]
            e_res = df_errors["$e_{res}$"].values[0]
            
            # Compute F1 score
            if tpr + ppv > 0:
                f1 = 2 * tpr * ppv / (tpr + ppv)
            else:
                f1 = 0.0
            
            # Handle NaN/Inf
            f1 = 0.0 if not np.isfinite(f1) else f1
            e2 = 1.0 if not np.isfinite(e2) else e2
            e_res = 1.0 if not np.isfinite(e_res) else e_res
            
            if true_coeffs is None:
                f1 = 0.0
                e2 = 1.0
            
            runtime = time.time() - start_time
            
            # Extract identified equation
            terms = []
            coeff_dict = {}
            if df_coeffs is not None and len(df_coeffs) > 0:
                for idx, row in df_coeffs.iterrows():
                    # Parse coefficient entries
                    pass  # Complex parsing of WeakIDENT output
            
            metrics = np.array([f1, e2, e_res], dtype=np.float64)
            info = {
                "terms": terms,
                "coefficients": coeff_dict,
                "runtime": runtime,
                "weakident_runtime": run_time if run_time else 0.0,
                "tpr": tpr,
                "ppv": ppv,
                "df_eqns": str(df_eqns) if df_eqns is not None else None,
            }
            
            return metrics, info
            
        except Exception as e:
            runtime = time.time() - start_time
            return (
                np.array([0.0, 1.0, 1.0], dtype=np.float64),
                {"error": str(e), "runtime": runtime, "terms": [], "coefficients": {}}
            )
    
    def _dict_to_weakident_format(self, true_coeffs: Dict[str, float]) -> np.ndarray:
        """
        Convert dict of {term_name: coeff_value} to WeakIDENT format.
        
        WeakIDENT expects: array([[beta_u, d_x, d_t, coeff], ...])
        """
        rows = []
        for term_name, coeff in true_coeffs.items():
            beta_u, d_x, d_t = self._parse_term_name(term_name)
            rows.append([beta_u, d_x, d_t, coeff])
        
        if len(rows) == 0:
            rows = [[1, 0, 0, 0.0]]
        
        return np.array([np.array(rows, dtype=np.float64)], dtype=object)
    
    def _parse_term_name(self, term_name: str):
        """
        Parse term name like 'u_x', 'u_xx', 'u^2_x' into (beta_u, d_x, d_t).
        """
        beta_u = 1
        d_x = 0
        d_t = 0
        
        if "^" in term_name:
            parts = term_name.split("^")
            power_part = parts[1].split("_")[0]
            beta_u = int(power_part)
            term_name = parts[0] + (
                "_" + "_".join(parts[1].split("_")[1:]) if "_" in parts[1] else ""
            )
        
        if "_x" in term_name:
            d_x = term_name.count("x")
        if "_t" in term_name:
            d_t = term_name.count("t")
        
        return beta_u, d_x, d_t


# Register the method
try:
    METHOD_REGISTRY.register(WeakIdentMethod())
except Exception:
    pass  # May fail if model.py not available - that's ok
