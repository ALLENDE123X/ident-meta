"""
Abstract base class for IDENT method adapters.

All PDE identification methods must implement this interface to be
compatible with the PDE-Selector framework.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any
import numpy as np


class IdentMethodBase(ABC):
    """
    Integration contract for IDENT methods.
    
    All PDE identification methods must inherit from this class and
    implement the required methods to work with the PDE-Selector.
    
    The selector uses the metrics returned by run() to determine
    which method performs best for a given data window.
    
    Metrics:
        m1 (F1 score): Structure accuracy - how well the method identifies
            the correct PDE terms (0 = wrong structure, 1 = perfect)
        m2 (coeff_err): Coefficient error - relative L2 error on identified
            coefficients (0 = perfect, higher = worse)
        m3 (residual): Residual MSE - cross-validation reconstruction error
            (0 = perfect fit, higher = worse)
    
    Example implementation:
        class WSINDyMethod(IdentMethodBase):
            @property
            def name(self) -> str:
                return "WSINDy"
            
            def run(self, u_win, dx, dt, **kwargs):
                # Run WSINDy algorithm
                result = pysindy.SINDy(...).fit(...)
                
                # Compute metrics
                f1 = self._compute_f1(result, kwargs.get("true_coeffs"))
                coeff_err = self._compute_coeff_error(result)
                residual = result.score(...)
                
                metrics = np.array([f1, coeff_err, residual])
                info = {
                    "terms": result.get_feature_names(),
                    "coefficients": dict(zip(terms, result.coef_)),
                    "runtime": elapsed_time,
                }
                return metrics, info
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """
        Unique identifier for this method.
        
        Returns:
            str: Method name (e.g., "WeakIDENT", "RobustIDENT", "WSINDy")
        """
        ...
    
    @abstractmethod
    def run(
        self,
        u_win: np.ndarray,
        dx: float,
        dt: float,
        **kwargs: Any,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Run the identification method on a spatiotemporal window.
        
        Args:
            u_win: np.ndarray of shape (nt, nx) - spatiotemporal data window
                Note: time axis is first (rows), space axis is second (columns)
            dx: float - spatial grid spacing
            dt: float - temporal grid spacing
            **kwargs: Additional method-specific parameters:
                - max_dx: int - maximum derivative order (default 4)
                - max_poly: int - maximum polynomial order (default 4)
                - tau: float - trimming/regularization threshold
                - true_coeffs: dict - ground truth coefficients (optional)
        
        Returns:
            metrics: np.ndarray of shape (3,)
                [0] = F1 score (structure accuracy, 0-1, higher is better)
                [1] = coefficient error (relative L2, 0+, lower is better)
                [2] = residual MSE (reconstruction error, 0+, lower is better)
            
            info: dict with additional output:
                - "terms": List[str] - identified PDE terms (e.g., ["u_x", "u_xx"])
                - "coefficients": Dict[str, float] - term coefficients
                - "runtime": float - execution time in seconds
                - Additional method-specific outputs
        
        Raises:
            ValueError: If input dimensions are invalid
            RuntimeError: If the method fails to converge
        """
        ...
    
    def validate_input(self, u_win: np.ndarray, dx: float, dt: float) -> None:
        """
        Validate input data before processing.
        
        Args:
            u_win: Input data window
            dx: Spatial spacing
            dt: Temporal spacing
        
        Raises:
            ValueError: If inputs are invalid
        """
        if u_win.ndim != 2:
            raise ValueError(f"u_win must be 2D, got {u_win.ndim}D")
        
        nt, nx = u_win.shape
        if nt < 4 or nx < 4:
            raise ValueError(f"Window too small: ({nt}, {nx}), need at least (4, 4)")
        
        if dx <= 0 or dt <= 0:
            raise ValueError(f"Grid spacings must be positive: dx={dx}, dt={dt}")
        
        if not np.all(np.isfinite(u_win)):
            raise ValueError("u_win contains NaN or Inf values")
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
