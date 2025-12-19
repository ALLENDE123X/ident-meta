"""
IDENT Methods Package

This package provides a plugin-style architecture for integrating
different PDE identification methods (WeakIDENT, RobustIDENT, WSINDy, etc.)
into the PDE-Selector framework.

Each method must implement the IdentMethodBase interface and register
itself in the METHOD_REGISTRY.

Example:
    from src.ident_methods import METHOD_REGISTRY, IdentMethodBase
    
    class MyNewMethod(IdentMethodBase):
        @property
        def name(self) -> str:
            return "MyNewMethod"
        
        def run(self, u_win, dx, dt, **kwargs):
            # Your implementation here
            metrics = np.array([f1, coeff_err, residual])
            info = {"terms": [...], "coefficients": {...}}
            return metrics, info
    
    METHOD_REGISTRY.register(MyNewMethod())
"""

from .base import IdentMethodBase
from .registry import METHOD_REGISTRY

# Import method implementations to auto-register them
try:
    from .lasso_sindy_method import LassoSINDyMethod
except ImportError:
    pass  # sklearn not installed

try:
    from .stlsq_method import STLSQMethod
except ImportError:
    pass  # Should always work (no external deps)

try:
    from .robustident_method import RobustIdentMethod
except ImportError:
    pass  # Should always work (no external deps)

try:
    from .weakident_method import WeakIdentMethod
except ImportError:
    pass  # May fail if model.py not available

# Optional: PySINDy-based methods (may have API issues)
try:
    from .pysindy_method import PySINDyMethod
except ImportError:
    pass

try:
    from .wsindy_method import WSINDyMethod
except ImportError:
    pass

__all__ = ["IdentMethodBase", "METHOD_REGISTRY"]

