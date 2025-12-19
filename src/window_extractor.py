"""
Window Extractor: Extract overlapping windows from PDE data.

Given a u(x,t) spatiotemporal field, extracts overlapping windows
that serve as individual samples for the selector training data.
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class Window:
    """A single spatiotemporal window."""
    data: np.ndarray  # (nt, nx) array
    x_start: int
    x_end: int
    t_start: int
    t_end: int
    window_id: str
    metadata: Dict[str, Any]


def extract_windows(
    u: np.ndarray,
    window_size: Tuple[int, int],
    stride: Tuple[int, int],
    pde_name: str,
    target_count: Optional[int] = None,
    dx: float = 1.0,
    dt: float = 1.0,
) -> List[Window]:
    """
    Extract overlapping windows from a spatiotemporal field.
    
    Args:
        u: np.ndarray of shape (nx, nt) - spatiotemporal data
           Note: If shape is (nt, nx), will be transposed internally
        window_size: (size_x, size_t) - window dimensions
        stride: (stride_x, stride_t) - step between windows
        pde_name: PDE identifier for window IDs
        target_count: Optional target number of windows (adjusts stride)
        dx: Spatial grid spacing
        dt: Temporal grid spacing
        
    Returns:
        List of Window objects
    """
    # Ensure correct orientation (nx, nt)
    if u.ndim != 2:
        raise ValueError(f"Expected 2D array, got {u.ndim}D")
    
    # Assume first dimension is space, second is time
    # If the data looks transposed (more time than space typically), flip it
    nx, nt = u.shape
    
    size_x, size_t = window_size
    stride_x, stride_t = stride
    
    # Validate
    if size_x > nx or size_t > nt:
        raise ValueError(
            f"Window size ({size_x}, {size_t}) exceeds data size ({nx}, {nt})"
        )
    
    # Calculate number of windows
    n_windows_x = (nx - size_x) // stride_x + 1
    n_windows_t = (nt - size_t) // stride_t + 1
    total_windows = n_windows_x * n_windows_t
    
    # Adjust stride if target_count specified
    if target_count is not None and total_windows < target_count:
        # Reduce stride to get more windows
        stride_x = max(1, (nx - size_x) // int(np.sqrt(target_count)) + 1)
        stride_t = max(1, (nt - size_t) // int(np.sqrt(target_count)) + 1)
        n_windows_x = (nx - size_x) // stride_x + 1
        n_windows_t = (nt - size_t) // stride_t + 1
    
    windows = []
    window_idx = 0
    
    for i_x in range(n_windows_x):
        for i_t in range(n_windows_t):
            x_start = i_x * stride_x
            x_end = x_start + size_x
            t_start = i_t * stride_t
            t_end = t_start + size_t
            
            # Extract window (transpose to nt, nx for IDENT methods)
            window_data = u[x_start:x_end, t_start:t_end].T  # (nt, nx)
            
            window = Window(
                data=window_data,
                x_start=x_start,
                x_end=x_end,
                t_start=t_start,
                t_end=t_end,
                window_id=f"{pde_name}_{window_idx:05d}",
                metadata={
                    "pde_name": pde_name,
                    "dx": dx,
                    "dt": dt,
                    "window_idx": window_idx,
                    "x_range": (x_start * dx, x_end * dx),
                    "t_range": (t_start * dt, t_end * dt),
                },
            )
            windows.append(window)
            window_idx += 1
            
            # Early stop if we have enough
            if target_count is not None and window_idx >= target_count:
                return windows
    
    return windows


def load_pde_data(filepath: str, data_format: str = "nested_object") -> np.ndarray:
    """
    Load PDE data from .npy file.
    
    Args:
        filepath: Path to .npy file
        data_format: Format of the data
            - "nested_object": np.load returns object array with nested array
            - "object_array": Data is directly in object array
            - "standard": Standard numpy array
            
    Returns:
        np.ndarray of shape (nx, nt)
    """
    data = np.load(filepath, allow_pickle=True)
    
    if data_format == "nested_object":
        # Extract inner array
        u = data[0]
        if isinstance(u, np.ndarray):
            return u
        else:
            raise ValueError(f"Expected nested array, got {type(u)}")
    
    elif data_format == "object_array":
        # Object array with floats, reshape
        if data.dtype == object:
            # Try to convert to float array
            u = np.array(data.tolist(), dtype=np.float64)
            # Remove leading dimension if present
            if u.ndim == 3 and u.shape[0] == 1:
                u = u[0]
            return u
        return data
    
    else:  # standard
        if data.ndim == 3 and data.shape[0] == 1:
            return data[0]
        return data


def compute_window_stats(window: Window) -> Dict[str, float]:
    """
    Compute basic statistics for a window (for debugging/validation).
    """
    data = window.data
    return {
        "mean": float(np.mean(data)),
        "std": float(np.std(data)),
        "min": float(np.min(data)),
        "max": float(np.max(data)),
        "shape_nt": data.shape[0],
        "shape_nx": data.shape[1],
    }
