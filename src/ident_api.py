"""
IDENT method adapters: unified interface for WeakIDENT, RobustIDENT, etc.

Provides run_ident_and_metrics() that runs a chosen method and returns
3 error metrics without leaking IDENT outputs into feature extraction.

Reference: pde-selector-implementation-plan.md §3
"""

import sys
import os
import numpy as np

# Add parent directory to path to import from model.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import weak_ident_pred


def run_ident_and_metrics(
    u_win,
    method,
    dx,
    dt,
    max_dx=4,
    max_poly=4,
    skip_x=4,
    skip_t=4,
    tau=0.05,
    true_coeffs=None,
):
    """
    Run chosen IDENT method on u_win and return 3 error metrics.

    Args:
        u_win: np.ndarray of shape (nt, nx), spatiotemporal window
        method: str, one of {"WeakIDENT", "RobustIDENT", ...}
        dx: float, spatial step
        dt: float, temporal step
        max_dx: int, max derivative order (default 4)
        max_poly: int, max polynomial order (default 4)
        skip_x: int, spatial downsampling (default 4)
        skip_t: int, temporal downsampling (default 4)
        tau: float, trimming threshold (default 0.05)
        true_coeffs: dict or None, ground truth coefficients for synthetic data

    Returns:
        np.ndarray of shape (3,): [m1, m2, m3]
            m1: structure accuracy (F1 score or Jaccard)
            m2: coefficient error (relative L2 on true support)
            m3: residual MSE (cross-validation error)
    """
    method = method.upper()

    if method == "WEAKIDENT":
        return _run_weakident(
            u_win, dx, dt, max_dx, max_poly, skip_x, skip_t, tau, true_coeffs
        )
    elif method == "ROBUSTIDENT":
        # Placeholder for RobustIDENT - to be implemented
        raise NotImplementedError(
            "RobustIDENT is not yet implemented. "
            "Please implement this method or use WeakIDENT only."
        )
    else:
        raise ValueError(
            f"Unknown method: {method}. Supported: ['WeakIDENT', 'RobustIDENT']"
        )


def _run_weakident(u_win, dx, dt, max_dx, max_poly, skip_x, skip_t, tau, true_coeffs):
    """
    Run WeakIDENT and extract 3 metrics.

    Returns:
        [structure_acc, coeff_error, residual_mse]
    """
    nt, nx = u_win.shape

    # Prepare inputs for weak_ident_pred
    u_hat = np.array([u_win.T])  # WeakIDENT expects (nx, nt) transposed
    x = np.linspace(0, nx * dx, nx).reshape(-1, 1)
    t = np.linspace(0, nt * dt, nt).reshape(1, -1)
    xs = np.array([x, t], dtype=object)

    # True coefficients - if not provided, create empty structure
    if true_coeffs is None:
        # No ground truth: return dummy values for m1, m2
        true_coefficients = np.array([np.array([[1.0, 1.0, 0.0, 0.0]])], dtype=object)
    else:
        # Convert dict to WeakIDENT format
        # true_coeffs format: {term_name: coefficient_value}
        # WeakIDENT format: array([[beta_u, d_x, d_t, coeff], ...])
        true_coefficients = _dict_to_weakident_format(true_coeffs)

    # Run WeakIDENT
    try:
        df_errors, df_eqns, df_coeffs, run_time = weak_ident_pred(
            u_hat=u_hat,
            xs=xs,
            true_coefficients=true_coefficients,
            max_dx=max_dx,
            max_poly=max_poly,
            skip_x=skip_x,
            skip_t=skip_t,
            use_cross_der=False,
            tau=tau,
        )

        # Extract metrics from df_errors
        # df_errors columns: ['$e_2$', '$e_{\\infty}$', '$e_{res}$', '$tpr$', '$ppv$']
        e2 = df_errors["$e_2$"].values[0]
        tpr = df_errors["$tpr$"].values[0]
        ppv = df_errors["$ppv$"].values[0]
        e_res = df_errors["$e_{res}$"].values[0]

        # Compute F1 score from TPR and PPV
        if tpr + ppv > 0:
            f1 = 2 * tpr * ppv / (tpr + ppv)
        else:
            f1 = 0.0

        # m1: structure accuracy (F1)
        # m2: coefficient error (e2)
        # m3: residual MSE (e_res)
        metrics = np.array([f1, e2, e_res], dtype=np.float64)

        # If no ground truth, set m1 and m2 to large values (since they're error metrics)
        if true_coeffs is None:
            metrics[0] = 0.0  # F1 unknown
            metrics[1] = 1.0  # coeff error unknown

        return metrics

    except Exception as e:
        # If IDENT fails, return worst-case metrics
        print(f"Warning: WeakIDENT failed with error: {e}")
        return np.array([0.0, 1.0, 1.0], dtype=np.float64)


def _dict_to_weakident_format(true_coeffs):
    """
    Convert dict of {term_name: coeff_value} to WeakIDENT format.

    WeakIDENT expects: array([[beta_u, d_x, d_t, coeff], ...])
    where beta_u is monomial degree, d_x is x-derivative order, d_t is t-derivative order.

    Example:
        {"u_x": -1.0, "u_xx": 0.01} ->
        [[1, 1, 0, -1.0], [1, 2, 0, 0.01]]
    """
    # Parse term names and build array
    rows = []
    for term_name, coeff in true_coeffs.items():
        beta_u, d_x, d_t = _parse_term_name(term_name)
        rows.append([beta_u, d_x, d_t, coeff])

    if len(rows) == 0:
        rows = [[1, 0, 0, 0.0]]  # dummy

    return np.array([np.array(rows, dtype=np.float64)], dtype=object)


def _parse_term_name(term_name):
    """
    Parse term name like 'u_x', 'u_xx', 'u^2_x' into (beta_u, d_x, d_t).

    Examples:
        'u' -> (1, 0, 0)
        'u_x' -> (1, 1, 0)
        'u_xx' -> (1, 2, 0)
        'u_t' -> (1, 0, 1)
        'u^2' -> (2, 0, 0)
        'u^2_x' -> (2, 1, 0)
    """
    beta_u = 1
    d_x = 0
    d_t = 0

    # Check for power
    if "^" in term_name:
        parts = term_name.split("^")
        power_part = parts[1].split("_")[0]
        beta_u = int(power_part)
        term_name = parts[0] + ("_" + "_".join(parts[1].split("_")[1:]) if "_" in parts[1] else "")

    # Count derivative orders
    if "_x" in term_name:
        d_x = term_name.count("x")
    if "_t" in term_name:
        d_t = term_name.count("t")

    return beta_u, d_x, d_t

