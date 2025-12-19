"""
Error metrics for PDE identification methods.

Provides 3 metrics:
    1. Structure accuracy: F1 score of term selection (synthetic only)
    2. Coefficient error: relative L2 error on nonzero coefficients (synthetic only)
    3. Residual MSE: mean squared residual (synthetic and real data)

Also provides aggregate() to combine metrics into a single score.

Reference: pde-selector-implementation-plan.md §4
"""

import numpy as np


def compute_structure_accuracy(pred_support, true_support):
    """
    Compute F1 score (or Jaccard) of term selection.

    Args:
        pred_support: set or list of predicted term indices
        true_support: set or list of true term indices

    Returns:
        float: F1 score in [0, 1]
    """
    pred_support = set(pred_support)
    true_support = set(true_support)

    tp = len(pred_support & true_support)
    fp = len(pred_support - true_support)
    fn = len(true_support - pred_support)

    # F1 = 2*TP / (2*TP + FP + FN)
    if tp + fp + fn == 0:
        return 1.0  # perfect if both empty

    f1 = 2 * tp / (2 * tp + fp + fn)
    return float(f1)


def compute_coefficient_error(pred_coeffs, true_coeffs, true_support):
    """
    Compute relative L2 error on the true support.

    Args:
        pred_coeffs: dict or array, predicted coefficients
        true_coeffs: dict or array, true coefficients
        true_support: list of indices in the true support

    Returns:
        float: relative L2 error
    """
    if len(true_support) == 0:
        return 0.0

    # Extract coefficients on true support
    if isinstance(pred_coeffs, dict) and isinstance(true_coeffs, dict):
        true_vals = np.array([true_coeffs.get(k, 0.0) for k in true_support])
        pred_vals = np.array([pred_coeffs.get(k, 0.0) for k in true_support])
    else:
        # Assume arrays
        true_vals = np.asarray(true_coeffs)[true_support]
        pred_vals = np.asarray(pred_coeffs)[true_support]

    norm_true = np.linalg.norm(true_vals)
    if norm_true < 1e-12:
        return 0.0

    error = np.linalg.norm(pred_vals - true_vals) / norm_true
    return float(error)


def compute_residual_mse(u, u_pred, mask=None):
    """
    Compute mean squared residual between true and predicted fields.

    Args:
        u: np.ndarray, true field
        u_pred: np.ndarray, predicted/reconstructed field
        mask: np.ndarray or None, optional mask for held-out points

    Returns:
        float: MSE
    """
    if mask is not None:
        diff = (u - u_pred)[mask]
    else:
        diff = u - u_pred

    mse = np.mean(diff ** 2)
    return float(mse)


def aggregate(y3, w=(0.5, 0.3, 0.2)):
    """
    Aggregate 3 metrics into a single score via weighted sum.

    Args:
        y3: np.ndarray of shape (3,), [m1, m2, m3]
            m1: structure accuracy (lower is worse, but we want to minimize error)
            m2: coefficient error (lower is better)
            m3: residual MSE (lower is better)
        w: tuple of 3 floats, weights for each metric

    Returns:
        float: aggregated score (lower is better)

    Note:
        - m1 is F1 score (higher is better), so we use (1 - m1) as error
        - m2 and m3 are errors (lower is better)
        - All metrics are errors to minimize
    """
    y3 = np.asarray(y3, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)

    # Convert m1 (F1 score) to error: error = 1 - F1
    # But the ident_api already returns m1 as F1 (higher is better)
    # For consistency with "lower is better", we invert it here
    m1_error = 1.0 - y3[0]  # F1 -> error
    m2_error = y3[1]  # coefficient error
    m3_error = y3[2]  # residual MSE

    errors = np.array([m1_error, m2_error, m3_error])

    # Weighted sum
    score = np.dot(w, errors)
    return float(score)


def aggregate_rank_based(y3_dict):
    """
    Alternative: rank-based aggregation (optional).

    Args:
        y3_dict: dict of {method_name: y3_array}

    Returns:
        dict of {method_name: rank_score}
    """
    # Convert to errors
    errors_dict = {}
    for method, y3 in y3_dict.items():
        m1_error = 1.0 - y3[0]
        errors = np.array([m1_error, y3[1], y3[2]])
        errors_dict[method] = errors

    # Rank each metric
    n_methods = len(errors_dict)
    methods = list(errors_dict.keys())
    ranks = {m: 0.0 for m in methods}

    for metric_idx in range(3):
        metric_vals = [(m, errors_dict[m][metric_idx]) for m in methods]
        metric_vals.sort(key=lambda x: x[1])  # sort by error (ascending)
        for rank, (method, _) in enumerate(metric_vals):
            ranks[method] += rank

    # Average rank (lower is better)
    ranks = {m: r / 3.0 for m, r in ranks.items()}
    return ranks

