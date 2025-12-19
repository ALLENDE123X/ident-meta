"""
Method selection with safety gate.

Provides:
- choose_method(): select best method(s) based on predicted metrics + uncertainty
- run_pipeline(): full pipeline (extract features, select, run IDENT, return best)

Safety gate: if predicted score is high OR uncertainty is high, run top-2 methods.

Reference: pde-selector-implementation-plan.md §5, §9
"""

import numpy as np
from .features import extract_tiny12
from .metrics import aggregate
from .ident_api import run_ident_and_metrics


def choose_method(phi_row, models, w=(0.5, 0.3, 0.2), tau=0.6, k_fallback=2):
    """
    Choose best method(s) based on predicted metrics and uncertainty.

    Args:
        phi_row: np.ndarray of shape (1, 12) or (12,), Tiny-12 features for one window
        models: dict of {method_name: PerMethodRegressor}
        w: tuple of 3 floats, weights for aggregation
        tau: float, safety threshold for predicted score
        k_fallback: int, number of top methods to run if safety triggered (default 2)

    Returns:
        list of str: method names to run (1 if confident, k_fallback if uncertain)
    """
    phi_row = np.asarray(phi_row, dtype=np.float64)
    if phi_row.ndim == 1:
        phi_row = phi_row.reshape(1, -1)

    # Predict scores and uncertainties for all methods
    scores = {}
    uncs = {}
    for name, model in models.items():
        yhat = model.predict(phi_row)[0]  # (3,)
        unc = model.predict_unc(phi_row)[0]  # (3,)

        # Aggregate into single score
        score = aggregate(yhat, w)
        scores[name] = score

        # Mean uncertainty across 3 metrics
        mean_unc = np.mean(unc)
        # treat NaN as high uncertainty (uncertain)
        if np.isnan(mean_unc):
            mean_unc = np.inf
        uncs[name] = mean_unc
    
    # Rank methods by predicted score (lower is better)
    ranked = sorted(scores.items(), key=lambda kv: kv[1])
    
    # Best method
    best_method, best_score = ranked[0]
    best_unc = uncs[best_method]
    
    # Safety gate: run top-k if score is high OR uncertainty is high
    # compute median ignoring inf (from NaN)
    unc_values = [u for u in uncs.values() if not np.isinf(u)]
    if len(unc_values) > 0:
        median_unc = np.median(unc_values)
    else:
        median_unc = np.inf  # all are NaN/inf
    
    if best_score > tau or best_unc > median_unc or np.isinf(best_unc):
        # Uncertain: run top-k methods
        chosen = [r[0] for r in ranked[:k_fallback]]
        return chosen
    else:
        # Confident: run only best method
        return [best_method]


def run_pipeline(
    u_win,
    dx,
    dt,
    models,
    w=(0.5, 0.3, 0.2),
    tau=0.6,
    true_coeffs=None,
    ident_params=None,
):
    """
    Full selector pipeline: extract features, choose method(s), run IDENT, return best.

    Args:
        u_win: np.ndarray of shape (nt, nx), spatiotemporal window
        dx: float, spatial step
        dt: float, temporal step
        models: dict of {method_name: PerMethodRegressor}
        w: tuple, aggregation weights
        tau: float, safety threshold
        true_coeffs: dict or None, ground truth for synthetic data
        ident_params: dict or None, extra params for IDENT methods

    Returns:
        tuple: (best_method_name, best_metrics, all_results)
            best_method_name: str
            best_metrics: np.ndarray of shape (3,)
            all_results: dict of {method_name: metrics} for methods that were run
    """
    if ident_params is None:
        ident_params = {}

    # 1. Extract Tiny-12 features
    phi = extract_tiny12(u_win, dx, dt).reshape(1, -1)

    # 2. Choose method(s) to run
    chosen_methods = choose_method(phi, models, w=w, tau=tau)

    # 3. Run chosen IDENT method(s) and collect TRUE metrics
    results = {}
    for method_name in chosen_methods:
        try:
            metrics = run_ident_and_metrics(
                u_win,
                method_name,
                dx,
                dt,
                true_coeffs=true_coeffs,
                **ident_params,
            )
            results[method_name] = metrics
        except Exception as e:
            print(f"Warning: {method_name} failed with error: {e}")
            # Assign worst-case metrics
            results[method_name] = np.array([0.0, 1.0, 1.0], dtype=np.float64)

    # 4. Pick best method by TRUE aggregated score
    best_method = None
    best_score = float("inf")
    best_metrics = None

    for method_name, metrics in results.items():
        score = aggregate(metrics, w)
        if score < best_score:
            best_score = score
            best_method = method_name
            best_metrics = metrics

    return best_method, best_metrics, results

