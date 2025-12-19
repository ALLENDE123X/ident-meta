#!/usr/bin/env python
"""
Choose and run IDENT method on a new spatiotemporal field.

Usage:
    python scripts/choose_and_run.py --npy data/u.npy --dx 0.0039 --dt 0.005 --cfg config/default.yaml

Reference: pde-selector-implementation-plan.md §12
"""

import argparse
import yaml
import sys
import os
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import PerMethodRegressor
from src.select_and_run import run_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Choose and run IDENT method on a new field"
    )
    parser.add_argument("--npy", type=str, required=True, help="Path to u.npy (nt, nx)")
    parser.add_argument("--dx", type=float, required=True, help="Spatial step")
    parser.add_argument("--dt", type=float, required=True, help="Temporal step")
    parser.add_argument("--cfg", type=str, required=True, help="Path to config YAML file")
    parser.add_argument(
        "--models", type=str, default="models", help="Directory containing trained models"
    )
    parser.add_argument(
        "--true-coeffs", type=str, default=None, help="Path to true coefficients JSON (optional)"
    )

    args = parser.parse_args()

    # Load config
    with open(args.cfg, "r") as f:
        config = yaml.safe_load(f)

    # Extract settings
    methods = config.get("methods", ["WeakIDENT"])
    agg_cfg = config.get("aggregation", {})
    weights = tuple(agg_cfg.get("weights", [0.5, 0.3, 0.2]))
    tau = agg_cfg.get("safety_tau", 0.6)

    print(f"\n{'=' * 60}")
    print("PDE Selector: Choose and Run")
    print(f"{'=' * 60}")
    print(f"Input: {args.npy}")
    print(f"dx: {args.dx}, dt: {args.dt}")
    print(f"Methods: {methods}")
    print(f"{'=' * 60}\n")

    # Load data
    print("Loading data...")
    u_win = np.load(args.npy)
    print(f"Data shape: {u_win.shape}")

    # Load true coefficients if provided
    true_coeffs = None
    if args.true_coeffs is not None:
        import json

        with open(args.true_coeffs, "r") as f:
            true_coeffs = json.load(f)
        print(f"True coefficients loaded: {true_coeffs}")

    # Load models
    print("\nLoading models...")
    models = {}
    for method in methods:
        model_path = os.path.join(args.models, f"{method}.joblib")
        models[method] = PerMethodRegressor.load(model_path)
        print(f"  Loaded {method}")

    # Run pipeline
    print("\nRunning selector pipeline...")
    best_method, best_metrics, all_results = run_pipeline(
        u_win, args.dx, args.dt, models, w=weights, tau=tau, true_coeffs=true_coeffs
    )

    # Print results
    print(f"\n{'=' * 60}")
    print("Results")
    print(f"{'=' * 60}")
    print(f"Chosen method: {best_method}")
    print(f"Metrics: [F1={best_metrics[0]:.4f}, CoeffErr={best_metrics[1]:.4f}, ResidualMSE={best_metrics[2]:.4f}]")

    print(f"\nAll methods run:")
    for method, metrics in all_results.items():
        print(f"  {method}: [F1={metrics[0]:.4f}, CoeffErr={metrics[1]:.4f}, ResidualMSE={metrics[2]:.4f}]")

    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()

