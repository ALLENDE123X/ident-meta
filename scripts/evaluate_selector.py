#!/usr/bin/env python
"""
Evaluate trained PDE selector on test set.

Usage:
    python scripts/evaluate_selector.py --cfg config/default.yaml
    python scripts/evaluate_selector.py --cfg config/default.yaml --model ridge_multi
    python scripts/evaluate_selector.py --cfg config/default.yaml --output-json results.json

Reference: pde-selector-implementation-plan.md §12
"""

import argparse
import yaml
import sys
import os
import json
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.label_dataset import load_dataset
from src.models import PerMethodRegressor
from src.metrics import aggregate
from src.eval import evaluate_selector, plot_regret_cdf, plot_confusion_matrix


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate PDE selector models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic evaluation
  python scripts/evaluate_selector.py --cfg config/default.yaml

  # Use specific model
  python scripts/evaluate_selector.py --cfg config/default.yaml --model ridge_multi

  # Save JSON results
  python scripts/evaluate_selector.py --cfg config/default.yaml --output-json results.json

  # Generate plots
  python scripts/evaluate_selector.py --cfg config/default.yaml --plot
        """,
    )
    parser.add_argument("--cfg", type=str, required=True, help="Path to config YAML file")
    parser.add_argument(
        "--data", type=str, default="artifacts", help="Directory containing datasets"
    )
    parser.add_argument(
        "--models", type=str, default="models", help="Directory containing trained models"
    )
    parser.add_argument(
        "--output", type=str, default="artifacts", help="Output directory for results"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["linear_ols", "ridge_multi", "regressor_chain_ridge", "rf_multi", "catboost_multi"],
        help="Model type to use (overrides config)"
    )
    parser.add_argument(
        "--output-json",
        type=str,
        metavar="FILE",
        help="Save evaluation results to JSON file",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate evaluation plots (regret CDF, confusion matrix)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed progress",
    )

    args = parser.parse_args()

    # Load config
    with open(args.cfg, "r") as f:
        config = yaml.safe_load(f)

    # Extract settings
    methods = config.get("methods", ["WeakIDENT"])
    model_cfg = config.get("model", {})
    model_name = args.model or model_cfg.get("name", "rf_multi")
    
    agg_cfg = config.get("aggregation", {})
    weights = tuple(agg_cfg.get("weights", [0.5, 0.3, 0.2]))
    tau = agg_cfg.get("safety_tau", 0.6)

    print(f"\n{'=' * 60}")
    print("PDE Selector Evaluation")
    print(f"{'=' * 60}")
    print(f"Config: {args.cfg}")
    print(f"Methods: {methods}")
    print(f"Model: {model_name}")
    print(f"Weights: {weights}")
    print(f"Safety tau: {tau}")
    print(f"{'=' * 60}\n")

    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset(methods, data_dir=args.data)
    X_features = dataset["X_features"]

    # Load test indices
    test_idx_path = os.path.join(args.models, "test_indices.npy")
    if not os.path.exists(test_idx_path):
        print(f"Warning: {test_idx_path} not found, using last 20% as test set")
        n = len(X_features)
        test_idx = np.arange(int(n * 0.8), n)
    else:
        test_idx = np.load(test_idx_path)
    X_test = X_features[test_idx]

    print(f"Test set size: {len(test_idx)}")

    # Load models
    print("\nLoading models...")
    models = {}
    for method in methods:
        # try new format first: {method}_{model_name}.joblib
        model_path = os.path.join(args.models, f"{method}_{model_name}.joblib")
        if not os.path.exists(model_path):
            # fallback to old format: {method}.joblib
            model_path = os.path.join(args.models, f"{method}.joblib")
        if not os.path.exists(model_path):
            print(f"Error: Model not found at {model_path}")
            sys.exit(1)
        models[method] = PerMethodRegressor.load(model_path)
        print(f"  Loaded {method} from {model_path}")

    # Evaluate on test set
    print("\nEvaluating selector...")

    chosen_methods = []
    chosen_scores = []
    all_scores = []
    n_methods_run = []

    for i in range(len(X_test)):
        phi = X_test[i : i + 1]

        # Predict scores for all methods
        scores_dict = {}
        uncs_dict = {}
        for method in methods:
            yhat = models[method].predict(phi)[0]
            unc = models[method].predict_unc(phi)[0]
            score = aggregate(yhat, weights)
            scores_dict[method] = score
            # treat NaN as high uncertainty
            mean_unc = np.mean(unc)
            uncs_dict[method] = np.inf if np.isnan(mean_unc) else mean_unc

        # Choose method(s) with safety gate
        ranked = sorted(scores_dict.items(), key=lambda kv: kv[1])
        best_method, best_score = ranked[0]
        best_unc = uncs_dict[best_method]
        # compute median ignoring inf (from NaN)
        unc_values = [u for u in uncs_dict.values() if not np.isinf(u)]
        if len(unc_values) > 0:
            median_unc = np.median(unc_values)
        else:
            median_unc = np.inf

        if best_score > tau or best_unc > median_unc or np.isinf(best_unc):
            # Run top-2
            n_run = min(2, len(methods))
            # For evaluation, we use oracle: just record the best predicted
            chosen_method = best_method
        else:
            # Run only best
            n_run = 1
            chosen_method = best_method

        chosen_methods.append(chosen_method)
        chosen_scores.append(best_score)
        all_scores.append(scores_dict)
        n_methods_run.append(n_run)

    # Compile results
    results = {
        "chosen_methods": chosen_methods,
        "chosen_scores": chosen_scores,
        "all_scores": all_scores,
        "n_methods_run": n_methods_run,
    }

    # Evaluate
    eval_results = evaluate_selector(results, methods, output_dir=args.output)

    # Add metadata
    eval_results["model_name"] = model_name
    eval_results["config_file"] = args.cfg
    eval_results["methods"] = methods
    eval_results["weights"] = list(weights)
    eval_results["safety_tau"] = tau

    # Save JSON output if requested
    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(eval_results, f, indent=2)
        print(f"\nResults saved to {args.output_json}")

    # Compute oracle best for plotting
    best_methods = []
    best_scores = []
    for scores_dict in all_scores:
        best_method = min(scores_dict, key=scores_dict.get)
        best_score = scores_dict[best_method]
        best_methods.append(best_method)
        best_scores.append(best_score)

    # Generate plots if requested (or by default for backwards compatibility)
    if args.plot:
        print("\nGenerating plots...")
        plot_regret_cdf(
            np.array(chosen_scores), np.array(best_scores), output_dir=args.output, model_name=model_name
        )
        plot_confusion_matrix(chosen_methods, best_methods, methods, output_dir=args.output, model_name=model_name)

        print(f"Plots saved to {args.output}/")
        print(f"  - regret_cdf_{model_name}.png")
        print(f"  - confusion_matrix_{model_name}.png")

    print(f"\n{'=' * 60}")
    print("Evaluation complete!")
    print(f"{'=' * 60}\n")

    return eval_results


if __name__ == "__main__":
    main()
