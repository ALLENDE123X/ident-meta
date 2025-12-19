#!/usr/bin/env python
"""
Compare ML models for PDE-Selector.

Trains and evaluates all 6 models on the same dataset, producing a
comparison table suitable for publication.

Usage:
    python scripts/compare_models.py --cfg config/default.yaml --output artifacts/model_comparison
    
    # Quick test with synthetic data
    python scripts/compare_models.py --synthetic --output artifacts/model_comparison
"""

import argparse
import json
import os
import sys
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.model_zoo import (
    get_available_models,
    compare_all_models,
    print_comparison_table,
    get_feature_importance,
)


def load_dataset(cfg_path: str, artifacts_dir: str = "artifacts"):
    """Load dataset from artifacts directory."""
    import yaml
    
    with open(cfg_path) as f:
        config = yaml.safe_load(f)
    
    # Load features and labels
    X = np.load(os.path.join(artifacts_dir, "X_features.npy"))
    
    # Load method scores - need to compute "best" method for each sample
    methods = config.get("methods", ["WeakIDENT"])
    Y_dict = {}
    for method in methods:
        y_path = os.path.join(artifacts_dir, f"Y_{method}.npy")
        if os.path.exists(y_path):
            Y_dict[method] = np.load(y_path)
    
    return X, Y_dict, methods


def generate_synthetic_data(
    n_samples: int = 1000,
    n_features: int = 12,
    n_methods: int = 4,
    random_state: int = 42,
):
    """
    Generate synthetic data for testing model comparison.
    
    Creates features and method scores with known structure.
    """
    np.random.seed(random_state)
    
    # Generate features
    X = np.random.randn(n_samples, n_features)
    
    # Generate method scores (lower is better)
    # Each method has different sensitivity to features
    methods = ["WeakIDENT", "PySINDy", "RobustIDENT", "WSINDy"][:n_methods]
    Y_dict = {}
    
    for i, method in enumerate(methods):
        # Base error
        base = 0.5 + 0.1 * i
        
        # Linear dependence on some features
        linear = 0.1 * X[:, i % n_features]
        
        # Nonlinear dependence
        nonlinear = 0.05 * np.sin(X[:, (i + 1) % n_features] * 2)
        
        # Noise
        noise = 0.1 * np.random.randn(n_samples)
        
        Y_dict[method] = np.clip(base + linear + nonlinear + noise, 0, 2)
    
    return X, Y_dict, methods


def run_comparison(
    X: np.ndarray,
    Y_dict: dict,
    methods: list,
    output_dir: str,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Run model comparison for each IDENT method.
    
    Trains each ML model to predict each IDENT method's error.
    """
    from sklearn.model_selection import train_test_split
    
    os.makedirs(output_dir, exist_ok=True)
    
    all_results = {}
    
    for method_name in methods:
        if method_name not in Y_dict:
            print(f"Skipping {method_name} (no labels found)")
            continue
        
        y = Y_dict[method_name]
        
        print(f"\n{'='*60}")
        print(f"Comparing models for: {method_name}")
        print(f"{'='*60}")
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Compare all models
        results = compare_all_models(X_train, y_train, X_test, y_test)
        
        # Print table
        print("\n" + print_comparison_table(results))
        
        # Find best model
        best_model = min(
            [(k, v) for k, v in results.items() if "error" not in v],
            key=lambda x: x[1]["metrics"]["mse"],
            default=(None, None)
        )
        
        if best_model[0]:
            print(f"\n✓ Best model: {best_model[1]['name']}")
            
            # Get feature importance if available
            importance = get_feature_importance(best_model[1]["pipeline"])
            if importance:
                print("\nTop 5 features:")
                top_features = sorted(
                    importance.items(), key=lambda x: -x[1]
                )[:5]
                for feat, imp in top_features:
                    print(f"  {feat}: {imp:.4f}")
        
        # Store results (without pipeline for JSON serialization)
        all_results[method_name] = {
            model_key: {
                k: v for k, v in data.items() 
                if k != "pipeline"
            }
            for model_key, data in results.items()
        }
    
    # Save results
    results_path = os.path.join(output_dir, "model_comparison.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")
    
    # Generate summary markdown
    summary = generate_summary_markdown(all_results, methods)
    summary_path = os.path.join(output_dir, "model_comparison.md")
    with open(summary_path, "w") as f:
        f.write(summary)
    print(f"Summary saved to {summary_path}")
    
    return all_results


def generate_summary_markdown(results: dict, methods: list) -> str:
    """Generate publication-ready markdown summary."""
    lines = [
        "# Model Comparison Results",
        "",
        "## Summary",
        "",
        "Comparison of 6 ML models for predicting IDENT method errors.",
        "",
    ]
    
    # Per-method tables
    for method in methods:
        if method not in results:
            continue
        
        lines.append(f"### {method}")
        lines.append("")
        lines.append("| Model | MSE | MAE | R² | Train Time |")
        lines.append("|-------|-----|-----|-----|------------|")
        
        sorted_models = sorted(
            results[method].items(),
            key=lambda x: x[1].get("metrics", {}).get("mse", float("inf"))
        )
        
        best_mse = float("inf")
        for model_key, data in sorted_models:
            if "error" in data:
                lines.append(f"| {data['name']} | ERROR | - | - | - |")
            else:
                m = data["metrics"]
                t = data["train_time"]
                marker = " **" if m["mse"] < best_mse else ""
                best_mse = min(best_mse, m["mse"])
                lines.append(
                    f"| {data['name']}{marker} | {m['mse']:.4f} | "
                    f"{m['mae']:.4f} | {m['r2']:.4f} | {t:.2f}s |"
                )
        
        lines.append("")
    
    # Overall best model
    lines.append("## Recommendation")
    lines.append("")
    
    overall_mse = {}
    for method, model_results in results.items():
        for model_key, data in model_results.items():
            if "metrics" in data:
                if model_key not in overall_mse:
                    overall_mse[model_key] = []
                overall_mse[model_key].append(data["metrics"]["mse"])
    
    avg_mse = {k: np.mean(v) for k, v in overall_mse.items()}
    if avg_mse:
        best = min(avg_mse.items(), key=lambda x: x[1])
        lines.append(f"**Best overall model: {best[0]}** (avg MSE: {best[1]:.4f})")
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Compare ML models for PDE-Selector"
    )
    parser.add_argument(
        "--cfg", type=str, help="Config file path"
    )
    parser.add_argument(
        "--output", type=str, default="artifacts/model_comparison",
        help="Output directory"
    )
    parser.add_argument(
        "--synthetic", action="store_true",
        help="Use synthetic data for testing"
    )
    parser.add_argument(
        "--n-samples", type=int, default=1000,
        help="Number of synthetic samples"
    )
    
    args = parser.parse_args()
    
    if args.synthetic:
        print("Generating synthetic data...")
        X, Y_dict, methods = generate_synthetic_data(n_samples=args.n_samples)
    else:
        if not args.cfg:
            parser.error("--cfg required when not using --synthetic")
        X, Y_dict, methods = load_dataset(args.cfg)
    
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Methods: {methods}")
    
    run_comparison(X, Y_dict, methods, args.output)


if __name__ == "__main__":
    main()
