#!/usr/bin/env python
"""
Train per-method regressors for PDE selector.

Usage:
    python scripts/train_selector.py --cfg config/default.yaml
    python scripts/train_selector.py --cfg config/default.yaml --model ridge_multi
    python scripts/train_selector.py --cfg config/default.yaml --model catboost_multi --params '{"iterations":600,"learning_rate":0.05}'

Reference: pde-selector-implementation-plan.md §12
"""

import argparse
import yaml
import json
import sys
import os
import numpy as np
from sklearn.model_selection import train_test_split

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.label_dataset import load_dataset
from src.models import PerMethodRegressor


def main():
    parser = argparse.ArgumentParser(description="Train PDE selector models")
    parser.add_argument("--cfg", type=str, required=True, help="Path to config YAML file")
    parser.add_argument(
        "--data", type=str, default="artifacts", help="Directory containing datasets"
    )
    parser.add_argument(
        "--output", type=str, default="models", help="Output directory for trained models"
    )
    parser.add_argument("--test-split", type=float, default=0.2, help="Test set fraction")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--model",
        type=str,
        choices=["linear_ols", "ridge_multi", "regressor_chain_ridge", "rf_multi", "catboost_multi"],
        help="Model type to use (overrides config)"
    )
    parser.add_argument(
        "--params",
        type=str,
        help="JSON string of model parameters to override (e.g., '{\"alpha\":1.0,\"max_depth\":10}')"
    )

    args = parser.parse_args()

    # Load config
    with open(args.cfg, "r") as f:
        config = yaml.safe_load(f)

    # Extract settings
    methods = config.get("methods", ["WeakIDENT"])
    model_cfg = config.get("model", {})
    
    # get model name: command line > config > default
    model_name = args.model or model_cfg.get("name", "rf_multi")
    
    # get model params: command line > config > defaults
    model_params = model_cfg.get("params", {}).copy()
    if args.params:
        try:
            override_params = json.loads(args.params)
            model_params.update(override_params)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in --params: {e}")
    
    # ensure random_state
    if "random_state" not in model_params:
        model_params["random_state"] = model_cfg.get("random_state", 0)

    print(f"\n{'=' * 60}")
    print("PDE Selector Training")
    print(f"{'=' * 60}")
    print(f"Config: {args.cfg}")
    print(f"Methods: {methods}")
    print(f"Model: {model_name}")
    print(f"Model params: {model_params}")
    print(f"Data directory: {args.data}")
    print(f"{'=' * 60}\n")

    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset(methods, data_dir=args.data)
    X_features = dataset["X_features"]

    print(f"Dataset loaded: {X_features.shape[0]} samples")

    # Split train/test
    indices = np.arange(len(X_features))
    train_idx, test_idx = train_test_split(
        indices, test_size=args.test_split, random_state=args.random_seed
    )

    X_train = X_features[train_idx]
    X_test = X_features[test_idx]

    print(f"Train size: {len(train_idx)}, Test size: {len(test_idx)}")

    # Save test indices for later evaluation
    os.makedirs(args.output, exist_ok=True)
    np.save(os.path.join(args.output, "test_indices.npy"), test_idx)

    # Train one model per method
    for method in methods:
        print(f"\n{'=' * 60}")
        print(f"Training model for {method}")
        print(f"{'=' * 60}")

        Y_all = dataset[method]
        Y_train = Y_all[train_idx]
        Y_test = Y_all[test_idx]

        # Initialize model
        model = PerMethodRegressor(model_name=model_name, **model_params)

        # Train
        print("Fitting model...")
        model.fit(X_train, Y_train)

        # Evaluate on train and test
        Y_train_pred = model.predict(X_train)
        Y_test_pred = model.predict(X_test)

        train_mse = np.mean((Y_train - Y_train_pred) ** 2)
        test_mse = np.mean((Y_test - Y_test_pred) ** 2)

        print(f"Train MSE: {train_mse:.4f}")
        print(f"Test MSE:  {test_mse:.4f}")

        # Feature importances
        importances = model.get_feature_importances()
        print("\nTop 5 feature importances:")
        feature_names = [
            "dx",
            "dt",
            "A",
            "Rx",
            "Rxx",
            "Rt",
            "SNR_dB",
            "out_frac",
            "k_centroid",
            "slope",
            "w_centroid",
            "rho_per",
        ]
        sorted_idx = np.argsort(importances)[::-1]
        for i in range(min(5, len(importances))):
            idx = sorted_idx[i]
            print(f"  {feature_names[idx]:12s}: {importances[idx]:.4f}")

        # Save model with model name in filename
        model_filename = f"{method}_{model_name}.joblib"
        model_path = os.path.join(args.output, model_filename)
        model.save(model_path)
        
        # Save metadata
        metadata = {
            "method": method,
            "model_name": model_name,
            "model_params": model_params,
            "train_mse": float(train_mse),
            "test_mse": float(test_mse),
            "n_train": len(train_idx),
            "n_test": len(test_idx),
        }
        metadata_path = os.path.join(args.output, f"{method}_{model_name}.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"Metadata saved to {metadata_path}")

    print(f"\n{'=' * 60}")
    print("Training complete!")
    print(f"Models saved to: {args.output}/")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()

