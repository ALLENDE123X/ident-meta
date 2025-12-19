"""
Model Zoo: Collection of ML models for PDE-Selector comparison.

Provides a unified interface for training and evaluating different
regression models to predict IDENT method errors.

Models Included:
    1. Ridge Regression - Linear baseline
    2. Random Forest - Tree ensemble (default)
    3. XGBoost - Gradient boosting
    4. SVR - Support Vector Regression
    5. KNN - K-Nearest Neighbors
    6. MLP - Multi-Layer Perceptron
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
import joblib
import time

# Sklearn models
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

# XGBoost (optional)
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


@dataclass
class ModelConfig:
    """Configuration for a model in the zoo."""
    name: str
    model_class: type
    params: Dict[str, Any]
    requires_scaling: bool = False
    description: str = ""


# Model configurations
MODEL_CONFIGS = {
    "ridge": ModelConfig(
        name="Ridge",
        model_class=Ridge,
        params={"alpha": 1.0},
        requires_scaling=True,
        description="Linear baseline with L2 regularization",
    ),
    "random_forest": ModelConfig(
        name="RandomForest",
        model_class=RandomForestRegressor,
        params={
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_split": 5,
            "n_jobs": -1,
            "random_state": 42,
        },
        requires_scaling=False,
        description="Tree ensemble, robust and interpretable",
    ),
    "xgboost": ModelConfig(
        name="XGBoost",
        model_class=XGBRegressor if XGBOOST_AVAILABLE else None,
        params={
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "n_jobs": -1,
        },
        requires_scaling=False,
        description="Gradient boosting, state-of-art for tabular",
    ),
    "svr": ModelConfig(
        name="SVR",
        model_class=SVR,
        params={
            "kernel": "rbf",
            "C": 1.0,
            "gamma": "scale",
        },
        requires_scaling=True,
        description="Support Vector Regression with RBF kernel",
    ),
    "knn": ModelConfig(
        name="KNN",
        model_class=KNeighborsRegressor,
        params={
            "n_neighbors": 5,
            "weights": "distance",
            "n_jobs": -1,
        },
        requires_scaling=True,
        description="K-Nearest Neighbors, non-parametric",
    ),
    "mlp": ModelConfig(
        name="MLP",
        model_class=MLPRegressor,
        params={
            "hidden_layer_sizes": (64, 32),
            "activation": "relu",
            "solver": "adam",
            "alpha": 0.001,
            "max_iter": 500,
            "early_stopping": True,
            "validation_fraction": 0.1,
            "random_state": 42,
        },
        requires_scaling=True,
        description="Multi-Layer Perceptron (neural network)",
    ),
}


def get_available_models() -> Dict[str, ModelConfig]:
    """Get dictionary of available models."""
    available = {}
    for key, config in MODEL_CONFIGS.items():
        if config.model_class is not None:
            available[key] = config
    return available


def create_model(model_name: str, **override_params) -> Pipeline:
    """
    Create a model pipeline by name.
    
    Args:
        model_name: One of 'ridge', 'random_forest', 'xgboost', 'svr', 'knn', 'mlp'
        **override_params: Override default parameters
        
    Returns:
        sklearn Pipeline with optional scaler and model
    """
    if model_name not in MODEL_CONFIGS:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available: {list(MODEL_CONFIGS.keys())}"
        )
    
    config = MODEL_CONFIGS[model_name]
    
    if config.model_class is None:
        raise ImportError(
            f"Model {model_name} requires additional packages. "
            f"Install with: pip install xgboost"
        )
    
    # Merge parameters
    params = {**config.params, **override_params}
    
    # Create model
    model = config.model_class(**params)
    
    # Create pipeline with optional scaling
    if config.requires_scaling:
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", model),
        ])
    else:
        pipeline = Pipeline([
            ("model", model),
        ])
    
    return pipeline


def train_model(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    **params
) -> Tuple[Pipeline, float]:
    """
    Train a model and return it with training time.
    
    Args:
        model_name: Model identifier
        X_train: Training features (n_samples, n_features)
        y_train: Training targets (n_samples,)
        **params: Additional model parameters
        
    Returns:
        (trained_pipeline, training_time_seconds)
    """
    pipeline = create_model(model_name, **params)
    
    start_time = time.time()
    pipeline.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    return pipeline, train_time


def evaluate_model(
    pipeline: Pipeline,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, float]:
    """
    Evaluate a trained model.
    
    Args:
        pipeline: Trained sklearn pipeline
        X_test: Test features
        y_test: Test targets
        
    Returns:
        Dict with MSE, MAE, R2 metrics
    """
    y_pred = pipeline.predict(X_test)
    
    mse = np.mean((y_test - y_pred) ** 2)
    mae = np.mean(np.abs(y_test - y_pred))
    
    # R2 score
    ss_res = np.sum((y_test - y_pred) ** 2)
    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    return {
        "mse": float(mse),
        "mae": float(mae),
        "r2": float(r2),
    }


def cross_validate_model(
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
    **params
) -> Dict[str, float]:
    """
    Cross-validate a model.
    
    Args:
        model_name: Model identifier
        X: Features
        y: Targets
        cv: Number of folds
        **params: Model parameters
        
    Returns:
        Dict with mean and std of cross-validation scores
    """
    pipeline = create_model(model_name, **params)
    
    scores = cross_val_score(
        pipeline, X, y, cv=cv, scoring="neg_mean_squared_error"
    )
    
    return {
        "cv_mse_mean": float(-np.mean(scores)),
        "cv_mse_std": float(np.std(scores)),
    }


def save_model(pipeline: Pipeline, filepath: str) -> None:
    """Save trained model to disk."""
    joblib.dump(pipeline, filepath)


def load_model(filepath: str) -> Pipeline:
    """Load trained model from disk."""
    return joblib.load(filepath)


def get_feature_importance(
    pipeline: Pipeline,
    feature_names: Optional[list] = None,
) -> Optional[Dict[str, float]]:
    """
    Extract feature importance if available.
    
    Works for Random Forest and XGBoost.
    
    Returns:
        Dict mapping feature name to importance, or None if not available
    """
    model = pipeline.named_steps.get("model")
    
    if model is None:
        return None
    
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(importances))]
        
        return dict(zip(feature_names, importances.tolist()))
    
    return None


# Convenience function for comparing all models
def compare_all_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    verbose: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """
    Train and evaluate all available models.
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        verbose: Print progress
        
    Returns:
        Dict mapping model name to results dict with:
            - metrics: evaluation metrics
            - train_time: training time
            - pipeline: trained pipeline
    """
    results = {}
    available = get_available_models()
    
    for model_key, config in available.items():
        if verbose:
            print(f"Training {config.name}...", end=" ", flush=True)
        
        try:
            pipeline, train_time = train_model(model_key, X_train, y_train)
            metrics = evaluate_model(pipeline, X_test, y_test)
            
            results[model_key] = {
                "name": config.name,
                "description": config.description,
                "metrics": metrics,
                "train_time": train_time,
                "pipeline": pipeline,
            }
            
            if verbose:
                print(f"MSE={metrics['mse']:.4f}, R2={metrics['r2']:.4f}")
                
        except Exception as e:
            if verbose:
                print(f"FAILED: {e}")
            results[model_key] = {"name": config.name, "error": str(e)}
    
    return results


def print_comparison_table(results: Dict[str, Dict[str, Any]]) -> str:
    """
    Format comparison results as a markdown table.
    
    Returns:
        Markdown string
    """
    lines = [
        "| Model | MSE | MAE | R² | Train Time |",
        "|-------|-----|-----|-----|------------|",
    ]
    
    for model_key, data in sorted(
        results.items(),
        key=lambda x: x[1].get("metrics", {}).get("mse", float("inf"))
    ):
        if "error" in data:
            lines.append(f"| {data['name']} | ERROR | - | - | - |")
        else:
            m = data["metrics"]
            t = data["train_time"]
            lines.append(
                f"| {data['name']} | {m['mse']:.4f} | {m['mae']:.4f} | "
                f"{m['r2']:.4f} | {t:.2f}s |"
            )
    
    return "\n".join(lines)
