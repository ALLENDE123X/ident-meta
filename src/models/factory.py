"""
Model factory for creating pluggable meta-regression models.

Supports 5 model types:
1. linear_ols: MultiOutputRegressor(LinearRegression())
2. ridge_multi: MultiOutputRegressor(Ridge(random_state=0))
3. regressor_chain_ridge: RegressorChain(base_estimator=Ridge(random_state=0))
4. rf_multi: RandomForestRegressor(n_estimators=300, max_depth=8, random_state=0)
5. catboost_multi: CatBoostRegressor(loss_function="MultiRMSE", verbose=False, random_state=0)
"""

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor, RegressorChain


def create_model(name: str, **params):
    """
    Create a model instance by name.
    
    Args:
        name: str, one of ["linear_ols", "ridge_multi", "regressor_chain_ridge", 
                           "rf_multi", "catboost_multi"]
        **params: dict, model-specific parameters (merged with defaults)
    
    Returns:
        estimator: sklearn-compatible estimator with fit/predict methods
    
    Raises:
        ValueError: if model name is not recognized
        ImportError: if catboost is required but not installed
    """
    # set random_state in params if not provided
    if "random_state" not in params:
        params["random_state"] = 0
    
    if name == "linear_ols":
        base = LinearRegression()
        return MultiOutputRegressor(base)
    
    elif name == "ridge_multi":
        alpha = params.pop("alpha", 1.0)
        base = Ridge(alpha=alpha, random_state=params["random_state"])
        return MultiOutputRegressor(base)
    
    elif name == "regressor_chain_ridge":
        alpha = params.pop("alpha", 1.0)
        order = params.pop("order", None)
        base = Ridge(alpha=alpha, random_state=params["random_state"])
        return RegressorChain(
            base_estimator=base, 
            order=order, 
            random_state=params["random_state"]
        )
    
    elif name == "rf_multi":
        n_estimators = params.pop("n_estimators", 300)
        max_depth = params.pop("max_depth", 8)
        n_jobs = params.pop("n_jobs", -1)
        return RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            n_jobs=n_jobs,
            random_state=params["random_state"]
        )
    
    elif name == "catboost_multi":
        try:
            from catboost import CatBoostRegressor
        except ImportError:
            raise ImportError(
                "catboost is not installed. Install it with: pip install catboost>=1.2"
            )
        
        iterations = params.pop("iterations", 1000)
        learning_rate = params.pop("learning_rate", 0.03)
        depth = params.pop("depth", 6)
        loss_function = params.pop("loss_function", "MultiRMSE")
        verbose = params.pop("verbose", False)
        
        return CatBoostRegressor(
            iterations=iterations,
            learning_rate=learning_rate,
            depth=depth,
            loss_function=loss_function,
            verbose=verbose,
            random_state=params["random_state"]
        )
    
    else:
        raise ValueError(
            f"Unknown model name: {name}. "
            f"Supported: ['linear_ols', 'ridge_multi', 'regressor_chain_ridge', "
            f"'rf_multi', 'catboost_multi']"
        )

