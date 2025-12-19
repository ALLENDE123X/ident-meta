"""
Models package for PDE selector.

This package contains:
- model_zoo: Collection of ML models for comparison
- factory: Model creation utilities
"""

# Lazy imports to avoid circular dependencies
__all__ = ["model_zoo", "factory", "create_selector_model"]


def create_selector_model(model_type: str = "random_forest", **kwargs):
    """
    Create a selector model for predicting IDENT method performance.
    
    Args:
        model_type: One of 'ridge', 'random_forest', 'xgboost', 'svr', 'knn', 'mlp'
        **kwargs: Model-specific parameters
        
    Returns:
        sklearn Pipeline with the model
    """
    from .model_zoo import create_model
    return create_model(model_type, **kwargs)
