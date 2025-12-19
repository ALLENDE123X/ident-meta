"""
Per-method regressors for PDE selector.

Provides PerMethodRegressor that:
- Takes Tiny-12 features as input
- Predicts 3 error metrics for a given IDENT method
- Supports pluggable models via factory
- Provides uncertainty estimates where available (RF variance; NaN otherwise)

Reference: pde-selector-implementation-plan.md §5, §8
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
from src.models.factory import create_model


class PerMethodRegressor:
    """
    Per-method multi-output regressor for predicting 3 error metrics.
    
    Supports pluggable models:
    - linear_ols: MultiOutputRegressor(LinearRegression())
    - ridge_multi: MultiOutputRegressor(Ridge())
    - regressor_chain_ridge: RegressorChain(Ridge())
    - rf_multi: RandomForestRegressor (native multi-output)
    - catboost_multi: CatBoostRegressor (native multi-output)
    
    Uses:
    - StandardScaler for feature normalization (linear/ridge/chain only)
    - log1p transform on targets (expm1 at prediction)
    - Uncertainty via tree variance (rf_multi) or NaN (others)
    """

    def __init__(self, model_name="rf_multi", **model_params):
        """
        Initialize the regressor.
        
        Args:
            model_name: str, one of ["linear_ols", "ridge_multi", "regressor_chain_ridge",
                                     "rf_multi", "catboost_multi"]
            **model_params: dict, model-specific parameters
        """
        self.model_name = model_name
        self.model_params = model_params.copy()
        
        # create model via factory
        self.model = create_model(model_name, **self.model_params)
        
        # use scaler for linear/ridge/chain models; skip for tree models
        self.use_scaler = model_name in ["linear_ols", "ridge_multi", "regressor_chain_ridge"]
        if self.use_scaler:
            self.scaler = StandardScaler()
        else:
            self.scaler = None
        
        self.fitted = False

    def fit(self, X, Y):
        """
        Fit the regressor.
        
        Args:
            X: np.ndarray of shape (n_samples, 12), Tiny-12 features
            Y: np.ndarray of shape (n_samples, 3), 3 error metrics
        
        Returns:
            self
        """
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        
        # standardize features for linear/ridge/chain models
        if self.use_scaler:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = X
        
        # log-transform targets (to handle wide range of error values)
        Y_log = np.log1p(Y)
        
        # fit model
        self.model.fit(X_scaled, Y_log)
        self.fitted = True
        
        return self

    def predict(self, X):
        """
        Predict 3 error metrics.
        
        Args:
            X: np.ndarray of shape (n_samples, 12), Tiny-12 features
        
        Returns:
            Y_pred: np.ndarray of shape (n_samples, 3), predicted metrics
        """
        if not self.fitted:
            raise RuntimeError("Model must be fitted before prediction.")
        
        X = np.asarray(X, dtype=np.float64)
        
        # scale features if needed
        if self.use_scaler:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        
        # predict in log space
        Y_log_pred = self.model.predict(X_scaled)
        
        # inverse transform
        Y_pred = np.expm1(Y_log_pred)
        
        # clip to reasonable range (metrics are non-negative)
        Y_pred = np.maximum(Y_pred, 0.0)
        
        return Y_pred

    def predict_unc(self, X):
        """
        Predict uncertainty (variance) for each metric.
        
        For rf_multi: computes variance across trees per target.
        For other models: returns NaN (safety gate handles it).
        
        Args:
            X: np.ndarray of shape (n_samples, 12), Tiny-12 features
        
        Returns:
            unc: np.ndarray of shape (n_samples, 3), uncertainty per metric
        """
        if not self.fitted:
            raise RuntimeError("Model must be fitted before prediction.")
        
        X = np.asarray(X, dtype=np.float64)
        
        # scale features if needed
        if self.use_scaler:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        
        # only rf_multi supports variance estimation
        if self.model_name == "rf_multi":
            # get predictions from all trees
            # for multi-output RF, each tree.predict() returns (n_samples, n_outputs)
            n_samples = X_scaled.shape[0]
            n_targets = 3
            tree_preds = []
            
            for tree in self.model.estimators_:
                pred = tree.predict(X_scaled)  # (n_samples, n_targets)
                tree_preds.append(pred)
            
            # stack: (n_trees, n_samples, n_targets)
            tree_preds = np.array(tree_preds)
            
            # compute variance across trees per sample per target
            var_y = np.var(tree_preds, axis=0)  # (n_samples, n_targets)
            
            return var_y
        else:
            # return NaN for models without uncertainty
            n_samples = X_scaled.shape[0]
            return np.full((n_samples, 3), np.nan)

    def save(self, filepath):
        """
        Save the trained model to disk.
        
        Args:
            filepath: str, path to save (e.g., 'models/WeakIDENT_rf_multi.joblib')
        """
        if not self.fitted:
            raise RuntimeError("Model must be fitted before saving.")
        
        joblib.dump(self, filepath)
        print(f"Model saved to {filepath}")

    @staticmethod
    def load(filepath):
        """
        Load a trained model from disk.
        
        Args:
            filepath: str, path to load
        
        Returns:
            PerMethodRegressor instance
        """
        model = joblib.load(filepath)
        if not isinstance(model, PerMethodRegressor):
            raise TypeError(f"Loaded object is not a PerMethodRegressor: {type(model)}")
        return model

    def get_feature_importances(self):
        """
        Get average feature importances across all outputs.
        
        Returns:
            np.ndarray of shape (12,), feature importances
        """
        if not self.fitted:
            raise RuntimeError("Model must be fitted before getting importances.")
        
        # rf_multi has native feature_importances_
        if self.model_name == "rf_multi":
            return self.model.feature_importances_
        
        # for multi-output wrappers, average across outputs
        if hasattr(self.model, "estimators_"):
            importances = []
            for estimator in self.model.estimators_:
                if hasattr(estimator, "feature_importances_"):
                    importances.append(estimator.feature_importances_)
                elif hasattr(estimator, "coef_"):
                    # linear models: use abs(coef) as importance
                    coef = np.abs(estimator.coef_)
                    if coef.ndim == 2:
                        coef = coef.mean(axis=0)
                    importances.append(coef)
            
            if importances:
                return np.mean(importances, axis=0)
        
        # fallback: return uniform importances
        return np.ones(12) / 12.0
