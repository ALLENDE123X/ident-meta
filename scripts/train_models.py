#!/usr/bin/env python
"""
Train and compare ML selector models on the PDE dataset.

Usage:
    docker compose run --rm weakident python scripts/train_models.py
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')


FEATURE_COLS = [f"feat_{i}" for i in range(12)]
TARGET_COL = "best_method"


def load_dataset(path: str = "data/results/full_dataset_4methods.csv") -> pd.DataFrame:
    """Load the dataset."""
    print(f"Loading dataset: {path}")
    df = pd.read_csv(path)
    print(f"  Loaded {len(df)} samples")
    print(f"  Classes: {df[TARGET_COL].value_counts().to_dict()}")
    return df


def prepare_data(df: pd.DataFrame):
    """Prepare features and labels."""
    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values
    
    # Handle missing/inf values
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y_encoded, le, scaler


def train_and_compare(X, y, le):
    """Train multiple models and compare."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Ridge Classifier": RidgeClassifier(random_state=42),
        "SVM (RBF)": SVC(kernel='rbf', random_state=42),
        "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
    }
    
    results = []
    best_model = None
    best_acc = 0
    
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        
        # Test accuracy
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X, y, cv=5)
        
        results.append({
            "model": name,
            "test_accuracy": acc,
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
        })
        
        print(f"  Test Accuracy: {acc:.4f}")
        print(f"  5-Fold CV: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        if acc > best_acc:
            best_acc = acc
            best_model = (name, model, y_pred)
    
    # Print results table
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    results_df = pd.DataFrame(results).sort_values("test_accuracy", ascending=False)
    print(results_df.to_string(index=False))
    
    # Detailed report for best model
    name, model, y_pred = best_model
    print(f"\n{'='*60}")
    print(f"BEST MODEL: {name}")
    print(f"{'='*60}")
    print("\nClassification Report:")
    unique_labels = np.unique(np.concatenate([y_test, y_pred]))
    target_names = [le.classes_[i] for i in unique_labels]
    print(classification_report(y_test, y_pred, labels=unique_labels, target_names=target_names))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"Classes: {le.classes_}")
    print(cm)
    
    # Feature importance (if available)
    if hasattr(model, 'feature_importances_'):
        print("\nFeature Importance:")
        importances = model.feature_importances_
        for i, imp in enumerate(importances):
            print(f"  feat_{i}: {imp:.4f}")
    
    return results_df, model, le


def compute_regret(df: pd.DataFrame, predictions: np.ndarray, le):
    """Compute regret metrics."""
    methods = ["LASSO", "STLSQ", "RobustIDENT", "WeakIDENT"]
    
    # Get e2 values for each method
    e2_cols = [f"{m}_e2" for m in methods]
    e2_matrix = df[e2_cols].values
    
    # Oracle: best e2 (min across methods)
    oracle_e2 = e2_matrix.min(axis=1)
    
    # Selector: e2 of predicted method
    pred_labels = le.inverse_transform(predictions)
    selector_e2 = np.array([
        df.iloc[i][f"{pred_labels[i]}_e2"] 
        for i in range(len(predictions))
    ])
    
    # Regret = selector_e2 - oracle_e2
    regret = selector_e2 - oracle_e2
    
    print(f"\n{'='*60}")
    print("REGRET ANALYSIS")
    print(f"{'='*60}")
    print(f"Mean Regret: {regret.mean():.4f}")
    print(f"Median Regret: {np.median(regret):.4f}")
    print(f"Max Regret: {regret.max():.4f}")
    print(f"Zero Regret (perfect): {(regret == 0).sum()} / {len(regret)} ({100*(regret == 0).mean():.1f}%)")
    
    return regret


def main():
    # Load data
    df = load_dataset()
    
    # Prepare
    X, y, le, scaler = prepare_data(df)
    print(f"\nFeatures: {X.shape[1]}")
    print(f"Classes: {len(le.classes_)} - {le.classes_}")
    
    # Train and compare
    results, best_model, le = train_and_compare(X, y, le)
    
    # Compute regret on test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    df_test = df.iloc[y_test]
    
    # For full dataset regret
    predictions = best_model.predict(X)
    regret = compute_regret(df, predictions, le)
    
    # Save results
    os.makedirs("data/results", exist_ok=True)
    results.to_csv("data/results/model_comparison.csv", index=False)
    print(f"\nResults saved to: data/results/model_comparison.csv")
    
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
