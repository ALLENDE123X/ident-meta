#!/usr/bin/env python
"""
Generate paper-ready visualizations for the PDE Selector.

Usage:
    docker compose run --rm weakident python scripts/generate_figures.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

FEATURE_COLS = [f"feat_{i}" for i in range(12)]
FEATURE_NAMES = [
    "u_x std", "u_xx std", "u_xxx std", 
    "u_t std", "u_tt std", "u_t max",
    "FFT low", "FFT mid", "FFT high",
    "u std", "nonlin ratio", "u range"
]
TARGET_COL = "best_method"
OUTPUT_DIR = "data/figures"


def load_and_prepare(path="data/results/full_dataset_4methods.csv"):
    """Load dataset and prepare for visualization."""
    df = pd.read_csv(path)
    
    X = df[FEATURE_COLS].values
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
    
    le = LabelEncoder()
    y = le.fit_transform(df[TARGET_COL].values)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return df, X_scaled, y, le


def train_model(X, y):
    """Train Random Forest model."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, X_train, X_test, y_train, y_test, y_pred


def plot_confusion_matrix(y_test, y_pred, le, output_path):
    """Generate confusion matrix figure."""
    # Only use labels that appear in the data
    unique_labels = np.unique(np.concatenate([y_test, y_pred]))
    class_names = [le.classes_[i] for i in unique_labels]
    
    cm = confusion_matrix(y_test, y_pred, labels=unique_labels)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)
    
    # Set ticks
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names, yticklabels=class_names)
    
    # Rotate tick labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    ax.set_xlabel('Predicted Method', fontsize=12)
    ax.set_ylabel('True Best Method', fontsize=12)
    ax.set_title('PDE Method Selector - Confusion Matrix', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_feature_importance(model, output_path):
    """Generate feature importance bar chart."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.Blues(np.linspace(0.4, 0.8, len(importances)))
    ax.barh(range(len(importances)), importances[indices[::-1]], 
            color=colors, edgecolor='black', linewidth=0.5)
    ax.set_yticks(range(len(importances)))
    ax.set_yticklabels([FEATURE_NAMES[i] for i in indices[::-1]])
    ax.set_xlabel('Feature Importance', fontsize=12)
    ax.set_title('Random Forest Feature Importance', fontsize=14)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_regret_cdf(df, model, X, le, output_path):
    """Generate regret CDF plot."""
    methods = ["LASSO", "STLSQ", "RobustIDENT", "WeakIDENT"]
    e2_cols = [f"{m}_e2" for m in methods]
    e2_matrix = df[e2_cols].values
    
    oracle_e2 = e2_matrix.min(axis=1)
    
    predictions = model.predict(X)
    pred_labels = le.inverse_transform(predictions)
    selector_e2 = np.array([
        df.iloc[i][f"{pred_labels[i]}_e2"] 
        for i in range(len(predictions))
    ])
    
    regret = selector_e2 - oracle_e2
    regret_sorted = np.sort(regret)
    cdf = np.arange(1, len(regret_sorted) + 1) / len(regret_sorted)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(regret_sorted, cdf, linewidth=2, color='#2E86AB')
    ax.axvline(x=0, color='green', linestyle='--', label='Zero Regret', alpha=0.7)
    ax.fill_between(regret_sorted, 0, cdf, alpha=0.2, color='#2E86AB')
    
    zero_regret_pct = (regret == 0).mean() * 100
    ax.axhline(y=zero_regret_pct/100, color='orange', linestyle=':', 
               label=f'Zero Regret: {zero_regret_pct:.1f}%', alpha=0.7)
    
    ax.set_xlabel('Regret (Selector e2 - Oracle e2)', fontsize=12)
    ax.set_ylabel('Cumulative Probability', fontsize=12)
    ax.set_title('Regret CDF - PDE Method Selector', fontsize=14)
    ax.set_xlim(-0.01, min(0.5, regret_sorted.max() * 1.1))
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_model_comparison(output_path):
    """Generate model comparison bar chart."""
    results = pd.read_csv("data/results/model_comparison.csv")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x = range(len(results))
    width = 0.35
    
    bars1 = ax.bar([i - width/2 for i in x], results['test_accuracy'], width, 
                   label='Test Accuracy', color='#2E86AB', edgecolor='black')
    bars2 = ax.bar([i + width/2 for i in x], results['cv_mean'], width,
                   label='5-Fold CV Mean', color='#A23B72', edgecolor='black')
    
    ax.errorbar([i + width/2 for i in x], results['cv_mean'], 
                yerr=results['cv_std'], fmt='none', color='black', capsize=3)
    
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('ML Model Comparison for PDE Method Selection', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(results['model'], rotation=30, ha='right')
    ax.legend()
    ax.set_ylim(0.7, 1.0)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_best_method_distribution(df, output_path):
    """Generate pie chart of best method distribution."""
    counts = df[TARGET_COL].value_counts()
    
    fig, ax = plt.subplots(figsize=(8, 8))
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    explode = [0.02] * len(counts)
    
    ax.pie(counts.values, labels=counts.index, autopct='%1.1f%%',
           colors=colors[:len(counts)], explode=explode[:len(counts)],
           startangle=90, shadow=True)
    ax.set_title('Best Method Distribution Across Windows', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Loading data...")
    df, X, y, le = load_and_prepare()
    
    print("Training model...")
    model, X_train, X_test, y_train, y_test, y_pred = train_model(X, y)
    
    print("\nGenerating figures...")
    
    # 1. Confusion Matrix
    plot_confusion_matrix(y_test, y_pred, le, 
                          f"{OUTPUT_DIR}/confusion_matrix.png")
    
    # 2. Feature Importance
    plot_feature_importance(model, 
                            f"{OUTPUT_DIR}/feature_importance.png")
    
    # 3. Regret CDF
    plot_regret_cdf(df, model, X, le, 
                    f"{OUTPUT_DIR}/regret_cdf.png")
    
    # 4. Model Comparison
    plot_model_comparison(f"{OUTPUT_DIR}/model_comparison.png")
    
    # 5. Best Method Distribution
    plot_best_method_distribution(df, 
                                   f"{OUTPUT_DIR}/method_distribution.png")
    
    print(f"\n{'='*60}")
    print("ALL FIGURES GENERATED")
    print(f"{'='*60}")
    print(f"Output directory: {OUTPUT_DIR}/")
    print("\nFigures created:")
    for f in os.listdir(OUTPUT_DIR):
        print(f"  - {f}")


if __name__ == "__main__":
    main()
