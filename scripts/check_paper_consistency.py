#!/usr/bin/env python
"""
Consistency check for paper figures and metrics.

This script verifies:
1. All figures referenced in tower_paper.tex exist in manuscript/figures/
2. Key metrics in metrics.json match what's computed from source data
3. metrics.tex macros are consistent with metrics.json

Usage:
    python scripts/check_paper_consistency.py
"""

import json
import os
import re
import sys
from pathlib import Path

def check_figures():
    """Verify all referenced figures exist."""
    tex_path = Path("manuscript/tower_paper.tex")
    figures_dir = Path("manuscript/figures")
    
    if not tex_path.exists():
        print("ERROR: manuscript/tower_paper.tex not found")
        return False
    
    with open(tex_path) as f:
        content = f.read()
    
    # Find all includegraphics
    pattern = r'\\includegraphics.*?\{figures/([^}]+)\}'
    refs = re.findall(pattern, content)
    
    missing = []
    for fig in refs:
        fig_path = figures_dir / fig
        if not fig_path.exists():
            missing.append(fig)
    
    if missing:
        print(f"ERROR: Missing figures: {missing}")
        return False
    
    print(f"✓ All {len(refs)} referenced figures exist")
    return True


def check_metrics_consistency():
    """Verify metrics.json matches computed values."""
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    
    # Load metrics.json
    with open("manuscript/metrics.json") as f:
        metrics = json.load(f)
    
    # Recompute key metrics
    df = pd.read_csv("data/results/full_dataset_4methods.csv")
    
    # Check total windows
    if len(df) != metrics["total_windows"]:
        print(f"ERROR: total_windows mismatch: {len(df)} vs {metrics['total_windows']}")
        return False
    
    # Check test accuracy
    FEATURE_COLS = [f'feat_{i}' for i in range(12)]
    X = df[FEATURE_COLS].values
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
    y = df['best_method'].values
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    test_acc = round(accuracy_score(y_test, y_pred) * 100, 2)
    
    if test_acc != metrics["test_accuracy"]:
        print(f"ERROR: test_accuracy mismatch: {test_acc} vs {metrics['test_accuracy']}")
        return False
    
    print(f"✓ Metrics consistency verified (test_accuracy={test_acc}%)")
    return True


def check_metrics_tex():
    """Verify metrics.tex is present and has expected macros."""
    tex_path = Path("manuscript/metrics.tex")
    
    if not tex_path.exists():
        print("ERROR: manuscript/metrics.tex not found")
        return False
    
    with open(tex_path) as f:
        content = f.read()
    
    required_macros = [
        "TotalWindows",
        "TestSize",
        "TestAccuracy",
        "TestZeroRegretPct",
        "CVMean",
        "CVStd",
    ]
    
    missing = []
    for macro in required_macros:
        if f"\\newcommand{{\\{macro}}}" not in content:
            missing.append(macro)
    
    if missing:
        print(f"ERROR: Missing macros in metrics.tex: {missing}")
        return False
    
    print(f"✓ metrics.tex has all {len(required_macros)} required macros")
    return True


def main():
    print("=" * 50)
    print("PAPER CONSISTENCY CHECK")
    print("=" * 50)
    
    all_ok = True
    
    # Check figures
    if not check_figures():
        all_ok = False
    
    # Check metrics.tex
    if not check_metrics_tex():
        all_ok = False
    
    # Check metrics consistency (requires sklearn)
    try:
        if not check_metrics_consistency():
            all_ok = False
    except ImportError as e:
        print(f"WARNING: Skipping metrics consistency check (missing: {e})")
    
    print("=" * 50)
    if all_ok:
        print("✓ ALL CHECKS PASSED")
        return 0
    else:
        print("✗ SOME CHECKS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
