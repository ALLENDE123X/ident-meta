#!/usr/bin/env python
"""
Merge WeakIDENT results with the fast methods dataset.

Usage:
    python scripts/merge_weakident.py
"""

import pandas as pd
import os


def main():
    fast_path = "data/results/full_dataset.csv"
    weak_path = "data/results/weakident_results.csv"
    output_path = "data/results/combined_dataset.csv"
    
    print(f"Loading fast methods: {fast_path}")
    df_fast = pd.read_csv(fast_path)
    print(f"  Loaded {len(df_fast)} rows, columns: {list(df_fast.columns)}")
    
    print(f"\nLoading WeakIDENT: {weak_path}")
    df_weak = pd.read_csv(weak_path)
    print(f"  Loaded {len(df_weak)} rows, columns: {list(df_weak.columns)}")
    
    # Merge on window_id
    print("\nMerging on window_id...")
    df_combined = df_fast.merge(df_weak, on="window_id", how="inner")
    print(f"  Merged: {len(df_combined)} rows")
    
    # Recalculate best_method including WeakIDENT
    print("\nRecalculating best_method including WeakIDENT...")
    method_cols = {
        "PySINDy": "PySINDy_e2",
        "RobustIDENT": "RobustIDENT_e2",
        "WSINDy": "WSINDy_e2",
        "WeakIDENT": "WeakIDENT_e2",
    }
    
    def get_best_method(row):
        best_e2 = float("inf")
        best_method = None
        for method, col in method_cols.items():
            if col in row and pd.notna(row[col]):
                if row[col] < best_e2:
                    best_e2 = row[col]
                    best_method = method
        return best_method, best_e2
    
    results = df_combined.apply(get_best_method, axis=1)
    df_combined["best_method"] = [r[0] for r in results]
    df_combined["oracle_e2"] = [r[1] for r in results]
    
    # Save
    print(f"\nSaving combined dataset: {output_path}")
    df_combined.to_csv(output_path, index=False)
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total samples: {len(df_combined)}")
    print(f"Columns: {len(df_combined.columns)}")
    print(f"\nBest method distribution:")
    print(df_combined["best_method"].value_counts())
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
