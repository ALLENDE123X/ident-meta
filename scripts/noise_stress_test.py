#!/usr/bin/env python
"""
Noise Stress Test: Evaluate IDENT methods under controlled noise/outlier corruption.

This script samples a subset of windows, applies controlled corruption (Gaussian noise
and/or outlier spikes), runs all 4 methods, and records which method performs best.

Goal: Demonstrate that WeakIDENT and RobustIDENT become competitive or dominant
under noisy/outlier conditions, while LASSO/STLSQ dominate on clean data.

Usage:
    python scripts/noise_stress_test.py           # Run with defaults (N=300)
    python scripts/noise_stress_test.py --n 100   # Quick test run
"""

import argparse
import os
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ident_methods import METHOD_REGISTRY

# Use same PDE configs as run_all_methods.py
PDE_CONFIGS = {
    "kdv": {
        "data_file": "data/raw/KdV.npy",
        "grid": {"dx": 0.05, "dt": 0.001},
        "windows": {"size_x": 64, "size_t": 100},
        "true_coefficients": {"u*u_x": -1.0, "u_xxx": -1.0}
    },
    "heat": {
        "data_file": "data/raw/heat.npy",
        "grid": {"dx": 0.0427, "dt": 0.002},
        "windows": {"size_x": 48, "size_t": 80},
        "true_coefficients": {"u_xx": 1.0}
    },
    "ks": {
        "data_file": "data/raw/KS.npy",
        "grid": {"dx": 0.39, "dt": 0.01},
        "windows": {"size_x": 64, "size_t": 80},
        "true_coefficients": {"u*u_x": -1.0, "u_xx": -1.0, "u_xxxx": -1.0}
    },
    "transport": {
        "data_file": "data/raw/transportDiff.npy",
        "grid": {"dx": 0.04, "dt": 0.01},
        "windows": {"size_x": 64, "size_t": 80},
        "true_coefficients": {"u_x": -1.0, "u_xx": 0.01}
    },
}

METHODS = ["LASSO", "STLSQ", "RobustIDENT", "WeakIDENT"]

# Corruption settings to test
CORRUPTION_SETTINGS = [
    {"name": "clean", "sigma": 0.0, "outlier_p": 0.0},
    {"name": "noise_2%", "sigma": 0.02, "outlier_p": 0.0},
    {"name": "noise_5%", "sigma": 0.05, "outlier_p": 0.0},
    {"name": "outlier_1%", "sigma": 0.0, "outlier_p": 0.01},
    {"name": "outlier_3%", "sigma": 0.0, "outlier_p": 0.03},
    {"name": "noise_5%_outlier_1%", "sigma": 0.05, "outlier_p": 0.01},
]


def load_pde_data(filepath: str) -> np.ndarray:
    """Load PDE data from .npy file."""
    data = np.load(filepath, allow_pickle=True)
    return np.array(data[0], dtype=np.float64)


def extract_windows_sampled(u: np.ndarray, pde_name: str, config: Dict, 
                            n_windows: int, rng: np.random.Generator) -> List[Dict]:
    """Extract random windows from PDE data."""
    w_cfg = config["windows"]
    size_x, size_t = w_cfg["size_x"], w_cfg["size_t"]
    dx, dt = config["grid"]["dx"], config["grid"]["dt"]
    
    nx, nt = u.shape
    
    if nx < size_x or nt < size_t:
        return []
    
    windows = []
    for _ in range(n_windows):
        i_x = rng.integers(0, nx - size_x)
        i_t = rng.integers(0, nt - size_t)
        window_data = u[i_x:i_x+size_x, i_t:i_t+size_t].T  # (nt, nx)
        windows.append({
            "pde_type": pde_name,
            "x_start": int(i_x),
            "t_start": int(i_t),
            "data": window_data.copy(),
            "dx": dx,
            "dt": dt,
            "true_coeffs": config.get("true_coefficients", {}),
        })
    return windows


def apply_corruption(data: np.ndarray, sigma: float, outlier_p: float, 
                     rng: np.random.Generator) -> np.ndarray:
    """Apply noise and/or outlier corruption to data."""
    corrupted = data.copy()
    data_std = np.std(data)
    
    # Additive Gaussian noise
    if sigma > 0:
        noise = rng.normal(0, sigma * data_std, data.shape)
        corrupted += noise
    
    # Outlier corruption: replace random values with large spikes
    if outlier_p > 0:
        n_outliers = int(outlier_p * data.size)
        if n_outliers > 0:
            flat = corrupted.flatten()
            indices = rng.choice(len(flat), size=n_outliers, replace=False)
            # Spike magnitude: 5-10x the standard deviation
            signs = rng.choice([-1, 1], size=n_outliers)
            magnitudes = rng.uniform(5, 10, size=n_outliers) * data_std
            flat[indices] = signs * magnitudes
            corrupted = flat.reshape(data.shape)
    
    return corrupted


def run_method_on_window(window_data: np.ndarray, dx: float, dt: float,
                         true_coeffs: Dict, method_name: str) -> Dict:
    """Run a single method on window data and return metrics."""
    method = METHOD_REGISTRY.get(method_name)
    if method is None:
        return {"e2": 1.0, "f1": 0.0, "runtime": 0.0, "error": "Method not found"}
    
    start = time.time()
    try:
        metrics, info = method.run(window_data, dx, dt, true_coeffs=true_coeffs)
        return {
            "e2": float(metrics[1]),
            "f1": float(metrics[0]),
            "runtime": time.time() - start,
        }
    except Exception as e:
        return {"e2": 1.0, "f1": 0.0, "runtime": time.time() - start, "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Noise stress test for IDENT methods")
    parser.add_argument("--n", type=int, default=300, help="Total windows to sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-csv", default="data/results/noise_stress_test_summary.csv")
    parser.add_argument("--output-fig", default="data/figures/noise_best_method_distribution.png")
    args = parser.parse_args()
    
    rng = np.random.default_rng(args.seed)
    
    print(f"Noise Stress Test")
    print(f"=================")
    print(f"Windows per PDE: {args.n // len(PDE_CONFIGS)}")
    print(f"Total windows: {args.n}")
    print(f"Corruption settings: {len(CORRUPTION_SETTINGS)}")
    print(f"Methods: {METHODS}")
    print()
    
    # Sample windows from each PDE
    windows_per_pde = args.n // len(PDE_CONFIGS)
    all_windows = []
    
    for pde_name, config in PDE_CONFIGS.items():
        print(f"Loading {pde_name}...")
        try:
            u = load_pde_data(config["data_file"])
            print(f"  Data shape: {u.shape}")
            windows = extract_windows_sampled(u, pde_name, config, windows_per_pde, rng)
            print(f"  Sampled {len(windows)} windows")
            all_windows.extend(windows)
        except Exception as e:
            print(f"  Error: {e}")
    
    print(f"\nTotal sampled windows: {len(all_windows)}")
    
    # Run experiment
    results = []
    total_runs = len(all_windows) * len(CORRUPTION_SETTINGS) * len(METHODS)
    run_count = 0
    start_time = time.time()
    
    for setting in CORRUPTION_SETTINGS:
        setting_name = setting["name"]
        sigma = setting["sigma"]
        outlier_p = setting["outlier_p"]
        
        print(f"\n--- Setting: {setting_name} (sigma={sigma}, outlier_p={outlier_p}) ---")
        
        e2_by_method = {m: [] for m in METHODS}
        best_counts = {m: 0 for m in METHODS}
        
        for i, window in enumerate(all_windows):
            # Apply corruption
            corrupted_data = apply_corruption(window["data"], sigma, outlier_p, rng)
            
            # Run all methods
            method_e2 = {}
            for method_name in METHODS:
                result = run_method_on_window(
                    corrupted_data, window["dx"], window["dt"],
                    window["true_coeffs"], method_name
                )
                method_e2[method_name] = result["e2"]
                e2_by_method[method_name].append(result["e2"])
                run_count += 1
            
            # Determine best method
            best_method = min(method_e2, key=method_e2.get)
            best_counts[best_method] += 1
            
            if (i + 1) % 50 == 0:
                elapsed = time.time() - start_time
                rate = run_count / elapsed
                eta = (total_runs - run_count) / rate if rate > 0 else 0
                print(f"  Progress: {i+1}/{len(all_windows)} windows (ETA: {eta/60:.1f}min)")
        
        # Record summary for this setting
        for method_name in METHODS:
            results.append({
                "setting": setting_name,
                "sigma": sigma,
                "outlier_p": outlier_p,
                "N": len(all_windows),
                "method": method_name,
                "best_count": best_counts[method_name],
                "best_frac": best_counts[method_name] / len(all_windows),
                "mean_e2": np.mean(e2_by_method[method_name]),
                "median_e2": np.median(e2_by_method[method_name]),
            })
        
        print(f"  Best method distribution: {best_counts}")
    
    # Save CSV
    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print(f"\nSaved results to {args.output_csv}")
    
    # Generate figure
    generate_figure(df, args.output_fig)
    
    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed/60:.1f} minutes")
    print(f"Total method runs: {run_count}")


def generate_figure(df: pd.DataFrame, output_path: str):
    """Generate grouped bar chart of best-method distribution."""
    settings = df["setting"].unique()
    methods = METHODS
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(settings))
    width = 0.2
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    
    for i, method in enumerate(methods):
        fracs = [df[(df["setting"] == s) & (df["method"] == method)]["best_frac"].values[0] 
                 for s in settings]
        offset = (i - len(methods)/2 + 0.5) * width
        bars = ax.bar(x + offset, fracs, width, label=method, color=colors[i], alpha=0.8)
        
        # Add value labels on bars > 5%
        for bar, frac in zip(bars, fracs):
            if frac >= 0.05:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{frac:.0%}', ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel("Corruption Setting", fontsize=11)
    ax.set_ylabel("Fraction of Windows Where Method is Best", fontsize=11)
    ax.set_title("Best-Method Distribution Under Different Noise/Outlier Conditions", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("_", "\n") for s in settings], fontsize=9)
    ax.legend(title="Method", loc="upper right")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved figure to {output_path}")
    plt.close()


if __name__ == "__main__":
    main()
