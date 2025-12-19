#!/usr/bin/env python
"""
Generate full dataset with 4 IDENT methods: LASSO, STLSQ, RobustIDENT, WeakIDENT.

Usage:
    # Test with 10 windows
    docker compose run --rm weakident python scripts/run_all_methods.py --test
    
    # Full run with 4 workers
    docker compose run --rm weakident python scripts/run_all_methods.py -j 4
"""

import argparse
import os
import sys
import time
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
import traceback

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ident_methods import METHOD_REGISTRY


# PDE configurations
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
CHECKPOINT_DIR = "data/checkpoints"
CHECKPOINT_INTERVAL = 50


def load_pde_data(filepath: str) -> np.ndarray:
    """Load PDE data from .npy file."""
    data = np.load(filepath, allow_pickle=True)
    return np.array(data[0], dtype=np.float64)


def extract_features(window_data: np.ndarray, dx: float, dt: float) -> np.ndarray:
    """Extract 12 features from a window."""
    nt, nx = window_data.shape
    features = np.zeros(12)
    
    try:
        u_x = np.gradient(window_data, dx, axis=1)
        u_xx = np.gradient(u_x, dx, axis=1)
        u_xxx = np.gradient(u_xx, dx, axis=1)
        features[0] = np.std(u_x)
        features[1] = np.std(u_xx)
        features[2] = np.std(u_xxx)
        
        u_t = np.gradient(window_data, dt, axis=0)
        u_tt = np.gradient(u_t, dt, axis=0)
        features[3] = np.std(u_t)
        features[4] = np.std(u_tt)
        features[5] = np.max(np.abs(u_t))
        
        fft_mag = np.abs(np.fft.fft2(window_data))
        features[6] = np.log1p(fft_mag[:nt//4, :nx//4].mean())
        features[7] = np.log1p(fft_mag[nt//4:nt//2, nx//4:nx//2].mean())
        features[8] = np.log1p(fft_mag[nt//2:, nx//2:].mean())
        
        features[9] = np.std(window_data)
        features[10] = np.mean(np.abs(u_xx)) / (np.std(window_data) + 1e-8)
        features[11] = np.max(window_data) - np.min(window_data)
    except:
        pass
    
    return features


def extract_windows(u: np.ndarray, pde_name: str, config: Dict, target_count: int = 2000) -> List[Dict]:
    """Extract windows from PDE data."""
    w_cfg = config["windows"]
    size_x, size_t = w_cfg["size_x"], w_cfg["size_t"]
    dx, dt = config["grid"]["dx"], config["grid"]["dt"]
    
    nx, nt = u.shape
    
    # Compute stride
    approx_stride = int(np.sqrt((nx * nt) / max(1, target_count)))
    stride_x = max(1, min(approx_stride, (nx - size_x) // 10 + 1))
    stride_t = max(1, min(approx_stride, (nt - size_t) // 10 + 1))
    
    windows = []
    idx = 0
    for i_x in range(0, max(1, nx - size_x), stride_x):
        for i_t in range(0, max(1, nt - size_t), stride_t):
            window_data = u[i_x:i_x+size_x, i_t:i_t+size_t].T  # (nt, nx)
            if window_data.shape == (size_t, size_x):
                windows.append({
                    "window_id": f"{pde_name}_{idx:05d}",
                    "pde_type": pde_name,
                    "x_start": i_x,
                    "t_start": i_t,
                    "data": window_data,
                    "dx": dx,
                    "dt": dt,
                    "true_coeffs": config.get("true_coefficients", {}),
                })
                idx += 1
                if idx >= target_count:
                    return windows
    return windows


def process_window(window: Dict, methods: List[str]) -> Dict[str, Any]:
    """Process a single window with all methods."""
    result = {
        "window_id": window["window_id"],
        "pde_type": window["pde_type"],
        "window_x_start": window["x_start"],
        "window_t_start": window["t_start"],
    }
    
    # Extract features
    features = extract_features(window["data"], window["dx"], window["dt"])
    for i, f in enumerate(features):
        result[f"feat_{i}"] = float(f)
    
    # Run each method
    best_e2 = float("inf")
    best_method = None
    
    for method_name in methods:
        method = METHOD_REGISTRY.get(method_name)
        if method is None:
            result[f"{method_name}_f1"] = 0.0
            result[f"{method_name}_e2"] = 1.0
            result[f"{method_name}_residual"] = 1.0
            result[f"{method_name}_runtime"] = 0.0
            continue
        
        start = time.time()
        try:
            metrics, info = method.run(
                window["data"], window["dx"], window["dt"],
                true_coeffs=window["true_coeffs"]
            )
            result[f"{method_name}_f1"] = float(metrics[0])
            result[f"{method_name}_e2"] = float(metrics[1])
            result[f"{method_name}_residual"] = float(metrics[2])
            result[f"{method_name}_runtime"] = time.time() - start
            
            if metrics[1] < best_e2:
                best_e2 = metrics[1]
                best_method = method_name
        except Exception as e:
            result[f"{method_name}_f1"] = 0.0
            result[f"{method_name}_e2"] = 1.0
            result[f"{method_name}_residual"] = 1.0
            result[f"{method_name}_runtime"] = time.time() - start
            result[f"{method_name}_error"] = str(e)
    
    result["best_method"] = best_method
    result["oracle_e2"] = best_e2
    
    return result


def load_checkpoint(path: str) -> Tuple[set, List[Dict]]:
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        return set(data["completed_ids"]), data["results"]
    return set(), []


def save_checkpoint(path: str, completed_ids: set, results: List[Dict]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({"completed_ids": list(completed_ids), "results": results}, f)


def main():
    parser = argparse.ArgumentParser(description="Run all IDENT methods")
    parser.add_argument("--output", default="data/results/full_dataset_4methods.csv")
    parser.add_argument("-j", "--jobs", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--windows-per-pde", type=int, default=2000)
    args = parser.parse_args()
    
    print(f"Methods: {METHODS}")
    print(f"PDEs: {list(PDE_CONFIGS.keys())}")
    
    # Extract all windows
    print("\nExtracting windows...")
    all_windows = []
    for pde_name, config in PDE_CONFIGS.items():
        print(f"  Loading {pde_name}...")
        u = load_pde_data(config["data_file"])
        print(f"    Data shape: {u.shape}")
        windows = extract_windows(u, pde_name, config, args.windows_per_pde)
        print(f"    Extracted {len(windows)} windows")
        all_windows.extend(windows)
    
    print(f"\nTotal windows: {len(all_windows)}")
    
    if args.test:
        all_windows = all_windows[:10]
        print(f"Test mode: using first 10 windows")
    
    # Checkpoint
    checkpoint_path = os.path.join(CHECKPOINT_DIR, "all_methods_checkpoint.json")
    if args.resume:
        completed_ids, results = load_checkpoint(checkpoint_path)
        print(f"Resuming: {len(completed_ids)} completed")
    else:
        completed_ids, results = set(), []
    
    # Filter
    windows_to_process = [w for w in all_windows if w["window_id"] not in completed_ids]
    print(f"Windows to process: {len(windows_to_process)}")
    
    if len(windows_to_process) == 0:
        print("All done!")
        return
    
    # Process
    start_time = time.time()
    
    if args.jobs == 1:
        for i, window in enumerate(windows_to_process):
            if i % 20 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                eta = (len(windows_to_process) - i) / rate if rate > 0 else 0
                print(f"Progress: {i+1}/{len(windows_to_process)} ({rate:.2f}/s, ETA: {eta/60:.1f}min)")
            
            result = process_window(window, METHODS)
            results.append(result)
            completed_ids.add(window["window_id"])
            
            if (i + 1) % CHECKPOINT_INTERVAL == 0:
                save_checkpoint(checkpoint_path, completed_ids, results)
    else:
        from joblib import Parallel, delayed
        
        batch_size = CHECKPOINT_INTERVAL
        for batch_start in range(0, len(windows_to_process), batch_size):
            batch = windows_to_process[batch_start:batch_start + batch_size]
            elapsed = time.time() - start_time
            eta = (len(windows_to_process) - batch_start) / (batch_start / elapsed) if batch_start > 0 else 0
            print(f"Batch {batch_start//batch_size + 1}: {batch_start+1}-{batch_start+len(batch)}/{len(windows_to_process)} (ETA: {eta/60:.1f}min)")
            
            batch_results = Parallel(n_jobs=args.jobs, verbose=0)(
                delayed(process_window)(w, METHODS) for w in batch
            )
            
            results.extend(batch_results)
            for r in batch_results:
                completed_ids.add(r["window_id"])
            
            save_checkpoint(checkpoint_path, completed_ids, results)
    
    # Save
    print(f"\nSaving {len(results)} results to {args.output}")
    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, index=False)
    
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print("COMPLETE")
    print(f"{'='*60}")
    print(f"Total: {len(results)} windows")
    print(f"Time: {elapsed/60:.1f} minutes")
    print(f"\nBest method distribution:")
    print(df["best_method"].value_counts())
    print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()
