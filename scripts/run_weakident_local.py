#!/usr/bin/env python
"""
Run WeakIDENT locally on existing dataset windows.

This script:
1. Reads the full_dataset.csv to get window positions
2. Loads PDE data and extracts those specific windows  
3. Runs WeakIDENT on each window (parallelized)
4. Saves checkpoints every N windows
5. Merges results into the existing dataset

Usage:
    # Test with 10 windows first
    docker compose run --rm weakident python scripts/run_weakident_local.py --test
    
    # Run full with 4 parallel workers
    docker compose run --rm weakident python scripts/run_weakident_local.py -j 4
    
    # Resume from checkpoint
    docker compose run --rm weakident python scripts/run_weakident_local.py -j 4 --resume
"""

import argparse
import os
import sys
import time
import json
import hashlib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import traceback

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class WindowSpec:
    """Specification for a window to process."""
    window_id: str
    pde_type: str
    x_start: int
    t_start: int


# PDE configurations (matching the Colab script)
PDE_CONFIGS = {
    "kdv": {
        "data_file": "data/raw/KdV.npy",
        "data_format": "nested_object",
        "grid": {"dx": 0.05, "dt": 0.001},
        "windows": {"size_x": 64, "size_t": 100},
        "true_coefficients": {"u*u_x": -1.0, "u_xxx": -1.0}
    },
    "heat": {
        "data_file": "data/raw/heat.npy",
        "data_format": "nested_object",
        "grid": {"dx": 0.0427, "dt": 0.002},
        "windows": {"size_x": 48, "size_t": 80},
        "true_coefficients": {"u_xx": 1.0}
    },
    "ks": {
        "data_file": "data/raw/KS.npy",
        "data_format": "nested_object",
        "grid": {"dx": 0.39, "dt": 0.01},
        "windows": {"size_x": 64, "size_t": 80},
        "true_coefficients": {"u*u_x": -1.0, "u_xx": -1.0, "u_xxxx": -1.0}
    },
    "transport": {
        "data_file": "data/raw/transportDiff.npy",
        "data_format": "nested_object",
        "grid": {"dx": 0.04, "dt": 0.01},
        "windows": {"size_x": 64, "size_t": 80},
        "true_coefficients": {"u_x": -1.0, "u_xx": 0.01}
    },
}

# Checkpoint configuration
CHECKPOINT_DIR = "data/checkpoints"
CHECKPOINT_INTERVAL = 50  # Save every N windows


def load_pde_data(filepath: str, data_format: str = "nested_object") -> np.ndarray:
    """Load PDE data from .npy file."""
    data = np.load(filepath, allow_pickle=True)
    
    if data_format == "nested_object":
        return np.array(data[0], dtype=np.float64)
    elif data_format == "object_array":
        u = np.array(data.tolist(), dtype=np.float64)
        return u[0] if u.ndim == 3 and u.shape[0] == 1 else u
    return data


def extract_window(u: np.ndarray, x_start: int, t_start: int, 
                   size_x: int, size_t: int) -> np.ndarray:
    """Extract a specific window from the data."""
    window_data = u[x_start:x_start+size_x, t_start:t_start+size_t].T  # (nt, nx)
    return window_data


def run_weakident_on_window(
    window_data: np.ndarray,
    dx: float,
    dt: float,
    true_coeffs: Dict[str, float],
) -> Dict[str, Any]:
    """Run WeakIDENT on a single window."""
    from src.ident_methods import METHOD_REGISTRY
    
    method = METHOD_REGISTRY.get("WeakIDENT")
    if method is None:
        return {
            "WeakIDENT_f1": 0.0,
            "WeakIDENT_e2": 1.0,
            "WeakIDENT_residual": 1.0,
            "WeakIDENT_runtime": 0.0,
            "WeakIDENT_error": "Method not found",
        }
    
    start = time.time()
    try:
        metrics, info = method.run(window_data, dx, dt, true_coeffs=true_coeffs)
        return {
            "WeakIDENT_f1": float(metrics[0]),
            "WeakIDENT_e2": float(metrics[1]),
            "WeakIDENT_residual": float(metrics[2]),
            "WeakIDENT_runtime": time.time() - start,
        }
    except Exception as e:
        return {
            "WeakIDENT_f1": 0.0,
            "WeakIDENT_e2": 1.0,
            "WeakIDENT_residual": 1.0,
            "WeakIDENT_runtime": time.time() - start,
            "WeakIDENT_error": str(e),
        }


def process_single_window(args: Tuple) -> Dict[str, Any]:
    """Process a single window (for parallel execution)."""
    window_id, pde_type, x_start, t_start, pde_data_cache = args
    
    config = PDE_CONFIGS.get(pde_type)
    if config is None:
        return {"window_id": window_id, "WeakIDENT_error": f"Unknown PDE: {pde_type}"}
    
    # Load data if not cached
    if pde_type not in pde_data_cache:
        pde_data_cache[pde_type] = load_pde_data(
            config["data_file"], config.get("data_format", "nested_object")
        )
    
    u = pde_data_cache[pde_type]
    w_cfg = config["windows"]
    
    # Extract window
    window_data = extract_window(u, x_start, t_start, w_cfg["size_x"], w_cfg["size_t"])
    
    # Run WeakIDENT
    result = run_weakident_on_window(
        window_data,
        config["grid"]["dx"],
        config["grid"]["dt"],
        config.get("true_coefficients", {}),
    )
    result["window_id"] = window_id
    
    return result


def load_checkpoint(checkpoint_path: str) -> Tuple[set, List[Dict]]:
    """Load checkpoint if exists."""
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path) as f:
            data = json.load(f)
        return set(data["completed_ids"]), data["results"]
    return set(), []


def save_checkpoint(checkpoint_path: str, completed_ids: set, results: List[Dict]):
    """Save checkpoint."""
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    with open(checkpoint_path, "w") as f:
        json.dump({
            "completed_ids": list(completed_ids),
            "results": results,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, f)


def main():
    parser = argparse.ArgumentParser(description="Run WeakIDENT locally on dataset")
    parser.add_argument(
        "--input", type=str, default="data/results/full_dataset.csv",
        help="Input CSV with window specifications"
    )
    parser.add_argument(
        "--output", type=str, default="data/results/weakident_results.csv",
        help="Output CSV with WeakIDENT results"
    )
    parser.add_argument(
        "-j", "--jobs", type=int, default=1,
        help="Number of parallel jobs (use 1 for debugging)"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from checkpoint"
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Run on first 10 windows only (for testing)"
    )
    parser.add_argument(
        "--checkpoint-interval", type=int, default=CHECKPOINT_INTERVAL,
        help="Save checkpoint every N windows"
    )
    
    args = parser.parse_args()
    
    # Load input dataset
    print(f"Loading input dataset: {args.input}")
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)
    
    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} windows")
    
    # Filter to valid PDEs
    valid_pdes = set(PDE_CONFIGS.keys())
    df = df[df["pde_type"].isin(valid_pdes)]
    print(f"Filtered to {len(df)} windows with valid PDE types: {valid_pdes}")
    
    # Test mode
    if args.test:
        df = df.head(10)
        print(f"Test mode: processing first {len(df)} windows only")
    
    # Checkpoint setup
    checkpoint_path = os.path.join(CHECKPOINT_DIR, "weakident_checkpoint.json")
    
    # Load checkpoint if resuming
    if args.resume:
        completed_ids, existing_results = load_checkpoint(checkpoint_path)
        print(f"Resuming from checkpoint: {len(completed_ids)} already completed")
    else:
        completed_ids, existing_results = set(), []
    
    # Filter to uncompleted windows
    windows_to_process = [
        (row["window_id"], row["pde_type"], row["window_x_start"], row["window_t_start"])
        for _, row in df.iterrows()
        if row["window_id"] not in completed_ids
    ]
    print(f"Windows to process: {len(windows_to_process)}")
    
    if len(windows_to_process) == 0:
        print("All windows already processed!")
        return
    
    # Pre-load PDE data to share across workers
    print("Pre-loading PDE data...")
    pde_data_cache = {}
    for pde_type in df["pde_type"].unique():
        if pde_type in PDE_CONFIGS:
            config = PDE_CONFIGS[pde_type]
            pde_data_cache[pde_type] = load_pde_data(
                config["data_file"], config.get("data_format", "nested_object")
            )
            print(f"  Loaded {pde_type}: {pde_data_cache[pde_type].shape}")
    
    # Process windows
    results = existing_results.copy()
    start_time = time.time()
    
    if args.jobs == 1:
        # Sequential processing (easier to debug)
        for i, (window_id, pde_type, x_start, t_start) in enumerate(windows_to_process):
            if i % 10 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                eta = (len(windows_to_process) - i) / rate if rate > 0 else 0
                print(f"Progress: {i+1}/{len(windows_to_process)} "
                      f"({rate:.2f}/s, ETA: {eta/60:.1f}min)")
            
            result = process_single_window(
                (window_id, pde_type, x_start, t_start, pde_data_cache)
            )
            results.append(result)
            completed_ids.add(window_id)
            
            # Save checkpoint
            if (i + 1) % args.checkpoint_interval == 0:
                save_checkpoint(checkpoint_path, completed_ids, results)
                print(f"  Checkpoint saved ({len(completed_ids)} completed)")
    else:
        # Parallel processing with joblib
        from joblib import Parallel, delayed
        
        def process_with_cache(window_id, pde_type, x_start, t_start):
            return process_single_window(
                (window_id, pde_type, x_start, t_start, pde_data_cache)
            )
        
        # Process in batches for checkpointing
        batch_size = args.checkpoint_interval
        for batch_start in range(0, len(windows_to_process), batch_size):
            batch = windows_to_process[batch_start:batch_start + batch_size]
            
            elapsed = time.time() - start_time
            completed = batch_start
            rate = completed / elapsed if elapsed > 0 else 0
            eta = (len(windows_to_process) - completed) / rate if rate > 0 else 0
            print(f"Processing batch {batch_start//batch_size + 1}: "
                  f"{batch_start+1}-{batch_start+len(batch)}/{len(windows_to_process)} "
                  f"(ETA: {eta/60:.1f}min)")
            
            batch_results = Parallel(n_jobs=args.jobs, verbose=0)(
                delayed(process_with_cache)(wid, ptype, xs, ts) 
                for wid, ptype, xs, ts in batch
            )
            
            results.extend(batch_results)
            for r in batch_results:
                completed_ids.add(r["window_id"])
            
            # Save checkpoint
            save_checkpoint(checkpoint_path, completed_ids, results)
            print(f"  Checkpoint saved ({len(completed_ids)} completed)")
    
    # Save final results
    print(f"\nSaving {len(results)} results to {args.output}")
    results_df = pd.DataFrame(results)
    results_df.to_csv(args.output, index=False)
    
    # Summary
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"COMPLETE")
    print(f"{'='*60}")
    print(f"Total windows: {len(results)}")
    print(f"Total time: {elapsed/60:.1f} minutes ({elapsed/len(results):.2f}s per window)")
    
    # Error summary
    if "WeakIDENT_error" in results_df.columns:
        errors = results_df["WeakIDENT_error"].notna().sum()
        print(f"Errors: {errors} ({100*errors/len(results):.1f}%)")
    
    print(f"\nResults saved to: {args.output}")
    print(f"\nTo merge with full dataset, run:")
    print(f"  python scripts/merge_weakident.py")


if __name__ == "__main__":
    main()
