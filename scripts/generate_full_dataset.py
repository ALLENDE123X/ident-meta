#!/usr/bin/env python
"""
Generate Full Dataset: Run all IDENT methods on PDE windows.

This script:
1. Loads each PDE's .npy data
2. Extracts overlapping windows
3. Runs 4 IDENT methods on each window
4. Extracts Tiny-12 features
5. Saves results to CSV

Usage:
    python scripts/generate_full_dataset.py --pdes burgers kdv heat ks
    python scripts/generate_full_dataset.py --pdes burgers --windows 100  # Quick test
"""

import argparse
import os
import sys
import time
import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import traceback

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.window_extractor import extract_windows, load_pde_data, Window
from src.ident_methods import METHOD_REGISTRY


def load_pde_config(pde_name: str, config_dir: str = "configs/pdes") -> Dict:
    """Load PDE configuration from YAML file."""
    config_path = os.path.join(config_dir, f"{pde_name}.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    
    with open(config_path) as f:
        return yaml.safe_load(f)


def extract_features(window_data: np.ndarray, dx: float, dt: float) -> np.ndarray:
    """
    Extract Tiny-12 features from a window.
    
    Features:
        0-2: Derivative statistics (u_x, u_xx, u_xxx magnitudes)
        3-5: Temporal derivatives
        6-8: Spectral content (low, mid, high frequency)
        9-11: Noise/smoothness indicators
    """
    nt, nx = window_data.shape
    features = np.zeros(12)
    
    try:
        # Spatial derivatives
        u_x = np.gradient(window_data, dx, axis=1)
        u_xx = np.gradient(u_x, dx, axis=1)
        u_xxx = np.gradient(u_xx, dx, axis=1)
        
        features[0] = np.std(u_x)
        features[1] = np.std(u_xx)
        features[2] = np.std(u_xxx)
        
        # Temporal derivatives
        u_t = np.gradient(window_data, dt, axis=0)
        u_tt = np.gradient(u_t, dt, axis=0)
        
        features[3] = np.std(u_t)
        features[4] = np.std(u_tt)
        features[5] = np.max(np.abs(u_t))
        
        # Spectral content
        fft = np.fft.fft2(window_data)
        fft_mag = np.abs(fft)
        
        # Divide into frequency bands
        low_freq = fft_mag[:nt//4, :nx//4].mean()
        mid_freq = fft_mag[nt//4:nt//2, nx//4:nx//2].mean()
        high_freq = fft_mag[nt//2:, nx//2:].mean()
        
        features[6] = np.log1p(low_freq)
        features[7] = np.log1p(mid_freq)
        features[8] = np.log1p(high_freq)
        
        # Noise/smoothness
        features[9] = np.std(window_data)
        features[10] = np.mean(np.abs(u_xx)) / (np.std(window_data) + 1e-8)
        features[11] = np.max(window_data) - np.min(window_data)
        
    except Exception as e:
        print(f"Feature extraction error: {e}")
        features = np.zeros(12)
    
    return features


def run_ident_method(
    method_name: str,
    window_data: np.ndarray,
    dx: float,
    dt: float,
    true_coeffs: Optional[Dict] = None,
    ident_params: Optional[Dict] = None,
) -> Dict[str, float]:
    """
    Run a single IDENT method on a window.
    
    Returns dict with f1, e2, residual, runtime.
    """
    method = METHOD_REGISTRY.get(method_name)
    if method is None:
        return {
            "f1": 0.0,
            "e2": 1.0,
            "residual": 1.0,
            "runtime": 0.0,
            "error": f"Method {method_name} not found",
        }
    
    params = ident_params or {}
    params["true_coeffs"] = true_coeffs
    
    try:
        metrics, info = method.run(window_data, dx, dt, **params)
        return {
            "f1": float(metrics[0]),
            "e2": float(metrics[1]),
            "residual": float(metrics[2]),
            "runtime": float(info.get("runtime", 0.0)),
        }
    except Exception as e:
        return {
            "f1": 0.0,
            "e2": 1.0,
            "residual": 1.0,
            "runtime": 0.0,
            "error": str(e),
        }


def process_window(
    window: Window,
    methods: List[str],
    true_coeffs: Dict,
    ident_params: Dict,
) -> Dict[str, Any]:
    """
    Process a single window: extract features and run all IDENT methods.
    """
    result = {
        "window_id": window.window_id,
        "pde_type": window.metadata["pde_name"],
        "window_x_start": window.x_start,
        "window_t_start": window.t_start,
    }
    
    dx = window.metadata["dx"]
    dt = window.metadata["dt"]
    
    # Extract features
    features = extract_features(window.data, dx, dt)
    for i, f in enumerate(features):
        result[f"feat_{i}"] = float(f)
    
    # Run all IDENT methods
    best_e2 = float("inf")
    best_method = None
    
    for method_name in methods:
        method_result = run_ident_method(
            method_name, window.data, dx, dt, true_coeffs, ident_params
        )
        
        for key, val in method_result.items():
            result[f"{method_name}_{key}"] = val
        
        if method_result["e2"] < best_e2:
            best_e2 = method_result["e2"]
            best_method = method_name
    
    result["best_method"] = best_method
    result["oracle_e2"] = best_e2
    
    return result


def process_pde(
    pde_name: str,
    config_dir: str,
    output_dir: str,
    methods: List[str],
    max_windows: Optional[int] = None,
    n_jobs: int = 1,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Process a single PDE: extract windows and run all methods.
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Processing: {pde_name.upper()}")
        print(f"{'='*60}")
    
    # Load config
    config = load_pde_config(pde_name, config_dir)
    
    # Load data
    data_path = config["data_file"]
    if verbose:
        print(f"Loading data from {data_path}...")
    u = load_pde_data(data_path, config.get("data_format", "nested_object"))
    if verbose:
        print(f"Data shape: {u.shape}")
    
    # Extract windows
    window_cfg = config["windows"]
    target = max_windows or window_cfg.get("target_count", 2500)
    
    if verbose:
        print(f"Extracting {target} windows...")
    
    windows = extract_windows(
        u=u,
        window_size=(window_cfg["size_x"], window_cfg["size_t"]),
        stride=(window_cfg["stride_x"], window_cfg["stride_t"]),
        pde_name=pde_name,
        target_count=target,
        dx=config["grid"]["dx"],
        dt=config["grid"]["dt"],
    )
    
    if verbose:
        print(f"Extracted {len(windows)} windows")
    
    # Process windows
    true_coeffs = config.get("true_coefficients", {})
    ident_params = config.get("ident_params", {})
    
    results = []
    start_time = time.time()
    
    if n_jobs == 1:
        # Sequential processing
        for i, window in enumerate(windows):
            if verbose and i % 100 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                eta = (len(windows) - i) / rate if rate > 0 else 0
                print(f"Progress: {i+1}/{len(windows)} ({rate:.1f}/s, ETA: {eta:.0f}s)")
            
            result = process_window(window, methods, true_coeffs, ident_params)
            results.append(result)
    else:
        # Parallel processing
        from joblib import Parallel, delayed
        
        def process_one(window):
            return process_window(window, methods, true_coeffs, ident_params)
        
        results = Parallel(n_jobs=n_jobs, verbose=10 if verbose else 0)(
            delayed(process_one)(w) for w in windows
        )
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Save to CSV
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{pde_name}_results.csv")
    df.to_csv(output_path, index=False)
    
    if verbose:
        elapsed = time.time() - start_time
        print(f"Saved {len(df)} samples to {output_path}")
        print(f"Time: {elapsed:.1f}s ({elapsed/len(df):.2f}s per window)")
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Generate full dataset from PDE windows"
    )
    parser.add_argument(
        "--pdes", nargs="+", default=["burgers", "kdv", "heat", "ks"],
        help="PDEs to process"
    )
    parser.add_argument(
        "--config-dir", type=str, default="configs/pdes",
        help="Directory with PDE config files"
    )
    parser.add_argument(
        "--output", type=str, default="data/results",
        help="Output directory for CSV files"
    )
    parser.add_argument(
        "--windows", type=int, default=None,
        help="Number of windows per PDE (default: from config)"
    )
    parser.add_argument(
        "--methods", nargs="+", 
        default=["WeakIDENT", "PySINDy", "WSINDy", "RobustIDENT"],
        help="IDENT methods to run"
    )
    parser.add_argument(
        "-j", "--jobs", type=int, default=1,
        help="Number of parallel jobs"
    )
    parser.add_argument(
        "--combine", action="store_true",
        help="Combine all PDEs into single CSV"
    )
    
    args = parser.parse_args()
    
    # Print available methods
    available = METHOD_REGISTRY.list_methods()
    print(f"Registered IDENT methods: {available}")
    
    # Filter to available methods
    methods = [m for m in args.methods if m in available]
    if not methods:
        print("Error: No IDENT methods available!")
        sys.exit(1)
    
    print(f"Using methods: {methods}")
    
    # Process each PDE
    all_dfs = []
    for pde_name in args.pdes:
        try:
            df = process_pde(
                pde_name=pde_name,
                config_dir=args.config_dir,
                output_dir=args.output,
                methods=methods,
                max_windows=args.windows,
                n_jobs=args.jobs,
            )
            all_dfs.append(df)
        except Exception as e:
            print(f"Error processing {pde_name}: {e}")
            traceback.print_exc()
    
    # Combine if requested
    if args.combine and all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined_path = os.path.join(args.output, "full_dataset.csv")
        combined.to_csv(combined_path, index=False)
        print(f"\nCombined dataset: {len(combined)} samples -> {combined_path}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
