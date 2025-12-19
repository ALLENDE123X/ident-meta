"""
Label dataset generation: build X_features.npy and Y_<method>.npy

Loops over:
- PDE families (Burgers, KdV)
- Parameter sweeps (nu, alpha, beta, etc.)
- Noise levels
- Windows

For each window:
1. Extract Tiny-12 features -> X row
2. Run each IDENT method -> Y_<method> row (3 metrics)

Features:
- Parallelization via joblib for 2-4x speedup
- Checkpointing for crash recovery
- Resume capability for long runs

Reference: pde-selector-implementation-plan.md §7
"""

import numpy as np
import os
import json
import tempfile
import shutil
from tqdm import tqdm
from joblib import Parallel, delayed
from .data_gen import simulate_burgers, simulate_kdv, make_windows, add_noise
from .features import extract_tiny12
from .ident_api import run_ident_and_metrics


# Default checkpoint interval (number of windows between saves)
DEFAULT_CHECKPOINT_INTERVAL = 50


def _process_single_window(
    u_win, dx, dt, methods, ident_cfg, true_coeffs=None
):
    """
    Process a single window: extract features and run IDENT methods.
    
    This is the unit of work for parallel processing.
    
    Args:
        u_win: np.ndarray of shape (nt_win, nx_win)
        dx, dt: grid spacings
        methods: list of method names
        ident_cfg: dict with {max_dx, max_poly, skip_x, skip_t, tau}
        true_coeffs: optional ground truth coefficients
    
    Returns:
        dict with:
            - features: np.ndarray of shape (12,) or None if failed
            - metrics: dict of {method: np.ndarray of shape (3,)}
            - error: str or None
    """
    result = {"features": None, "metrics": {}, "error": None}
    
    # Extract features
    try:
        phi = extract_tiny12(u_win, dx, dt)
        result["features"] = phi
    except Exception as e:
        result["error"] = f"Feature extraction failed: {e}"
        return result
    
    # Run each IDENT method
    for method in methods:
        try:
            metrics = run_ident_and_metrics(
                u_win,
                method,
                dx,
                dt,
                max_dx=ident_cfg.get("max_dx", 4),
                max_poly=ident_cfg.get("max_poly", 4),
                skip_x=ident_cfg.get("skip_x", 4),
                skip_t=ident_cfg.get("skip_t", 4),
                tau=ident_cfg.get("tau", 0.05),
                true_coeffs=true_coeffs,
            )
            result["metrics"][method] = metrics
        except Exception as e:
            result["metrics"][method] = np.array([0.0, 1.0, 1.0])
            if result["error"] is None:
                result["error"] = f"{method} failed: {e}"
    
    return result


def _save_checkpoint(
    output_dir, X_features, Y_dict, processed_windows, config_hash, methods
):
    """
    Save checkpoint to disk atomically.
    
    Uses atomic write pattern: write to temp file, then rename.
    """
    checkpoint_path = os.path.join(output_dir, ".checkpoint.npz")
    metadata_path = os.path.join(output_dir, ".checkpoint_meta.json")
    
    # Create temp files
    temp_npz = tempfile.NamedTemporaryFile(delete=False, suffix=".npz")
    temp_json = tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode="w")
    
    try:
        # Save arrays
        save_dict = {"X_features": np.array(X_features, dtype=np.float64)}
        for method in methods:
            save_dict[f"Y_{method}"] = np.array(Y_dict[method], dtype=np.float64)
        np.savez(temp_npz.name, **save_dict)
        temp_npz.close()
        
        # Save metadata
        metadata = {
            "processed_windows": processed_windows,
            "config_hash": config_hash,
            "methods": methods,
            "n_samples": len(X_features),
        }
        json.dump(metadata, temp_json)
        temp_json.close()
        
        # Atomic rename
        shutil.move(temp_npz.name, checkpoint_path)
        shutil.move(temp_json.name, metadata_path)
        
    except Exception:
        # Cleanup temp files on failure
        for f in [temp_npz.name, temp_json.name]:
            try:
                os.unlink(f)
            except OSError:
                pass
        raise


def _load_checkpoint(output_dir, methods, config_hash):
    """
    Load checkpoint from disk if it exists and matches config.
    
    Returns:
        tuple: (X_features_list, Y_dict, processed_windows) or ([], {}, set())
    """
    checkpoint_path = os.path.join(output_dir, ".checkpoint.npz")
    metadata_path = os.path.join(output_dir, ".checkpoint_meta.json")
    
    if not os.path.exists(checkpoint_path) or not os.path.exists(metadata_path):
        return [], {method: [] for method in methods}, set()
    
    try:
        # Load metadata
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        
        # Check config hash matches
        if metadata.get("config_hash") != config_hash:
            print("Warning: Checkpoint config mismatch, starting fresh")
            return [], {method: [] for method in methods}, set()
        
        # Load arrays
        data = np.load(checkpoint_path)
        X_features = list(data["X_features"])
        Y_dict = {}
        for method in methods:
            key = f"Y_{method}"
            if key in data:
                Y_dict[method] = list(data[key])
            else:
                Y_dict[method] = []
        
        processed_windows = set(metadata.get("processed_windows", []))
        
        print(f"Resumed from checkpoint: {len(X_features)} samples, "
              f"{len(processed_windows)} windows processed")
        
        return X_features, Y_dict, processed_windows
        
    except Exception as e:
        print(f"Warning: Failed to load checkpoint: {e}")
        return [], {method: [] for method in methods}, set()


def _compute_config_hash(config, methods):
    """Compute a hash of the config for checkpoint validation."""
    import hashlib
    config_str = json.dumps(config, sort_keys=True) + str(sorted(methods))
    return hashlib.md5(config_str.encode()).hexdigest()[:16]


def _create_window_tasks(config, verbose=True):
    """
    Pre-generate all window tasks with metadata.
    
    Returns:
        list of dicts with {window_id, u_win, dx, dt, pde, params, noise_level}
    """
    tasks = []
    
    pdes = config.get("pdes", ["burgers", "kdv"])
    noise_levels = config.get("noise_levels", [0.0, 0.01, 0.02, 0.05])
    nx = config.get("nx", 256)
    nt = config.get("nt", 200)
    
    window_cfg = config.get("window", {})
    nx_win = window_cfg.get("nx_win", 128)
    nt_win = window_cfg.get("nt_win", 64)
    stride_x = window_cfg.get("stride_x", 64)
    stride_t = window_cfg.get("stride_t", 32)
    
    window_id = 0
    
    for pde_name in pdes:
        if pde_name.lower() == "burgers":
            params_list = config.get("burgers_params", [{"nu": 0.01, "ic_type": "sine"}])
            simulator = simulate_burgers
        elif pde_name.lower() == "kdv":
            params_list = config.get("kdv_params", [{"alpha": 1.0, "beta": 0.01, "ic_type": "sech"}])
            simulator = simulate_kdv
        else:
            continue
        
        for params in params_list:
            try:
                u_clean, dx_sim, dt_sim = simulator(nx=nx, nt=nt, L=1.0, T=1.0, **params)
            except Exception as e:
                if verbose:
                    print(f"Error simulating {pde_name} with {params}: {e}")
                continue
            
            for noise_level in noise_levels:
                u_noisy = add_noise(u_clean, noise_level=noise_level)
                windows = make_windows(u_noisy, nt_win, nx_win, stride_t, stride_x)
                
                for u_win in windows:
                    tasks.append({
                        "window_id": window_id,
                        "u_win": u_win,
                        "dx": dx_sim,
                        "dt": dt_sim,
                        "pde": pde_name,
                        "params": params,
                        "noise_level": noise_level,
                    })
                    window_id += 1
    
    return tasks


def generate_dataset(
    config,
    methods,
    output_dir="artifacts",
    verbose=True,
    n_jobs=1,
    checkpoint_interval=DEFAULT_CHECKPOINT_INTERVAL,
    resume=False,
):
    """
    Generate labeled dataset for training selector.

    Args:
        config: dict with keys:
            - pdes: list of PDE names ["burgers", "kdv"]
            - noise_levels: list of noise fractions [0.0, 0.01, 0.02, 0.05]
            - nx, nt: grid sizes
            - dx, dt: grid spacings
            - window: dict with {nx_win, nt_win, stride_x, stride_t}
            - burgers_params: list of dicts with {nu, ic_type, ic_params}
            - kdv_params: list of dicts with {alpha, beta, ic_type, ic_params}
            - ident: dict with {max_dx, max_poly, skip_x, skip_t, tau}
        methods: list of str, IDENT method names ["WeakIDENT"]
        output_dir: str, directory to save datasets
        verbose: bool, print progress
        n_jobs: int, number of parallel jobs (1 = sequential, -1 = all cores)
        checkpoint_interval: int, save checkpoint every N windows
        resume: bool, resume from checkpoint if available

    Returns:
        dict with keys:
            - X_features: np.ndarray of shape (n_windows, 12)
            - Y_<method>: np.ndarray of shape (n_windows, 3) for each method
    """
    os.makedirs(output_dir, exist_ok=True)
    
    config_hash = _compute_config_hash(config, methods)
    ident_cfg = config.get("ident", {})
    
    # Try to resume from checkpoint
    if resume:
        X_features, Y_dict, processed_windows = _load_checkpoint(
            output_dir, methods, config_hash
        )
    else:
        X_features = []
        Y_dict = {method: [] for method in methods}
        processed_windows = set()
    
    # Create all window tasks
    if verbose:
        print(f"\n{'=' * 60}")
        print("PDE Selector Dataset Generation")
        print(f"{'=' * 60}")
        print(f"  Parallel jobs: {n_jobs}")
        print(f"  Checkpoint interval: {checkpoint_interval}")
        print(f"  Resume mode: {resume}")
    
    tasks = _create_window_tasks(config, verbose=verbose)
    
    if verbose:
        print(f"  Total windows: {len(tasks)}")
        print(f"  Already processed: {len(processed_windows)}")
        print(f"{'=' * 60}\n")
    
    # Filter out already processed windows
    pending_tasks = [t for t in tasks if t["window_id"] not in processed_windows]
    
    if len(pending_tasks) == 0:
        if verbose:
            print("All windows already processed!")
    
    elif n_jobs == 1:
        # Sequential processing with checkpointing
        for i, task in enumerate(tqdm(pending_tasks, desc="Processing windows", disable=not verbose)):
            result = _process_single_window(
                task["u_win"],
                task["dx"],
                task["dt"],
                methods,
                ident_cfg,
            )
            
            if result["features"] is not None:
                X_features.append(result["features"])
                for method in methods:
                    Y_dict[method].append(result["metrics"].get(method, np.array([0.0, 1.0, 1.0])))
                processed_windows.add(task["window_id"])
            
            # Checkpoint periodically
            if (i + 1) % checkpoint_interval == 0:
                _save_checkpoint(
                    output_dir, X_features, Y_dict,
                    list(processed_windows), config_hash, methods
                )
                if verbose:
                    tqdm.write(f"  Checkpoint saved: {len(X_features)} samples")
    
    else:
        # Parallel processing in batches
        batch_size = max(checkpoint_interval, n_jobs * 4)
        
        for batch_start in range(0, len(pending_tasks), batch_size):
            batch = pending_tasks[batch_start:batch_start + batch_size]
            
            if verbose:
                print(f"Processing batch {batch_start // batch_size + 1}: "
                      f"windows {batch_start} to {batch_start + len(batch)}")
            
            results = Parallel(n_jobs=n_jobs, backend="loky")(
                delayed(_process_single_window)(
                    t["u_win"], t["dx"], t["dt"], methods, ident_cfg
                )
                for t in tqdm(batch, desc="  Batch progress", disable=not verbose)
            )
            
            # Collect results
            for task, result in zip(batch, results):
                if result["features"] is not None:
                    X_features.append(result["features"])
                    for method in methods:
                        Y_dict[method].append(
                            result["metrics"].get(method, np.array([0.0, 1.0, 1.0]))
                        )
                    processed_windows.add(task["window_id"])
            
            # Checkpoint after each batch
            _save_checkpoint(
                output_dir, X_features, Y_dict,
                list(processed_windows), config_hash, methods
            )
            if verbose:
                print(f"  Checkpoint saved: {len(X_features)} samples")
    
    # Final save
    _save_checkpoint(
        output_dir, X_features, Y_dict,
        list(processed_windows), config_hash, methods
    )
    
    # Convert to arrays
    X_features = np.array(X_features, dtype=np.float64)
    for method in methods:
        Y_dict[method] = np.array(Y_dict[method], dtype=np.float64)
    
    if verbose:
        print(f"\n{'=' * 60}")
        print("Dataset generation complete!")
        print(f"  Total windows: {len(X_features)}")
        print(f"  X_features shape: {X_features.shape}")
        for method in methods:
            print(f"  Y_{method} shape: {Y_dict[method].shape}")
        print(f"{'=' * 60}")
    
    # Save final datasets
    np.save(os.path.join(output_dir, "X_features.npy"), X_features)
    for method in methods:
        np.save(os.path.join(output_dir, f"Y_{method}.npy"), Y_dict[method])
    
    if verbose:
        print(f"\nDatasets saved to {output_dir}/")
    
    return {"X_features": X_features, **Y_dict}


def load_dataset(methods, data_dir="artifacts"):
    """
    Load pre-generated dataset from disk.

    Args:
        methods: list of str, IDENT method names
        data_dir: str, directory containing datasets

    Returns:
        dict with keys:
            - X_features: np.ndarray
            - Y_<method>: np.ndarray for each method
    """
    X_features = np.load(os.path.join(data_dir, "X_features.npy"))
    Y_dict = {}
    for method in methods:
        Y_dict[method] = np.load(os.path.join(data_dir, f"Y_{method}.npy"))

    return {"X_features": X_features, **Y_dict}
