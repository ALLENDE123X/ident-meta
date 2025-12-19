#!/usr/bin/env python
"""
Run full benchmark suite for PDE-Selector.

This script runs the complete pipeline:
1. Generate dataset (if not exists)
2. Train selector models
3. Evaluate on test set
4. Compute metrics (regret, top-1 accuracy, compute saved)
5. Generate figures
6. Save provenance info (git hash, environment)

Usage:
    python scripts/run_benchmark.py --cfg config/default.yaml
    python scripts/run_benchmark.py --cfg experiments/baseline_benchmark/config.yaml --quick

Reference: pde-selector-implementation-plan.md §M2
"""

import argparse
import yaml
import sys
import os
import json
import subprocess
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_git_info():
    """Get current git commit hash and status."""
    try:
        commit = subprocess.check_output(
            ["git", "log", "-1", "--format=%H %s"], text=True
        ).strip()
        diff = subprocess.check_output(
            ["git", "diff", "--stat", "HEAD"], text=True
        ).strip()
        return {"commit": commit, "has_uncommitted_changes": len(diff) > 0}
    except Exception as e:
        return {"error": str(e)}


def get_environment_info():
    """Get Python and package versions."""
    try:
        pip_freeze = subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"], text=True
        )
        return {
            "python_version": sys.version,
            "packages": pip_freeze.strip().split("\n"),
        }
    except Exception as e:
        return {"error": str(e)}


def run_command(cmd, description, verbose=True):
    """Run a shell command and handle errors."""
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Step: {description}")
        print(f"Command: {' '.join(cmd)}")
        print(f"{'=' * 60}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error: {description} failed")
        print(f"stdout: {result.stdout}")
        print(f"stderr: {result.stderr}")
        return False
    
    if verbose:
        print(result.stdout)
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Run full PDE-Selector benchmark suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full benchmark (may take 1+ hours)
  python scripts/run_benchmark.py --cfg config/default.yaml

  # Quick benchmark (reduced parameters)
  python scripts/run_benchmark.py --cfg config/default.yaml --quick

  # Specify output directory
  python scripts/run_benchmark.py --cfg config/default.yaml --output experiments/run_001
        """,
    )
    parser.add_argument(
        "--cfg", type=str, required=True, help="Path to config YAML file"
    )
    parser.add_argument(
        "--output", type=str, default="artifacts/benchmark",
        help="Output directory for benchmark results"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Run quick benchmark with reduced dataset (for testing)"
    )
    parser.add_argument(
        "--skip-dataset", action="store_true",
        help="Skip dataset generation (use existing)"
    )
    parser.add_argument(
        "--skip-train", action="store_true",
        help="Skip model training (use existing)"
    )
    parser.add_argument(
        "--parallel", "-j", type=int, default=1,
        help="Number of parallel jobs for dataset generation"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print detailed progress"
    )

    args = parser.parse_args()

    # Setup output directory
    os.makedirs(args.output, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"\n{'#' * 60}")
    print(f"# PDE-Selector Benchmark Suite")
    print(f"# Started: {timestamp}")
    print(f"# Output: {args.output}")
    print(f"{'#' * 60}")

    # Save provenance info
    provenance = {
        "timestamp": timestamp,
        "config_file": args.cfg,
        "quick_mode": args.quick,
        "git": get_git_info(),
        "environment": get_environment_info(),
    }
    
    with open(os.path.join(args.output, "provenance.json"), "w") as f:
        json.dump(provenance, f, indent=2)
    
    print(f"\nProvenance saved to {args.output}/provenance.json")

    # Copy config to output
    subprocess.run(["cp", args.cfg, os.path.join(args.output, "config.yaml")])

    # Modify config for quick mode
    config_to_use = args.cfg
    if args.quick:
        print("\nQuick mode: creating reduced config...")
        with open(args.cfg, "r") as f:
            config = yaml.safe_load(f)
        
        # Reduce dataset size
        if "data" in config:
            config["data"]["noise_levels"] = [0.0, 0.02]
            if "burgers_params" in config["data"]:
                config["data"]["burgers_params"] = config["data"]["burgers_params"][:1]
            if "kdv_params" in config["data"]:
                config["data"]["kdv_params"] = config["data"]["kdv_params"][:1]
        
        quick_config_path = os.path.join(args.output, "config_quick.yaml")
        with open(quick_config_path, "w") as f:
            yaml.dump(config, f)
        config_to_use = quick_config_path
        print(f"Quick config saved to {quick_config_path}")

    # Step 1: Generate dataset
    data_dir = os.path.join(args.output, "data")
    if not args.skip_dataset:
        cmd = [
            sys.executable, "scripts/make_dataset.py",
            "--cfg", config_to_use,
            "--output", data_dir,
            "--parallel", str(args.parallel),
            "--verbose",
        ]
        if not run_command(cmd, "Generate Dataset", verbose=args.verbose):
            print("Dataset generation failed!")
            sys.exit(1)
    else:
        print("\nSkipping dataset generation (--skip-dataset)")
        if not os.path.exists(data_dir):
            data_dir = "artifacts"  # fallback

    # Step 2: Train models
    models_dir = os.path.join(args.output, "models")
    if not args.skip_train:
        cmd = [
            sys.executable, "scripts/train_selector.py",
            "--cfg", config_to_use,
            "--data", data_dir,
            "--output", models_dir,
        ]
        if not run_command(cmd, "Train Models", verbose=args.verbose):
            print("Model training failed!")
            sys.exit(1)
    else:
        print("\nSkipping model training (--skip-train)")
        if not os.path.exists(models_dir):
            models_dir = "models"  # fallback

    # Step 3: Evaluate
    results_file = os.path.join(args.output, "benchmark_results.json")
    cmd = [
        sys.executable, "scripts/evaluate_selector.py",
        "--cfg", config_to_use,
        "--data", data_dir,
        "--models", models_dir,
        "--output", args.output,
        "--output-json", results_file,
        "--plot",
    ]
    if not run_command(cmd, "Evaluate Selector", verbose=args.verbose):
        print("Evaluation failed!")
        sys.exit(1)

    # Load and display results
    with open(results_file, "r") as f:
        results = json.load(f)

    print(f"\n{'#' * 60}")
    print("# Benchmark Results")
    print(f"{'#' * 60}")
    print(f"Mean Regret:        {results['regret']:.4f}")
    print(f"Top-1 Accuracy:     {results['top1_accuracy'] * 100:.2f}%")
    print(f"Compute Saved:      {results['compute_saved']['frac_saved'] * 100:.2f}%")
    print(f"Mean Methods Run:   {results['compute_saved']['mean_methods_run']:.2f}")
    print(f"Windows Evaluated:  {results['n_windows']}")
    print(f"{'#' * 60}")

    print(f"\nBenchmark outputs saved to {args.output}/")
    print(f"  - benchmark_results.json")
    print(f"  - provenance.json")
    print(f"  - regret_cdf_*.png")
    print(f"  - confusion_matrix_*.png")

    print(f"\n{'#' * 60}")
    print(f"# Benchmark Complete!")
    print(f"{'#' * 60}\n")

    return results


if __name__ == "__main__":
    main()
