#!/usr/bin/env python
"""
Generate labeled dataset for PDE selector training.

Usage:
    python scripts/make_dataset.py --cfg config/default.yaml
    python scripts/make_dataset.py --cfg config/default.yaml --parallel 4
    python scripts/make_dataset.py --cfg config/default.yaml --parallel 4 --resume

Reference: pde-selector-implementation-plan.md §12
"""

import argparse
import yaml
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.label_dataset import generate_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Generate labeled dataset for PDE selector",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Sequential processing (default)
  python scripts/make_dataset.py --cfg config/default.yaml --verbose

  # Parallel processing with 4 workers
  python scripts/make_dataset.py --cfg config/default.yaml --parallel 4 --verbose

  # Resume from checkpoint after crash
  python scripts/make_dataset.py --cfg config/default.yaml --parallel 4 --resume

  # Use all available CPU cores
  python scripts/make_dataset.py --cfg config/default.yaml --parallel -1
        """,
    )
    parser.add_argument(
        "--cfg", type=str, required=True, help="Path to config YAML file"
    )
    parser.add_argument(
        "--output", type=str, default="artifacts", help="Output directory for datasets"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Print detailed progress"
    )
    parser.add_argument(
        "--parallel", "-j",
        type=int,
        default=1,
        metavar="N",
        help="Number of parallel jobs (1=sequential, -1=all cores, default: 1)",
    )
    parser.add_argument(
        "--resume", "-r",
        action="store_true",
        help="Resume from checkpoint if available (matches config hash)",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=50,
        metavar="N",
        help="Save checkpoint every N windows (default: 50)",
    )

    args = parser.parse_args()

    # Load config
    with open(args.cfg, "r") as f:
        config = yaml.safe_load(f)

    # Extract methods
    methods = config.get("methods", ["WeakIDENT"])

    print(f"\n{'=' * 60}")
    print("PDE Selector Dataset Generation")
    print(f"{'=' * 60}")
    print(f"Config: {args.cfg}")
    print(f"Methods: {methods}")
    print(f"Output: {args.output}")
    print(f"Parallel jobs: {args.parallel}")
    print(f"Resume mode: {args.resume}")
    print(f"Checkpoint interval: {args.checkpoint_interval}")
    print(f"{'=' * 60}\n")

    # Generate dataset
    dataset = generate_dataset(
        config=config.get("data", {}),
        methods=methods,
        output_dir=args.output,
        verbose=args.verbose,
        n_jobs=args.parallel,
        checkpoint_interval=args.checkpoint_interval,
        resume=args.resume,
    )

    print(f"\nDataset generation complete!")
    print(f"Files saved to: {args.output}/")
    print(f"  - X_features.npy: {dataset['X_features'].shape}")
    for method in methods:
        print(f"  - Y_{method}.npy: {dataset[method].shape}")


if __name__ == "__main__":
    main()
