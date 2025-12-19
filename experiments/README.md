# Experiments Directory

This directory stores dated experiment runs with full provenance tracking.

## Directory Structure

Each experiment should follow this structure:

```
experiments/<experiment_name>/
├── config.yaml         # Experiment configuration (copy of config used)
├── git_info.txt        # Output of `git log -1` and `git diff HEAD`
├── environment.txt     # Output of `pip freeze`
├── outputs/            # Results, models, and artifacts
│   ├── benchmark_results.json
│   ├── models/
│   └── figures/
└── README.md           # What this experiment tests and key findings
```

## Usage

1. Create a new experiment folder with a descriptive name and date:
   ```bash
   mkdir -p experiments/2025-01-15_baseline_benchmark
   ```

2. Record provenance before running:
   ```bash
   cd experiments/2025-01-15_baseline_benchmark
   git log -1 > git_info.txt
   git diff HEAD >> git_info.txt
   pip freeze > environment.txt
   cp ../../config/default.yaml config.yaml
   ```

3. Run the experiment with output directed to this folder:
   ```bash
   python scripts/run_benchmark.py --cfg config.yaml --output outputs/
   ```

4. Document findings in README.md.

## Naming Convention

Format: `YYYY-MM-DD_<experiment_name>`

Examples:
- `2025-01-15_baseline_benchmark`
- `2025-01-20_noise_sensitivity`
- `2025-01-25_kdv_only`

## Guidelines

- **Never modify** completed experiment folders
- **Always commit** experiment results to git
- **Include** all configuration and code version info for reproducibility
