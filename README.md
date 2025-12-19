# PDE-Selector: Meta-Learning for PDE Identification

A meta-learning framework that automatically selects the best PDE identification method for a given dataset without running all candidate algorithms.

> **Acknowledgment**: This repository extends the [WeakIdent](https://github.com/sunghakang/WeakIdent) codebase originally developed by **Mengyi Tang Rajchel**. The WeakIdent method and its Python implementation are based on the paper:
>
> Tang, M., Liao, W., Kuske, R., & Kang, S. H. (2023). *WeakIdent: Weak formulation for Identifying Differential Equation using Narrow-fit and Trimming*. Journal of Computational Physics. [DOI](https://doi.org/10.1016/j.jcp.2023.112069)

## Overview

Given noisy spatiotemporal data, PDE-Selector:
1. Extracts **12 inexpensive features** (Tiny-12) from the raw data
2. Uses a trained **Random Forest classifier** to predict which identification method will perform best
3. Achieves **97% accuracy** in selecting the optimal method on a held-out test set

This avoids the computational cost of running all methods on every data window.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Reproduce paper results (from frozen data)
make reproduce

# Or run the full pipeline (~25 min)
make full
```

## Methods Compared

| Method | Description |
|--------|-------------|
| **LASSO** | L1-regularized sparse regression |
| **STLSQ** | Sequentially Thresholded Least Squares (SINDy) |
| **WeakIDENT** | Weak-form identification with narrow-fit trimming |
| **RobustIDENT** | Robust identification using trimmed least squares |

## Key Results

- **Test accuracy**: 97.06% (1,124/1,158 windows match oracle selection)
- **Zero-regret rate**: 97.06% (predictions achieve same error as oracle)
- **Under noise**: RobustIDENT becomes optimal for ~20% of windows under 2% Gaussian noise

## Repository Structure

```
├── src/                    # Core library
│   ├── features.py         # Tiny-12 feature extraction
│   ├── ident_methods/      # Method implementations
│   └── models/             # Selector models
├── scripts/                # Runnable experiments
├── manuscript/             # Paper source and figures
├── notebooks/              # Guided reproduction notebook
├── data/                   # Raw data and results
└── configs/                # Configuration files
```

## Reproduction

```bash
# Generate dataset from scratch
python scripts/run_all_methods.py    # ~25 min

# Train selector and generate figures
python scripts/train_models.py       # ~2 min
python scripts/generate_figures.py   # ~1 min

# Verify paper consistency
python scripts/check_paper_consistency.py
```

## Citation

If you use this code, please cite both works:

```bibtex
@article{tang2023weakident,
  title={WeakIdent: Weak formulation for Identifying Differential Equation using Narrow-fit and Trimming},
  author={Tang, Mengyi and Liao, Wenjing and Kuske, Rachel and Kang, Sung Ha},
  journal={Journal of Computational Physics},
  pages={112069},
  year={2023},
  publisher={Elsevier}
}
```

## License

MIT License. See [LICENSE](LICENSE).

## Original WeakIdent

For the standalone WeakIdent implementation without the meta-learning framework, see the original repository: https://github.com/sunghakang/WeakIdent
