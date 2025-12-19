# WeakIdent-Python

Script for **identifying differential equations** using WeakIdent.

> **Note**: This repository extends the original [WeakIdent](https://github.com/sunghakang/WeakIdent) implementation with a **PDE-Selector** meta-learning framework for automated method selection.

This repo provides implementation details of WeakIdent using Python. 

Copyright 2022, All Rights Reserved

**Code author:  Mengyi Tang Rajchel**

For Paper, "[:link:WeakIdent: Weak formulation for Identifying Differential Equation using Narrow-fit and Trimming](https://doi.org/10.1016/j.jcp.2023.112069)" by Mengyi Tang, Wenjing Liao, Rachel Kuske and Sung Ha Kang.

:blush: If you found WeakIdent useful in your research, please consider cite us:

```
@article{tang2023weakident,
  title={WeakIdent: Weak formulation for Identifying Differential Equation using Narrow-fit and Trimming},
  author={Tang, Mengyi and Liao, Wenjing and Kuske, Rachel and Kang, Sung Ha},
  journal={Journal of Computational Physics},
  pages={112069},
  year={2023},
  publisher={Elsevier}
}
```

## What  does WeakIdent do?
WeakIdent is a general and robust framework to recover differential equations using a weak formulation, for both ordinary and partial differential equations (ODEs and PDEs). 
Noisy time series data are taken in with spacing as input and output a governing equation for this data.




## Environment set-up

### Required packages
`sys, yaml, argparse, time, typing, pandas, tabular, numpy, numpy_index, Scipy`

### Set-up
[Option 1] If you do not have `conda` installed, you can use `pip install` to install the packages listed above.

[Option 2] (1) run `conda env create -f environment.yml` to create the environment. (2) run `conda activate test_env1` to activate the environment.


## Datasets
Sample datasets from various type of equations including true coefficients can be found in the folder `dataset-Python`. For each dataset, there exists a 
configuration file in `configs` that specifies the input argument to run WeakIdent. The following table provides the name of equations of each dataset:

| config file  index       | Equation name      | 
|:-------------:|-------------|
|1     |  Transport Equation |  
| 2     | Reaction Diffusion Equation    | 
| 3 | Anisotropic Porous Medium (PM) Equation    |
| 4 | Heat Equation | 
| 5 | Korteweg-de Vires (KdV) Equation | 
| 6 | Kuramoto-Sivashinsky (KS) Equation | 
| 7 | Nonlinear Schrodinger (NLS) | 
| 8 | 2D Linear System | 
| 9 | Nonlinear System (Van der Pol) | 
| 10 | Nonlinear System (Duffing) | 
| 11 | Noninear System (Lotka-Volterra) | 
|12| Nonlinear System (Lorenz) | 
|13| Noninear System 2D (Lotka-Volterra) |

We refer details of each dataset to the experimental result section in our paper *WeakIdent: Weak formulation for Identifying Differential Equation using Narrow-fit and Trimming*

### Remark: 
The dataset for reaction diffusion type equation and Nonlinear Lotka-Volterro equation are sligher larger (100-200 M). They are not provided in `dataset-Python`. Instead, I provided auto-simulation for these two dataset when running WeakIdent on these equations. 

- To run WeakIdent on reaction diffusion type equation, run `python main.py --config configs/config_2.yaml`. The auto simulation takes 1-3 mintues.

- To run WeakIdent on Nonlinear System 2D (Lotka-Volterra), please run `python main.py --config configs/config_13.yaml`. The auto simulation takes 1 second.

- To run other examples, WeakIdent directly takes provided datasets as input. Running time various between 1 - 30 seconds. 

## Run WeakIdent on provided datasets
There are 13 datasets provided in `dataset-Python`. To run WeakIdent on each indivial dataset, 
run `python main.py --config configs/config_$n$ file name$.yaml` to identify differential equation using a pre-simulated dataset specified in `configs/config_$n$.yaml`. 
### An example of running WeakIdent on Transport Equation with diffusion
Run `python main.py --config configs/config_1.yaml` to see the output:

```
Start loading arguments and dataset transportDiff.npy for Transport Equation
Start building feature matrix W:
[===========================================] 100.0% 
Start building scale matrix S:
[===========================================] 100.0% 
The number of rows in the highly dynamic region is  845

 Start finding support: 
[=========] 100.0% 
WeakIdent finished support trimming and narrow-fit for variable no.1 . A support is found this variable.

 ------------- coefficient vector overview ------noise-signal-ratio : 0.5  -------
+----+----------------+------------+------------+
|    | feature        |   true u_t |   pred u_t |
+====+================+============+============+
|  0 | 1              |       0    |  0         |
+----+----------------+------------+------------+
|  1 | u              |       0    |  0         |
+----+----------------+------------+------------+
|  2 | u_{x}          |      -1    | -1.00707   |
+----+----------------+------------+------------+
|  3 | u_{xx}         |       0.05 |  0.0526434 |
+----+----------------+------------+------------+
                    ......
+----+----------------+------------+------------+
| 42 | (u^6)_{xxxxxx} |       0    |  0         |
+----+----------------+------------+------------+

 ------------- equation overview ------noise-signal-ratio : 0.5  -------------------
+----+---------------------------------+------------------------------------+
|    | True equation                   | Predicted equation                 |
+====+=================================+====================================+
|  0 | u_t = - 1.0 u_{x} + 0.05 u_{xx} | u_t = - 1.007 u_{x} + 0.053 u_{xx} |
+----+---------------------------------+------------------------------------+

 ------------------------------ CPU time: 0.7 seconds ------------------------------

 Identification error for Transport Equation from WeakIdent: 
+----+-----------+----------------+-------------+---------+---------+
|    |     $e_2$ |   $e_{\infty}$ |   $e_{res}$ |   $tpr$ |   $ppv$ |
+====+===========+================+=============+=========+=========+
|  0 | 0.0075372 |      0.0528683 |    0.417006 |       1 |       1 |
+----+-----------+----------------+-------------+---------+---------+
```

## More sample output for each dataset
We provide sample output for each equation(dataset) in  the folder `output`.

## Credit/note
Build feature matrix through convolution (using fft), this part of the code is modified from `get_lib_columns()` (originally Matlab version) from [WeakSindyPde](https://github.com/dm973/WSINDy_PDE).

---

## PDE-Selector: Algorithm-Selection Meta-Learner

**NEW**: This repository now includes a **PDE-Selector** framework that automatically chooses the best IDENT method for a given spatiotemporal dataset using machine learning.

### Quick Start with PDE-Selector

**Prerequisites**: Python 3.8+, numpy, scipy, scikit-learn, joblib, matplotlib, pyyaml, numpy-indexed

```bash
# 1. Install dependencies (required for all PDE-Selector features)
pip install -r requirements.txt

# 2. Generate labeled training dataset
python scripts/make_dataset.py --cfg config/default.yaml --verbose

# 3. Train per-method regressors (default: rf_multi)
python scripts/train_selector.py --cfg config/default.yaml

# Train with different models:
python scripts/train_selector.py --cfg config/default.yaml --model ridge_multi
python scripts/train_selector.py --cfg config/default.yaml --model catboost_multi --params '{"iterations":600,"learning_rate":0.05,"depth":8}'

# 4. Evaluate selector on test set
python scripts/evaluate_selector.py --cfg config/default.yaml

# Evaluate with specific model:
python scripts/evaluate_selector.py --cfg config/default.yaml --model ridge_multi

# 5. Choose and run IDENT on a new field
python scripts/choose_and_run.py --npy data/u.npy --dx 0.0039 --dt 0.005 --cfg config/default.yaml
```

### What is PDE-Selector?

PDE-Selector is an **algorithm-selection meta-learner** that:
- Extracts **12 characteristic features** (Tiny-12) from spatiotemporal data without running IDENT
- Predicts which IDENT method (e.g., WeakIDENT, RobustIDENT) will perform best
- Uses a **safety gate** to run multiple methods when predictions are uncertain
- Saves computation by avoiding unnecessary method runs

**Key Benefits:**
- ✅ **Faster**: Runs only 1-2 methods instead of all methods
- ✅ **Smarter**: Learns which method works best for different data characteristics
- ✅ **Robust**: Falls back to top-2 methods when uncertain

### Architecture

```
1. Data Generation (src/data_gen.py)
   - Simulate Burgers, KdV equations
   - Add noise at multiple levels
   - Extract windows

2. Feature Extraction (src/features.py)
   - Tiny-12 features (NO IDENT leakage)
   - Sampling, derivatives, noise, spectrum, periodicity

3. Labeling (src/label_dataset.py)
   - Run WeakIDENT on each window
   - Compute 3 metrics: F1, CoeffErr, ResidualMSE
   - Save X_features.npy, Y_WeakIDENT.npy

4. Training (src/models.py, src/models/factory.py)
   - Pluggable meta-regression models (5 types supported)
   - Per-method regressors predicting 3 metrics
   - Uncertainty estimation (RF variance; NaN for others)

5. Selection (src/select_and_run.py)
   - Choose best method based on predicted score
   - Safety gate: run top-2 if score > tau or uncertainty high

6. Evaluation (src/eval.py)
   - Regret, Top-1 accuracy, Compute saved
```

### Supported Models

PDE-Selector supports 5 pluggable meta-regression models:

1. **linear_ols**: MultiOutputRegressor(LinearRegression()) - Fast baseline
2. **ridge_multi**: MultiOutputRegressor(Ridge()) - Regularized linear model
3. **regressor_chain_ridge**: RegressorChain(Ridge()) - Chain-based multi-output
4. **rf_multi**: RandomForestRegressor - Tree ensemble with variance estimation (default)
5. **catboost_multi**: CatBoostRegressor - Gradient boosting with multi-output support

**Uncertainty**: Only `rf_multi` provides variance estimates; others return NaN (safety gate handles it).

### Configuration

Edit `config/default.yaml` to customize:
- PDE families (Burgers, KdV, KS)
- Noise levels
- Window sizes and strides
- Model selection (`model.name`) and hyperparameters (`model.params`)
- Cross-validation settings (optional)
- Aggregation weights for 3 metrics
- Safety threshold (tau)

**Example model configuration:**
```yaml
model:
  name: rf_multi  # or linear_ols, ridge_multi, regressor_chain_ridge, catboost_multi
  params:
    n_estimators: 300
    max_depth: 8
  random_state: 0
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/test_features.py -v
```

### Directory Structure

```
WeakIdent-Python/
├── src/               # PDE-Selector core modules
│   ├── features.py       # Tiny-12 feature extraction
│   ├── data_gen.py       # Burgers/KdV simulators
│   ├── ident_api.py      # IDENT method adapters
│   ├── metrics.py        # 3 error metrics + aggregation
│   ├── label_dataset.py  # Dataset generation
│   ├── models.py         # PerMethodRegressor (pluggable models)
│   ├── models/           # Model factory
│   │   └── factory.py    # create_model() for 5 model types
│   ├── select_and_run.py # Selector + safety gate
│   └── eval.py           # Evaluation metrics
├── scripts/           # CLI scripts
│   ├── make_dataset.py
│   ├── train_selector.py
│   ├── evaluate_selector.py
│   └── choose_and_run.py
├── tests/             # Unit tests
├── config/            # Configuration files
│   └── default.yaml
├── models/            # Trained .joblib models
├── artifacts/         # Datasets and outputs
└── logs/              # Execution logs
```

### Reference

For full implementation details, see:
- `pde-selector-implementation-plan.md` - Complete specification
- `RUNLOG.md` - Development log and gap analysis

---

## Dataset file format (u, xs, true_coefficients)

Each dataset file in `dataset-Python/` is a single `.npy` file containing three consecutive NumPy arrays saved with `allow_pickle=True`:

- `u`: object array of length `n` (number of variables). For 1D-in-space PDEs, `u[0]` has shape `(Nx, Nt)`.
- `xs`: object array of length `dim_x + 1`. For 1D-in-space, `xs = [x, t]` where `x.shape == (Nx, 1)` and `t.shape == (1, Nt)`.
- `true_coefficients`: object array of length `n`. Each entry is a 2D float array where each row is `[beta_u, d_x, d_t, coeff]` for `n=1, dim_x=1`. For example, viscous Burgers `u_t = - (u^2)_x/2 + \nu u_{xx}` is encoded as:

```
[[2., 1., 0., -0.5],   # (u^2)_x with coefficient -1/2
 [1., 2., 0.,  0.01]]  # u_{xx} with viscosity 0.01
```

### Regenerating Burgers (viscous) dataset

- Run:

```
python make_burgers_viscous.py
```

- This writes `dataset-Python/burgers_viscous.npy` with the correct structure for `main.py --config configs/burgers_viscous.yaml`.
