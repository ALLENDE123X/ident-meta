#!/bin/bash
# reproduce_results.sh - Reproduce PDE-Selector results end-to-end
#
# Usage:
#   ./scripts/reproduce_results.sh           # Local execution (no PySINDy)
#   ./scripts/reproduce_results.sh --docker  # Docker execution (with PySINDy)
#
# Expected runtime: ~30 minutes
# Expected outputs:
#   - data/results/full_dataset_4methods.csv  (5786 rows)
#   - data/results/model_comparison.csv       (6 models)
#   - data/figures/*.png                      (5 figures)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "========================================"
echo "PDE-Selector Reproduction Script"
echo "========================================"
echo "Project directory: $PROJECT_DIR"
echo "Start time: $(date)"
echo ""

USE_DOCKER=false
if [[ "$1" == "--docker" ]]; then
    USE_DOCKER=true
fi

# Function to run python command
run_python() {
    if $USE_DOCKER; then
        docker compose run --rm weakident python "$@"
    else
        python "$@"
    fi
}

# Check prerequisites
if $USE_DOCKER; then
    echo "Mode: Docker"
    if ! command -v docker &> /dev/null; then
        echo "ERROR: Docker not found. Install Docker or run without --docker flag."
        exit 1
    fi
    echo "Building Docker image..."
    docker compose build
else
    echo "Mode: Local Python"
    if [[ ! -d "venv" ]]; then
        echo "Creating virtual environment..."
        python3 -m venv venv
    fi
    echo "Activating venv..."
    source venv/bin/activate
    echo "Installing dependencies..."
    pip install -q -r requirements.txt
fi

echo ""
echo "========================================"
echo "Step 1: Dataset Generation"
echo "========================================"
echo "Running: scripts/run_all_methods.py"
echo "This processes 4 PDEs with 4 methods (~25 min)..."

START_TIME=$(date +%s)
run_python scripts/run_all_methods.py
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo "Dataset generation completed in $((ELAPSED / 60)) min $((ELAPSED % 60)) sec"

# Verify output
if [[ -f "data/results/full_dataset_4methods.csv" ]]; then
    ROWS=$(wc -l < data/results/full_dataset_4methods.csv)
    echo "✓ Created: data/results/full_dataset_4methods.csv ($((ROWS - 1)) data rows)"
else
    echo "✗ ERROR: Dataset file not created!"
    exit 1
fi

echo ""
echo "========================================"
echo "Step 2: Model Training"
echo "========================================"
echo "Running: scripts/train_models.py"
echo "Training 6 classifiers..."

START_TIME=$(date +%s)
run_python scripts/train_models.py
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo "Training completed in $ELAPSED sec"

# Verify output
if [[ -f "data/results/model_comparison.csv" ]]; then
    MODELS=$(wc -l < data/results/model_comparison.csv)
    echo "✓ Created: data/results/model_comparison.csv ($((MODELS - 1)) models)"
else
    echo "✗ ERROR: Model comparison file not created!"
    exit 1
fi

echo ""
echo "========================================"
echo "Step 3: Figure Generation"
echo "========================================"
echo "Running: scripts/generate_figures.py"
echo "Creating 5 publication-ready figures..."

START_TIME=$(date +%s)
run_python scripts/generate_figures.py
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo "Figure generation completed in $ELAPSED sec"

# Verify outputs
FIGURES=("confusion_matrix.png" "feature_importance.png" "regret_cdf.png" "model_comparison.png" "method_distribution.png")
MISSING=0
for fig in "${FIGURES[@]}"; do
    if [[ -f "data/figures/$fig" ]]; then
        SIZE=$(ls -lh "data/figures/$fig" | awk '{print $5}')
        echo "✓ Created: data/figures/$fig ($SIZE)"
    else
        echo "✗ Missing: data/figures/$fig"
        MISSING=$((MISSING + 1))
    fi
done

echo ""
echo "========================================"
echo "Verification"
echo "========================================"

# Quick verification
run_python -c "
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('data/results/full_dataset_4methods.csv')
print(f'Dataset rows: {len(df)} (expected: 5786)')

FEATURE_COLS = [f'feat_{i}' for i in range(12)]
X = np.nan_to_num(df[FEATURE_COLS].values)
le = LabelEncoder()
y = le.fit_transform(df['best_method'].values)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

acc = np.mean(model.predict(X_test) == y_test)
print(f'Test Accuracy: {acc:.4f} (expected: ~0.971)')

# Zero regret
methods = ['LASSO', 'STLSQ', 'RobustIDENT', 'WeakIDENT']
e2_matrix = df[[f'{m}_e2' for m in methods]].values
oracle_e2 = e2_matrix.min(axis=1)
preds = model.predict(X_scaled)
pred_labels = le.inverse_transform(preds)
selector_e2 = np.array([df.iloc[i][f'{pred_labels[i]}_e2'] for i in range(len(preds))])
regret = selector_e2 - oracle_e2
zr = 100 * (regret == 0).mean()
print(f'Zero-Regret Rate: {zr:.1f}% (expected: 99.4%)')
"

echo ""
echo "========================================"
echo "Summary"
echo "========================================"
echo "End time: $(date)"
echo ""
echo "Outputs created:"
echo "  - data/results/full_dataset_4methods.csv"
echo "  - data/results/model_comparison.csv"
echo "  - data/figures/confusion_matrix.png"
echo "  - data/figures/feature_importance.png"
echo "  - data/figures/regret_cdf.png"
echo "  - data/figures/model_comparison.png"
echo "  - data/figures/method_distribution.png"

if [[ $MISSING -eq 0 ]]; then
    echo ""
    echo "✓ All results reproduced successfully!"
    exit 0
else
    echo ""
    echo "✗ $MISSING figure(s) missing - check logs above"
    exit 1
fi

