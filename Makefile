.PHONY: install test reproduce figures clean help

# Default target
help:
	@echo "PDE-Selector Makefile"
	@echo "====================="
	@echo ""
	@echo "Targets:"
	@echo "  make install     - Install dependencies in current venv"
	@echo "  make reproduce   - Reproduce paper results from frozen data (fast)"
	@echo "  make full        - Re-run full pipeline including dataset generation (~25 min)"
	@echo "  make figures     - Generate figures from existing data"
	@echo "  make test        - Run tests"
	@echo "  make notebook    - Launch Jupyter with reproducibility notebook"
	@echo "  make clean       - Remove generated files"
	@echo ""
	@echo "Paper Run Artifacts: experiments/paper_run_2025-12-18/"

# Install dependencies
install:
	pip install -r requirements.txt
	pip install -e .

# Reproduce from frozen data (fast, no dataset regeneration)
reproduce:
	@echo "=== Reproducing from frozen paper run ==="
	python scripts/train_models.py
	python scripts/generate_figures.py
	@echo "✅ Reproduction complete. Figures in data/figures/"

# Full pipeline including dataset generation
full:
	@echo "=== Full pipeline (this will take ~25 minutes) ==="
	python scripts/run_all_methods.py
	python scripts/train_models.py
	python scripts/generate_figures.py
	@echo "✅ Full pipeline complete."

# Just generate figures
figures:
	python scripts/generate_figures.py

# Run tests
test:
	python -m pytest tests/ -v

# Launch Jupyter notebook
notebook:
	jupyter notebook notebooks/reproduce_paper.ipynb

# Clean generated files (but not frozen paper run)
clean:
	rm -rf data/results/*.csv
	rm -rf data/figures/*.png
	rm -rf models/*.joblib
	rm -rf __pycache__ */__pycache__ */*/__pycache__
	@echo "Cleaned generated files. Frozen paper run preserved."
