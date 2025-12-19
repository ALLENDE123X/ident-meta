"""
Evaluation metrics and visualization for PDE selector.

Computes:
- Regret: E[score_chosen - score_best]
- Top-1 accuracy: fraction where chosen = best
- Compute saved: fraction of windows where only 1 method was run

Also provides visualization functions.

Reference: pde-selector-implementation-plan.md §10
"""

import numpy as np
import matplotlib.pyplot as plt
import os


def compute_regret(chosen_scores, best_scores):
    """
    Compute mean regret: E[score_chosen - score_best]

    Args:
        chosen_scores: np.ndarray of shape (n,), scores of chosen methods
        best_scores: np.ndarray of shape (n,), scores of best methods (oracle)

    Returns:
        float: mean regret
    """
    regrets = chosen_scores - best_scores
    mean_regret = np.mean(regrets)
    return float(mean_regret)


def compute_top1_accuracy(chosen_methods, best_methods):
    """
    Compute top-1 accuracy: fraction where chosen == best.

    Args:
        chosen_methods: list or array of chosen method names
        best_methods: list or array of best method names (oracle)

    Returns:
        float: top-1 accuracy in [0, 1]
    """
    correct = np.sum(np.array(chosen_methods) == np.array(best_methods))
    accuracy = correct / len(chosen_methods)
    return float(accuracy)


def compute_compute_saved(n_methods_run, n_methods_total):
    """
    Compute compute saved: fraction of windows where < n_methods_total were run.

    Args:
        n_methods_run: list or array of ints, number of methods run per window
        n_methods_total: int, total number of available methods

    Returns:
        dict: {
            "frac_saved": float, fraction where methods were saved,
            "mean_methods_run": float, average number of methods run,
            "frac_single": float, fraction where only 1 method was run
        }
    """
    n_methods_run = np.array(n_methods_run)
    frac_saved = np.mean(n_methods_run < n_methods_total)
    mean_methods_run = np.mean(n_methods_run)
    frac_single = np.mean(n_methods_run == 1)

    return {
        "frac_saved": float(frac_saved),
        "mean_methods_run": float(mean_methods_run),
        "frac_single": float(frac_single),
    }


def evaluate_selector(results, methods, output_dir="artifacts"):
    """
    Evaluate selector performance on test set.

    Args:
        results: dict with keys:
            - "chosen_methods": list of chosen method names
            - "chosen_scores": list of aggregated scores for chosen methods
            - "all_scores": list of dicts {method: score} for each window
            - "n_methods_run": list of ints, number of methods run per window
        methods: list of str, all available method names
        output_dir: str, directory to save results

    Returns:
        dict with evaluation metrics
    """
    chosen_methods = results["chosen_methods"]
    chosen_scores = np.array(results["chosen_scores"])
    all_scores = results["all_scores"]
    n_methods_run = results["n_methods_run"]

    # Compute oracle best for each window
    best_methods = []
    best_scores = []
    for scores_dict in all_scores:
        best_method = min(scores_dict, key=scores_dict.get)
        best_score = scores_dict[best_method]
        best_methods.append(best_method)
        best_scores.append(best_score)
    best_scores = np.array(best_scores)

    # Compute metrics
    regret = compute_regret(chosen_scores, best_scores)
    top1_acc = compute_top1_accuracy(chosen_methods, best_methods)
    compute_stats = compute_compute_saved(n_methods_run, len(methods))

    eval_results = {
        "regret": regret,
        "top1_accuracy": top1_acc,
        "compute_saved": compute_stats,
        "n_windows": len(chosen_methods),
    }

    # Print summary
    print(f"\n{'=' * 60}")
    print("Selector Evaluation Results")
    print(f"{'=' * 60}")
    print(f"Number of windows: {eval_results['n_windows']}")
    print(f"Mean regret: {regret:.4f}")
    print(f"Top-1 accuracy: {top1_acc * 100:.2f}%")
    print(f"Fraction with compute saved: {compute_stats['frac_saved'] * 100:.2f}%")
    print(f"Mean methods run per window: {compute_stats['mean_methods_run']:.2f}")
    print(f"Fraction with only 1 method run: {compute_stats['frac_single'] * 100:.2f}%")
    print(f"{'=' * 60}\n")

    # Save to file
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "eval_results.txt"), "w") as f:
        f.write(f"Selector Evaluation Results\n")
        f.write(f"{'=' * 60}\n")
        f.write(f"Number of windows: {eval_results['n_windows']}\n")
        f.write(f"Mean regret: {regret:.4f}\n")
        f.write(f"Top-1 accuracy: {top1_acc * 100:.2f}%\n")
        f.write(f"Fraction with compute saved: {compute_stats['frac_saved'] * 100:.2f}%\n")
        f.write(f"Mean methods run per window: {compute_stats['mean_methods_run']:.2f}\n")
        f.write(f"Fraction with only 1 method run: {compute_stats['frac_single'] * 100:.2f}%\n")

    return eval_results


def plot_regret_cdf(chosen_scores, best_scores, output_dir="artifacts", model_name="rf_multi"):
    """
    Plot CDF of regret values.

    Args:
        chosen_scores: np.ndarray
        best_scores: np.ndarray
        output_dir: str
        model_name: str, model name for filename
    """
    regrets = chosen_scores - best_scores
    sorted_regrets = np.sort(regrets)
    cdf = np.arange(1, len(sorted_regrets) + 1) / len(sorted_regrets)

    plt.figure(figsize=(8, 6))
    plt.plot(sorted_regrets, cdf, linewidth=2)
    plt.xlabel("Regret", fontsize=12)
    plt.ylabel("CDF", fontsize=12)
    plt.title(f"Cumulative Distribution of Regret ({model_name})", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    filename = f"regret_cdf_{model_name}.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=150)
    plt.close()


def plot_confusion_matrix(chosen_methods, best_methods, methods, output_dir="artifacts", model_name="rf_multi"):
    """
    Plot confusion matrix: chosen vs. oracle best.

    Args:
        chosen_methods: list of str
        best_methods: list of str
        methods: list of str, all method names
        output_dir: str
        model_name: str, model name for filename
    """
    n = len(methods)
    method_to_idx = {m: i for i, m in enumerate(methods)}

    # Build confusion matrix
    conf_matrix = np.zeros((n, n), dtype=int)
    for chosen, best in zip(chosen_methods, best_methods):
        i = method_to_idx[chosen]
        j = method_to_idx[best]
        conf_matrix[i, j] += 1

    # Plot
    plt.figure(figsize=(8, 6))
    plt.imshow(conf_matrix, cmap="Blues", aspect="auto")
    plt.colorbar(label="Count")
    plt.xticks(range(n), methods, rotation=45, ha="right")
    plt.yticks(range(n), methods)
    plt.xlabel("Oracle Best Method", fontsize=12)
    plt.ylabel("Chosen Method", fontsize=12)
    plt.title(f"Confusion Matrix: Chosen vs. Best ({model_name})", fontsize=14)

    # Annotate cells
    for i in range(n):
        for j in range(n):
            plt.text(j, i, str(conf_matrix[i, j]), ha="center", va="center", color="black")

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    filename = f"confusion_matrix_{model_name}.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=150)
    plt.close()


def plot_top1_by_noise(results_by_noise, output_dir="artifacts"):
    """
    Plot top-1 accuracy vs. noise level.

    Args:
        results_by_noise: dict of {noise_level: top1_accuracy}
        output_dir: str
    """
    noise_levels = sorted(results_by_noise.keys())
    accuracies = [results_by_noise[nl] for nl in noise_levels]

    plt.figure(figsize=(8, 6))
    plt.plot(noise_levels, accuracies, marker="o", linewidth=2, markersize=8)
    plt.xlabel("Noise Level", fontsize=12)
    plt.ylabel("Top-1 Accuracy", fontsize=12)
    plt.title("Top-1 Accuracy vs. Noise Level", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "top1_by_noise.png"), dpi=150)
    plt.close()

