"""
compare_models.py
-----------------
Reads saved results for both models and produces:
  - A side-by-side summary table printed to stdout
  - results/training_accuracy_comparison.png  (training accuracy bar chart)
  - results/inference_accuracy_comparison.png (inference accuracy bar chart)
  - results/model_comparison.json             (combined stats for reference)

Each bar chart annotates:
  • absolute accuracy for each model
  • absolute percent gain (fine-tuned − base)
  • relative percent gain ((fine-tuned − base) / base × 100)

Does NOT load any model or run any inference.
"""

import json
import os
import sys
import matplotlib.pyplot as plt

RESULTS_DIR  = "results"
BASE_TRAIN   = os.path.join(RESULTS_DIR, "base_model_train_stats.json")
FT_TRAIN     = os.path.join(RESULTS_DIR, "finetune_model_train_stats.json")
BASE_INFER   = os.path.join(RESULTS_DIR, "base_model_inference.json")
FT_INFER     = os.path.join(RESULTS_DIR, "finetune_model_inference.json")
OUT_TRAIN_PNG = os.path.join(RESULTS_DIR, "training_accuracy_comparison.png")
OUT_INFER_PNG = os.path.join(RESULTS_DIR, "inference_accuracy_comparison.png")
OUT_JSON      = os.path.join(RESULTS_DIR, "model_comparison.json")


def load_json(path: str) -> dict:
    if not os.path.exists(path):
        print(f"[ERROR] Required file not found: {path}")
        print("  Run the training and inference scripts first.")
        sys.exit(1)
    with open(path, "r") as f:
        return json.load(f)


def gain_labels(base_acc: float, ft_acc: float) -> tuple:
    """Return (absolute_gain_str, relative_gain_str)."""
    abs_gain = ft_acc - base_acc
    rel_gain = (abs_gain / base_acc) * 100.0 if base_acc > 0 else 0.0
    abs_sign = "+" if abs_gain >= 0 else ""
    rel_sign = "+" if rel_gain >= 0 else ""
    return (
        f"{abs_sign}{abs_gain:.2f} pp",       # percentage-point gain
        f"{rel_sign}{rel_gain:.2f}% relative", # relative gain
    )


def print_table(base_train: dict, ft_train: dict,
                base_infer: dict, ft_infer: dict) -> None:
    """Print a side-by-side comparison table to stdout."""
    col_w = [24, 16, 16]
    sep = "┼".join("─" * w for w in col_w)
    top = "┬".join("─" * w for w in col_w)
    bot = "┴".join("─" * w for w in col_w)

    def row(c1, c2, c3):
        return f"│ {c1:<{col_w[0]-2}} │ {c2:<{col_w[1]-2}} │ {c3:<{col_w[2]-2}} │"

    # Training accuracy
    base_tr  = base_train.get("train_accuracy",    base_train.get("best_val_accuracy", 0))
    ft_tr    = ft_train.get("train_accuracy",      ft_train.get("best_val_accuracy", 0))
    abs_t, rel_t = gain_labels(base_tr, ft_tr)

    # Inference accuracy
    base_inf = base_infer["test_accuracy"]
    ft_inf   = ft_infer["test_accuracy"]
    abs_i, rel_i = gain_labels(base_inf, ft_inf)

    print(f"\n{'='*62}")
    print(" Model Comparison — CIFAR-10")
    print(f"{'='*62}")
    print(f"┌{top}┐")
    print(row("Metric", "Base (scratch)", "Fine-Tuned"))
    print(f"├{sep}┤")
    print(row("Training Accuracy",  f"{base_tr:.2f}%",  f"{ft_tr:.2f}%"))
    print(row("  abs gain",         "—",                abs_t))
    print(row("  rel gain",         "—",                rel_t))
    print(f"├{sep}┤")
    print(row("Inference Accuracy", f"{base_inf:.2f}%", f"{ft_inf:.2f}%"))
    print(row("  abs gain",         "—",                abs_i))
    print(row("  rel gain",         "—",                rel_i))
    print(f"└{bot}┘\n")


def save_bar_chart(base_acc: float, ft_acc: float,
                   title: str, ylabel: str, out_path: str) -> None:
    """Save a bar chart comparing base vs fine-tuned for one accuracy metric."""
    abs_gain, rel_gain = gain_labels(base_acc, ft_acc)
    labels = ["Base (scratch)", "Fine-Tuned"]
    values = [base_acc, ft_acc]
    colors = ["steelblue", "darkorange"]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, values, color=colors, width=0.45)
    ax.set_title(title, fontsize=13, pad=14)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, min(values[1] * 1.25 + 5, 100))

    # Value labels above each bar
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                val + 0.8,
                f"{val:.2f}%",
                ha="center", va="bottom", fontsize=11, fontweight="bold")

    # Gain annotation below the chart title
    gain_text = f"Gain:  {abs_gain}   |   {rel_gain}"
    ax.annotate(
        gain_text,
        xy=(0.5, 0.97), xycoords="axes fraction",
        ha="center", va="top", fontsize=10,
        color="dimgray",
        bbox=dict(boxstyle="round,pad=0.3", fc="#f0f0f0", ec="none"),
    )

    plt.tight_layout()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Chart saved → {out_path}")


def save_combined_json(base_train: dict, ft_train: dict,
                       base_infer: dict, ft_infer: dict) -> None:
    base_tr  = base_train.get("train_accuracy", base_train.get("best_val_accuracy", 0))
    ft_tr    = ft_train.get("train_accuracy",   ft_train.get("best_val_accuracy", 0))
    abs_t,  rel_t  = gain_labels(base_tr,           ft_tr)
    abs_i,  rel_i  = gain_labels(base_infer["test_accuracy"], ft_infer["test_accuracy"])

    combined = {
        "training_accuracy": {
            "base":          base_tr,
            "fine_tuned":    ft_tr,
            "abs_gain":      abs_t,
            "rel_gain":      rel_t,
        },
        "inference_accuracy": {
            "base":          base_infer["test_accuracy"],
            "fine_tuned":    ft_infer["test_accuracy"],
            "abs_gain":      abs_i,
            "rel_gain":      rel_i,
        },
        "inference_time_sec": {
            "base":       base_infer["inference_time_sec"],
            "fine_tuned": ft_infer["inference_time_sec"],
        },
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"Combined JSON saved → {OUT_JSON}")


def main():
    base_train = load_json(BASE_TRAIN)
    ft_train   = load_json(FT_TRAIN)
    base_infer = load_json(BASE_INFER)
    ft_infer   = load_json(FT_INFER)

    print_table(base_train, ft_train, base_infer, ft_infer)

    # Chart 1 — Training Accuracy
    base_tr = base_train.get("train_accuracy", base_train.get("best_val_accuracy", 0))
    ft_tr   = ft_train.get("train_accuracy",   ft_train.get("best_val_accuracy", 0))
    save_bar_chart(
        base_tr, ft_tr,
        title="Training Accuracy — Base vs Fine-Tuned (CIFAR-10)",
        ylabel="Training Accuracy (%)",
        out_path=OUT_TRAIN_PNG,
    )

    # Chart 2 — Inference Accuracy
    save_bar_chart(
        base_infer["test_accuracy"],
        ft_infer["test_accuracy"],
        title="Inference Accuracy — Base vs Fine-Tuned (CIFAR-10)",
        ylabel="Inference Accuracy (%)",
        out_path=OUT_INFER_PNG,
    )

    save_combined_json(base_train, ft_train, base_infer, ft_infer)


if __name__ == "__main__":
    main()
