"""Regenerate figures from saved CSV files, without rerunning the experiment."""
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/sofia-kv-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statistics


def main():
    figures = Path("results/figures"); figures.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False})
    with open("results/training.csv") as f:
        training = list(csv.DictReader(f))
    fig, ax = plt.subplots(figsize=(8, 3.5), layout="constrained")
    steps = [int(r["step"]) for r in training]
    for name, label, color in [("train_loss", "Training", "#777777"), ("val_loss", "Validation", "#245a81")]:
        ax.plot(steps, [float(r[name]) for r in training], marker="o", markersize=3, label=label, color=color)
    ax.set(xlabel="Optimizer steps", ylabel="Cross-entropy (nats / character)")
    ax.legend(frameon=False)
    fig.savefig(figures / "training.png", dpi=180); plt.close(fig)
    with open("results/benchmark.csv") as f:
        rows = list(csv.DictReader(f))
    fig, ax = plt.subplots(figsize=(9, 3.4), layout="constrained")
    for mode, delta, color, label in [("uncached", -0.18, "#999999", "Recompute history"),
                                      ("cached", 0.18, "#245a81", "KV cache")]:
        groups = [[63_000 / float(r["decode_ms"]) for r in rows
                   if r["mode"] == mode and int(r["prompt_tokens"]) == p] for p in (16, 32, 64)]
        medians = [statistics.median(g) for g in groups]
        errors = [[m - min(g) for m, g in zip(medians, groups)],
                  [max(g) - m for m, g in zip(medians, groups)]]
        bars = ax.bar([i + delta for i in range(3)], medians, width=0.34, color=color,
                      label=label, yerr=errors, capsize=4)
        ax.bar_label(bars, fmt="%.0f", padding=5, fontsize=11)
    ax.set_xticks(range(3), ["16", "32", "64"])
    ax.set(xlabel="Prompt length (characters)", ylabel="Decode predictions / second")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
    ax.legend(frameon=False, loc="upper left")
    fig.savefig(figures / "decode_speed.png", dpi=180); plt.close(fig)
    print("Saved training.png and decode_speed.png")


if __name__ == "__main__":
    main()
