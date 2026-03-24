from __future__ import annotations

import argparse
import os
from pathlib import Path

if __package__ in (None, ""):
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))

REPO_ROOT = Path(__file__).resolve().parents[1]
mpl_config_dir = REPO_ROOT / ".cache" / "matplotlib"
mpl_config_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from scripts.analyzer import analyze_methods
from scripts.helpers import get_results_root, load_config


def create_figures(config_path: str | Path = "config.yaml") -> dict[str, Path]:
    config = load_config(config_path)
    analyze_methods(config_path)

    results_root = get_results_root(config)
    figures_dir = results_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    final_metrics = pd.read_csv(results_root / "final_metrics.csv")
    complexity_metrics = pd.read_csv(results_root / "complexity_metrics.csv")
    per_prompt = pd.read_csv(results_root / "per_prompt_summary.csv")

    sns.set_theme(style="whitegrid")
    outputs: dict[str, Path] = {}

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=final_metrics, x="method", y="mean_recall", ax=ax)
    ax.set_title("Mean Recall by Method")
    ax.set_xlabel("Method")
    ax.set_ylabel("Mean Recall")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    recall_path = figures_dir / "mean_recall_by_method.png"
    fig.savefig(recall_path, dpi=200)
    plt.close(fig)
    outputs["mean_recall"] = recall_path

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=final_metrics, x="method", y="avg_generations", ax=ax)
    ax.set_title("Average Generations by Method")
    ax.set_xlabel("Method")
    ax.set_ylabel("Average Generations")
    fig.tight_layout()
    generations_path = figures_dir / "avg_generations_by_method.png"
    fig.savefig(generations_path, dpi=200)
    plt.close(fig)
    outputs["avg_generations"] = generations_path

    ordered_complexity = ["simple", "medium", "hard"]
    complexity_metrics["complexity"] = pd.Categorical(
        complexity_metrics["complexity"],
        categories=ordered_complexity,
        ordered=True,
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(
        data=complexity_metrics.sort_values("complexity"),
        x="complexity",
        y="mean_recall",
        hue="method",
        marker="o",
        ax=ax,
    )
    ax.set_title("Mean Recall by Complexity")
    ax.set_xlabel("Complexity")
    ax.set_ylabel("Mean Recall")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    complexity_path = figures_dir / "mean_recall_by_complexity.png"
    fig.savefig(complexity_path, dpi=200)
    plt.close(fig)
    outputs["mean_recall_by_complexity"] = complexity_path

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=final_metrics, x="method", y="success_rate", ax=ax)
    ax.set_title("Success Rate by Method")
    ax.set_xlabel("Method")
    ax.set_ylabel("Success Rate")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    success_path = figures_dir / "success_rate_by_method.png"
    fig.savefig(success_path, dpi=200)
    plt.close(fig)
    outputs["success_rate"] = success_path

    if "recall_score" in per_prompt.columns and "num_objects" in per_prompt.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(data=per_prompt, x="num_objects", y="recall_score", hue="method", ax=ax)
        ax.set_title("Recall Distribution by Object Count")
        ax.set_xlabel("Number of Requested Objects")
        ax.set_ylabel("Recall Score")
        ax.set_ylim(0, 1)
        fig.tight_layout()
        distribution_path = figures_dir / "recall_distribution_by_object_count.png"
        fig.savefig(distribution_path, dpi=200)
        plt.close(fig)
        outputs["recall_distribution"] = distribution_path

    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Create report-ready figures.")
    parser.add_argument("--config", default="config.yaml", help="Path to config file.")
    args = parser.parse_args()

    outputs = create_figures(args.config)
    for label, output_path in outputs.items():
        print(f"{label}: {output_path}")


if __name__ == "__main__":
    main()
