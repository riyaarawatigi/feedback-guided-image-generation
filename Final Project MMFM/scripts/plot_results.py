from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from object_omission.helpers import resolve_repo_path
from object_omission.metrics import compare_methods, load_method_csv


def _final_rows(df: pd.DataFrame) -> pd.DataFrame:
    if "selected" in df.columns:
        selected = df[df["selected"] == True].copy()
        if not selected.empty:
            return selected
    if {"prompt_id", "attempt"}.issubset(df.columns):
        return df.sort_values(["prompt_id", "attempt"]).groupby("prompt_id", as_index=False).last()
    return df.copy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Create comparison plots from result CSVs.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--vanilla", default="results/vanilla_detection.csv")
    parser.add_argument("--structured", default="results/structured_detection.csv")
    parser.add_argument("--rejection", default="results/rejection_5_results_CLEAN.csv")
    parser.add_argument("--feedback", default="results/feedback_realtime_results.csv")
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    method_files = {
        "Vanilla": resolve_repo_path(args.vanilla),
        "Structured": resolve_repo_path(args.structured),
        "Rejection k=5": resolve_repo_path(args.rejection),
        "Feedback": resolve_repo_path(args.feedback),
    }

    summary_df = compare_methods(method_files)
    out_dir = resolve_repo_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Figure 1: success vs computational cost
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.scatter(summary_df["Avg Generations"], summary_df["Success Rate (%)"], s=140)
    for _, row in summary_df.iterrows():
        ax.text(row["Avg Generations"] + 0.03, row["Success Rate (%)"] + 1.0, row["Method"])
    ax.set_xlabel("Average Generations per Prompt")
    ax.set_ylabel("Success Rate (%)")
    ax.set_title("Accuracy vs Computational Cost")
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "accuracy_vs_cost.png", dpi=200)
    plt.close(fig)

    # Figure 2: complexity breakdown for success rates
    complexity_rows = []
    order = {"simple": 0, "medium": 1, "hard": 2}
    for method_name, csv_path in method_files.items():
        df = _final_rows(load_method_csv(csv_path))
        if "complexity" not in df.columns:
            continue
        for complexity, group in df.groupby("complexity"):
            complexity_rows.append(
                {
                    "Method": method_name,
                    "Complexity": complexity,
                    "Success Rate (%)": (group["success"].mean() * 100.0),
                }
            )
    if complexity_rows:
        complexity_df = pd.DataFrame(complexity_rows)
        complexity_df["sort_key"] = complexity_df["Complexity"].map(order).fillna(999)
        complexity_df = complexity_df.sort_values(["sort_key", "Method"])
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        for method_name, group in complexity_df.groupby("Method"):
            group = group.sort_values("sort_key")
            ax.plot(group["Complexity"], group["Success Rate (%)"], marker="o", label=method_name)
        ax.set_xlabel("Prompt Complexity")
        ax.set_ylabel("Success Rate (%)")
        ax.set_title("Success Rate by Prompt Complexity")
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "success_by_complexity.png", dpi=200)
        plt.close(fig)

    # Figure 3: bar chart summary
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111)
    ax.bar(summary_df["Method"], summary_df["Success Rate (%)"])
    ax.set_ylabel("Success Rate (%)")
    ax.set_title("Method Comparison")
    ax.set_ylim(0, 100)
    ax.tick_params(axis="x", rotation=20)
    for i, value in enumerate(summary_df["Success Rate (%)"]):
        ax.text(i, value + 1.0, f"{value:.1f}%", ha="center")
    fig.tight_layout()
    fig.savefig(out_dir / "method_comparison_bar.png", dpi=200)
    plt.close(fig)

    print(f"Saved plots to {out_dir}")


if __name__ == "__main__":
    main()
