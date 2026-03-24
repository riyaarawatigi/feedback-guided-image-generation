from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

if __package__ in (None, ""):
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.analyzer import analyze_methods
from scripts.helpers import get_results_root, load_config


def create_tables(config_path: str | Path = "config.yaml") -> dict[str, Path]:
    config = load_config(config_path)
    analyze_methods(config_path)

    results_root = get_results_root(config)
    tables_dir = results_root / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    final_metrics = pd.read_csv(results_root / "final_metrics.csv")
    complexity_metrics = pd.read_csv(results_root / "complexity_metrics.csv")

    main_table = final_metrics.rename(
        columns={
            "method": "Method",
            "mean_recall": "Mean Recall",
            "success_rate": "Success Rate",
            "avg_generations": "Avg Generations",
            "num_prompts": "Num Prompts",
        }
    )
    complexity_table = complexity_metrics.rename(
        columns={
            "method": "Method",
            "complexity": "Complexity",
            "mean_recall": "Mean Recall",
            "success_rate": "Success Rate",
            "avg_generations": "Avg Generations",
            "num_prompts": "Num Prompts",
        }
    )

    main_path = tables_dir / "main_comparison_table.csv"
    complexity_path = tables_dir / "complexity_breakdown_table.csv"
    main_table.to_csv(main_path, index=False)
    complexity_table.to_csv(complexity_path, index=False)

    return {
        "main_table": main_path,
        "complexity_table": complexity_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create report-ready comparison tables.")
    parser.add_argument("--config", default="config.yaml", help="Path to config file.")
    args = parser.parse_args()

    outputs = create_tables(args.config)
    for label, output_path in outputs.items():
        print(f"{label}: {output_path}")


if __name__ == "__main__":
    main()
