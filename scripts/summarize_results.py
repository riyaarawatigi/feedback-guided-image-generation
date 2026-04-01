from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from object_omission.helpers import load_config, resolve_repo_path
from object_omission.metrics import compare_methods, complexity_breakdown, load_method_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize experiment CSV files.")
    parser.add_argument("--config", default="config.yaml", help="Path to config file.")
    parser.add_argument("--vanilla", default="results/vanilla_detection.csv")
    parser.add_argument("--structured", default="results/structured_detection.csv")
    parser.add_argument("--rejection", default="results/rejection_5_results_CLEAN.csv")
    parser.add_argument("--feedback", default="results/feedback_realtime_results.csv")
    parser.add_argument("--output", default="results/summary_table.csv")
    parser.add_argument("--complexity-output", default="results/complexity_breakdown.csv")
    args = parser.parse_args()

    method_files = {
        "Vanilla (SD 1.5)": resolve_repo_path(args.vanilla),
        "Structured Prompt": resolve_repo_path(args.structured),
        "Rejection Sampling (k=5)": resolve_repo_path(args.rejection),
        "Feedback-Guided (Ours)": resolve_repo_path(args.feedback),
    }
    summary_df = compare_methods(method_files)
    output_path = resolve_repo_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_path, index=False)
    print("\n--- SUMMARY TABLE ---")
    print(summary_df.to_string(index=False))
    print(f"\nSaved summary table to {output_path}")

    breakdown_frames = []
    for method_name, csv_path in method_files.items():
        df = load_method_csv(csv_path)
        if "complexity" in df.columns:
            breakdown_frames.append(complexity_breakdown(df, method_name))
    if breakdown_frames:
        complexity_df = pd.concat(breakdown_frames, ignore_index=True)
        complexity_output = resolve_repo_path(args.complexity_output)
        complexity_df.to_csv(complexity_output, index=False)
        print(f"Saved complexity breakdown to {complexity_output}")


if __name__ == "__main__":
    main()
