from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

if __package__ in (None, ""):
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.helpers import (
    compute_method_metrics,
    get_results_root,
    load_config,
    load_prompts,
)


def _results_candidate_paths(config: dict, method: str) -> list[Path]:
    results_root = get_results_root(config)
    repo_root = Path(config["paths"]["prompts"]).resolve().parents[1]

    mapping = {
        "vanilla": [
            results_root / "vanilla_results.csv",
            repo_root / "data" / "vanilla_detection.csv",
        ],
        "structured": [
            results_root / "structured_results.csv",
            repo_root / "data" / "structured_detection.csv",
        ],
        "rejection_5": [results_root / "rejection_5_results.csv"],
        "rejection_10": [results_root / "rejection_10_results.csv"],
        "feedback": [results_root / "feedback_results.csv"],
    }
    return mapping[method]


def load_method_results(config: dict, method: str) -> pd.DataFrame:
    for candidate in _results_candidate_paths(config, method):
        if candidate.exists():
            df = pd.read_csv(candidate)
            df["source_path"] = str(candidate)
            return normalize_method_results(df, method)
    return pd.DataFrame()


def normalize_method_results(df: pd.DataFrame, method: str) -> pd.DataFrame:
    if df.empty:
        return df

    normalized = df.copy()
    normalized["method"] = method
    normalized["prompt_id"] = normalized["prompt_id"].astype(int)
    normalized["recall_score"] = normalized["recall_score"].astype(float)

    if "requested_objects" not in normalized.columns:
        normalized["requested_objects"] = ""
    if "detected_objects" not in normalized.columns:
        normalized["detected_objects"] = ""
    if "missing_objects" not in normalized.columns:
        normalized["missing_objects"] = ""

    if "success" not in normalized.columns:
        normalized["success"] = normalized["recall_score"] >= 1.0
    else:
        normalized["success"] = normalized["success"].astype(bool)

    if "total_generations" not in normalized.columns:
        defaults = {
            "vanilla": 1,
            "structured": 1,
            "rejection_5": 5,
            "rejection_10": 10,
            "feedback": 3,
        }
        normalized["total_generations"] = defaults[method]

    if method.startswith("rejection_") and "selected" in normalized.columns:
        normalized = normalized[normalized["selected"].astype(bool)]
    elif method == "feedback" and "final_selected" in normalized.columns:
        normalized = normalized[normalized["final_selected"].astype(bool)]

    normalized = normalized.drop_duplicates(subset=["prompt_id"], keep="first")
    return normalized.reset_index(drop=True)


def analyze_methods(config_path: str | Path = "config.yaml") -> dict[str, Path]:
    config = load_config(config_path)
    prompts_df = load_prompts(config)[["prompt_id", "complexity", "num_objects"]]

    method_frames = []
    for method in ["vanilla", "structured", "rejection_5", "rejection_10", "feedback"]:
        method_df = load_method_results(config, method)
        if not method_df.empty:
            method_frames.append(method_df)

    if not method_frames:
        raise ValueError("No result files were found to analyze.")

    combined = pd.concat(method_frames, ignore_index=True)
    combined = combined.merge(prompts_df, on="prompt_id", how="left")

    overall_rows = []
    for method, method_df in combined.groupby("method"):
        metrics = compute_method_metrics(method_df)
        overall_rows.append({"method": method, **metrics})
    overall_df = pd.DataFrame(overall_rows).sort_values("method")

    complexity_rows = []
    for (method, complexity), subset in combined.groupby(["method", "complexity"]):
        metrics = compute_method_metrics(subset)
        complexity_rows.append(
            {
                "method": method,
                "complexity": complexity,
                **metrics,
            }
        )
    complexity_df = pd.DataFrame(complexity_rows).sort_values(["method", "complexity"])

    results_root = get_results_root(config)
    results_root.mkdir(parents=True, exist_ok=True)
    per_prompt_path = results_root / "per_prompt_summary.csv"
    metrics_path = results_root / "final_metrics.csv"
    complexity_path = results_root / "complexity_metrics.csv"

    combined.to_csv(per_prompt_path, index=False)
    overall_df.to_csv(metrics_path, index=False)
    complexity_df.to_csv(complexity_path, index=False)

    return {
        "per_prompt_summary": per_prompt_path,
        "final_metrics": metrics_path,
        "complexity_metrics": complexity_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate experiment metrics across all methods.")
    parser.add_argument("--config", default="config.yaml", help="Path to config file.")
    args = parser.parse_args()

    outputs = analyze_methods(args.config)
    for label, output_path in outputs.items():
        print(f"{label}: {output_path}")


if __name__ == "__main__":
    main()
