from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_method_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "recall_score" not in df.columns and "recall" in df.columns:
        df = df.rename(columns={"recall": "recall_score"})
    if "success" not in df.columns and "recall_score" in df.columns:
        df["success"] = df["recall_score"] >= 1.0
    return df


def select_final_rows(df: pd.DataFrame) -> pd.DataFrame:
    working = df.copy()
    if "selected" in working.columns:
        selected = working[working["selected"] == True].copy()
        if not selected.empty:
            return selected.reset_index(drop=True)

    if {"prompt_id", "attempt"}.issubset(working.columns):
        return (
            working.sort_values(["prompt_id", "attempt"])
            .groupby("prompt_id", as_index=False)
            .last()
            .reset_index(drop=True)
        )

    return working.reset_index(drop=True)


def summarize_method_df(df: pd.DataFrame, name: str, avg_gens_fixed: float | None = None) -> dict:
    final_rows = select_final_rows(df)

    avg_recall = float(final_rows["recall_score"].mean()) if "recall_score" in final_rows.columns else 0.0
    success_rate = float((final_rows["success"]).mean() * 100.0) if "success" in final_rows.columns else 0.0

    if avg_gens_fixed is not None:
        avg_gens = float(avg_gens_fixed)
    elif "total_gens" in final_rows.columns:
        avg_gens = float(final_rows["total_gens"].mean())
    elif "attempt" in final_rows.columns:
        avg_gens = float(final_rows["attempt"].mean())
    else:
        avg_gens = 1.0

    efficiency = success_rate / avg_gens if avg_gens else 0.0

    return {
        "Method": name,
        "Avg Recall": round(avg_recall, 3),
        "Success Rate (%)": round(success_rate, 1),
        "Avg Generations": round(avg_gens, 2),
        "Efficiency": round(efficiency, 2),
    }


def summarize_method_file(path: str | Path, name: str, avg_gens_fixed: float | None = None) -> dict:
    return summarize_method_df(load_method_csv(path), name, avg_gens_fixed=avg_gens_fixed)


def compare_methods(method_files: dict[str, str | Path]) -> pd.DataFrame:
    rows = []
    for name, path in method_files.items():
        avg_fixed = None
        lowered = name.lower()
        if "vanilla" in lowered or "structured" in lowered:
            avg_fixed = 1
        elif "rejection" in lowered:
            avg_fixed = 5
        rows.append(summarize_method_file(path, name, avg_gens_fixed=avg_fixed))
    return pd.DataFrame(rows)


def complexity_breakdown(df: pd.DataFrame, name: str) -> pd.DataFrame:
    final_rows = select_final_rows(df)
    if "complexity" not in final_rows.columns:
        raise ValueError("DataFrame does not contain a 'complexity' column.")

    def _avg_gens(group: pd.DataFrame) -> float:
        if "total_gens" in group.columns:
            return float(group["total_gens"].mean())
        if "attempt" in group.columns:
            return float(group["attempt"].mean())
        return 1.0

    rows = []
    for complexity, group in final_rows.groupby("complexity"):
        rows.append(
            {
                "Method": name,
                "Complexity": complexity,
                "Success Rate (%)": round(float(group["success"].mean() * 100.0), 1),
                "Avg Recall": round(float(group["recall_score"].mean()), 3),
                "Avg Generations": round(_avg_gens(group), 2),
            }
        )
    return pd.DataFrame(rows).sort_values("Complexity").reset_index(drop=True)
