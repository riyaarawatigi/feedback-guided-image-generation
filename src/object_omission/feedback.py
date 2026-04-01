from __future__ import annotations

from pathlib import Path

import pandas as pd

from .detection import compare_objects, detect_objects
from .generator import generate_for_prompt
from .helpers import (
    get_results_path,
    load_config,
    load_prompts,
    seed_for_attempt,
    stringify_items,
)


def refine_prompt_text(original_prompt, missing_objects):
    if not missing_objects:
        return original_prompt
    missing_str = ", ".join(missing_objects)
    return f"{original_prompt}. Make sure to clearly include the {missing_str}."


def run_feedback_experiment(
    config_path: str | Path | None = None,
    *,
    limit_ids: list[int] | None = None,
    output_csv: str | Path | None = None,
) -> Path:
    config = load_config(config_path)
    prompts_df = load_prompts(config)

    if limit_ids:
        prompts_df = prompts_df[prompts_df["prompt_id"].isin(limit_ids)].copy()

    conf_threshold = float(config["detection"]["confidence_threshold"])
    detector_model = str(config["detection"]["model"])
    max_attempts = int(config["experiments"]["max_feedback_attempts"])
    base_seed = int(config["generation"]["seed"])

    all_rows: list[dict] = []

    for _, row in prompts_df.iterrows():
        prompt_id = int(row["prompt_id"])
        original_prompt = str(row["prompt_text"])
        current_prompt = original_prompt
        requested_objects = list(row["requested_objects"])
        complexity = row.get("complexity", "")
        prompt_rows = []
        best_idx = None
        best_recall = -1.0
        selected_idx = None
        final_status = "max_attempts"

        print(f"\n[feedback] Prompt {prompt_id}: {original_prompt}")

        for attempt in range(1, max_attempts + 1):
            seed = seed_for_attempt(prompt_id, attempt, base_seed)
            image_path = generate_for_prompt(
                "feedback",
                prompt_id,
                current_prompt,
                seed,
                attempt_number=attempt,
                config=config,
            )

            detected = detect_objects(
                image_path=image_path,
                conf_threshold=conf_threshold,
                model_name=detector_model,
            )
            comparison = compare_objects(requested_objects, detected)
            recall = comparison["recall_score"]
            success = recall == 1.0

            row_result = {
                "method": "feedback",
                "prompt_id": prompt_id,
                "complexity": complexity,
                "attempt": attempt,
                "recall": recall,
                "recall_score": recall,
                "success": success,
                "current_prompt": current_prompt,
                "image_path": str(image_path),
                "detected_objects": stringify_items(comparison["detected_objects"]),
                "missing_objects": stringify_items(comparison["missing_objects"]),
                "selected": False,
                "total_gens": None,
                "status": "success" if success else "continue",
            }
            prompt_rows.append(row_result)

            if recall > best_recall:
                best_recall = recall
                best_idx = len(prompt_rows) - 1

            if success:
                selected_idx = len(prompt_rows) - 1
                final_status = "success"
                print(f"  success on attempt {attempt}")
                break

            current_prompt = refine_prompt_text(original_prompt, comparison["missing_objects"])
            print(f"  attempt {attempt}: missing={comparison['missing_objects']} -> refine")

        if selected_idx is None:
            selected_idx = best_idx if best_idx is not None else len(prompt_rows) - 1

        total_gens = len(prompt_rows)
        for idx, result_row in enumerate(prompt_rows):
            result_row["selected"] = idx == selected_idx
            result_row["total_gens"] = total_gens
            if idx == selected_idx:
                result_row["status"] = final_status
            elif result_row["status"] != "success":
                result_row["status"] = "non_selected"

        all_rows.extend(prompt_rows)

    results_df = pd.DataFrame(all_rows)
    output_path = get_results_path(config, output_csv or "feedback_realtime_results.csv")
    results_df.to_csv(output_path, index=False)
    print(f"\nSaved feedback results to {output_path}")
    return output_path
