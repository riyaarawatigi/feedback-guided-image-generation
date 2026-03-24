from __future__ import annotations

import argparse
from pathlib import Path

if __package__ in (None, ""):
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.detector import detect_image
from scripts.generator import generate_for_prompt
from scripts.helpers import (
    choose_best_result,
    load_config,
    load_prompts,
    refine_prompt_text,
    results_output_path,
    save_results,
)


def run_feedback_method(
    *,
    config_path: str | Path = "config.yaml",
    limit: int | None = None,
) -> Path:
    config = load_config(config_path)
    prompts_df = load_prompts(config)
    if limit is not None:
        prompts_df = prompts_df.head(limit)

    max_rounds = int(config["experiments"]["feedback_rounds"])
    batch_size = int(config["experiments"]["batch_size"])
    base_seed = int(config["generation"]["seed"])
    all_rows: list[dict] = []

    for _, prompt_row in prompts_df.iterrows():
        prompt_id = int(prompt_row["prompt_id"])
        original_prompt = str(prompt_row["prompt_text"])
        current_prompt = original_prompt
        requested_objects = list(prompt_row["requested_objects"])
        prompt_rows: list[dict] = []
        stopped_reason = "max_rounds"
        best_overall: dict | None = None

        for round_number in range(1, max_rounds + 1):
            round_rows: list[dict] = []
            for attempt_number in range(1, batch_size + 1):
                seed = base_seed + prompt_id * 1000 + round_number * 100 + attempt_number
                image_path = generate_for_prompt(
                    "feedback",
                    prompt_id,
                    current_prompt,
                    seed,
                    round_number=round_number,
                    attempt_number=attempt_number,
                    config=config,
                )
                detection = detect_image(
                    image_path,
                    requested_objects=requested_objects,
                    config=config,
                )
                row = {
                    "method": "feedback",
                    "prompt_id": prompt_id,
                    "prompt_text": original_prompt,
                    "refined_prompt": current_prompt,
                    "complexity": prompt_row["complexity"],
                    "round_number": round_number,
                    "attempt_number": attempt_number,
                    "requested_objects": requested_objects,
                    "detected_objects": detection["detected_objects"],
                    "missing_objects": detection["missing_objects"],
                    "recall_score": detection["recall_score"],
                    "success": detection["success"],
                    "image_path": str(image_path),
                    "seed": seed,
                    "selected_in_round": False,
                    "final_selected": False,
                    "total_generations": 0,
                    "stopped_reason": "",
                }
                round_rows.append(row)

            round_best = choose_best_result(round_rows)
            for row in round_rows:
                row["selected_in_round"] = row is round_best

            if best_overall is None:
                best_overall = round_best
            else:
                best_overall = choose_best_result([best_overall, round_best])

            prompt_rows.extend(round_rows)

            if round_best["success"]:
                stopped_reason = "success"
                break

            current_prompt = refine_prompt_text(current_prompt, round_best["missing_objects"])

        total_generations = len(prompt_rows)
        if best_overall is None:
            continue

        for row in prompt_rows:
            row["total_generations"] = total_generations
            row["stopped_reason"] = stopped_reason
            row["final_selected"] = row is best_overall
        all_rows.extend(prompt_rows)

    output_path = results_output_path(config, "feedback_results.csv")
    return save_results(all_rows, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the feedback-guided image generation method.")
    parser.add_argument("--config", default="config.yaml", help="Path to config file.")
    parser.add_argument("--limit", type=int, help="Limit to the first N prompts.")
    args = parser.parse_args()

    output = run_feedback_method(config_path=args.config, limit=args.limit)
    print(output)


if __name__ == "__main__":
    main()
