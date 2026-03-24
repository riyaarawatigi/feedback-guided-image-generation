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
    results_output_path,
    save_results,
)


def run_rejection_sampling(
    *,
    config_path: str | Path = "config.yaml",
    limit: int | None = None,
) -> list[Path]:
    config = load_config(config_path)
    prompts_df = load_prompts(config)
    if limit is not None:
        prompts_df = prompts_df.head(limit)

    base_seed = int(config["generation"]["seed"])
    output_paths: list[Path] = []

    for rejection_k in config["experiments"]["rejection_k"]:
        rows: list[dict] = []
        for _, prompt_row in prompts_df.iterrows():
            prompt_id = int(prompt_row["prompt_id"])
            prompt_text = str(prompt_row["prompt_text"])
            requested_objects = list(prompt_row["requested_objects"])
            per_prompt_rows: list[dict] = []

            for attempt_number in range(1, int(rejection_k) + 1):
                seed = base_seed + prompt_id * 100 + attempt_number
                image_path = generate_for_prompt(
                    f"rejection_{rejection_k}",
                    prompt_id,
                    prompt_text,
                    seed,
                    attempt_number=attempt_number,
                    config=config,
                )
                detection = detect_image(
                    image_path,
                    requested_objects=requested_objects,
                    config=config,
                )
                per_prompt_rows.append(
                    {
                        "method": f"rejection_{rejection_k}",
                        "prompt_id": prompt_id,
                        "prompt_text": prompt_text,
                        "prompt_used": prompt_text,
                        "complexity": prompt_row["complexity"],
                        "attempt_number": attempt_number,
                        "requested_objects": requested_objects,
                        "detected_objects": detection["detected_objects"],
                        "missing_objects": detection["missing_objects"],
                        "recall_score": detection["recall_score"],
                        "success": detection["success"],
                        "image_path": str(image_path),
                        "seed": seed,
                        "selected": False,
                        "total_generations": int(rejection_k),
                    }
                )

            best_row = choose_best_result(per_prompt_rows)
            for row in per_prompt_rows:
                row["selected"] = row is best_row
            rows.extend(per_prompt_rows)

        output_path = results_output_path(config, f"rejection_{rejection_k}_results.csv")
        output_paths.append(save_results(rows, output_path))

    return output_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Run rejection sampling baselines.")
    parser.add_argument("--config", default="config.yaml", help="Path to config file.")
    parser.add_argument("--limit", type=int, help="Limit to the first N prompts.")
    args = parser.parse_args()

    outputs = run_rejection_sampling(config_path=args.config, limit=args.limit)
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
