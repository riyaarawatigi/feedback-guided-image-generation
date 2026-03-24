from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

if __package__ in (None, ""):
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.detector import detect_image
from scripts.generator import generate_for_prompt
from scripts.helpers import load_config, load_prompts, results_output_path, save_results


PromptTransform = Callable[[str, list[str]], str]


def run_baseline(
    *,
    method: str = "vanilla",
    transform_prompt: PromptTransform | None = None,
    output_filename: str | None = None,
    config_path: str | Path = "config.yaml",
    limit: int | None = None,
) -> Path:
    config = load_config(config_path)
    prompts_df = load_prompts(config)
    if limit is not None:
        prompts_df = prompts_df.head(limit)

    output_path = results_output_path(
        config,
        output_filename or f"{method}_results.csv",
    )
    rows: list[dict] = []
    base_seed = int(config["generation"]["seed"])

    for _, prompt_row in prompts_df.iterrows():
        prompt_id = int(prompt_row["prompt_id"])
        requested_objects = list(prompt_row["requested_objects"])
        original_prompt = str(prompt_row["prompt_text"])
        prompt_used = (
            transform_prompt(original_prompt, requested_objects)
            if transform_prompt is not None
            else original_prompt
        )
        seed = base_seed + prompt_id
        image_path = generate_for_prompt(
            method,
            prompt_id,
            prompt_used,
            seed,
            config=config,
        )
        detection = detect_image(
            image_path,
            requested_objects=requested_objects,
            config=config,
        )
        rows.append(
            {
                "method": method,
                "prompt_id": prompt_id,
                "prompt_text": original_prompt,
                "prompt_used": prompt_used,
                "complexity": prompt_row["complexity"],
                "requested_objects": requested_objects,
                "detected_objects": detection["detected_objects"],
                "missing_objects": detection["missing_objects"],
                "recall_score": detection["recall_score"],
                "success": detection["success"],
                "image_path": str(image_path),
                "seed": seed,
                "total_generations": 1,
            }
        )

    return save_results(rows, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the vanilla one-shot baseline.")
    parser.add_argument("--config", default="config.yaml", help="Path to config file.")
    parser.add_argument("--limit", type=int, help="Limit to the first N prompts.")
    args = parser.parse_args()

    output_path = run_baseline(config_path=args.config, limit=args.limit)
    print(output_path)


if __name__ == "__main__":
    main()
