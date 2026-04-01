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


def structured_prompt_text(prompt_text: str, num_objects: int | None = None) -> str:
    if num_objects is None:
        return f"An image containing: {prompt_text}. Show all clearly."
    return f"An image containing exactly {int(num_objects)} objects: {prompt_text}. Show all clearly."


def _evaluate_image(
    *,
    method: str,
    prompt_id: int,
    complexity,
    attempt: int,
    prompt_text: str,
    requested_objects: list[str],
    image_path,
    conf_threshold: float,
    detector_model: str,
    total_gens: int,
    selected: bool,
) -> dict:
    detected = detect_objects(
        image_path=image_path,
        conf_threshold=conf_threshold,
        model_name=detector_model,
    )
    comparison = compare_objects(requested_objects, detected)
    return {
        "method": method,
        "prompt_id": prompt_id,
        "complexity": complexity,
        "attempt": attempt,
        "recall": comparison["recall_score"],
        "recall_score": comparison["recall_score"],
        "success": comparison["recall_score"] == 1.0,
        "current_prompt": prompt_text,
        "image_path": str(image_path),
        "detected_objects": stringify_items(comparison["detected_objects"]),
        "missing_objects": stringify_items(comparison["missing_objects"]),
        "selected": selected,
        "total_gens": total_gens,
        "status": "success" if comparison["recall_score"] == 1.0 else "evaluated",
    }


def run_vanilla_experiment(
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
    base_seed = int(config["generation"]["seed"])

    rows = []
    for _, row in prompts_df.iterrows():
        prompt_id = int(row["prompt_id"])
        prompt_text = str(row["prompt_text"])
        seed = seed_for_attempt(prompt_id, 1, base_seed)
        image_path = generate_for_prompt("vanilla", prompt_id, prompt_text, seed, attempt_number=1, config=config)
        rows.append(
            _evaluate_image(
                method="vanilla",
                prompt_id=prompt_id,
                complexity=row.get("complexity", ""),
                attempt=1,
                prompt_text=prompt_text,
                requested_objects=list(row["requested_objects"]),
                image_path=image_path,
                conf_threshold=conf_threshold,
                detector_model=detector_model,
                total_gens=1,
                selected=True,
            )
        )

    df = pd.DataFrame(rows)
    output_path = get_results_path(config, output_csv or "vanilla_detection.csv")
    df.to_csv(output_path, index=False)
    print(f"Saved vanilla results to {output_path}")
    return output_path


def run_structured_experiment(
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
    base_seed = int(config["generation"]["seed"])

    rows = []
    for _, row in prompts_df.iterrows():
        prompt_id = int(row["prompt_id"])
        prompt_text = structured_prompt_text(str(row["prompt_text"]), row.get("num_objects"))
        seed = seed_for_attempt(prompt_id, 1, base_seed)
        image_path = generate_for_prompt(
            "structured",
            prompt_id,
            prompt_text,
            seed,
            attempt_number=1,
            config=config,
        )
        rows.append(
            _evaluate_image(
                method="structured",
                prompt_id=prompt_id,
                complexity=row.get("complexity", ""),
                attempt=1,
                prompt_text=prompt_text,
                requested_objects=list(row["requested_objects"]),
                image_path=image_path,
                conf_threshold=conf_threshold,
                detector_model=detector_model,
                total_gens=1,
                selected=True,
            )
        )

    df = pd.DataFrame(rows)
    output_path = get_results_path(config, output_csv or "structured_detection.csv")
    df.to_csv(output_path, index=False)
    print(f"Saved structured results to {output_path}")
    return output_path


def run_rejection_experiment(
    config_path: str | Path | None = None,
    *,
    limit_ids: list[int] | None = None,
    output_csv: str | Path | None = None,
    k: int | None = None,
) -> Path:
    config = load_config(config_path)
    prompts_df = load_prompts(config)
    if limit_ids:
        prompts_df = prompts_df[prompts_df["prompt_id"].isin(limit_ids)].copy()

    conf_threshold = float(config["detection"]["confidence_threshold"])
    detector_model = str(config["detection"]["model"])
    base_seed = int(config["generation"]["seed"])
    k = int(k or config["experiments"]["rejection_k"])

    all_rows = []
    for _, row in prompts_df.iterrows():
        prompt_id = int(row["prompt_id"])
        prompt_text = str(row["prompt_text"])
        prompt_rows = []

        for attempt in range(1, k + 1):
            seed = seed_for_attempt(prompt_id, attempt, base_seed)
            image_path = generate_for_prompt(
                "rejection_k5" if k == 5 else f"rejection_k{k}",
                prompt_id,
                prompt_text,
                seed,
                attempt_number=attempt,
                config=config,
            )
            prompt_rows.append(
                _evaluate_image(
                    method="rejection_k5" if k == 5 else f"rejection_k{k}",
                    prompt_id=prompt_id,
                    complexity=row.get("complexity", ""),
                    attempt=attempt,
                    prompt_text=prompt_text,
                    requested_objects=list(row["requested_objects"]),
                    image_path=image_path,
                    conf_threshold=conf_threshold,
                    detector_model=detector_model,
                    total_gens=k,
                    selected=False,
                )
            )

        if prompt_rows:
            best_idx = max(range(len(prompt_rows)), key=lambda idx: prompt_rows[idx]["recall_score"])
            prompt_rows[best_idx]["selected"] = True
        all_rows.extend(prompt_rows)

    df = pd.DataFrame(all_rows)
    output_path = get_results_path(config, output_csv or "rejection_5_results_CLEAN.csv")
    df.to_csv(output_path, index=False)
    print(f"Saved rejection results to {output_path}")
    return output_path
