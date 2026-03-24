from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config.yaml"

DEFAULT_CONFIG: dict[str, Any] = {
    "generation": {
        "model": "stabilityai/stable-diffusion-2-1",
        "image_size": 512,
        "steps": 50,
        "guidance": 7.5,
        "device": "auto",
        "dtype": "auto",
        "seed": 42,
    },
    "detection": {
        "model": "yolov8n.pt",
        "confidence": 0.25,
    },
    "experiments": {
        "rejection_k": [5, 10],
        "feedback_rounds": 3,
        "batch_size": 3,
    },
    "paths": {
        "prompts": "data/prompts.csv",
        "images": "generated_images",
        "results": "results",
    },
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def resolve_repo_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    config_file = resolve_repo_path(config_path or DEFAULT_CONFIG_PATH)
    with config_file.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    config = _deep_merge(DEFAULT_CONFIG, loaded)

    paths = config.setdefault("paths", {})
    for key in ("prompts", "images", "results"):
        paths[key] = str(resolve_repo_path(paths[key]))

    return config


def ensure_standard_directories(config: dict[str, Any]) -> None:
    get_images_root(config).mkdir(parents=True, exist_ok=True)
    get_results_root(config).mkdir(parents=True, exist_ok=True)


def get_images_root(config: dict[str, Any]) -> Path:
    return Path(config["paths"]["images"])


def get_results_root(config: dict[str, Any]) -> Path:
    return Path(config["paths"]["results"])


def get_prompts_path(config: dict[str, Any]) -> Path:
    return Path(config["paths"]["prompts"])


def normalize_object_name(name: Any) -> str:
    return str(name).strip().lower().replace("_", " ")


def parse_object_list(object_string: Any) -> list[str]:
    if pd.isna(object_string) or object_string in (None, ""):
        return []
    if isinstance(object_string, list):
        return [normalize_object_name(item) for item in object_string]
    return [
        normalize_object_name(part)
        for part in str(object_string).split("|")
        if str(part).strip()
    ]


def serialize_object_list(objects: Iterable[Any]) -> str:
    normalized = [normalize_object_name(obj) for obj in objects if str(obj).strip()]
    return "|".join(normalized)


def load_prompts(config: dict[str, Any]) -> pd.DataFrame:
    prompts_path = get_prompts_path(config)
    prompts_df = pd.read_csv(prompts_path)
    required_columns = {
        "prompt_id",
        "prompt_text",
        "num_objects",
        "object_list",
        "complexity",
    }
    missing = required_columns - set(prompts_df.columns)
    if missing:
        raise ValueError(f"prompts.csv is missing required columns: {sorted(missing)}")

    prompts_df["prompt_id"] = prompts_df["prompt_id"].astype(int)
    prompts_df["requested_objects"] = prompts_df["object_list"].apply(parse_object_list)
    prompts_df["num_objects"] = prompts_df["num_objects"].astype(int)

    inconsistent = prompts_df[
        prompts_df["requested_objects"].apply(len) != prompts_df["num_objects"]
    ]
    if not inconsistent.empty:
        bad_ids = inconsistent["prompt_id"].tolist()
        raise ValueError(f"Prompt object counts do not match num_objects for ids: {bad_ids}")

    return prompts_df


def build_image_filename(
    method: str,
    prompt_id: int,
    seed: int,
    *,
    attempt_number: int | None = None,
    round_number: int | None = None,
    extension: str = "png",
) -> str:
    parts = [method, f"prompt_{int(prompt_id):03d}"]
    if round_number is not None:
        parts.append(f"round_{int(round_number):02d}")
    if attempt_number is not None:
        parts.append(f"attempt_{int(attempt_number):02d}")
    parts.append(f"seed_{int(seed)}")
    return "_".join(parts) + f".{extension}"


def build_image_path(
    config: dict[str, Any],
    method: str,
    prompt_id: int,
    seed: int,
    *,
    attempt_number: int | None = None,
    round_number: int | None = None,
    extension: str = "png",
) -> Path:
    method_dir = get_images_root(config) / method
    method_dir.mkdir(parents=True, exist_ok=True)
    return method_dir / build_image_filename(
        method,
        prompt_id,
        seed,
        attempt_number=attempt_number,
        round_number=round_number,
        extension=extension,
    )


def compare_object_lists(
    requested_objects: Iterable[Any],
    detected_objects: Iterable[Any],
) -> dict[str, Any]:
    requested = [normalize_object_name(obj) for obj in requested_objects]
    detected = [normalize_object_name(obj) for obj in detected_objects]
    found = [obj for obj in requested if obj in detected]
    missing = [obj for obj in requested if obj not in detected]
    recall = len(found) / len(requested) if requested else 0.0
    return {
        "requested_objects": requested,
        "detected_objects": detected,
        "found_objects": found,
        "missing_objects": missing,
        "recall_score": round(recall, 4),
        "success": len(missing) == 0 and bool(requested),
    }


def choose_best_result(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Cannot choose a best result from an empty list.")

    def sort_key(row: dict[str, Any]) -> tuple[float, int, int]:
        recall = float(row.get("recall_score", 0.0))
        missing_count = len(parse_object_list(row.get("missing_objects", [])))
        detected_count = len(parse_object_list(row.get("detected_objects", [])))
        return (recall, detected_count, -missing_count)

    return max(rows, key=sort_key)


def results_output_path(config: dict[str, Any], filename: str) -> Path:
    path = get_results_root(config) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def normalize_result_row(row: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(row)
    for key in ("requested_objects", "detected_objects", "found_objects", "missing_objects"):
        if key in normalized:
            normalized[key] = serialize_object_list(parse_object_list(normalized[key]))
    if "success" in normalized:
        normalized["success"] = bool(normalized["success"])
    return normalized


def save_results(rows: list[dict[str, Any]], output_path: str | Path) -> Path:
    if not rows:
        raise ValueError("No rows supplied to save_results.")
    normalized_rows = [normalize_result_row(row) for row in rows]
    output = resolve_repo_path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(normalized_rows).to_csv(output, index=False)
    return output


def structured_prompt(prompt_text: str, requested_objects: Iterable[Any]) -> str:
    objects = [normalize_object_name(obj) for obj in requested_objects]
    if not objects:
        return prompt_text
    object_phrase = ", ".join(objects[:-1])
    if len(objects) > 1:
        object_phrase = f"{object_phrase}, and {objects[-1]}" if object_phrase else objects[-1]
    else:
        object_phrase = objects[0]
    return (
        f"An image containing exactly {len(objects)} objects: {object_phrase}. "
        "Show all of them clearly and prominently."
    )


def refine_prompt_text(prompt_text: str, missing_objects: Iterable[Any]) -> str:
    missing = [normalize_object_name(obj) for obj in missing_objects]
    if not missing:
        return prompt_text

    if len(missing) == 1:
        emphasis = missing[0]
    elif len(missing) == 2:
        emphasis = f"{missing[0]} and {missing[1]}"
    else:
        emphasis = ", ".join(missing[:-1]) + f", and {missing[-1]}"

    return f"{prompt_text}. Make sure to clearly include {emphasis}."


def compute_method_metrics(results_df: pd.DataFrame) -> dict[str, Any]:
    if results_df.empty:
        return {
            "num_prompts": 0,
            "mean_recall": 0.0,
            "success_rate": 0.0,
            "avg_generations": 0.0,
        }

    metrics = {
        "num_prompts": int(results_df["prompt_id"].nunique()),
        "mean_recall": round(float(results_df["recall_score"].mean()), 4),
        "success_rate": round(float(results_df["success"].mean()), 4),
    }
    if "total_generations" in results_df.columns:
        metrics["avg_generations"] = round(
            float(results_df["total_generations"].mean()),
            4,
        )
    else:
        metrics["avg_generations"] = 1.0
    return metrics
