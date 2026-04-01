from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
import yaml


DEFAULT_CONFIG = {
    "generation": {
        "model": "runwayml/stable-diffusion-v1-5",
        "image_size": 512,
        "steps": 50,
        "guidance": 7.5,
        "seed": 42,
        "device": "auto",
        "dtype": "auto",
    },
    "detection": {
        "model": "yolov8l.pt",
        "confidence_threshold": 0.25,
    },
    "experiments": {
        "rejection_k": 5,
        "max_feedback_attempts": 9,
    },
    "paths": {
        "prompts": "data/prompts.csv",
        "output_dir": "results",
        "images_dir": "generated_images",
    },
}


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _deep_merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path | None = None) -> dict:
    config_path = Path(path) if path else project_root() / "config.yaml"
    if not config_path.exists():
        return DEFAULT_CONFIG.copy()

    with config_path.open("r", encoding="utf-8") as handle:
        user_config = yaml.safe_load(handle) or {}

    config = _deep_merge(DEFAULT_CONFIG, user_config)

    paths = config.get("paths", {})
    for key in ("prompts", "output_dir", "images_dir"):
        if key in paths:
            paths[key] = str(resolve_repo_path(paths[key]))
    config["paths"] = paths
    return config


def resolve_repo_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return project_root() / path


def ensure_standard_directories(config: dict) -> None:
    resolve_repo_path(config["paths"]["output_dir"]).mkdir(parents=True, exist_ok=True)
    resolve_repo_path(config["paths"]["images_dir"]).mkdir(parents=True, exist_ok=True)


def load_prompts(config: dict | str | Path | None = None) -> pd.DataFrame:
    if isinstance(config, (str, Path)) or config is None:
        config = load_config(config)
    prompts_path = resolve_repo_path(config["paths"]["prompts"])
    df = pd.read_csv(prompts_path)
    required = {"prompt_id", "prompt_text", "object_list"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"prompts.csv is missing required columns: {sorted(missing)}")

    if "complexity" not in df.columns:
        df["complexity"] = df.get("num_objects", df["object_list"].fillna("").str.count(r"\|") + 1)
    df["requested_objects"] = df["object_list"].fillna("").apply(parse_pipe_list)
    return df


def parse_pipe_list(value: object) -> list[str]:
    if pd.isna(value):
        return []
    items = []
    for part in str(value).split("|"):
        cleaned = str(part).strip()
        if cleaned:
            items.append(cleaned)
    return items


def build_image_path(
    config: dict,
    method: str,
    prompt_id: int,
    seed: int,
    attempt_number: int | None = None,
    round_number: int | None = None,
) -> Path:
    images_root = resolve_repo_path(config["paths"]["images_dir"])
    folder = images_root / method
    folder.mkdir(parents=True, exist_ok=True)

    filename = f"{method}_prompt_{int(prompt_id):03d}"
    if round_number is not None:
        filename += f"_round_{int(round_number):02d}"
    if attempt_number is not None:
        filename += f"_attempt_{int(attempt_number):02d}"
    filename += f"_seed_{int(seed)}.png"
    return folder / filename


def get_results_path(config: dict, filename: str | Path) -> Path:
    output_root = resolve_repo_path(config["paths"]["output_dir"])
    output_root.mkdir(parents=True, exist_ok=True)
    return output_root / Path(filename).name


def seed_for_attempt(prompt_id: int, attempt: int, base_seed: int = 42) -> int:
    return int(base_seed) + int(prompt_id) * 100 + int(attempt)


def stringify_items(values: Iterable[object]) -> str:
    return "|".join(str(v) for v in values)
