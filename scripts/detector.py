from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

if __package__ in (None, ""):
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.detector_utils import analyze_single_image, detect_objects
from scripts.helpers import (
    compare_object_lists,
    load_config,
    load_prompts,
    normalize_result_row,
    parse_object_list,
    save_results,
)


def detect_image(
    image_path: str | Path,
    *,
    requested_objects: list[str] | None = None,
    config: dict | None = None,
    config_path: str | Path | None = None,
) -> dict:
    config = config or load_config(config_path)
    detected = detect_objects(
        str(image_path),
        conf_threshold=float(config["detection"]["confidence"]),
        model_name=config["detection"]["model"],
    )
    if requested_objects is None:
        return {"detected_objects": detected}
    comparison = compare_object_lists(requested_objects, detected)
    return normalize_result_row(comparison)


def analyze_image_file(
    image_path: str | Path,
    *,
    config: dict | None = None,
    config_path: str | Path | None = None,
) -> dict:
    config = config or load_config(config_path)
    prompts_df = load_prompts(config)
    image_path = Path(image_path)
    result = analyze_single_image(
        image_path.name,
        str(image_path.parent),
        prompts_df,
        conf_threshold=float(config["detection"]["confidence"]),
        model_name=config["detection"]["model"],
    )
    if "error" in result:
        raise ValueError(result["error"])
    return normalize_result_row(result)


def detect_directory(
    image_dir: str | Path,
    output_path: str | Path,
    *,
    config: dict | None = None,
    config_path: str | Path | None = None,
) -> Path:
    config = config or load_config(config_path)
    prompts_df = load_prompts(config)
    image_dir = Path(image_dir)
    rows: list[dict] = []
    for image_path in sorted(image_dir.glob("*.png")):
        result = analyze_single_image(
            image_path.name,
            str(image_dir),
            prompts_df,
            conf_threshold=float(config["detection"]["confidence"]),
            model_name=config["detection"]["model"],
        )
        if "error" not in result:
            rows.append(normalize_result_row(result))
    return save_results(rows, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run YOLO detection on generated images.")
    parser.add_argument("--image", help="Path to a single image.")
    parser.add_argument("--image-dir", help="Path to a directory of PNG images.")
    parser.add_argument("--output", help="Optional CSV output path for directory mode.")
    parser.add_argument("--objects", help="Pipe-delimited requested objects for single-image mode.")
    parser.add_argument("--config", default="config.yaml", help="Path to config file.")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.image:
        requested = parse_object_list(args.objects) if args.objects else None
        result = detect_image(args.image, requested_objects=requested, config=config)
        print(pd.Series(result).to_json())
        return

    if args.image_dir and args.output:
        output = detect_directory(args.image_dir, args.output, config=config)
        print(output)
        return

    raise SystemExit("Provide --image or both --image-dir and --output.")


if __name__ == "__main__":
    main()
