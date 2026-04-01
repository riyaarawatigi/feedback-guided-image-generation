from __future__ import annotations

import os
import re
from pathlib import Path

import pandas as pd

from .helpers import load_config


_DETECTOR_CACHE = {}


def get_detector(model_name: str = "yolov8l.pt"):
    from ultralytics import YOLO

    if model_name not in _DETECTOR_CACHE:
        _DETECTOR_CACHE[model_name] = YOLO(model_name)
    return _DETECTOR_CACHE[model_name]


def normalize_object_name(name):
    name = str(name).lower().strip()
    name = name.replace("_", " ")
    return name


def parse_object_list(object_string):
    if pd.isna(object_string):
        return []
    return [normalize_object_name(obj) for obj in str(object_string).split("|") if str(obj).strip()]


def extract_prompt_id(filename):
    match = re.search(r"prompt_(\d+)", str(filename).lower())
    if match:
        return int(match.group(1))
    return None


def detect_objects(image_path, conf_threshold=0.25, model_name: str = "yolov8l.pt"):
    model = get_detector(model_name)
    results = model(str(image_path), conf=conf_threshold, verbose=False)

    detected_objects = []

    for result in results:
        boxes = result.boxes
        names = result.names

        if boxes is not None:
            for box in boxes:
                cls_id = int(box.cls.item())
                class_name = normalize_object_name(names[cls_id])
                detected_objects.append(class_name)

    unique_detected = list(dict.fromkeys(detected_objects))
    return unique_detected


def compare_objects(requested_objects, detected_objects):
    requested_norm = [normalize_object_name(obj) for obj in requested_objects]
    detected_norm = [normalize_object_name(obj) for obj in detected_objects]

    missing_objects = [obj for obj in requested_norm if obj not in detected_norm]
    recall = (
        len([obj for obj in requested_norm if obj in detected_norm]) / len(requested_norm)
        if len(requested_norm) > 0
        else 0.0
    )

    return {
        "requested_objects": requested_norm,
        "detected_objects": detected_norm,
        "missing_objects": missing_objects,
        "recall_score": round(recall, 4),
    }


def analyze_single_image(image_filename, image_folder, prompts_df, conf_threshold=0.25, model_name="yolov8l.pt"):
    image_path = os.path.join(image_folder, image_filename)
    prompt_id = extract_prompt_id(image_filename)

    if prompt_id is None:
        return {"error": "Could not extract prompt_id"}

    matched_rows = prompts_df[prompts_df["prompt_id"] == prompt_id]

    if len(matched_rows) == 0:
        return {"error": "No matching prompt_id in prompts.csv"}

    row = matched_rows.iloc[0]
    requested_objects = parse_object_list(row["object_list"])
    detected_objects = detect_objects(image_path, conf_threshold=conf_threshold, model_name=model_name)
    comparison = compare_objects(requested_objects, detected_objects)

    return {
        "prompt_id": prompt_id,
        "requested_objects": comparison["requested_objects"],
        "detected_objects": comparison["detected_objects"],
        "missing_objects": comparison["missing_objects"],
        "recall_score": comparison["recall_score"],
    }


def detect_with_project_defaults(image_path, config: dict | None = None):
    config = config or load_config()
    return detect_objects(
        image_path=image_path,
        conf_threshold=float(config["detection"]["confidence_threshold"]),
        model_name=str(config["detection"]["model"]),
    )
