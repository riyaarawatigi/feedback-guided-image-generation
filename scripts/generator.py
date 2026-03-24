from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

if __package__ in (None, ""):
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.helpers import build_image_path, ensure_standard_directories, load_config


_PIPELINE_CACHE = {}


def _resolve_device(config: dict) -> str:
    import torch

    configured = str(config["generation"].get("device", "auto")).lower()
    if configured != "auto":
        return configured
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _resolve_dtype(config: dict, device: str):
    import torch

    configured = str(config["generation"].get("dtype", "auto")).lower()
    if configured == "float32":
        return torch.float32
    if configured == "float16":
        return torch.float16
    return torch.float16 if device in {"cuda", "mps"} else torch.float32


def get_generator_pipeline(config: dict):
    from diffusers import StableDiffusionPipeline

    device = _resolve_device(config)
    cache_key = (config["generation"]["model"], device, str(_resolve_dtype(config, device)))
    if cache_key in _PIPELINE_CACHE:
        return _PIPELINE_CACHE[cache_key], device

    dtype = _resolve_dtype(config, device)
    pipeline = StableDiffusionPipeline.from_pretrained(
        config["generation"]["model"],
        torch_dtype=dtype,
    )
    pipeline.set_progress_bar_config(disable=True)
    pipeline = pipeline.to(device)
    _PIPELINE_CACHE[cache_key] = pipeline
    return pipeline, device


def generate_image(
    prompt_text: str,
    output_path: str | Path,
    seed: int,
    *,
    config: dict | None = None,
    config_path: str | Path | None = None,
) -> Image.Image:
    import torch

    config = config or load_config(config_path)
    ensure_standard_directories(config)
    pipeline, device = get_generator_pipeline(config)

    generator_device = "cpu" if device == "mps" else device
    torch_generator = torch.Generator(device=generator_device).manual_seed(int(seed))

    result = pipeline(
        prompt_text,
        height=int(config["generation"]["image_size"]),
        width=int(config["generation"]["image_size"]),
        num_inference_steps=int(config["generation"]["steps"]),
        guidance_scale=float(config["generation"]["guidance"]),
        generator=torch_generator,
    )
    image = result.images[0]
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    return image


def generate_for_prompt(
    method: str,
    prompt_id: int,
    prompt_text: str,
    seed: int,
    *,
    attempt_number: int | None = None,
    round_number: int | None = None,
    config: dict | None = None,
    config_path: str | Path | None = None,
) -> Path:
    config = config or load_config(config_path)
    output_path = build_image_path(
        config,
        method,
        prompt_id,
        seed,
        attempt_number=attempt_number,
        round_number=round_number,
    )
    generate_image(prompt_text, output_path, seed, config=config)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a single image with Stable Diffusion.")
    parser.add_argument("--prompt", required=True, help="Prompt text to render.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--output", help="Explicit output image path.")
    parser.add_argument("--method", default="manual", help="Method name for default output path.")
    parser.add_argument("--prompt-id", type=int, default=0, help="Prompt id for default output path.")
    parser.add_argument("--config", default="config.yaml", help="Path to config file.")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.output:
        output_path = Path(args.output)
        generate_image(args.prompt, output_path, args.seed, config=config)
    else:
        output_path = generate_for_prompt(
            args.method,
            args.prompt_id,
            args.prompt,
            args.seed,
            config=config,
        )
    print(output_path)


if __name__ == "__main__":
    main()
