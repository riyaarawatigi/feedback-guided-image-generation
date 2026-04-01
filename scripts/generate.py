from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from object_omission.generator import generate_for_prompt, generate_image
from object_omission.helpers import load_config


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
