from __future__ import annotations

import argparse
from pathlib import Path

if __package__ in (None, ""):
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.baseline_vanilla import run_baseline
from scripts.helpers import structured_prompt


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the structured-prompt baseline.")
    parser.add_argument("--config", default="config.yaml", help="Path to config file.")
    parser.add_argument("--limit", type=int, help="Limit to the first N prompts.")
    args = parser.parse_args()

    output_path = run_baseline(
        method="structured",
        transform_prompt=structured_prompt,
        output_filename="structured_results.csv",
        config_path=args.config,
        limit=args.limit,
    )
    print(output_path)


if __name__ == "__main__":
    main()
