from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from object_omission.baselines import (
    run_rejection_experiment,
    run_structured_experiment,
    run_vanilla_experiment,
)


def parse_limit_ids(raw: str | None):
    if not raw:
        return None
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run vanilla / structured / rejection baselines.")
    parser.add_argument("--config", default="config.yaml", help="Path to config file.")
    parser.add_argument(
        "--methods",
        default="vanilla,structured,rejection",
        help="Comma-separated subset: vanilla,structured,rejection",
    )
    parser.add_argument("--limit-ids", help="Comma-separated prompt IDs for a smaller test run.")
    parser.add_argument("--rejection-k", type=int, help="Override rejection sampling k.")
    args = parser.parse_args()

    methods = {part.strip().lower() for part in args.methods.split(",") if part.strip()}
    limit_ids = parse_limit_ids(args.limit_ids)

    if "vanilla" in methods:
        run_vanilla_experiment(config_path=args.config, limit_ids=limit_ids)
    if "structured" in methods:
        run_structured_experiment(config_path=args.config, limit_ids=limit_ids)
    if "rejection" in methods:
        run_rejection_experiment(config_path=args.config, limit_ids=limit_ids, k=args.rejection_k)


if __name__ == "__main__":
    main()
