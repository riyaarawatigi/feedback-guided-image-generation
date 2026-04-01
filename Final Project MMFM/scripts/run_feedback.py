from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from object_omission.feedback import run_feedback_experiment


def parse_limit_ids(raw: str | None):
    if not raw:
        return None
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run feedback-guided generation.")
    parser.add_argument("--config", default="config.yaml", help="Path to config file.")
    parser.add_argument("--limit-ids", help="Comma-separated prompt IDs for a smaller test run.")
    parser.add_argument("--output", help="Optional output CSV path.")
    args = parser.parse_args()

    output_path = run_feedback_experiment(
        config_path=args.config,
        limit_ids=parse_limit_ids(args.limit_ids),
        output_csv=args.output,
    )
    print(output_path)


if __name__ == "__main__":
    main()
