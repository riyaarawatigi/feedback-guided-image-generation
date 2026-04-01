from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all experiments end-to-end.")
    parser.add_argument("--config", default="config.yaml", help="Path to config file.")
    parser.add_argument("--limit-ids", help="Comma-separated prompt IDs for a quick smoke test.")
    parser.add_argument("--skip-plots", action="store_true", help="Skip figure generation.")
    args = parser.parse_args()

    common = [sys.executable]

    subprocess.run(
        common + [str(SCRIPTS / "run_baselines.py"), "--config", args.config]
        + (["--limit-ids", args.limit_ids] if args.limit_ids else []),
        check=True,
    )
    subprocess.run(
        common + [str(SCRIPTS / "run_feedback.py"), "--config", args.config]
        + (["--limit-ids", args.limit_ids] if args.limit_ids else []),
        check=True,
    )
    subprocess.run(
        common + [str(SCRIPTS / "summarize_results.py"), "--config", args.config],
        check=True,
    )
    if not args.skip_plots:
        subprocess.run(
            common + [str(SCRIPTS / "plot_results.py"), "--config", args.config],
            check=True,
        )


if __name__ == "__main__":
    main()
