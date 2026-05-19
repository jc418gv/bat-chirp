from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import json
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from batpipe.pipeline import run_night_pipeline
from batpipe.site_config import load_site_config


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Run the full overnight pipeline from raw WAVs through detection, summary, and review export.")
    parser.add_argument("--config", required=True, help="Path to the site JSON configuration file.")
    parser.add_argument("--dry-run", action="store_true", help="Write the detection manifest and print the planned detection command without executing the pipeline.")
    parser.add_argument("--skip-detection", action="store_true", help="Reuse existing BatDetect2 JSON files and skip inference.")
    parser.add_argument("--skip-summary", action="store_true", help="Skip CSV summary generation even if summary_output_dir is configured.")
    parser.add_argument("--skip-review", "--skip-validation", dest="skip_review", action="store_true", help="Skip review export generation even if review_output_dir is configured.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = load_site_config(Path(args.config).expanduser().resolve())
    result = run_night_pipeline(
        config,
        dry_run=args.dry_run,
        skip_detection=args.skip_detection,
        skip_summary=args.skip_summary,
        skip_review=args.skip_review,
    )
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())