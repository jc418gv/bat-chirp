from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from batpipe.aggregate import summarize_detection_directory


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Summarize BatDetect2 JSON outputs into hourly and nightly CSVs.")
    parser.add_argument("--input-dir", required=True, help="Directory containing BatDetect2 JSON outputs.")
    parser.add_argument("--output-dir", required=True, help="Directory for summary CSVs.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    outputs = summarize_detection_directory(
        Path(args.input_dir).expanduser().resolve(),
        Path(args.output_dir).expanduser().resolve(),
    )
    for label, path in outputs.items():
        print(f"{label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())