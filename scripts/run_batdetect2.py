from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from batpipe.detect import build_detection_plan, command_as_shell_string, run_detection_plan, write_detection_plan


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Run BatDetect2 over an AudioMoth directory or a limited subset.")
    parser.add_argument("--input-dir", required=True, help="Directory containing AudioMoth WAV files.")
    parser.add_argument("--output-dir", required=True, help="Directory for BatDetect2 outputs.")
    parser.add_argument("--batdetect2-bin", default="batdetect2", help="BatDetect2 executable name or path.")
    parser.add_argument("--model", default=None, help="Optional BatDetect2 model checkpoint or alias.")
    parser.add_argument("--detection-threshold", type=float, default=None, help="Optional BatDetect2 detection threshold override.")
    parser.add_argument("--limit", type=int, default=None, help="Limit processing to the first N sorted recordings.")
    parser.add_argument(
        "--name-contains",
        action="append",
        default=[],
        help="Require the recording filename to contain this substring. Repeat to combine filters.",
    )
    parser.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help="Extra argument to append to the BatDetect2 CLI. Repeat as needed.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Write the manifest and print the command without executing it.")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    plan = build_detection_plan(
        input_dir=input_dir,
        output_dir=output_dir,
        batdetect2_bin=args.batdetect2_bin,
        model=args.model,
        detection_threshold=args.detection_threshold,
        limit=args.limit,
        name_filters=args.name_contains,
        extra_args=args.extra_arg,
    )
    manifest_path = output_dir / "run_manifest.json"
    write_detection_plan(plan, manifest_path)

    print(f"Discovered {plan.audio_file_count} audio files.")
    print(f"Selected {plan.selected_file_count} file(s) using mode: {plan.invocation_mode}.")
    print(f"Manifest: {manifest_path}")
    print(command_as_shell_string(plan.batdetect2_command))

    run_detection_plan(plan, dry_run=args.dry_run)
    if args.dry_run:
        print("Dry run only. BatDetect2 was not executed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
