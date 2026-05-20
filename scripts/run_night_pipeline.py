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
    parser.add_argument("--json", dest="json_output", action="store_true", help="Print the final result as JSON instead of a human-readable summary.")
    return parser


def _print_progress(event: str, payload: dict[str, object]) -> None:
    if event == "detection_started":
        print(
            f"Running BatDetect2 inference on {payload.get('selected_audio_files', 0)} file(s)...",
            file=sys.stderr,
            flush=True,
        )
        return
    if event == "summary_started":
        print("Summarizing detections...", file=sys.stderr, flush=True)
        return
    if event == "review_started":
        print("Review export: preparing matched files...", file=sys.stderr, flush=True)
        return
    if event == "batch_started":
        print(
            f"Review export: {payload.get('matched_job_count', 0)} clip(s) matched, "
            f"{payload.get('missing_json_count', 0)} missing JSON. Analyzing",
            file=sys.stderr,
            end="",
            flush=True,
        )
        return
    if event in {"item_completed", "item_failed"}:
        print(".", file=sys.stderr, end="", flush=True)
        return
    if event == "batch_completed":
        print(
            f" done ({payload.get('exported_count', 0)} exported, {payload.get('failed_count', 0)} failed).",
            file=sys.stderr,
            flush=True,
        )
        return
    if event == "review_site_started":
        print("Building review site...", file=sys.stderr, flush=True)
        return


def _render_human_summary(result: dict[str, object]) -> str:
    lines = [
        f"Night pipeline complete for {result.get('night_token')}",
        f"Audio files: {result.get('selected_audio_files', 0)} selected from {result.get('discovered_audio_files', 0)} discovered",
        f"Detection manifest: {result.get('detection_manifest')}",
    ]

    summary_outputs = result.get("summary_outputs")
    if isinstance(summary_outputs, dict):
        lines.append(f"Nightly summary CSV: {summary_outputs.get('nightly_summary')}")
        lines.append(f"Review queue CSV: {summary_outputs.get('review_queue')}")

    review_outputs = result.get("review_outputs")
    if isinstance(review_outputs, dict):
        lines.append(
            "Review export: "
            f"{review_outputs.get('exported_count', 0)} exported, "
            f"{review_outputs.get('missing_json_count', 0)} missing JSON, "
            f"{review_outputs.get('failed_count', 0)} failed"
        )
        lines.append(f"Review index: {review_outputs.get('review_index_html')}")
        lines.append(f"Batch summary: {review_outputs.get('summary_json')}")

    return "\n".join(lines)


def main() -> int:
    args = build_parser().parse_args()
    config = load_site_config(Path(args.config).expanduser().resolve())
    result = run_night_pipeline(
        config,
        dry_run=args.dry_run,
        skip_detection=args.skip_detection,
        skip_summary=args.skip_summary,
        skip_review=args.skip_review,
        progress_callback=_print_progress,
    )
    if args.json_output:
        print(json.dumps(result, indent=2, default=str))
    else:
        print(_render_human_summary(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())