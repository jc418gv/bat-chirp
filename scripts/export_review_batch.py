from __future__ import annotations

import json
import sys
from argparse import ArgumentParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from batpipe.review import export_review_batch


def _print_progress(event: str, payload: dict[str, object]) -> None:
    if event == "batch_started":
        print(
            "Starting review batch: "
            f"{payload.get('matched_job_count', 0)} matched file(s), "
            f"{payload.get('missing_json_count', 0)} missing JSON, "
            f"output {payload.get('night_output_dir')}",
            file=sys.stderr,
        )
        return
    if event == "item_started":
        print(
            f"[{payload.get('index')}/{payload.get('total')}] Exporting "
            f"{Path(str(payload.get('audio_file'))).name}",
            file=sys.stderr,
        )
        return
    if event == "item_completed":
        print(
            f"[{payload.get('index')}/{payload.get('total')}] Finished "
            f"{Path(str(payload.get('audio_file'))).name} -> {payload.get('output_dir')}",
            file=sys.stderr,
        )
        return
    if event == "item_failed":
        print(
            f"[{payload.get('index')}/{payload.get('total')}] Failed "
            f"{Path(str(payload.get('audio_file'))).name}: {payload.get('error')}",
            file=sys.stderr,
        )
        return
    if event == "batch_completed":
        print(
            "Batch complete: "
            f"{payload.get('exported_count', 0)} exported, "
            f"{payload.get('failed_count', 0)} failed, "
            f"summary {payload.get('summary_json')}",
            file=sys.stderr,
        )


def _render_human_summary(result: dict[str, object]) -> str:
    return "\n".join(
        [
            f"Review batch complete for {result.get('night')}",
            f"Audio files discovered: {result.get('discovered_audio_files', 0)}",
            f"Matched review clips: {result.get('matched_job_count', 0)}",
            f"Exported: {result.get('exported_count', 0)}",
            f"Missing JSON: {result.get('missing_json_count', 0)}",
            f"Failed: {result.get('failed_count', 0)}",
            f"Batch summary: {result.get('summary_json')}",
            f"Review index: {result.get('review_index_html')}",
        ]
    )


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Export review clips, MP3s, and spectrograms for a batch of AudioMoth recordings.")
    parser.add_argument("--audio-dir", required=True, help="Directory containing AudioMoth WAV files.")
    parser.add_argument("--json-dir", required=True, help="Directory containing BatDetect2 JSON files named <wav>.json.")
    parser.add_argument("--output-dir", required=True, help="Directory where per-file review exports and the batch summary are written.")
    parser.add_argument("--clip-start", type=float, default=None, help="Optional explicit clip start time in seconds for every file.")
    parser.add_argument("--clip-duration", type=float, default=None, help="Optional explicit clip duration in seconds for every file.")
    parser.add_argument("--padding-before", type=float, default=5.0, help="Seconds of context to include before the selected bout.")
    parser.add_argument("--padding-after", type=float, default=4.0, help="Seconds of context to include after the selected bout.")
    parser.add_argument("--minimum-duration", type=float, default=10.0, help="Minimum clip duration when auto-selecting a window.")
    parser.add_argument("--bout-gap", type=float, default=0.5, help="Maximum gap in seconds for grouping detections into one bout.")
    parser.add_argument("--slowdown-factor", type=int, default=8, help="Integer factor used to time-expand the audible review files.")
    parser.add_argument("--max-freq-hz", type=float, default=120000.0, help="Upper frequency limit for the plotted spectrogram.")
    parser.add_argument("--name-contains", action="append", default=[], help="Require the WAV filename to contain this substring. Repeat to combine filters.")
    parser.add_argument("--limit", type=int, default=None, help="Limit processing to the first N sorted WAV files after filtering.")
    parser.add_argument("--ffmpeg-bin", default="ffmpeg", help="ffmpeg executable name or path for MP3 encoding.")
    parser.add_argument("--mp3-bitrate", default="192k", help="MP3 bitrate passed to ffmpeg.")
    parser.add_argument("--no-mp3", action="store_true", help="Skip MP3 generation and export only WAV plus spectrogram assets.")
    parser.add_argument("--fail-fast", action="store_true", help="Stop on the first export failure instead of continuing.")
    parser.add_argument("--json", dest="json_output", action="store_true", help="Print the final result as JSON instead of a human-readable summary.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = export_review_batch(
        audio_dir=Path(args.audio_dir).expanduser().resolve(),
        json_dir=Path(args.json_dir).expanduser().resolve(),
        output_dir=Path(args.output_dir).expanduser().resolve(),
        clip_start_s=args.clip_start,
        clip_duration_s=args.clip_duration,
        padding_before_s=args.padding_before,
        padding_after_s=args.padding_after,
        minimum_duration_s=args.minimum_duration,
        bout_gap_s=args.bout_gap,
        slowdown_factor=args.slowdown_factor,
        max_freq_hz=args.max_freq_hz,
        name_filters=args.name_contains,
        limit=args.limit,
        write_mp3=not args.no_mp3,
        ffmpeg_bin=args.ffmpeg_bin,
        mp3_bitrate=args.mp3_bitrate,
        continue_on_error=not args.fail_fast,
        progress_callback=_print_progress,
    )
    if args.json_output:
        print(json.dumps(result, indent=2))
    else:
        print(_render_human_summary(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())