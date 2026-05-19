from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import json
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from batpipe.validate import export_validation_clip


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Export a review clip and annotated spectrogram for one BatDetect2 JSON result.")
    parser.add_argument("--audio-file", required=True, help="AudioMoth WAV file to read.")
    parser.add_argument("--json-file", required=True, help="BatDetect2 JSON output that corresponds to the WAV file.")
    parser.add_argument("--output-dir", required=True, help="Directory for the clip, spectrogram, and report.")
    parser.add_argument("--clip-start", type=float, default=None, help="Optional clip start time in seconds relative to the WAV file.")
    parser.add_argument("--clip-duration", type=float, default=None, help="Optional clip duration in seconds.")
    parser.add_argument("--padding-before", type=float, default=5.0, help="Seconds of context to include before the first detection when auto-selecting a clip.")
    parser.add_argument("--padding-after", type=float, default=4.0, help="Seconds of context to include after the selected detection bout when auto-selecting a clip.")
    parser.add_argument("--minimum-duration", type=float, default=10.0, help="Minimum clip duration when auto-selecting a window.")
    parser.add_argument("--bout-gap", type=float, default=0.5, help="Maximum gap in seconds for grouping nearby detections into one bout.")
    parser.add_argument("--slowdown-factor", type=int, default=8, help="Integer factor used to time-expand the audible review WAV.")
    parser.add_argument("--max-freq-hz", type=float, default=120000.0, help="Upper frequency limit for the plotted spectrogram.")
    parser.add_argument("--write-mp3", action="store_true", help="Also write regular-speed and x8 audible MP3 review files using ffmpeg.")
    parser.add_argument("--ffmpeg-bin", default="ffmpeg", help="ffmpeg executable name or path used for MP3 encoding.")
    parser.add_argument("--mp3-bitrate", default="192k", help="MP3 bitrate passed to ffmpeg.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = export_validation_clip(
        audio_path=Path(args.audio_file).expanduser().resolve(),
        json_path=Path(args.json_file).expanduser().resolve(),
        output_dir=Path(args.output_dir).expanduser().resolve(),
        clip_start_s=args.clip_start,
        clip_duration_s=args.clip_duration,
        padding_before_s=args.padding_before,
        padding_after_s=args.padding_after,
        minimum_duration_s=args.minimum_duration,
        bout_gap_s=args.bout_gap,
        slowdown_factor=args.slowdown_factor,
        max_freq_hz=args.max_freq_hz,
        write_mp3=args.write_mp3,
        ffmpeg_bin=args.ffmpeg_bin,
        mp3_bitrate=args.mp3_bitrate,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())