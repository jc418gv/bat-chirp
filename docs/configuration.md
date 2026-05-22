# Configuration

The runtime config is a plain JSON file. JSON does not support comments, so this document is the source of truth for what each field means and which fields should stay stable versus which ones are injected per run.

## Recommended Base Config

The normal host-level config should be small and stable:

```json
{
  "recording_input_dir": "/path/to/audiomoth/recordings",
  "work_root_dir": "/path/to/audiomoth/work",
  "night_runs_dir": "/path/to/audiomoth/work/night-runs",
  "night_start_hour": 18,
  "night_end_hour": 12,
  "batdetect2_bin": "batdetect2",
  "detection_threshold": 0.5,
  "write_mp3": true,
  "ffmpeg_bin": "ffmpeg"
}
```

In most cases, that is enough.

## Core Fields

- `recording_input_dir`
  Directory containing the source AudioMoth WAV files.
- `work_root_dir`
  Base work area for shared outputs. If you provide this field, the loader can derive `detections`, `summary`, and `review` roots automatically.
- `night_runs_dir`
  Base directory for per-night browse bundles. If omitted, it defaults to `work_root_dir/night-runs`.
- `night_start_hour`
  Hour on the anchor date when an overnight session begins. `18` means 6 PM.
- `night_end_hour`
  Hour on the following boundary when the overnight session stops. `12` means noon, and the effective window is `[18:00, 12:00 next day)`.
- `batdetect2_bin`
  Command or absolute path used to run BatDetect2.
- `detection_threshold`
  BatDetect2 detection threshold.
- `write_mp3`
  Whether review export also writes MP3 copies in addition to WAV.
- `ffmpeg_bin`
  Path or command name for `ffmpeg` when MP3 export is enabled.

## Optional Advanced Fields

- `model`
  Optional BatDetect2 model override.
- `subset_limit`
  Optional cap on the number of audio files selected for a run.
- `name_contains`
  Additional filename substring filters applied before the night window filter.
- `extra_args`
  Extra BatDetect2 command-line arguments.
- `noise_reduction_enabled`
  If true, the pipeline first writes enhanced copies of the selected WAV files and runs BatDetect2 on those copies. Review clips and spectrograms still use the original recordings because the enhanced files are only a detection aid.
- `noise_reduction_output_dir`
  Optional directory for enhanced WAV files. If omitted, it defaults beside the detection output as `noise-reduced`.
- `noise_reduction_mode`
  Enhancement method. `spectral_subtract` is the default and current recommended mode for noisy ultrasonic clips. `soft_gate` keeps the older sigmoid mask if you want to compare behavior.
- `noise_reduction_n_fft`, `noise_reduction_hop`
  STFT size and hop used for noise reduction. For ultrasonic AudioMoth files, start with `1024/128` or `2048/256`.
- `noise_reduction_percentile`
  Per-frequency noise floor percentile. Lower values are more conservative; `20.0` is the default starting point.
- `noise_reduction_spectral_subtract_oversubtract`, `noise_reduction_spectral_subtract_floor_ratio`, `noise_reduction_spectral_subtract_smoothing_bins`
  Spectral subtraction controls. Higher oversubtraction removes more stationary floor, floor ratio prevents hard nulling, and smoothing bins stabilize the estimated floor across adjacent frequencies.
- `noise_reduction_margin_db`, `noise_reduction_softness_db`, `noise_reduction_floor_gain`
  Soft spectral gate controls used only when `noise_reduction_mode` is `soft_gate`.
- `clip_start_s`
  Fixed clip start override for review export.
- `clip_duration_s`
  Fixed clip duration override for review export.
- `padding_before_s`
  Seconds of review context before the selected bout.
- `padding_after_s`
  Seconds of review context after the selected bout.
- `minimum_duration_s`
  Minimum exported clip length.
- `bout_gap_s`
  Gap threshold used when grouping nearby detections into one review bout.
- `slowdown_factor`
  Time expansion factor for audible review audio.
- `max_freq_hz`
  Top frequency shown in exported spectrograms.
- `mp3_bitrate`
  Bitrate passed to `ffmpeg` for MP3 output.
- `continue_on_error`
  Whether batch review export continues if one file fails.

## Runtime Environment Overrides

Some runtime behavior is controlled by environment variables rather than JSON config fields:

- `CUDA_VISIBLE_DEVICES`
  Passed through directly to BatDetect2 when already set in the parent environment.
- `BATPIPE_BATDETECT2_CUDA_VISIBLE_DEVICES`
  Batpipe-specific override for BatDetect2 device visibility when `CUDA_VISIBLE_DEVICES` is not already set. The default routine behavior is `0`, which pins inference to one visible GPU. Use `all`, `ALL`, or `*` to leave GPU visibility unrestricted.

## Derived Output Paths

If `work_root_dir` is present, the loader derives these internal paths automatically unless you explicitly override them:

- `detection_output_dir` -> `work_root_dir/detections`
- `summary_output_dir` -> `work_root_dir/summary`
- `review_output_dir` -> `work_root_dir/review`

Those fields still exist internally because different pipeline stages need concrete directories, but most users should not need to set all three in the base config.

## Per-Night Runtime Fields

These values are normally supplied by [../scripts/run_night_for_date.sh](../scripts/run_night_for_date.sh) in a temporary config created for a specific run:

- `night_token`
- `detection_output_dir` for that night run
- `summary_output_dir` for that night run
- `review_output_dir` for that night run

Example: when you run `./scripts/run_night_for_date.sh 20260518 config/site.json`, the wrapper creates a temporary config with outputs under `night-runs/20260518/` and sets `night_token` to `20260518`.

## Legacy Compatibility

Older configs that define these fields still work:

- `detection_output_dir`
- `summary_output_dir`
- `review_output_dir`
- `validation_output_dir`

That compatibility exists to avoid breaking older host setups, but the preferred direction is a stable base config with `work_root_dir` and, optionally, `night_runs_dir`.