# Usage

## Typical Flow On A Linux GPU Host

Recommended setup on the processing host:

- use [../scripts/setup_host.sh](../scripts/setup_host.sh)
- prefer running on a Linux host with an NVIDIA GPU available
- keep the PyTorch stack aligned with the CUDA runtime available on that host

1. Run [../scripts/setup_host.sh](../scripts/setup_host.sh) to create or reuse `.venv`.
2. Let that script install the local package, BatDetect2, and the PyTorch stack into the same environment.
3. Copy the example config and replace its placeholder paths with your own host-specific folders and binaries.
4. Run the full pipeline from raw WAVs to review exports.

The simplest base config shape is:

- `recording_input_dir`: where the AudioMoth WAV files live
- `work_root_dir`: base work area for shared outputs such as `detections`, `summary`, and `review`
- `night_runs_dir`: optional override for per-night browse bundles; if omitted it defaults to `work_root_dir/night-runs`

You do not normally keep a fixed `night_token` in the site JSON. The nightly wrapper takes that token as an argument and creates a temporary config for that specific run.

For field-by-field configuration details, see [configuration.md](configuration.md).

Example:

```bash
cp config/site.example.json config/site.local.json

cd /path/to/bat-chirp
./scripts/setup_host.sh
. .venv/bin/activate

PYTHONPATH=src python scripts/run_night_pipeline.py --config config/site.local.json
```

After setup, confirm the detector entry point exists before running a night batch:

```bash
which batdetect2
batdetect2 --help >/dev/null
```

Most users should do that directly over SSH on the Linux host where the recordings and GPU live, rather than driving the pipeline remotely from Windows.

If you want a simple one-night trigger after SSH'ing into the host, use:

```bash
./scripts/run_night_for_date.sh 20260518 config/site.local.json
```

That wrapper creates a temporary night-specific config, selects recordings inside the requested overnight window, and writes outputs under a dedicated `night-runs/YYYYMMDD/` work area.

To validate the orchestration without running inference, use:

```bash
PYTHONPATH=src python scripts/run_night_pipeline.py --config config/site.local.json --dry-run
```

To reuse existing BatDetect2 JSON files and only regenerate summaries plus review outputs, use:

```bash
PYTHONPATH=src python scripts/run_night_pipeline.py --config config/site.local.json --skip-detection
```

The same shell wrapper accepts any extra `run_night_pipeline.py` flags, for example:

```bash
./scripts/run_night_for_date.sh 20260518 config/site.local.json --dry-run
./scripts/run_night_for_date.sh 20260518 config/site.local.json --skip-detection
```

## AudioMoth Filename Handling

The summarizer expects AudioMoth-style filenames such as `20260517_200000T.WAV` and uses that timestamp to build hourly summaries.

## GPU Note

The setup script currently installs CUDA 12.4 PyTorch wheels because that is a common fit for practical Linux GPU hosts running this workflow. If your host uses a different CUDA runtime, adjust the wheel index and versions accordingly.

CPU-only execution is possible for development and small tests, but the intended production workflow is GPU-backed inference on a Linux host.

## Outputs

The summarizer writes:

- `detections_flat.csv`: one row per detected event
- `file_summary.csv`: one row per recording file
- `review_queue.csv`: positive files ranked for manual review
- `hourly_summary.csv`: activity summary per hour
- `nightly_summary.csv`: single-row summary for the batch

## Review Outputs

The review export writes one night folder named for the date the selected recordings began, for example `review/20260518/`.

Inside each per-recording folder it writes:

- `clip_original_HHMMSS.wav`: original ultrasonic excerpt at the original sample rate, named with the local sample clock time
- `clip_audible_x8_HHMMSS.wav`: time-expanded excerpt for listening review
- `clip_original_HHMMSS.mp3`: regular-speed MP3 derived from the extracted clip for quick browsing
- `clip_audible_x8_HHMMSS.mp3`: x8 slowed MP3 for listening review
- `spectrogram_HHMMSS.png`: spectrogram with BatDetect2 bat-candidate boxes overlaid; any model class label is shown only as untrusted reference
- `detections_HHMMSS.json`: clip metadata, the detections that fall inside the exported window, and any expanded candidate train segments inferred around the anchored bout

At the night root it writes:

- `batch_summary.json`: machine-readable manifest for the night review bundle
- `review_assets.csv`: flat asset index for the exported review clips
- `index.html`: a static thumbnail sheet with spectrogram previews and direct links to audio, spectrograms, and reports
- `hour-*.html`: hour-specific review pages for faster browsing on larger nights

After a run, `night-runs/YYYYMMDD/` itself is the main browse folder. It contains `index.html`, `hour-*.html`, the review CSV/JSON manifests, and one subdirectory per recording with the spectrogram, audio clips, and review JSON for that detection window.

The nightly `index.html` is grouped by hour so you can browse detections in expandable hour sections before drilling into the hour-specific pages.

MP3 generation requires `ffmpeg`; on many Linux hosts that will simply be available as `ffmpeg` on `PATH`, or at a path such as `/usr/bin/ffmpeg`.

The review exporter groups nearby detections into one primary review bout and exports context around that bout rather than spanning every detection in the file. It then uses a denoised, band-limited peak search around the anchored bout to underline one or more candidate chirp segments instead of forcing a single dashed span. Connected chirp segments can now knit together up to `2.5 ×` the average inter-peak interval within the detected train, which better preserves visibly continuous call series without collapsing obviously separate bursts into one candidate. Use this step to compare visible non-bat noise in the clip against the narrower annotated regions that BatDetect2 marked as echolocation. In North American review, treat any BatDetect2 species name as a raw upstream model label rather than a valid local species ID.

Current limitation: if the selected bout lands at the start or end of the source WAV, the exported clip can still truncate at the file boundary because adjacent-file stitching is not implemented yet.