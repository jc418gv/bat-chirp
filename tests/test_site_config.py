import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from batpipe.site_config import load_site_config


class SiteConfigTests(unittest.TestCase):
    def test_load_site_config_derives_output_dirs_from_work_root(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "site.json"
            config_path.write_text(
                """
{
  "recording_input_dir": "/tmp/recordings",
  "work_root_dir": "/tmp/work",
  "night_start_hour": 18,
  "night_end_hour": 12,
  "detection_threshold": 0.5,
  "subset_limit": 10,
  "name_contains": ["20260518"],
    "noise_reduction_enabled": true,
    "noise_reduction_n_fft": 2048,
    "noise_reduction_hop": 256,
    "noise_reduction_percentile": 15.0,
    "noise_reduction_margin_db": 5.0,
    "noise_reduction_softness_db": 2.5,
    "noise_reduction_floor_gain": 0.1,
  "write_mp3": false,
  "ffmpeg_bin": "/usr/bin/ffmpeg"
}
                """.strip(),
                encoding="utf-8",
            )

            config = load_site_config(config_path)

            self.assertEqual(config.recording_input_dir, str(Path("/tmp/recordings").expanduser().resolve()))
            self.assertEqual(config.work_root_dir, str(Path("/tmp/work").expanduser().resolve()))
            self.assertEqual(config.night_runs_dir, str(Path("/tmp/work/night-runs").expanduser().resolve()))
            self.assertEqual(config.detection_output_dir, str(Path("/tmp/work/detections").expanduser().resolve()))
            self.assertEqual(config.summary_output_dir, str(Path("/tmp/work/summary").expanduser().resolve()))
            self.assertEqual(config.review_output_dir, str(Path("/tmp/work/review").expanduser().resolve()))
            self.assertIsNone(config.night_token)
            self.assertEqual(config.night_start_hour, 18)
            self.assertEqual(config.night_end_hour, 12)
            self.assertEqual(config.detection_threshold, 0.5)
            self.assertEqual(config.subset_limit, 10)
            self.assertEqual(config.name_contains, ["20260518"])
            self.assertTrue(config.noise_reduction_enabled)
            self.assertEqual(config.noise_reduction_n_fft, 2048)
            self.assertEqual(config.noise_reduction_hop, 256)
            self.assertEqual(config.noise_reduction_percentile, 15.0)
            self.assertEqual(config.noise_reduction_margin_db, 5.0)
            self.assertEqual(config.noise_reduction_softness_db, 2.5)
            self.assertEqual(config.noise_reduction_floor_gain, 0.1)
            self.assertFalse(config.write_mp3)
            self.assertEqual(config.ffmpeg_bin, "/usr/bin/ffmpeg")

    def test_load_site_config_accepts_explicit_output_dirs(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "site.json"
            config_path.write_text(
                """
{
  "recording_input_dir": "/tmp/recordings",
  "detection_output_dir": "/tmp/detections",
  "summary_output_dir": "/tmp/summary",
  "review_output_dir": "/tmp/review",
  "night_runs_dir": "/tmp/night-runs"
}
                """.strip(),
                encoding="utf-8",
            )

            config = load_site_config(config_path)

            self.assertEqual(config.detection_output_dir, str(Path("/tmp/detections").expanduser().resolve()))
            self.assertEqual(config.summary_output_dir, str(Path("/tmp/summary").expanduser().resolve()))
            self.assertEqual(config.review_output_dir, str(Path("/tmp/review").expanduser().resolve()))
            self.assertEqual(config.night_runs_dir, str(Path("/tmp/night-runs").expanduser().resolve()))

    def test_load_site_config_resolves_relative_paths_against_config_file(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_dir = root / "config"
            config_dir.mkdir()
            config_path = config_dir / "site.json"
            config_path.write_text(
                """
{
  "recording_input_dir": "../recordings",
  "work_root_dir": "../work"
}
                """.strip(),
                encoding="utf-8",
            )

            config = load_site_config(config_path)

            self.assertEqual(config.recording_input_dir, str((root / "recordings").resolve()))
            self.assertEqual(config.detection_output_dir, str((root / "work" / "detections").resolve()))
            self.assertEqual(config.summary_output_dir, str((root / "work" / "summary").resolve()))
            self.assertEqual(config.review_output_dir, str((root / "work" / "review").resolve()))
            self.assertEqual(config.night_runs_dir, str((root / "work" / "night-runs").resolve()))

    def test_load_site_config_accepts_legacy_validation_output_dir_alias(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "site.json"
            config_path.write_text(
                """
{
  "recording_input_dir": "/tmp/recordings",
  "detection_output_dir": "/tmp/detections",
  "validation_output_dir": "/tmp/review-legacy"
}
                """.strip(),
                encoding="utf-8",
            )

            config = load_site_config(config_path)

            self.assertEqual(config.review_output_dir, str(Path("/tmp/review-legacy").expanduser().resolve()))
