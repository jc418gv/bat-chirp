from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest
from unittest.mock import patch

from batpipe.review.batch import _resolve_night_output_dir, derive_night_token, discover_review_jobs, export_review_batch


class ReviewBatchTests(unittest.TestCase):
    def test_discover_review_jobs_matches_wavs_to_jsons(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            audio_dir = root / "audio"
            json_dir = root / "json"
            output_dir = root / "out"
            audio_dir.mkdir()
            json_dir.mkdir()

            for name in ["20260518_020000T.WAV", "20260518_020100T.WAV", "20260518_030000T.wav"]:
                (audio_dir / name).write_bytes(b"wav")
            (json_dir / "20260518_020000T.WAV.json").write_text("{}", encoding="utf-8")
            (json_dir / "20260518_030000T.wav.json").write_text("{}", encoding="utf-8")

            jobs, missing_json_paths, discovered_count, night_output_dir = discover_review_jobs(
                audio_dir=audio_dir,
                json_dir=json_dir,
                output_dir=output_dir,
                name_filters=["20260518_0"],
                limit=None,
            )

            self.assertEqual(discovered_count, 3)
            self.assertEqual([job.audio_path.name for job in jobs], ["20260518_020000T.WAV", "20260518_030000T.wav"])
            self.assertEqual([path.name for path in missing_json_paths], ["20260518_020100T.WAV"])
            self.assertEqual(night_output_dir, output_dir / "20260518")
            self.assertEqual(jobs[0].output_dir, output_dir / "20260518" / "20260518_020000T")

    def test_derive_night_token_uses_earliest_recording(self) -> None:
        token = derive_night_token([
            Path("20260519_001500T.WAV"),
            Path("20260518_235900T.WAV"),
            Path("20260519_000100T.WAV"),
        ])

        self.assertEqual(token, "20260518")

    def test_derive_night_token_prefers_requested_token(self) -> None:
        token = derive_night_token([Path("20260519_001500T.WAV")], requested_night_token="20260518")

        self.assertEqual(token, "20260518")

    def test_resolve_night_output_dir_avoids_duplicate_date_folder(self) -> None:
        target = _resolve_night_output_dir(Path("c:/tmp/night-runs/20260518"), [Path("20260518_020000T.WAV")])
        self.assertEqual(target, Path("c:/tmp/night-runs/20260518"))

    def test_discover_review_jobs_filters_to_night_window(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            audio_dir = root / "audio"
            json_dir = root / "json"
            output_dir = root / "out"
            audio_dir.mkdir()
            json_dir.mkdir()

            for name in [
                "20260518_175900T.WAV",
                "20260518_235500T.WAV",
                "20260519_000500T.WAV",
                "20260519_120100T.WAV",
            ]:
                (audio_dir / name).write_bytes(b"wav")
                (json_dir / f"{name}.json").write_text("{}", encoding="utf-8")

            jobs, missing_json_paths, discovered_count, night_output_dir = discover_review_jobs(
                audio_dir=audio_dir,
                json_dir=json_dir,
                output_dir=output_dir,
                requested_night_token="20260518",
                night_start_hour=18,
                night_end_hour=12,
            )

            self.assertEqual(discovered_count, 2)
            self.assertEqual(missing_json_paths, [])
            self.assertEqual([job.audio_path.name for job in jobs], ["20260518_235500T.WAV", "20260519_000500T.WAV"])
            self.assertEqual(night_output_dir, output_dir / "20260518")

    def test_export_review_batch_writes_summary(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            audio_dir = root / "audio"
            json_dir = root / "json"
            output_dir = root / "out"
            audio_dir.mkdir()
            json_dir.mkdir()

            audio_path = audio_dir / "20260518_220000T.WAV"
            json_path = json_dir / "20260518_220000T.WAV.json"
            audio_path.write_bytes(b"wav")
            json_path.write_text("{}", encoding="utf-8")

            with patch("batpipe.review.batch.export_review_clip") as mock_export:
                mock_export.return_value = {
                    "sample_local_time": "220000",
                    "clip_wav": "clip.wav",
                    "audible_wav": "audible.wav",
                    "clip_mp3": "clip.mp3",
                    "audible_mp3": "audible.mp3",
                    "spectrogram_png": "spec.png",
                    "report_json": "report.json",
                    "clip_start_s": 0.0,
                    "clip_end_s": 10.0,
                    "selected_bout_start_s": 5.0,
                    "selected_bout_end_s": 5.2,
                    "expanded_train_start_s": 0.2,
                    "expanded_train_end_s": 6.2,
                    "expanded_train_segment_count": 1,
                    "detections_in_clip": 2,
                }

                result = export_review_batch(
                    audio_dir=audio_dir,
                    json_dir=json_dir,
                    output_dir=output_dir,
                    write_mp3=True,
                    requested_night_token="20260518",
                    night_start_hour=18,
                    night_end_hour=12,
                )

            self.assertEqual(result["discovered_audio_files"], 1)
            self.assertEqual(result["matched_job_count"], 1)
            self.assertEqual(result["exported_count"], 1)
            self.assertEqual(result["failed_count"], 0)
            self.assertEqual(result["missing_json_count"], 0)
            night_output_dir = output_dir / "20260518"
            self.assertTrue((night_output_dir / "batch_summary.json").exists())
            self.assertTrue((night_output_dir / "review_assets.csv").exists())
            summary = json.loads((night_output_dir / "batch_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["items"][0]["audio_file"], str(audio_path))
            self.assertEqual(summary["items"][0]["json_file"], str(json_path))
            self.assertEqual(summary["items"][0]["clip_mp3"], "clip.mp3")
            self.assertEqual(summary["night_output_dir"], str(night_output_dir))
            self.assertEqual(summary["requested_night_token"], "20260518")
            self.assertEqual(summary["review_assets_csv"], str(night_output_dir / "review_assets.csv"))
            assets_csv = (night_output_dir / "review_assets.csv").read_text(encoding="utf-8")
            self.assertIn("sample_local_time", assets_csv)
            self.assertIn("220000", assets_csv)
            mock_export.assert_called_once()