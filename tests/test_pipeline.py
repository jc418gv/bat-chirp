from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest
from unittest.mock import patch

from batpipe.site_config import SiteConfig
from batpipe.pipeline import run_night_pipeline


class NightPipelineTests(unittest.TestCase):
    def test_run_night_pipeline_executes_detection_summary_and_review(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            recordings = root / "recordings"
            detections = root / "detections"
            summary = root / "summary"
            review = root / "review"
            recordings.mkdir()
            (recordings / "20260518_220000T.WAV").write_bytes(b"wav")

            config = SiteConfig(
                recording_input_dir=str(recordings),
                detection_output_dir=str(detections),
                summary_output_dir=str(summary),
                review_output_dir=str(review),
                night_token="20260518",
                night_start_hour=18,
                night_end_hour=12,
                detection_threshold=0.5,
                subset_limit=1,
            )

            with (
                patch("batpipe.pipeline.run_detection_plan") as mock_run_detection,
                patch("batpipe.pipeline.summarize_detection_directory") as mock_summarize,
                patch("batpipe.pipeline.export_review_batch") as mock_export_review,
                patch("batpipe.pipeline.build_review_site") as mock_build_review_site,
            ):
                mock_summarize.return_value = {"nightly_summary_csv": summary / "nightly_summary.csv"}
                mock_export_review.return_value = {
                    "exported_count": 1,
                    "summary_json": str(review / "20260518" / "batch_summary.json"),
                    "night_output_dir": str(review / "20260518"),
                    "items": [{"audio_file": str(recordings / "20260518_220000T.WAV")}],
                }
                mock_build_review_site.return_value = {"review_index_html": str(review / "20260518" / "summary" / "index.html")}

                result = run_night_pipeline(config)

            self.assertTrue((detections / "run_manifest.json").exists())
            mock_run_detection.assert_called_once()
            mock_summarize.assert_called_once_with(detections.resolve(), summary.resolve())
            mock_export_review.assert_called_once()
            mock_build_review_site.assert_called_once()
            self.assertEqual(result["selected_audio_files"], 1)
            self.assertIn("review_outputs", result)
            self.assertEqual(result["night_token"], "20260518")
            json.dumps(result)

    def test_run_night_pipeline_dry_run_skips_follow_on_stages(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            recordings = root / "recordings"
            detections = root / "detections"
            recordings.mkdir()
            (recordings / "20260518_020000T.WAV").write_bytes(b"wav")

            config = SiteConfig(
                recording_input_dir=str(recordings),
                detection_output_dir=str(detections),
            )

            with (
                patch("batpipe.pipeline.run_detection_plan") as mock_run_detection,
                patch("batpipe.pipeline.summarize_detection_directory") as mock_summarize,
                patch("batpipe.pipeline.export_review_batch") as mock_export_review,
            ):
                result = run_night_pipeline(config, dry_run=True)

            mock_run_detection.assert_called_once()
            mock_summarize.assert_not_called()
            mock_export_review.assert_not_called()
            self.assertTrue(result["dry_run"])

    def test_run_night_pipeline_reports_unwritable_detection_output_dir(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            recordings = root / "recordings"
            detections = root / "detections"
            recordings.mkdir()
            (recordings / "20260518_020000T.WAV").write_bytes(b"wav")

            config = SiteConfig(
                recording_input_dir=str(recordings),
                detection_output_dir=str(detections),
            )

            with patch("batpipe.pipeline.Path.mkdir", side_effect=PermissionError("denied")):
                with self.assertRaisesRegex(ValueError, "detection_output_dir is not writable"):
                    run_night_pipeline(config)

    def test_run_night_pipeline_forwards_progress_events_to_review_export(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            recordings = root / "recordings"
            detections = root / "detections"
            summary = root / "summary"
            review = root / "review"
            recordings.mkdir()
            (recordings / "20260518_220000T.WAV").write_bytes(b"wav")

            config = SiteConfig(
                recording_input_dir=str(recordings),
                detection_output_dir=str(detections),
                summary_output_dir=str(summary),
                review_output_dir=str(review),
                night_token="20260518",
                night_start_hour=18,
                night_end_hour=12,
                detection_threshold=0.5,
                subset_limit=1,
            )

            events: list[str] = []

            def capture_progress(event: str, payload: dict[str, object]) -> None:
                events.append(event)

            def fake_export_review_batch(*args, **kwargs):
                progress_callback = kwargs.get("progress_callback")
                if progress_callback is not None:
                    progress_callback("batch_started", {"matched_job_count": 1, "missing_json_count": 0})
                    progress_callback("item_completed", {"index": 1, "total": 1, "audio_file": "clip.wav"})
                    progress_callback("batch_completed", {"exported_count": 1, "failed_count": 0})
                return {
                    "exported_count": 1,
                    "summary_json": str(review / "20260518" / "batch_summary.json"),
                    "night_output_dir": str(review / "20260518"),
                    "items": [{"audio_file": str(recordings / "20260518_220000T.WAV")}],
                }

            with (
                patch("batpipe.pipeline.run_detection_plan") as mock_run_detection,
                patch("batpipe.pipeline.summarize_detection_directory") as mock_summarize,
                patch("batpipe.pipeline.export_review_batch", side_effect=fake_export_review_batch) as mock_export_review,
                patch("batpipe.pipeline.build_review_site") as mock_build_review_site,
            ):
                mock_summarize.return_value = {"nightly_summary_csv": summary / "nightly_summary.csv"}
                mock_build_review_site.return_value = {"review_index_html": str(review / "20260518" / "summary" / "index.html")}

                run_night_pipeline(config, progress_callback=capture_progress)

            mock_run_detection.assert_called_once()
            mock_export_review.assert_called_once()
            self.assertEqual(
                events,
                [
                    "detection_started",
                    "summary_started",
                    "review_started",
                    "batch_started",
                    "item_completed",
                    "batch_completed",
                    "review_site_started",
                    "pipeline_completed",
                ],
            )