from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from batpipe.review_site import build_review_site


class ReviewSiteTests(unittest.TestCase):
    def test_build_review_site_writes_html_pages_with_spectrogram_links(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            night_dir = root / "review" / "20260518"
            item_dir = night_dir / "20260518_020000T"
            item_dir.mkdir(parents=True)
            spectrogram = item_dir / "spectrogram_020000.png"
            report = item_dir / "detections_020000.json"
            clip_wav = item_dir / "clip_original_020000.wav"
            clip_mp3 = item_dir / "clip_original_020000.mp3"
            audible_wav = item_dir / "clip_audible_x8_020000.wav"
            audible_mp3 = item_dir / "clip_audible_x8_020000.mp3"
            for path in [spectrogram, report, clip_wav, clip_mp3, audible_wav, audible_mp3]:
                path.write_text("x", encoding="utf-8")

            result = build_review_site(
                night_output_dir=night_dir,
                review_items=[
                    {
                        "audio_file": str(root / "recordings" / "20260518_020000T.WAV"),
                        "sample_local_time": "020000",
                        "spectrogram_png": str(spectrogram),
                        "report_json": str(report),
                        "clip_wav": str(clip_wav),
                        "clip_mp3": str(clip_mp3),
                        "audible_wav": str(audible_wav),
                        "audible_mp3": str(audible_mp3),
                        "clip_start_s": 0.0,
                        "clip_end_s": 10.0,
                        "expanded_train_segment_count": 1,
                        "detections_in_clip": 2,
                    }
                ],
            )

            index_html = Path(str(result["review_index_html"]))
            hour_html = Path(str(result["review_hour_pages"][0]))
            self.assertTrue(index_html.exists())
            self.assertTrue(hour_html.exists())
            self.assertEqual(index_html.parent, night_dir)
            html = index_html.read_text(encoding="utf-8")
            self.assertIn("Night Review 20260518", html)
            self.assertIn("<details class=\"hour-group\" open>", html)
            self.assertIn("Hour 2026-05-18 02:00", html)
            self.assertIn("hour-26051802.html", html)
            self.assertIn("spectrogram_020000.png", html)
            self.assertIn("clip_original_020000.mp3", html)

    def test_build_review_site_orders_overnight_hours_by_full_timestamp(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            night_dir = root / "review" / "20260518"
            first_item_dir = night_dir / "20260518_210000T"
            second_item_dir = night_dir / "20260519_000500T"
            first_item_dir.mkdir(parents=True)
            second_item_dir.mkdir(parents=True)

            first_spectrogram = first_item_dir / "spectrogram_210000.png"
            second_spectrogram = second_item_dir / "spectrogram_000500.png"
            first_report = first_item_dir / "detections_210000.json"
            second_report = second_item_dir / "detections_000500.json"
            first_clip_wav = first_item_dir / "clip_original_210000.wav"
            second_clip_wav = second_item_dir / "clip_original_000500.wav"
            first_clip_mp3 = first_item_dir / "clip_original_210000.mp3"
            second_clip_mp3 = second_item_dir / "clip_original_000500.mp3"
            first_audible_wav = first_item_dir / "clip_audible_x8_210000.wav"
            second_audible_wav = second_item_dir / "clip_audible_x8_000500.wav"
            first_audible_mp3 = first_item_dir / "clip_audible_x8_210000.mp3"
            second_audible_mp3 = second_item_dir / "clip_audible_x8_000500.mp3"

            for path in [
                first_spectrogram,
                second_spectrogram,
                first_report,
                second_report,
                first_clip_wav,
                second_clip_wav,
                first_clip_mp3,
                second_clip_mp3,
                first_audible_wav,
                second_audible_wav,
                first_audible_mp3,
                second_audible_mp3,
            ]:
                path.write_text("x", encoding="utf-8")

            result = build_review_site(
                night_output_dir=night_dir,
                review_items=[
                    {
                        "audio_file": str(root / "recordings" / "20260519_000500T.WAV"),
                        "sample_local_time": "000500",
                        "spectrogram_png": str(second_spectrogram),
                        "report_json": str(second_report),
                        "clip_wav": str(second_clip_wav),
                        "clip_mp3": str(second_clip_mp3),
                        "audible_wav": str(second_audible_wav),
                        "audible_mp3": str(second_audible_mp3),
                        "clip_start_s": 0.0,
                        "clip_end_s": 10.0,
                        "expanded_train_segment_count": 1,
                        "detections_in_clip": 1,
                    },
                    {
                        "audio_file": str(root / "recordings" / "20260518_210000T.WAV"),
                        "sample_local_time": "210000",
                        "spectrogram_png": str(first_spectrogram),
                        "report_json": str(first_report),
                        "clip_wav": str(first_clip_wav),
                        "clip_mp3": str(first_clip_mp3),
                        "audible_wav": str(first_audible_wav),
                        "audible_mp3": str(first_audible_mp3),
                        "clip_start_s": 0.0,
                        "clip_end_s": 10.0,
                        "expanded_train_segment_count": 1,
                        "detections_in_clip": 1,
                    },
                ],
            )

            html = Path(str(result["review_index_html"])).read_text(encoding="utf-8")
            self.assertLess(html.index("Hour 2026-05-18 21:00"), html.index("Hour 2026-05-19 00:00"))
            self.assertIn("hour-26051821.html", html)
            self.assertIn("hour-26051900.html", html)