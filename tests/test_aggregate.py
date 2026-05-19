from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest

from batpipe.aggregate import summarize_detection_directory


class AggregateTests(unittest.TestCase):
    def test_summarize_detection_directory(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()

            payload = {
                "annotated": False,
                "annotation": [
                    {
                        "class": "unknown",
                        "class_prob": 0.81,
                        "det_prob": 0.92,
                        "start_time": 0.1,
                        "end_time": 0.15,
                        "low_freq": 32000,
                        "high_freq": 47000,
                    },
                    {
                        "class": "unknown",
                        "class_prob": 0.66,
                        "det_prob": 0.74,
                        "start_time": 0.3,
                        "end_time": 0.34,
                        "low_freq": 28000,
                        "high_freq": 43000,
                    },
                ],
            }
            (input_dir / "20260518_020000T.WAV.json").write_text(
                json.dumps(payload),
                encoding="utf-8",
            )
            (input_dir / "20260518_020100T.WAV.json").write_text(
                json.dumps({"annotated": False, "annotation": []}),
                encoding="utf-8",
            )

            outputs = summarize_detection_directory(input_dir, output_dir)

            review_queue = Path(outputs["review_queue"]).read_text(encoding="utf-8")
            hourly_summary = Path(outputs["hourly_summary"]).read_text(encoding="utf-8")
            nightly_summary = Path(outputs["nightly_summary"]).read_text(encoding="utf-8")

            self.assertIn("20260518_020000T.WAV", review_queue)
            self.assertIn("2026-05-18 02:00:00", hourly_summary)
            self.assertIn("total_files", nightly_summary)


if __name__ == "__main__":
    unittest.main()