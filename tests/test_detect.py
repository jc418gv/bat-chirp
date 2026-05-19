from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from batpipe.detect import build_detection_plan


class DetectPlanTests(unittest.TestCase):
    def test_build_detection_plan_with_name_filter_and_limit(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            output_dir.mkdir()

            for name in [
                "20260518_020000T.WAV",
                "20260518_020100T.WAV",
                "20260518_030000T.WAV",
            ]:
                (input_dir / name).write_bytes(b"wav")

            plan = build_detection_plan(
                input_dir=input_dir,
                output_dir=output_dir,
                batdetect2_bin="batdetect2",
                model=None,
                detection_threshold=0.4,
                limit=1,
                name_filters=["20260518_020"],
                extra_args=[],
            )

            self.assertEqual(plan.audio_file_count, 2)
            self.assertEqual(plan.selected_file_count, 1)
            self.assertEqual(plan.invocation_mode, "file_list")
            self.assertEqual(plan.name_filters, ["20260518_020"])
            self.assertIsNotNone(plan.selected_files_manifest)
            manifest_path = Path(plan.selected_files_manifest or "")
            self.assertTrue(manifest_path.exists())
            contents = manifest_path.read_text(encoding="utf-8")
            self.assertIn("20260518_020000T.WAV", contents)

    def test_build_detection_plan_selects_one_overnight_session(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            output_dir.mkdir()

            for name in [
                "20260518_175900T.WAV",
                "20260518_235500T.WAV",
                "20260519_000500T.WAV",
                "20260519_120100T.WAV",
            ]:
                (input_dir / name).write_bytes(b"wav")

            plan = build_detection_plan(
                input_dir=input_dir,
                output_dir=output_dir,
                batdetect2_bin="batdetect2",
                model=None,
                detection_threshold=0.4,
                limit=None,
                name_filters=[],
                extra_args=[],
                night_token="20260518",
                night_start_hour=18,
                night_end_hour=12,
            )

            self.assertEqual(plan.audio_file_count, 2)
            self.assertEqual(plan.selected_file_count, 2)
            self.assertEqual(plan.invocation_mode, "file_list")
            manifest_path = Path(plan.selected_files_manifest or "")
            self.assertTrue(manifest_path.exists())
            contents = manifest_path.read_text(encoding="utf-8")
            self.assertIn("20260518_235500T.WAV", contents)
            self.assertIn("20260519_000500T.WAV", contents)
            self.assertNotIn("20260518_175900T.WAV", contents)
            self.assertNotIn("20260519_120100T.WAV", contents)


if __name__ == "__main__":
    unittest.main()
