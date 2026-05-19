from datetime import datetime
import unittest

from batpipe.audiomoth import normalize_recording_name, parse_audiomoth_timestamp


class AudioMothTests(unittest.TestCase):
    def test_parse_timestamp_from_wav_name(self) -> None:
        self.assertEqual(
            parse_audiomoth_timestamp("20260518_020000T.WAV"),
            datetime(2026, 5, 18, 2, 0, 0),
        )

    def test_parse_timestamp_from_json_name(self) -> None:
        self.assertEqual(
            parse_audiomoth_timestamp("20260518_020100T.WAV.json"),
            datetime(2026, 5, 18, 2, 1, 0),
        )

    def test_normalize_recording_name(self) -> None:
        self.assertEqual(
            normalize_recording_name("20260518_020100T.WAV.json"),
            "20260518_020100T.WAV",
        )


if __name__ == "__main__":
    unittest.main()
