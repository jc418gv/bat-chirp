import unittest

import numpy as np
from scipy import signal

from batpipe.noise_floor import NoiseReductionConfig, reduce_noise_floor


def _spectrogram_metrics(audio: np.ndarray, sample_rate_hz: int) -> tuple[float, float]:
    frequencies_hz, _, spectrum = signal.spectrogram(audio, fs=sample_rate_hz, nperseg=1024, noverlap=768)
    spectrum_db = 10.0 * np.log10(spectrum + 1e-20)
    low_band = frequencies_hz < 12_000.0
    bat_band = (frequencies_hz >= 24_000.0) & (frequencies_hz <= 48_000.0)
    low_floor_db = float(np.median(spectrum_db[low_band]))
    bat_peak_db = float(np.percentile(spectrum_db[bat_band], 99.0))
    return low_floor_db, bat_peak_db


class NoiseFloorTests(unittest.TestCase):
    def test_spectral_subtract_improves_band_contrast_more_than_soft_gate(self) -> None:
        sample_rate_hz = 256_000
        duration_s = 1.0
        time_s = np.arange(int(sample_rate_hz * duration_s), dtype=np.float32) / sample_rate_hz
        rng = np.random.default_rng(1234)

        low_noise = 0.08 * np.sin(2.0 * np.pi * 3_500.0 * time_s)
        broadband_noise = 0.025 * rng.standard_normal(time_s.shape, dtype=np.float32)
        bat_call = np.zeros_like(time_s)
        active_slice = slice(int(0.45 * sample_rate_hz), int(0.55 * sample_rate_hz))
        bat_call[active_slice] = 0.12 * np.sin(2.0 * np.pi * 32_000.0 * time_s[active_slice])
        audio = (low_noise + broadband_noise + bat_call).astype(np.float32)

        original_low_floor_db, original_bat_peak_db = _spectrogram_metrics(audio, sample_rate_hz)

        spectral_subtract_audio = reduce_noise_floor(
            audio,
            sample_rate_hz,
            NoiseReductionConfig(
                mode="spectral_subtract",
                spectral_subtract_oversubtract=2.5,
                spectral_subtract_floor_ratio=0.01,
                spectral_subtract_smoothing_bins=7,
            ),
        )
        spectral_subtract_low_floor_db, spectral_subtract_bat_peak_db = _spectrogram_metrics(spectral_subtract_audio, sample_rate_hz)

        original_contrast_db = original_bat_peak_db - original_low_floor_db
        spectral_subtract_contrast_db = spectral_subtract_bat_peak_db - spectral_subtract_low_floor_db

        self.assertLess(spectral_subtract_low_floor_db, original_low_floor_db - 2.0)
        self.assertGreater(spectral_subtract_bat_peak_db, original_bat_peak_db - 4.0)
        self.assertGreater(spectral_subtract_contrast_db, original_contrast_db)

    def test_reduce_noise_floor_rejects_unknown_mode(self) -> None:
        with self.assertRaisesRegex(ValueError, "noise reduction mode"):
            reduce_noise_floor(np.zeros(128, dtype=np.float32), 256_000, NoiseReductionConfig(mode="unknown"))


if __name__ == "__main__":
    unittest.main()