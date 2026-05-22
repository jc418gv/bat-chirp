from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from batpipe.review.model_detection import DetectionBout
from batpipe.review.model_review import SpectrogramConfig


@dataclass(slots=True)
class SpectrogramAnalysis:
    waveform: Any
    clip_duration_s: float
    frequencies_hz: Any
    times_s: Any
    spectrum_db: Any
    noise_floor_db: Any
    excess_db: Any


@dataclass(slots=True)
class DetectionBandAnalysis:
    band_low_hz: float
    band_high_hz: float
    band_frequencies_hz: Any
    band_spectrum_db: Any
    band_envelope_db: Any
    dominant_bin_share: Any
    normalized_entropy: Any
    concentration_score: Any


def compute_per_frequency_excess_db(spectrum_db, percentile: float):
    import numpy as np

    if not 0.0 <= percentile <= 100.0:
        raise ValueError("noise_floor_percentile must be between 0 and 100.")

    spectrum = np.asarray(spectrum_db, dtype=np.float32)
    finite_mask = np.isfinite(spectrum)
    if not finite_mask.any():
        noise_floor_db = np.zeros((spectrum.shape[0],), dtype=np.float32)
        excess_db = np.zeros_like(spectrum, dtype=np.float32)
        return noise_floor_db, excess_db

    safe_spectrum = spectrum.copy()
    global_floor = float(np.nanpercentile(safe_spectrum[finite_mask], percentile))
    safe_spectrum[~finite_mask] = global_floor
    noise_floor_db = np.nanpercentile(safe_spectrum, percentile, axis=1).astype(np.float32)
    excess_db = (safe_spectrum - noise_floor_db[:, np.newaxis]).astype(np.float32)
    np.maximum(excess_db, 0.0, out=excess_db)
    return noise_floor_db, excess_db


def compute_spectrogram_db(audio, sample_rate_hz: int, config: SpectrogramConfig) -> SpectrogramAnalysis:
    import numpy as np
    from scipy import signal

    waveform = np.asarray(audio)
    if waveform.size == 0:
        raise ValueError("Clip audio is empty.")

    if np.issubdtype(waveform.dtype, np.integer):
        info = np.iinfo(waveform.dtype)
        scale = max(abs(info.min), info.max)
        waveform = waveform.astype(np.float32) / float(scale)
    else:
        waveform = waveform.astype(np.float32)

    clip_duration_s = waveform.size / float(sample_rate_hz)
    nperseg = min(config.nperseg, waveform.size)
    noverlap = max(0, int(nperseg * config.noverlap_ratio))
    frequencies_hz, times_s, spectrum = signal.spectrogram(
        waveform,
        fs=sample_rate_hz,
        nperseg=nperseg,
        noverlap=noverlap,
        mode="magnitude",
    )
    spectrum_db = 20.0 * np.log10(np.maximum(spectrum, 1e-12))
    noise_floor_db, excess_db = compute_per_frequency_excess_db(
        spectrum_db,
        percentile=config.noise_floor_percentile,
    )
    return SpectrogramAnalysis(
        waveform=waveform,
        clip_duration_s=clip_duration_s,
        frequencies_hz=frequencies_hz,
        times_s=times_s,
        spectrum_db=spectrum_db,
        noise_floor_db=noise_floor_db,
        excess_db=excess_db,
    )


def estimate_band_envelope_db(
    spectrum_db,
    frequencies_hz,
    selected_bout: DetectionBout,
    max_freq_hz: float,
    config: SpectrogramConfig,
):
    import numpy as np
    from scipy import ndimage

    band_low_hz = max(0.0, (selected_bout.min_low_freq_hz or 0.0) - config.band_margin_hz)
    band_high_hz = min(max_freq_hz, (selected_bout.max_high_freq_hz or max_freq_hz) + config.band_margin_hz)
    band_mask = (frequencies_hz >= band_low_hz) & (frequencies_hz <= band_high_hz)
    if not band_mask.any():
        return None

    # The caller passes excess-above-floor dB here, not raw spectrogram dB.
    # This is the audio equivalent of subtracting a background image: each
    # frequency bin is normalized against its own robust low-percentile floor,
    # so broad raised noise does not masquerade as activity merely because the
    # whole selected band is louder than usual.
    band_spectrum = spectrum_db[band_mask].copy()
    np.maximum(band_spectrum, 0.0, out=band_spectrum)
    band_energy = np.maximum(np.expm1(np.log(10.0) * np.clip(band_spectrum, 0.0, 80.0) / 10.0), 1e-12)
    frame_energy = np.sum(band_energy, axis=0)
    zero_energy_mask = frame_energy <= 1e-9
    frame_energy = np.where(zero_energy_mask, 1.0, frame_energy)
    probabilities = band_energy / frame_energy[np.newaxis, :]
    dominant_bin_share = np.max(probabilities, axis=0)
    entropy = -np.sum(probabilities * np.log(probabilities), axis=0)
    max_entropy = np.log(float(max(2, band_energy.shape[0])))
    normalized_entropy = np.clip(entropy / max_entropy, 0.0, 1.0)
    concentration_score = 0.5 * (dominant_bin_share + (1.0 - normalized_entropy))
    dominant_bin_share[zero_energy_mask] = 0.0
    normalized_entropy[zero_energy_mask] = 1.0
    concentration_score[zero_energy_mask] = 0.0
    band_envelope_db = np.nanpercentile(band_spectrum, config.envelope_percentile, axis=0)
    band_envelope_db = ndimage.gaussian_filter1d(band_envelope_db, sigma=config.gaussian_sigma, mode="nearest")
    return DetectionBandAnalysis(
        band_low_hz=band_low_hz,
        band_high_hz=band_high_hz,
        band_frequencies_hz=frequencies_hz[band_mask],
        band_spectrum_db=band_spectrum,
        band_envelope_db=band_envelope_db,
        dominant_bin_share=dominant_bin_share,
        normalized_entropy=normalized_entropy,
        concentration_score=concentration_score,
    )