from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from batpipe.review.models import DetectionBout, SpectrogramConfig


@dataclass(slots=True)
class SpectrogramAnalysis:
    waveform: Any
    clip_duration_s: float
    frequencies_hz: Any
    times_s: Any
    spectrum_db: Any


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
    return SpectrogramAnalysis(
        waveform=waveform,
        clip_duration_s=clip_duration_s,
        frequencies_hz=frequencies_hz,
        times_s=times_s,
        spectrum_db=spectrum_db,
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

    band_spectrum = spectrum_db[band_mask].copy()
    band_spectrum -= np.nanmean(band_spectrum, axis=1, keepdims=True)
    np.maximum(band_spectrum, 0.0, out=band_spectrum)
    band_energy = np.maximum(band_spectrum, 1e-12)
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