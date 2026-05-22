from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import istft, stft

EPS = 1e-20


@dataclass(slots=True)
class NoiseReductionConfig:
    n_fft: int = 1024
    hop: int = 128
    noise_floor_percentile: float = 20.0
    margin_db: float = 6.0
    softness_db: float = 3.0
    floor_gain: float = 0.05


def _as_float_audio(audio: np.ndarray) -> tuple[np.ndarray, np.dtype | None]:
    original_dtype = audio.dtype
    if np.issubdtype(original_dtype, np.integer):
        info = np.iinfo(original_dtype)
        scale = float(max(abs(info.min), info.max))
        return audio.astype(np.float32) / scale, original_dtype
    return audio.astype(np.float32), None


def _from_float_audio(audio: np.ndarray, original_dtype: np.dtype | None) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    audio = np.clip(audio, -1.0, 1.0)
    if original_dtype is None:
        return audio
    info = np.iinfo(original_dtype)
    peak = float(min(abs(info.min), info.max))
    return np.clip(audio * peak, info.min, info.max).astype(original_dtype)


def reduce_noise_floor(audio: np.ndarray, sample_rate_hz: int, config: NoiseReductionConfig | None = None) -> np.ndarray:
    config = config or NoiseReductionConfig()
    if audio.size == 0:
        return audio.astype(np.float32)
    if config.n_fft <= 0:
        raise ValueError("noise reduction n_fft must be positive.")
    if config.hop <= 0:
        raise ValueError("noise reduction hop must be positive.")
    if config.hop > config.n_fft:
        raise ValueError("noise reduction hop must be less than or equal to n_fft.")
    if not 0.0 <= config.noise_floor_percentile <= 100.0:
        raise ValueError("noise reduction noise_floor_percentile must be between 0 and 100.")

    waveform = np.asarray(audio, dtype=np.float32)
    if waveform.ndim == 1:
        return _reduce_noise_floor_channel(waveform, sample_rate_hz, config)
    channels = [
        _reduce_noise_floor_channel(waveform[:, channel], sample_rate_hz, config)
        for channel in range(waveform.shape[1])
    ]
    return np.stack(channels, axis=1).astype(np.float32)


def _reduce_noise_floor_channel(waveform: np.ndarray, sample_rate_hz: int, config: NoiseReductionConfig) -> np.ndarray:
    nperseg = min(config.n_fft, int(waveform.size))
    if nperseg < 2:
        return waveform.astype(np.float32)
    hop = min(config.hop, nperseg)
    noverlap = nperseg - hop

    _, _, spectrum = stft(
        waveform,
        fs=sample_rate_hz,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        boundary="zeros",
        padded=True,
    )
    power = np.abs(spectrum) ** 2
    noise_power = np.percentile(power, config.noise_floor_percentile, axis=1)
    power_db = 10.0 * np.log10(power + EPS)
    noise_db = 10.0 * np.log10(noise_power[:, np.newaxis] + EPS)
    excess_db = power_db - noise_db

    mask = 1.0 / (1.0 + np.exp(-(excess_db - config.margin_db) / max(config.softness_db, 1e-6)))
    mask = config.floor_gain + (1.0 - config.floor_gain) * mask
    filtered_spectrum = spectrum * mask

    _, enhanced = istft(
        filtered_spectrum,
        fs=sample_rate_hz,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        input_onesided=True,
        boundary=True,
    )
    if enhanced.size > waveform.size:
        enhanced = enhanced[: waveform.size]
    elif enhanced.size < waveform.size:
        enhanced = np.pad(enhanced, (0, waveform.size - enhanced.size))
    return enhanced.astype(np.float32)


def reduce_noise_floor_wav(source_path: Path, output_path: Path, config: NoiseReductionConfig | None = None) -> Path:
    sample_rate_hz, audio = wavfile.read(source_path)
    float_audio, original_dtype = _as_float_audio(np.asarray(audio))
    enhanced = reduce_noise_floor(float_audio, sample_rate_hz, config=config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(output_path, sample_rate_hz, _from_float_audio(enhanced, original_dtype))
    return output_path


def reduce_noise_for_files(
    audio_paths: list[Path],
    input_dir: Path,
    output_dir: Path,
    config: NoiseReductionConfig | None = None,
) -> list[Path]:
    written_paths: list[Path] = []
    for audio_path in audio_paths:
        try:
            relative_path = audio_path.relative_to(input_dir)
        except ValueError:
            relative_path = Path(audio_path.name)
        output_path = output_dir / relative_path
        written_paths.append(reduce_noise_floor_wav(audio_path, output_path, config=config))
    return written_paths