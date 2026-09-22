"""
Audio analysis module for osu!mania 4K auto-mapper.

Performs waveform parsing, short-time energy calculation, transient/onset detection,
and tempo/BPM autocorrelation grid estimation with millisecond precision.
"""

from __future__ import annotations

import math
import struct
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple


@dataclass
class AudioRhythmResult:
    bpm: float
    offset_ms: int
    duration_sec: float
    duration_ms: int
    sample_rate: int
    onsets_ms: List[int] = field(default_factory=list)
    rms_profile: List[Tuple[int, float]] = field(default_factory=list)  # (time_ms, normalized_rms)


def read_wav_mono(wav_path: str) -> Tuple[List[float], int]:
    """Reads a WAV file and returns normalized mono floating-point samples [-1.0, 1.0] and sample rate."""
    with wave.open(str(wav_path), "rb") as wf:
        num_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        num_frames = wf.getnframes()
        raw_bytes = wf.readframes(num_frames)

    # Decode audio frames based on sample width
    samples: List[float] = []
    if sample_width == 2:  # 16-bit PCM
        total_samples = len(raw_bytes) // 2
        ints = struct.unpack(f"<{total_samples}h", raw_bytes)
        if num_channels == 1:
            samples = [x / 32768.0 for x in ints]
        else:
            # Average stereo / multi-channel to mono
            samples = []
            for i in range(0, len(ints), num_channels):
                mono_val = sum(ints[i:i + num_channels]) / (num_channels * 32768.0)
                samples.append(mono_val)
    elif sample_width == 1:  # 8-bit unsigned
        if num_channels == 1:
            samples = [(x - 128) / 128.0 for x in raw_bytes]
        else:
            samples = []
            for i in range(0, len(raw_bytes), num_channels):
                chunk = raw_bytes[i:i + num_channels]
                mono_val = sum((x - 128) / 128.0 for x in chunk) / num_channels
                samples.append(mono_val)
    elif sample_width == 3:  # 24-bit PCM
        samples = []
        bytes_per_sample = 3 * num_channels
        total_frames = len(raw_bytes) // bytes_per_sample
        for f in range(total_frames):
            frame_vals = []
            for ch in range(num_channels):
                idx = f * bytes_per_sample + ch * 3
                b = raw_bytes[idx:idx + 3]
                val = int.from_bytes(b, byteorder="little", signed=True)
                frame_vals.append(val / 8388608.0)
            samples.append(sum(frame_vals) / num_channels)
    else:
        raise ValueError(f"Unsupported WAV sample width: {sample_width} bytes")

    return samples, sample_rate


def create_synthetic_click_wav(
    output_path: str,
    duration_sec: float = 60.0,
    click_times_ms: List[int] = None,
    sample_rate: int = 44100,
    bpm: float = 120.0,
) -> str:
    """Generates a synthetic WAV with sharp transient clicks at designated millisecond positions."""
    if click_times_ms is None:
        interval_ms = 60000.0 / bpm
        click_times_ms = [int(round(i * interval_ms)) for i in range(int(duration_sec * 1000 / interval_ms))]

    total_samples = int(duration_sec * sample_rate)
    buffer = [0.0] * total_samples

    # Inject sharp 10ms click impulses with exponential decay
    click_len_samples = int(sample_rate * 0.010)
    for t_ms in click_times_ms:
        start_idx = int((t_ms / 1000.0) * sample_rate)
        for i in range(click_len_samples):
            idx = start_idx + i
            if idx < total_samples:
                decay = math.exp(-i / (sample_rate * 0.002))
                # 1 kHz sinusoidal burst
                val = 0.95 * math.sin(2.0 * math.pi * 1000.0 * (i / sample_rate)) * decay
                buffer[idx] += val
                if buffer[idx] > 1.0:
                    buffer[idx] = 1.0
                elif buffer[idx] < -1.0:
                    buffer[idx] = -1.0

    # Write WAV 16-bit PCM
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        int_data = [max(-32768, min(32767, int(s * 32767.0))) for s in buffer]
        raw = struct.pack(f"<{len(int_data)}h", *int_data)
        wf.writeframes(raw)

    return output_path


def detect_onsets(samples: List[float], sample_rate: int, hop_size_ms: int = 5) -> List[int]:
    """Detects sharp transient peaks (onsets) with millisecond timestamps."""
    hop_samples = max(1, int(sample_rate * (hop_size_ms / 1000.0)))
    num_hops = len(samples) // hop_samples

    energies: List[float] = []
    for h in range(num_hops):
        start = h * hop_samples
        end = start + hop_samples
        chunk = samples[start:end]
        if chunk:
            rms = math.sqrt(sum(s * s for s in chunk) / len(chunk))
        else:
            rms = 0.0
        energies.append(rms)

    if not energies:
        return []

    # Spectral/energy flux (half-wave rectified difference)
    flux: List[float] = [0.0]
    for i in range(1, len(energies)):
        diff = energies[i] - energies[i - 1]
        flux.append(max(0.0, diff))

    # Adaptive thresholding
    mean_flux = sum(flux) / len(flux)
    std_flux = math.sqrt(sum((f - mean_flux) ** 2 for f in flux) / len(flux)) if len(flux) > 1 else 0.001
    thresh = mean_flux + 1.2 * std_flux

    onsets_ms: List[int] = []
    min_dist_hops = int(60 / hop_size_ms)  # at least 60ms between distinct onsets
    last_onset_hop = -min_dist_hops

    for i in range(1, len(flux) - 1):
        if flux[i] > thresh and flux[i] > flux[i - 1] and flux[i] >= flux[i + 1]:
            if (i - last_onset_hop) >= min_dist_hops:
                # Precise sample peak within this hop window
                start_sample = (i - 1) * hop_samples
                end_sample = min(len(samples), (i + 2) * hop_samples)
                best_sub = start_sample
                best_val = -1.0
                for s_idx in range(start_sample, end_sample):
                    mag = abs(samples[s_idx])
                    if mag > best_val:
                        best_val = mag
                        best_sub = s_idx
                
                onset_ms = int(round((best_sub / sample_rate) * 1000.0))
                onsets_ms.append(onset_ms)
                last_onset_hop = i

    return onsets_ms


def analyze_audio_rhythm(wav_path: str) -> AudioRhythmResult:
    """Analyzes a WAV audio file to extract BPM, initial offset, transients, and RMS dynamics."""
    samples, sample_rate = read_wav_mono(wav_path)
    total_samples = len(samples)
    duration_sec = total_samples / float(sample_rate) if sample_rate > 0 else 0.0
    duration_ms = int(round(duration_sec * 1000.0))

    onsets = detect_onsets(samples, sample_rate, hop_size_ms=5)

    # Estimate tempo (BPM) via inter-onset intervals (IOI) histogram & grid scoring
    detected_bpm = 120.0
    detected_offset = 0

    if len(onsets) >= 2:
        intervals: List[int] = []
        for i in range(len(onsets)):
            for j in range(i + 1, min(len(onsets), i + 8)):
                diff = onsets[j] - onsets[i]
                if 180 <= diff <= 1200:  # corresponds to 50 - 333 BPM
                    intervals.append(diff)

        # Candidate BPM evaluation from 70 to 240 in 0.5 steps
        best_score = -1.0
        best_bpm = 120.0
        for candidate in [b * 0.5 for b in range(140, 480)]:
            beat_ms = 60000.0 / candidate
            score = 0.0
            for diff in intervals:
                # Proximity to multiples 0.5, 1, 2, 4
                for mult in [0.5, 1.0, 2.0]:
                    target = beat_ms * mult
                    err = abs(diff - target)
                    if err <= 15.0:
                        score += (1.0 - err / 15.0)
            if score > best_score:
                best_score = score
                best_bpm = candidate

        detected_bpm = round(best_bpm, 2)

        # Estimate phase / initial offset: pick the onset that minimizes grid error for first 16 beats
        beat_interval = 60000.0 / detected_bpm
        best_phase_err = float("inf")
        best_offset = onsets[0]

        for cand_offset in onsets[:min(10, len(onsets))]:
            total_err = 0.0
            matched = 0
            for k in range(min(32, int(duration_ms / beat_interval))):
                expected = cand_offset + k * beat_interval
                # Find closest detected onset
                closest_dist = min(abs(expected - o) for o in onsets)
                if closest_dist < 40.0:
                    total_err += closest_dist
                    matched += 1
            if matched > 0:
                avg_err = total_err / matched - (matched * 2.0)
                if avg_err < best_phase_err:
                    best_phase_err = avg_err
                    best_offset = cand_offset

        detected_offset = int(round(best_offset))
    elif onsets:
        detected_offset = onsets[0]

    # Calculate local RMS profile across 250ms chunks for dynamic SV mapping
    chunk_size = int(sample_rate * 0.25)
    rms_profile: List[Tuple[int, float]] = []
    max_rms = 0.0001
    cur_sample = 0
    raw_rms_list: List[Tuple[int, float]] = []

    while cur_sample < total_samples:
        chunk = samples[cur_sample:cur_sample + chunk_size]
        t_ms = int((cur_sample / sample_rate) * 1000.0)
        rms = math.sqrt(sum(x * x for x in chunk) / len(chunk)) if chunk else 0.0
        raw_rms_list.append((t_ms, rms))
        if rms > max_rms:
            max_rms = rms
        cur_sample += chunk_size

    for t_ms, rms in raw_rms_list:
        norm = rms / max_rms
        rms_profile.append((t_ms, round(norm, 4)))

    return AudioRhythmResult(
        bpm=detected_bpm,
        offset_ms=detected_offset,
        duration_sec=duration_sec,
        duration_ms=duration_ms,
        sample_rate=sample_rate,
        onsets_ms=onsets,
        rms_profile=rms_profile,
    )


def predict_bpm_offset_with_model(
    wav_path: str,
    checkpoint_path: str,
    device: str = "auto",
) -> Tuple[float, int]:
    """
    Uses trained SyncBPMOffsetModel (Model 1) to regress BPM and downbeat offset (ms) directly
    from the audio Mel spectrogram representation.
    """
    try:
        import torch
        from osu_mapper.models.sync_model import SyncBPMOffsetModel
        from osu_mapper.dataset.feature_extractor import compute_mel_spectrogram, extract_audio_waveform
    except ImportError as e:
        raise RuntimeError(f"PyTorch is required for model-driven rhythm prediction: {e}")

    if device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)

    model = SyncBPMOffsetModel()
    checkpoint = torch.load(checkpoint_path, map_location=dev)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(dev)
    model.eval()

    # Extract audio waveform and compute Mel spectrogram
    audio = extract_audio_waveform(wav_path, target_sr=16000)
    mel = compute_mel_spectrogram(audio, sr=16000, n_mels=80)

    # Use first sequence window (up to 1024 frames = ~16s)
    seq_len = min(1024, mel.shape[0])
    inp = mel[:seq_len].unsqueeze(0).to(dev)  # [1, seq_len, 80]

    with torch.no_grad():
        bpm_out, offset_out = model(inp)
        pred_bpm = float(bpm_out.squeeze().item())
        pred_offset = int(round(float(offset_out.squeeze().item())))

    pred_bpm = max(45.0, min(360.0, pred_bpm))
    pred_offset = max(0, pred_offset)
    return pred_bpm, pred_offset
