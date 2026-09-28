"""
Feature extraction pipeline for osu!mania 4K beatmap dataset.
"""

from __future__ import annotations

import math
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None  # type: ignore
    F = None  # type: ignore

from osu_mapper.dataset.difficulty_calculator import calculate_osu_mania_star_rating
from osu_mapper.pipeline.timing import (
    MusicalGridTick,
    generate_musical_grid_ticks_from_timing_points,
)


def find_ffmpeg(custom_path: Optional[str] = None) -> Optional[str]:
    if custom_path:
        cand = Path(custom_path)
        if cand.is_file():
            return str(cand.resolve())
        which = shutil.which(custom_path)
        if which:
            return which
        return None

    for env_var in ("FFMPEG_PATH", "FFMPEG_BINARY", "FFMPEG_EXECUTABLE"):
        val = os.environ.get(env_var)
        if val and Path(val).is_file():
            return str(Path(val).resolve())

    cwd = Path.cwd()
    module_dir = Path(__file__).resolve().parent
    repo_root = module_dir.parents[1]

    candidates = [
        cwd / "ffmpeg/ffmpeg.exe",
        cwd / "ffmpeg/ffmpeg",
        cwd / "ffmpeg.exe",
        repo_root / "ffmpeg/ffmpeg.exe",
        repo_root / "ffmpeg/ffmpeg",
    ]

    if os.name == "nt" or sys.platform.startswith("win"):
        candidates.extend([
            Path(r"C:\ffmpeg\bin\ffmpeg.exe"),
            Path(r"C:\ffmpeg\ffmpeg.exe"),
            Path(r"C:\Program Files\ffmpeg\bin\ffmpeg.exe"),
        ])

    for c in candidates:
        if c.is_file():
            return str(c.resolve())

    which_exe = shutil.which("ffmpeg.exe") or shutil.which("ffmpeg")
    if which_exe:
        return which_exe

    return None


def keys_to_token(k0: int, k1: int, k2: int, k3: int) -> int:
    return (int(bool(k0))) | (int(bool(k1)) << 1) | (int(bool(k2)) << 2) | (int(bool(k3)) << 3)


def token_to_keys(token: int) -> Tuple[int, int, int, int]:
    return (token & 1, (token >> 1) & 1, (token >> 2) & 1, (token >> 3) & 1)


def create_mel_filterbank(
    sr: int = 16000,
    n_fft: int = 512,
    n_mels: int = 80,
    f_min: float = 20.0,
    f_max: float = 8000.0,
) -> torch.Tensor:
    def hz_to_mel(hz: float) -> float:
        return 2595.0 * math.log10(1.0 + hz / 700.0)

    def mel_to_hz(mel: float) -> float:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    mel_min = hz_to_mel(f_min)
    mel_max = hz_to_mel(f_max)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)

    num_bins = n_fft // 2 + 1
    fft_freqs = np.linspace(0.0, sr / 2.0, num_bins)

    filterbank = np.zeros((n_mels, num_bins), dtype=np.float32)
    for i in range(n_mels):
        left, center, right = hz_points[i], hz_points[i + 1], hz_points[i + 2]
        up_idx = np.where((fft_freqs >= left) & (fft_freqs <= center))[0]
        if len(up_idx) > 0 and center > left:
            filterbank[i, up_idx] = (fft_freqs[up_idx] - left) / (center - left)
        down_idx = np.where((fft_freqs >= center) & (fft_freqs <= right))[0]
        if len(down_idx) > 0 and right > center:
            filterbank[i, down_idx] = (right - fft_freqs[down_idx]) / (right - center)

    enorm = 2.0 / (hz_points[2 : n_mels + 2] - hz_points[:n_mels])
    filterbank *= enorm[:, np.newaxis]
    return torch.from_numpy(filterbank)


def compute_mel_spectrogram(
    waveform: torch.Tensor,
    sr: int = 16000,
    n_fft: int = 512,
    hop_length: int = 256,
    n_mels: int = 80,
    filterbank: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    window = torch.hann_window(n_fft, device=waveform.device)
    stft = torch.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=window,
        return_complex=True,
    )
    mag_spec = torch.abs(stft).squeeze(0)

    if filterbank is None:
        filterbank = create_mel_filterbank(sr=sr, n_fft=n_fft, n_mels=n_mels).to(waveform.device)
    else:
        filterbank = filterbank.to(waveform.device)

    mel_spec = torch.matmul(filterbank, mag_spec)
    log_mel = torch.log(torch.clamp(mel_spec, min=1e-5))
    return log_mel.transpose(0, 1)


def extract_audio_waveform(
    audio_input: bytes | str | Path,
    file_ext: str = ".mp3",
    target_sr: int = 16000,
    ffmpeg_path: Optional[str] = None,
) -> Optional[torch.Tensor]:
    if np is None or torch is None:
        raise ImportError("NumPy and PyTorch are required to decode audio waveforms.")

    if isinstance(audio_input, (str, Path)):
        p = Path(audio_input)
        file_ext = p.suffix.lower()
        audio_bytes = p.read_bytes()
    else:
        audio_bytes = audio_input

    if file_ext.lower() == ".wav":
        try:
            import io
            import struct
            import wave
            with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
                num_channels = wf.getnchannels()
                sample_width = wf.getsampwidth()
                in_sr = wf.getframerate()
                num_frames = wf.getnframes()
                raw_bytes = wf.readframes(num_frames)

            if sample_width == 2:
                total_samples = len(raw_bytes) // 2
                ints = struct.unpack(f"<{total_samples}h", raw_bytes)
                if num_channels == 1:
                    samples = [x / 32768.0 for x in ints]
                else:
                    samples = [
                        sum(ints[i : i + num_channels]) / (num_channels * 32768.0)
                        for i in range(0, len(ints), num_channels)
                    ]
                arr = np.array(samples, dtype=np.float32)
                if in_sr != target_sr:
                    num_target = int(len(arr) * target_sr / in_sr)
                    arr = np.interp(
                        np.linspace(0, len(arr), num_target, endpoint=False),
                        np.arange(len(arr)),
                        arr,
                    ).astype(np.float32)
                return torch.from_numpy(arr)
        except Exception:
            pass

    resolved_ffmpeg = find_ffmpeg(ffmpeg_path)
    if not resolved_ffmpeg:
        raise FileNotFoundError(
            "FFmpeg не найден на компьютере! Положите 'ffmpeg.exe' в папку 'ffmpeg/' или задайте PATH."
        )

    with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as in_file, \
         tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as out_wav:
        in_file.write(audio_bytes)
        in_file.flush()
        in_file.close()
        out_wav.close()

        cmd = [
            resolved_ffmpeg,
            "-y",
            "-v", "error",
            "-i", in_file.name,
            "-ar", str(target_sr),
            "-ac", "1",
            "-f", "f32le",
            out_wav.name,
        ]
        try:
            res = subprocess.run(cmd, capture_output=True)
            if res.returncode == 0 and os.path.exists(out_wav.name) and os.path.getsize(out_wav.name) > 0:
                raw_bytes = Path(out_wav.name).read_bytes()
                samples = np.frombuffer(raw_bytes, dtype=np.float32)
                return torch.from_numpy(samples.copy())
            else:
                stderr_msg = res.stderr.decode("utf-8", errors="ignore") if res.stderr else "Unknown error"
                raise RuntimeError(f"FFmpeg decoding failed: {stderr_msg}")
        finally:
            if os.path.exists(in_file.name):
                os.unlink(in_file.name)
            if os.path.exists(out_wav.name):
                os.unlink(out_wav.name)
    return None


@dataclass
class ParsedBeatmapDetails:
    osu_file_name: str
    version: str
    mode: int
    circle_size: float
    overall_difficulty: float
    audio_filename: str
    bpm: float
    offset_ms: int
    star_rating: float
    timing_points: List[Tuple[int, float, bool]]
    hit_objects: List[Tuple[int, int, int, bool]]


def parse_osu_for_training(content: str, filename: str) -> Optional[ParsedBeatmapDetails]:
    lines = content.splitlines()
    general: Dict[str, str] = {}
    difficulty: Dict[str, str] = {}
    metadata: Dict[str, str] = {}
    timing_lines: List[str] = []
    hit_lines: List[str] = []

    current_section = ""
    for line in lines:
        line_str = line.strip()
        if not line_str or line_str.startswith("//"):
            continue
        if line_str.startswith("[") and line_str.endswith("]"):
            current_section = line_str[1:-1].strip()
            continue

        if current_section == "General" and ":" in line_str:
            k, v = line_str.split(":", 1)
            general[k.strip()] = v.strip()
        elif current_section == "Difficulty" and ":" in line_str:
            k, v = line_str.split(":", 1)
            difficulty[k.strip()] = v.strip()
        elif current_section == "Metadata" and ":" in line_str:
            k, v = line_str.split(":", 1)
            metadata[k.strip()] = v.strip()
        elif current_section == "TimingPoints":
            timing_lines.append(line_str)
        elif current_section == "HitObjects":
            hit_lines.append(line_str)

    try:
        mode = int(general.get("Mode", "0"))
        cs = float(difficulty.get("CircleSize", "4"))
    except ValueError:
        return None

    if mode != 3 or round(cs) != 4:
        return None

    od = float(difficulty.get("OverallDifficulty", "8"))
    audio_file = general.get("AudioFilename", "")
    version = metadata.get("Version", "Normal")

    timing_points: List[Tuple[int, float, bool]] = []
    base_bpm = 120.0
    first_offset = 0
    found_first_uninherited = False

    for t_str in timing_lines:
        parts = t_str.split(",")
        if len(parts) < 2:
            continue
        try:
            t_ms = int(round(float(parts[0])))
            beat_len = float(parts[1])
            uninherited = True if len(parts) <= 6 else (parts[6].strip() == "1")
            timing_points.append((t_ms, beat_len, uninherited))

            if uninherited and not found_first_uninherited and beat_len > 0:
                base_bpm = 60000.0 / beat_len
                first_offset = t_ms
                found_first_uninherited = True
        except ValueError:
            continue

    hit_objects: List[Tuple[int, int, int, bool]] = []
    for h_str in hit_lines:
        parts = h_str.split(",")
        if len(parts) < 5:
            continue
        try:
            x = int(float(parts[0]))
            col = max(0, min(3, int(x * 4 / 512)))
            start_ms = int(round(float(parts[2])))
            obj_type = int(parts[3])
            is_ln = bool(obj_type & 128)
            end_ms = start_ms

            if is_ln and len(parts) >= 6:
                end_str = parts[5].split(":")[0]
                try:
                    end_ms = int(round(float(end_str)))
                except ValueError:
                    end_ms = start_ms

            hit_objects.append((col, start_ms, end_ms, is_ln))
        except ValueError:
            continue

    if not hit_objects:
        return None

    hit_objects.sort(key=lambda o: (o[1], o[0]))

    sr = calculate_osu_mania_star_rating(
        osu_text=content,
        hit_objects=hit_objects,
        overall_difficulty=od,
        keys=4,
    )

    return ParsedBeatmapDetails(
        osu_file_name=filename,
        version=version,
        mode=mode,
        circle_size=cs,
        overall_difficulty=od,
        audio_filename=audio_file,
        bpm=round(base_bpm, 2),
        offset_ms=first_offset,
        star_rating=round(sr, 2),
        timing_points=timing_points,
        hit_objects=hit_objects,
    )


def quantize_beatmap_to_grid(
    details: ParsedBeatmapDetails,
    num_frames: int,
    frame_ms: float = 16.0,
    audio_mel: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[MusicalGridTick]]:
    duration_ms = float(num_frames * frame_ms)
    ticks = generate_musical_grid_ticks_from_timing_points(
        timing_points=details.timing_points,
        duration_ms=duration_ms,
        default_bpm=details.bpm,
        default_offset_ms=details.offset_ms,
    )

    num_ticks = len(ticks)
    if num_ticks == 0:
        ticks = [
            MusicalGridTick(
                time_ms=0.0,
                bpm=details.bpm,
                beat_len_ms=60000.0 / max(30.0, details.bpm),
                meter=4,
                divisor=1,
                divisor_fraction=1.0,
                subdivision_index=0,
                subdivisions_per_beat=8,
                beat_in_measure=0,
                bar_phase=0.0,
                is_downbeat=True,
                is_beat_start=True,
                is_ternary=False,
            )
        ]
        num_ticks = 1

    tick_times = np.array([t.time_ms for t in ticks], dtype=np.float32)

    key_states = np.zeros((num_ticks, 4), dtype=np.int64)
    ln_active = np.zeros((num_ticks, 4), dtype=np.float32)
    lane_states = np.zeros((num_ticks, 4), dtype=np.int64)
    sv_curve = np.ones(num_ticks, dtype=np.float32)

    mean_e = 1.0
    tick_energies = np.ones(num_ticks, dtype=np.float32)
    if audio_mel is not None:
        try:
            with torch.no_grad():
                frame_e = audio_mel.exp().mean(dim=-1).cpu().numpy()
                mean_e = max(1e-4, float(np.mean(frame_e)))
                for i, t_obj in enumerate(ticks):
                    f = max(0, min(len(frame_e) - 1, int(round(t_obj.time_ms / frame_ms))))
                    tick_energies[i] = frame_e[f]
        except Exception:
            mean_e = 1.0

    for col, start_ms, end_ms, is_ln in details.hit_objects:
        start_idx = int(np.argmin(np.abs(tick_times - start_ms)))
        if 0 <= start_idx < num_ticks:
            key_states[start_idx, col] = 1
            lane_states[start_idx, col] = 2 if is_ln else 1

        if is_ln:
            end_idx = int(np.argmin(np.abs(tick_times - end_ms)))
            if end_idx <= start_idx:
                end_idx = min(num_ticks - 1, start_idx + 1)
            # Long note trigger at start_idx
            ln_active[start_idx, col] = 1.0
            for idx in range(start_idx + 1, min(num_ticks, end_idx)):
                ln_active[idx, col] = 1.0
                lane_states[idx, col] = 3
            if 0 <= end_idx < num_ticks:
                ln_active[end_idx, col] = 1.0
                lane_states[end_idx, col] = 4

    grid_metrics = np.zeros((num_ticks, 8), dtype=np.float32)
    for i, t_obj in enumerate(ticks):
        progress = float(i / max(1, num_ticks))
        time_to_end = float((num_ticks - i) / max(1, num_ticks))
        e_ratio = float(tick_energies[i] / mean_e) if mean_e > 0 else 1.0

        grid_metrics[i, 0] = float(t_obj.divisor_fraction)
        grid_metrics[i, 1] = float(t_obj.bar_phase)
        grid_metrics[i, 2] = 1.0 if t_obj.is_downbeat else 0.0
        grid_metrics[i, 3] = progress
        grid_metrics[i, 4] = time_to_end
        grid_metrics[i, 5] = e_ratio
        grid_metrics[i, 6] = float(t_obj.meter)
        grid_metrics[i, 7] = 1.0 if t_obj.is_ternary else 0.0

    for t_ms, val, uninherited in details.timing_points:
        if not uninherited:
            sv = max(0.1, min(10.0, -100.0 / val)) if val < 0 else 1.0
            idx = int(np.argmin(np.abs(tick_times - t_ms)))
            if 0 <= idx < num_ticks:
                sv_curve[idx:] = sv

    tokens = np.zeros(num_ticks, dtype=np.int64)
    for i in range(num_ticks):
        tokens[i] = keys_to_token(
            key_states[i, 0],
            key_states[i, 1],
            key_states[i, 2],
            key_states[i, 3],
        )

    return (
        torch.from_numpy(tokens),
        torch.from_numpy(ln_active),
        torch.from_numpy(sv_curve),
        torch.from_numpy(lane_states),
        torch.from_numpy(grid_metrics),
        ticks,
    )


def compute_multiscale_pyramid_features(
    mel: torch.Tensor,
    tick_times_ms: np.ndarray | torch.Tensor | List[float],
    frame_ms: float = 16.0,
    meso_sec: float = 10.0,
    macro_sec: float = 25.0,
) -> torch.Tensor:
    """
    Computes 3-scale Acoustic Pyramid features for every tick:
      1. Micro (instantaneous tick t): LogMel (80) + Delta (80) + OnsetFlux (80) = 240 dims
      2. Meso (±10.0s window, 20s total): Past/Future spectrum, energy gradient, quadrant energies, dynamic range = 96 dims
      3. Macro (±25.0s window, 50s total): Macro 4-band spectrum across 4 quadrants, track boundary ratio = 32 dims
      Total dimension = 368 dims per tick.
    
    CRITICAL: Audio before 0.0s and after track end is padded with genuine silence (zero-padded log-mel),
    ensuring fixed tensor dimensions and 100% accurate temporal centering at all points.
    """
    if not isinstance(mel, torch.Tensor):
        mel = torch.tensor(mel, dtype=torch.float32)

    num_frames = mel.shape[0]
    device = mel.device
    dtype = mel.dtype

    if isinstance(tick_times_ms, torch.Tensor):
        t_ms_arr = tick_times_ms.cpu().numpy()
    elif isinstance(tick_times_ms, np.ndarray):
        t_ms_arr = tick_times_ms
    else:
        t_ms_arr = np.array(tick_times_ms, dtype=np.float32)

    num_ticks = len(t_ms_arr)
    if num_ticks == 0:
        return torch.zeros(0, 368, dtype=dtype, device=device)

    pad_macro = int(round((macro_sec * 1000.0) / frame_ms))  # 1562 frames (25s)
    pad_meso = int(round((meso_sec * 1000.0) / frame_ms))    # 625 frames (10s)

    # Pad spectrogram with silence (0.0) on both sides
    padded_mel = F.pad(mel, (0, 0, pad_macro, pad_macro), value=0.0)
    L = padded_mel.shape[0]

    # Pre-compute pooled 40-bin mel and integral image for O(1) interval queries
    mel_40 = padded_mel.view(L, 40, 2).mean(dim=-1)
    integral_mel_40 = torch.cumsum(F.pad(mel_40, (0, 0, 1, 0)), dim=0)

    # Coarse 4-band mel for macro scale (0-20, 20-40, 40-60, 60-80)
    mel_4 = padded_mel.view(L, 4, 20).mean(dim=-1)
    integral_mel_4 = torch.cumsum(F.pad(mel_4, (0, 0, 1, 0)), dim=0)

    padded_e = padded_mel.exp().mean(dim=-1)
    integral_e = torch.cumsum(F.pad(padded_e, (1, 0)), dim=0)

    # Track boundary mask: 1.0 inside valid song, 0.0 in outer silence padding
    valid_track = torch.zeros(L, dtype=dtype, device=device)
    valid_track[pad_macro : pad_macro + num_frames] = 1.0
    integral_valid = torch.cumsum(F.pad(valid_track, (1, 0)), dim=0)

    # Frame indices for all ticks
    raw_frames = np.clip(np.round(t_ms_arr / frame_ms).astype(int), 0, max(0, num_frames - 1))
    centers = raw_frames + pad_macro

    c_t = torch.from_numpy(centers).to(device)

    # 1. Micro features (240 dims)
    cur_mel = padded_mel[c_t]                        # [num_ticks, 80]
    prev_mel = padded_mel[c_t - 1]                   # [num_ticks, 80]
    micro_delta = cur_mel - prev_mel                 # [num_ticks, 80]
    micro_flux = F.relu(micro_delta)                 # [num_ticks, 80]
    micro_feat = torch.cat([cur_mel, micro_delta, micro_flux], dim=-1)  # [num_ticks, 240]

    # 2. Meso features (±10s = 625 frames past & future) -> 96 dims
    past_spec = (integral_mel_40[c_t + 1] - integral_mel_40[c_t - pad_meso]) / pad_meso          # 40
    future_spec = (integral_mel_40[c_t + pad_meso + 1] - integral_mel_40[c_t]) / pad_meso        # 40
    past_e = (integral_e[c_t + 1] - integral_e[c_t - pad_meso]) / pad_meso                        # 1
    future_e = (integral_e[c_t + pad_meso + 1] - integral_e[c_t]) / pad_meso                    # 1
    e_slope = (future_e - past_e) / (past_e + future_e + 1e-4)                                    # 1
    past_valid = (integral_valid[c_t + 1] - integral_valid[c_t - pad_meso]) / pad_meso            # 1
    future_valid = (integral_valid[c_t + pad_meso + 1] - integral_valid[c_t]) / pad_meso          # 1

    # 4 temporal sub-quadrants of Meso (each ~156 frames / 2.5s)
    q_len = pad_meso // 2
    q1_e = (integral_e[c_t - q_len + 1] - integral_e[c_t - pad_meso]) / q_len
    q2_e = (integral_e[c_t + 1] - integral_e[c_t - q_len]) / q_len
    q3_e = (integral_e[c_t + q_len + 1] - integral_e[c_t]) / q_len
    q4_e = (integral_e[c_t + pad_meso + 1] - integral_e[c_t + q_len]) / q_len

    meso_e_stats = torch.stack([
        past_e, future_e, e_slope, past_valid, future_valid,
        q1_e, q2_e, q3_e, q4_e,
        (q3_e - q2_e) / (q2_e + q3_e + 1e-4),
        (q4_e - q1_e) / (q1_e + q4_e + 1e-4),
        torch.clamp(future_e / (past_e + 1e-4), 0.0, 10.0),
        torch.clamp(past_e / (future_e + 1e-4), 0.0, 10.0),
        torch.abs(future_e - past_e),
        torch.max(q1_e, q2_e),
        torch.max(q3_e, q4_e),
    ], dim=-1)  # 16 dims

    meso_feat = torch.cat([past_spec, future_spec, meso_e_stats], dim=-1)  # 40 + 40 + 16 = 96 dims

    # 3. Macro features (±25s = 1562 frames past & future) -> 32 dims
    # 4 intervals: [-25s, -12.5s], [-12.5s, 0s], [0s, +12.5s], [+12.5s, +25s]
    h_macro = pad_macro // 2  # 781 frames
    m1_spec = (integral_mel_4[c_t - h_macro + 1] - integral_mel_4[c_t - pad_macro]) / h_macro    # 4
    m2_spec = (integral_mel_4[c_t + 1] - integral_mel_4[c_t - h_macro]) / h_macro                # 4
    m3_spec = (integral_mel_4[c_t + h_macro + 1] - integral_mel_4[c_t]) / h_macro                # 4
    m4_spec = (integral_mel_4[c_t + pad_macro + 1] - integral_mel_4[c_t + h_macro]) / h_macro    # 4
    macro_spec = torch.cat([m1_spec, m2_spec, m3_spec, m4_spec], dim=-1)                         # 16

    m1_e = (integral_e[c_t - h_macro + 1] - integral_e[c_t - pad_macro]) / h_macro
    m2_e = (integral_e[c_t + 1] - integral_e[c_t - h_macro]) / h_macro
    m3_e = (integral_e[c_t + h_macro + 1] - integral_e[c_t]) / h_macro
    m4_e = (integral_e[c_t + pad_macro + 1] - integral_e[c_t + h_macro]) / h_macro

    macro_past_valid = (integral_valid[c_t + 1] - integral_valid[c_t - pad_macro]) / pad_macro
    macro_future_valid = (integral_valid[c_t + pad_macro + 1] - integral_valid[c_t]) / pad_macro

    prog_scalar = torch.from_numpy((raw_frames / max(1, num_frames)).astype(np.float32)).to(device)
    end_scalar = 1.0 - prog_scalar

    macro_stats = torch.stack([
        m1_e, m2_e, m3_e, m4_e,
        macro_past_valid, macro_future_valid,
        (macro_past_valid + macro_future_valid) * 0.5,
        torch.clamp(m3_e / (m2_e + 1e-4), 0.0, 10.0),
        torch.clamp(m4_e / (m1_e + 1e-4), 0.0, 10.0),
        (m3_e + m4_e - m1_e - m2_e) / (m1_e + m2_e + m3_e + m4_e + 1e-4),
        prog_scalar, end_scalar,
        torch.max(torch.stack([m1_e, m2_e, m3_e, m4_e], dim=-1), dim=-1)[0],
        torch.min(torch.stack([m1_e, m2_e, m3_e, m4_e], dim=-1), dim=-1)[0],
        torch.abs(m3_e - m2_e),
        torch.abs(m4_e - m1_e),
    ], dim=-1)  # 16 dims

    macro_feat = torch.cat([macro_spec, macro_stats], dim=-1)  # 16 + 16 = 32 dims

    # Total Multi-Scale Pyramid: 240 (Micro) + 96 (Meso) + 32 (Macro) = 368 dims
    multiscale_tensor = torch.cat([micro_feat, meso_feat, macro_feat], dim=-1)
    return multiscale_tensor


def compute_chunk_time_deltas(
    chunk_tok: torch.Tensor,
    ticks_slice: List[MusicalGridTick],
    beat_len_ms: float,
) -> torch.Tensor:
    """Computes exact continuous time distance (in ms) to next active hit object event."""
    chunk_size = len(chunk_tok)
    deltas = torch.zeros(chunk_size, 1, dtype=torch.float32)
    tok_arr = chunk_tok.cpu().numpy() if isinstance(chunk_tok, torch.Tensor) else chunk_tok
    note_indices = [i for i, tok in enumerate(tok_arr) if tok > 0]

    if not note_indices:
        deltas.fill_(float(beat_len_ms))
        return deltas

    times_ms = [t.time_ms for t in ticks_slice]
    for i in range(chunk_size):
        next_notes = [idx for idx in note_indices if idx > i]
        if next_notes:
            nxt = next_notes[0]
            dt = max(10.0, times_ms[nxt] - times_ms[i])
        else:
            dt = max(10.0, (times_ms[-1] - times_ms[i]) + beat_len_ms)
        deltas[i, 0] = float(dt)
    return deltas


def slice_beatmap_into_musical_chunks(
    mel: torch.Tensor,
    tokens: torch.Tensor,
    ln_active: torch.Tensor,
    sv_curve: torch.Tensor,
    details: ParsedBeatmapDetails,
    chunk_size: int = 256,
    hop_size: int = 64,
    frame_ms: float = 16.0,
    align_mode: str = "beat",
    bars_per_step: int = 1,
    timing_source: str = "map",
    timing_override: Optional[Tuple[float, float]] = None,
    lane_states: Optional[torch.Tensor] = None,
    grid_metrics: Optional[torch.Tensor] = None,
    ticks: Optional[List[MusicalGridTick]] = None,
) -> List[Dict[str, Any]]:
    num_ticks = tokens.shape[0]
    num_frames = mel.shape[0]
    chunks: List[Dict[str, Any]] = []

    if ticks is None or len(ticks) != num_ticks:
        duration_ms = float(num_frames * frame_ms)
        ticks = generate_musical_grid_ticks_from_timing_points(
            timing_points=details.timing_points,
            duration_ms=duration_ms,
            default_bpm=details.bpm,
            default_offset_ms=details.offset_ms,
        )
        if len(ticks) > num_ticks:
            ticks = ticks[:num_ticks]
        elif len(ticks) < num_ticks:
            while len(ticks) < num_ticks:
                last_t = ticks[-1].time_ms if ticks else 0.0
                ticks.append(MusicalGridTick(
                    time_ms=last_t + 50.0,
                    bpm=details.bpm,
                    beat_len_ms=500.0,
                    meter=4,
                    divisor=1,
                    divisor_fraction=1.0,
                    subdivision_index=0,
                    subdivisions_per_beat=8,
                    beat_in_measure=0,
                    bar_phase=0.0,
                    is_downbeat=False,
                    is_beat_start=False,
                    is_ternary=False,
                ))

    tick_audio_mel = torch.zeros(num_ticks, mel.shape[-1], dtype=mel.dtype, device=mel.device)
    for i, t_obj in enumerate(ticks):
        f = max(0, min(num_frames - 1, int(round(t_obj.time_ms / frame_ms))))
        tick_audio_mel[i] = mel[f]

    tick_times_arr = np.array([t.time_ms for t in ticks], dtype=np.float32)
    multiscale_pyramid_all = compute_multiscale_pyramid_features(mel, tick_times_arr, frame_ms=frame_ms)

    downbeat_indices = [i for i, t in enumerate(ticks) if t.is_downbeat]
    if not downbeat_indices:
        downbeat_indices = list(range(0, num_ticks, max(1, hop_size)))

    step_step = max(1, bars_per_step)
    for db_i in range(0, len(downbeat_indices), step_step):
        start_idx = downbeat_indices[db_i]
        end_idx = start_idx + chunk_size

        if end_idx > num_ticks:
            break

        chunk_mel = tick_audio_mel[start_idx:end_idx]
        chunk_multiscale = multiscale_pyramid_all[start_idx:end_idx]
        chunk_tok = tokens[start_idx:end_idx]
        chunk_ln = ln_active[start_idx:end_idx]
        chunk_sv = sv_curve[start_idx:end_idx]
        chunk_lane = lane_states[start_idx:end_idx] if lane_states is not None else torch.zeros(chunk_size, 4, dtype=torch.int64)
        chunk_grid = grid_metrics[start_idx:end_idx] if grid_metrics is not None else torch.zeros(chunk_size, 8, dtype=torch.float32)

        prev_start = start_idx - chunk_size
        if prev_start >= 0:
            prev_tok = tokens[prev_start:start_idx].clone()
        else:
            if start_idx > 0:
                valid = tokens[0:start_idx]
                pad = torch.zeros(chunk_size - len(valid), dtype=torch.int64)
                prev_tok = torch.cat([pad, valid])
            else:
                prev_tok = torch.zeros(chunk_size, dtype=torch.int64)

        t_start_ms = ticks[start_idx].time_ms
        t_end_ms = ticks[end_idx - 1].time_ms
        b_len_ms = round(60000.0 / max(30.0, float(details.bpm)), 2)
        chunk_deltas = compute_chunk_time_deltas(chunk_tok, ticks[start_idx:end_idx], b_len_ms)

        chunks.append({
            "audio_mel": chunk_mel,
            "multiscale_audio": chunk_multiscale,
            "tokens": chunk_tok,
            "prev_tokens": prev_tok,
            "lane_states": chunk_lane,
            "grid_metrics": chunk_grid,
            "ln_mask": chunk_ln,
            "sv_curve": chunk_sv,
            "target_time_deltas": chunk_deltas,
            "star_rating": torch.tensor([details.star_rating], dtype=torch.float32),
            "bpm": torch.tensor([details.bpm], dtype=torch.float32),
            "offset_ms": torch.tensor([details.offset_ms], dtype=torch.float32),
            "timing_source": timing_source,
            "start_ms": round(t_start_ms, 2),
            "end_ms": round(t_end_ms, 2),
            "bar_index": db_i,
            "beat_len_ms": b_len_ms,
        })

    if not chunks and num_ticks >= chunk_size:
        for start_idx in range(0, num_ticks - chunk_size + 1, max(1, hop_size)):
            end_idx = start_idx + chunk_size
            chunk_mel = tick_audio_mel[start_idx:end_idx]
            chunk_multiscale = multiscale_pyramid_all[start_idx:end_idx]
            chunk_tok = tokens[start_idx:end_idx]
            chunk_ln = ln_active[start_idx:end_idx]
            chunk_sv = sv_curve[start_idx:end_idx]
            chunk_lane = lane_states[start_idx:end_idx] if lane_states is not None else torch.zeros(chunk_size, 4, dtype=torch.int64)
            chunk_grid = grid_metrics[start_idx:end_idx] if grid_metrics is not None else torch.zeros(chunk_size, 8, dtype=torch.float32)

            prev_start = start_idx - chunk_size
            if prev_start >= 0:
                prev_tok = tokens[prev_start:start_idx].clone()
            else:
                if start_idx > 0:
                    valid = tokens[0:start_idx]
                    pad = torch.zeros(chunk_size - len(valid), dtype=torch.int64)
                    prev_tok = torch.cat([pad, valid])
                else:
                    prev_tok = torch.zeros(chunk_size, dtype=torch.int64)

            t_start_ms = ticks[start_idx].time_ms
            t_end_ms = ticks[end_idx - 1].time_ms
            b_len_ms = round(60000.0 / max(30.0, float(details.bpm)), 2)
            chunk_deltas = compute_chunk_time_deltas(chunk_tok, ticks[start_idx:end_idx], b_len_ms)

            chunks.append({
                "audio_mel": chunk_mel,
                "multiscale_audio": chunk_multiscale,
                "tokens": chunk_tok,
                "prev_tokens": prev_tok,
                "lane_states": chunk_lane,
                "grid_metrics": chunk_grid,
                "ln_mask": chunk_ln,
                "sv_curve": chunk_sv,
                "target_time_deltas": chunk_deltas,
                "star_rating": torch.tensor([details.star_rating], dtype=torch.float32),
                "bpm": torch.tensor([details.bpm], dtype=torch.float32),
                "offset_ms": torch.tensor([details.offset_ms], dtype=torch.float32),
                "timing_source": timing_source,
                "start_ms": round(t_start_ms, 2),
                "end_ms": round(t_end_ms, 2),
                "bar_index": 0,
                "beat_len_ms": b_len_ms,
            })

    return chunks
