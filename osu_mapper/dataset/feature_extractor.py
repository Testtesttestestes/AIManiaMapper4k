"""
Feature extraction pipeline for osu!mania 4K beatmap dataset.

Converts .osz / .osu and audio files into PyTorch-ready training tensors:
- 80-bin Mel-spectrograms from 16kHz mono audio
- 16-chord discrete token sequences (2^4 key combinations for 4K)
- Long Note (LN) duration masks
- BPM and first offset extraction (Model 1)
- Continuous Slider Velocity (SV) profile (Model 2)
- Star Rating and note density statistics (Model 4)
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
from typing import Dict, List, Optional, Tuple, Any

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore

try:
    import torch
except ImportError:
    torch = None  # type: ignore


def find_ffmpeg(custom_path: Optional[str] = None) -> Optional[str]:
    """
    Locates FFmpeg executable across Windows, Linux, and macOS.
    Checks:
    1. Explicit custom_path if passed (strictly checked).
    2. Environment variables FFMPEG_PATH, FFMPEG_BINARY, FFMPEG_EXECUTABLE.
    3. Common project-relative folders (e.g. ffmpeg/ffmpeg.exe, bin/ffmpeg.exe).
    4. Common Windows paths (C:\\ffmpeg\\bin\\ffmpeg.exe, C:\\ffmpeg\\ffmpeg.exe, etc.).
    5. System PATH via shutil.which("ffmpeg") / shutil.which("ffmpeg.exe").
    """
    if custom_path:
        cand = Path(custom_path)
        if cand.is_file():
            return str(cand.resolve())
        which = shutil.which(custom_path)
        if which:
            return which
        return None

    candidates: List[Path] = []

    for env_var in ("FFMPEG_PATH", "FFMPEG_BINARY", "FFMPEG_EXECUTABLE"):
        val = os.environ.get(env_var)
        if val:
            candidates.append(Path(val))

    # Current working dir and module dir
    cwd = Path.cwd()
    module_dir = Path(__file__).resolve().parent
    repo_root = module_dir.parents[1]  # osu_mapper/dataset -> root

    search_roots = [cwd, repo_root, module_dir]
    relative_patterns = [
        "ffmpeg/ffmpeg.exe",
        "ffmpeg/bin/ffmpeg.exe",
        "ffmpeg/ffmpeg",
        "ffmpeg/bin/ffmpeg",
        "bin/ffmpeg.exe",
        "bin/ffmpeg",
        "ffmpeg.exe",
        "ffmpeg",
    ]

    for root in search_roots:
        for rel in relative_patterns:
            candidates.append(root / rel)

    # Windows-specific typical install locations
    if os.name == "nt" or sys.platform.startswith("win"):
        candidates.extend([
            Path(r"C:\ffmpeg\bin\ffmpeg.exe"),
            Path(r"C:\ffmpeg\ffmpeg.exe"),
            Path(r"C:\Program Files\ffmpeg\bin\ffmpeg.exe"),
            Path(r"C:\Program Files\ffmpeg\ffmpeg.exe"),
            Path(r"C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe"),
        ])
        user_profile = os.environ.get("USERPROFILE")
        if user_profile:
            candidates.extend([
                Path(user_profile) / "ffmpeg" / "bin" / "ffmpeg.exe",
                Path(user_profile) / "ffmpeg" / "ffmpeg.exe",
                Path(user_profile) / "scoop" / "apps" / "ffmpeg" / "current" / "bin" / "ffmpeg.exe",
            ])

    for cand in candidates:
        try:
            if cand.is_file():
                return str(cand.resolve())
        except Exception:
            continue

    # Fall back to PATH
    which_exe = shutil.which("ffmpeg.exe")
    if which_exe:
        return which_exe
    which_bin = shutil.which("ffmpeg")
    if which_bin:
        return which_bin

    return None


def keys_to_token(k0: int, k1: int, k2: int, k3: int) -> int:
    """Encodes 4 binary keys into chord token [0..15]."""
    return (int(bool(k0))) | (int(bool(k1)) << 1) | (int(bool(k2)) << 2) | (int(bool(k3)) << 3)


def token_to_keys(token: int) -> Tuple[int, int, int, int]:
    """Decodes chord token [0..15] into 4 binary keys."""
    return (token & 1, (token >> 1) & 1, (token >> 2) & 1, (token >> 3) & 1)


def create_mel_filterbank(
    sr: int = 16000,
    n_fft: int = 512,
    n_mels: int = 80,
    f_min: float = 20.0,
    f_max: float = 8000.0,
) -> torch.Tensor:
    """Constructs a standard triangular Mel-filterbank matrix [n_mels, n_fft // 2 + 1]."""
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
        # Up-slope
        up_idx = np.where((fft_freqs >= left) & (fft_freqs <= center))[0]
        if len(up_idx) > 0 and center > left:
            filterbank[i, up_idx] = (fft_freqs[up_idx] - left) / (center - left)
        # Down-slope
        down_idx = np.where((fft_freqs >= center) & (fft_freqs <= right))[0]
        if len(down_idx) > 0 and right > center:
            filterbank[i, down_idx] = (right - fft_freqs[down_idx]) / (right - center)

    # Normalize area for energy conservation
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
    """
    Computes log-mel spectrogram [TimeFrames, n_mels] from 1D audio waveform.
    hop_length = 256 @ 16000 Hz corresponds to 16.0 ms temporal resolution.
    """
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
    mag_spec = torch.abs(stft).squeeze(0)  # [FreqBins, TimeFrames]

    if filterbank is None:
        filterbank = create_mel_filterbank(sr=sr, n_fft=n_fft, n_mels=n_mels).to(waveform.device)
    else:
        filterbank = filterbank.to(waveform.device)

    mel_spec = torch.matmul(filterbank, mag_spec)  # [n_mels, TimeFrames]
    log_mel = torch.log(torch.clamp(mel_spec, min=1e-5))  # [n_mels, TimeFrames]
    
    # Transpose to [TimeFrames, n_mels]
    return log_mel.transpose(0, 1)


def extract_audio_waveform(
    audio_bytes: bytes,
    file_ext: str = ".mp3",
    target_sr: int = 16000,
    ffmpeg_path: Optional[str] = None,
) -> Optional[torch.Tensor]:
    """Decodes audio bytes (mp3, ogg, wav) to a 16kHz mono float32 PyTorch tensor using ffmpeg or wave."""
    if np is None or torch is None:
        raise ImportError(
            "NumPy and PyTorch are required to decode audio waveforms. "
            "Please install them via: pip install numpy torch"
        )

    # If audio is already a WAV file, decode natively with Python wave without external ffmpeg
    if file_ext.lower() == ".wav":
        try:
            import io
            import wave
            import struct
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
            pass  # fallback to ffmpeg

    resolved_ffmpeg = find_ffmpeg(ffmpeg_path)
    if not resolved_ffmpeg:
        raise FileNotFoundError(
            "FFmpeg не найден на компьютере! Невозможно декодировать сжатый аудиофайл (.mp3 / .ogg).\n"
            "Пожалуйста, положите 'ffmpeg.exe' в папку 'ffmpeg/' проекта или укажите путь через параметр --ffmpeg."
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
                raise RuntimeError(f"FFmpeg decoding failed ({res.returncode}): {stderr_msg}")
        except FileNotFoundError as fnf:
            raise FileNotFoundError(
                f"Не удалось запустить FFmpeg по пути '{resolved_ffmpeg}': [WinError 2]. "
                "Проверьте, что файл существует и доступен для запуска."
            ) from fnf
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
    timing_points: List[Tuple[int, float, bool]]  # (time_ms, beat_len_or_sv, uninherited)
    hit_objects: List[Tuple[int, int, int, bool]]  # (col, start_ms, end_ms, is_ln)


def parse_osu_for_training(content: str, filename: str) -> Optional[ParsedBeatmapDetails]:
    """Parses .osu file text for 4K mania training."""
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

    # Parse timing points
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

    # Parse hit objects
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

    # Star Rating approximation from OD, NPS, and chords
    duration_s = max(1.0, (hit_objects[-1][2] - hit_objects[0][1]) / 1000.0)
    avg_nps = len(hit_objects) / duration_s
    sr = min(9.9, max(1.0, 0.45 * avg_nps + 0.1 * od))

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
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Quantizes hit objects and SV dynamic curves onto the temporal frame grid.
    Returns:
    - tokens: [num_frames] integer tensor in [0..15]
    - ln_mask: [num_frames, 4] float tensor in [0, 1] indicating active long notes
    - sv_curve: [num_frames] float tensor representing local SV multiplier
    """
    key_states = np.zeros((num_frames, 4), dtype=np.int64)
    ln_active = np.zeros((num_frames, 4), dtype=np.float32)
    sv_curve = np.ones(num_frames, dtype=np.float32)

    # Fill HitObjects
    for col, start_ms, end_ms, is_ln in details.hit_objects:
        start_frame = int(round(start_ms / frame_ms))
        if 0 <= start_frame < num_frames:
            key_states[start_frame, col] = 1

        if is_ln:
            end_frame = int(round(end_ms / frame_ms))
            for f in range(max(0, start_frame), min(num_frames, end_frame + 1)):
                ln_active[f, col] = 1.0

    # Fill SV Curve from TimingPoints
    for t_ms, val, uninherited in details.timing_points:
        f_idx = int(round(t_ms / frame_ms))
        if 0 <= f_idx < num_frames:
            if not uninherited:  # Green line (SV)
                sv = max(0.1, min(10.0, -100.0 / val)) if val < 0 else 1.0
                sv_curve[f_idx:] = sv

    # Encode chords into tokens [0..15]
    tokens = np.zeros(num_frames, dtype=np.int64)
    for f in range(num_frames):
        tokens[f] = keys_to_token(
            key_states[f, 0],
            key_states[f, 1],
            key_states[f, 2],
            key_states[f, 3],
        )

    return (
        torch.from_numpy(tokens),
        torch.from_numpy(ln_active),
        torch.from_numpy(sv_curve),
    )
