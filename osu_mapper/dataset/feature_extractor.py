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
    - lane_states: [num_frames, 4] int64 tensor (0=Free, 1=Tap, 2=Hold, 3=Release)
    - grid_metrics: [num_frames, 3] float tensor (beat_divisor, bar_phase, is_downbeat)
    """
    key_states = np.zeros((num_frames, 4), dtype=np.int64)
    ln_active = np.zeros((num_frames, 4), dtype=np.float32)
    lane_states = np.zeros((num_frames, 4), dtype=np.int64)
    sv_curve = np.ones(num_frames, dtype=np.float32)

    # Fill HitObjects & 5-State Physical Lanes (0=Free, 1=Tap, 2=LN_Start, 3=LN_Hold, 4=LN_Release)
    for col, start_ms, end_ms, is_ln in details.hit_objects:
        start_frame = int(round(start_ms / frame_ms))
        if 0 <= start_frame < num_frames:
            key_states[start_frame, col] = 1
            lane_states[start_frame, col] = 2 if is_ln else 1  # 2 for LN start, 1 for Tap

        if is_ln:
            end_frame = int(round(end_ms / frame_ms))
            for f in range(max(0, start_frame + 1), min(num_frames, end_frame)):
                ln_active[f, col] = 1.0
                lane_states[f, col] = 3  # 3 for LN hold sustain
            if 0 <= end_frame < num_frames:
                ln_active[end_frame, col] = 1.0
                lane_states[end_frame, col] = 4  # 4 for LN release

    # Fill Musical Grid Metrics (Beat Divisor, Bar Position, Downbeat Flag)
    grid_metrics = np.zeros((num_frames, 3), dtype=np.float32)
    bpm = max(30.0, float(details.bpm))
    beat_len_ms = 60000.0 / bpm
    bar_len_ms = 4.0 * beat_len_ms
    offset_ms = float(details.offset_ms)

    for f in range(num_frames):
        t_ms = f * frame_ms
        rel_ms = t_ms - offset_ms
        
        # Bar phase fraction [0.0 to 1.0)
        bar_phase = (rel_ms % bar_len_ms) / bar_len_ms if bar_len_ms > 0 else 0.0
        grid_metrics[f, 1] = float(bar_phase)

        # Downbeat detection (within 16ms of 1/1 bar start)
        dist_to_downbeat = abs(rel_ms % bar_len_ms)
        is_downbeat = 1.0 if (dist_to_downbeat < 16.0 or abs(dist_to_downbeat - bar_len_ms) < 16.0) else 0.0
        grid_metrics[f, 2] = is_downbeat

        # Beat divisor identification (1/1, 1/2, 1/3, 1/4, 1/6, 1/8)
        best_div = 0.0
        min_delta = 999.0
        for div in (1, 2, 4, 3, 6, 8):
            step = beat_len_ms / float(div)
            cand_dist = abs(rel_ms % step)
            dist = min(cand_dist, abs(cand_dist - step))
            if dist < min_delta and dist <= (frame_ms * 0.75):
                min_delta = dist
                best_div = float(div)
        grid_metrics[f, 0] = best_div

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
        torch.from_numpy(lane_states),
        torch.from_numpy(grid_metrics),
    )


def slice_beatmap_into_musical_chunks(
    mel: torch.Tensor,
    tokens: torch.Tensor,
    ln_active: torch.Tensor,
    sv_curve: torch.Tensor,
    details: ParsedBeatmapDetails,
    chunk_size: int = 256,
    hop_size: int = 128,
    frame_ms: float = 16.0,
    align_mode: str = "beat",
    bars_per_step: int = 2,
    timing_source: str = "map",
    timing_override: Optional[Tuple[float, float]] = None,
    lane_states: Optional[torch.Tensor] = None,
    grid_metrics: Optional[torch.Tensor] = None,
) -> List[Dict[str, Any]]:
    """
    Slices beatmap audio representations and tokens into synchronized training/inference chunks.

    Parameters:
    - timing_source: "map" (Ground Truth directly from .osu file) or "model1" (AI predicted BPM & Offset).
    - timing_override: Optional (bpm, offset_ms) tuple to override beatmap timing when using Model 1 sync.
    - align_mode="beat" (Default):
        Audio slices are strictly synchronized to the song's Offset and BPM.
        Each chunk begins on an exact musical downbeat (beat 1 of a measure/square):
        t_start = first_bar_ms + bar_idx * (bars_per_step * bar_len_ms).
        This guarantees zero phase drift and aligns Model 3's context with Model 1 (BPM/Offset)
        and Model 2 (SV Dynamics).
        Also preserves autoregressive context ('prev_tokens') from the preceding measures/squares.

    - align_mode="frame":
        Fallback unaligned sliding window advancing by fixed frame increments (hop_size).
    """
    num_frames = mel.shape[0]
    chunks: List[Dict[str, Any]] = []

    if align_mode == "beat":
        # Determine active BPM and Offset (either ground truth from map or override from Model 1)
        if timing_override is not None:
            bpm_val, offset_val = timing_override
        else:
            bpm_val, offset_val = float(details.bpm), float(details.offset_ms)

        bpm = max(30.0, float(bpm_val))
        beat_len_ms = 60000.0 / bpm
        bar_len_ms = 4.0 * beat_len_ms  # 1 measure / ход (in 4/4)

        # Align to first non-negative musical bar start
        offset_ms = float(offset_val)
        first_bar_ms = offset_ms
        while first_bar_ms >= bar_len_ms:
            first_bar_ms -= bar_len_ms
        while first_bar_ms < 0:
            first_bar_ms += bar_len_ms

        step_idx = 0
        while True:
            chunk_start_ms = first_bar_ms + step_idx * (bars_per_step * bar_len_ms)
            start_frame = int(round(chunk_start_ms / frame_ms))
            end_frame = start_frame + chunk_size

            if end_frame > num_frames:
                break

            if start_frame >= 0:
                chunk_mel = mel[start_frame:end_frame]
                chunk_tok = tokens[start_frame:end_frame]
                chunk_ln = ln_active[start_frame:end_frame]
                chunk_sv = sv_curve[start_frame:end_frame]
                chunk_lane = lane_states[start_frame:end_frame] if lane_states is not None else torch.zeros(chunk_size, 4, dtype=torch.int64)
                chunk_grid = grid_metrics[start_frame:end_frame] if grid_metrics is not None else torch.zeros(chunk_size, 3, dtype=torch.float32)

                # Retain autoregressive context of previous measures/squares
                prev_start_frame = start_frame - chunk_size
                if prev_start_frame >= 0:
                    prev_tok = tokens[prev_start_frame:start_frame].clone()
                else:
                    if start_frame > 0:
                        valid = tokens[0:start_frame]
                        pad = torch.zeros(chunk_size - len(valid), dtype=torch.int64)
                        prev_tok = torch.cat([pad, valid])
                    else:
                        prev_tok = torch.zeros(chunk_size, dtype=torch.int64)

                chunks.append({
                    "audio_mel": chunk_mel,
                    "tokens": chunk_tok,
                    "prev_tokens": prev_tok,
                    "lane_states": chunk_lane,
                    "grid_metrics": chunk_grid,
                    "ln_mask": chunk_ln,
                    "sv_curve": chunk_sv,
                    "star_rating": torch.tensor([details.star_rating], dtype=torch.float32),
                    "bpm": torch.tensor([bpm_val], dtype=torch.float32),
                    "offset_ms": torch.tensor([offset_val], dtype=torch.float32),
                    "timing_source": timing_source,
                    "start_ms": round(chunk_start_ms, 2),
                    "end_ms": round(chunk_start_ms + chunk_size * frame_ms, 2),
                    "bar_index": step_idx * bars_per_step,
                    "beat_len_ms": round(beat_len_ms, 2),
                })

            step_idx += 1

        if not chunks and num_frames >= chunk_size:
            # Fallback to frame mode if beat-anchoring yielded 0 windows
            return slice_beatmap_into_musical_chunks(
                mel, tokens, ln_active, sv_curve, details, chunk_size, hop_size, frame_ms, align_mode="frame",
                timing_source=timing_source, timing_override=timing_override
            )

    else:  # "frame" mode
        for start_idx in range(0, num_frames - chunk_size + 1, hop_size):
            end_idx = start_idx + chunk_size
            chunk_mel = mel[start_idx:end_idx]
            chunk_tok = tokens[start_idx:end_idx]
            chunk_ln = ln_active[start_idx:end_idx]
            chunk_sv = sv_curve[start_idx:end_idx]
            chunk_lane = lane_states[start_idx:end_idx] if lane_states is not None else torch.zeros(chunk_size, 4, dtype=torch.int64)
            chunk_grid = grid_metrics[start_idx:end_idx] if grid_metrics is not None else torch.zeros(chunk_size, 3, dtype=torch.float32)

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

            chunks.append({
                "audio_mel": chunk_mel,
                "tokens": chunk_tok,
                "prev_tokens": prev_tok,
                "lane_states": chunk_lane,
                "grid_metrics": chunk_grid,
                "ln_mask": chunk_ln,
                "sv_curve": chunk_sv,
                "star_rating": torch.tensor([details.star_rating], dtype=torch.float32),
                "bpm": torch.tensor([details.bpm], dtype=torch.float32),
                "offset_ms": torch.tensor([details.offset_ms], dtype=torch.float32),
                "start_ms": round(start_idx * frame_ms, 2),
                "end_ms": round(end_idx * frame_ms, 2),
                "bar_index": 0,
                "beat_len_ms": round(60000.0 / max(30.0, float(details.bpm)), 2),
            })

    return chunks
