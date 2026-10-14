"""
Multi-representation audio front-end ("tokenizers") for the hybrid network.

Every representation looks at the same audio from a different angle, all aligned to the
10 ms parameter frames (frame f covers [10f, 10f+10) ms):

    mel_fine   64  log-mel, 21 ms window      - sharp transients, attack positions
    mel_mid    96  log-mel, 85 ms window      - timbre / instrument identity
    pitch      96  semitone log-frequency spectrum (A0..A8), 341 ms window - melody / bass pitch
    chroma     12  pitch-class energy share   - harmony
    onset      10  band-wise spectral flux + superflux - where something starts
    hpss       34  percussive vs harmonic energy (median-filter HPSS) - drums vs sustained sounds
    fine_env   20  1 ms energy envelope inside each frame + its derivative - millisecond timing
    spectral    8  loudness, centroid, bandwidth, rolloff, flatness, ZCR, low/high balance, flux

Optional external representations (neural codecs such as EnCodec/DAC, self-supervised models
such as MERT, Demucs stems) can be added as extra ``.npy`` files, see
``extract_external_features.py``; they are resampled to the frame rate by :func:`load_external_rep`.

Everything is computed block-wise (60 s blocks with margins), so memory does not depend on the
track length, and per-song normalized, so the absolute loudness of a file does not matter.
"""

from __future__ import annotations

import math
import os
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - torch is a hard requirement of the hybrid pipeline
    torch = None  # type: ignore
    F = None  # type: ignore

from .beatmap import RawTimingPoint

FEATURE_VERSION = "hybrid-features-v1"
DEFAULT_REPRESENTATIONS: Tuple[str, ...] = (
    "mel_fine", "mel_mid", "pitch", "chroma", "onset", "hpss", "fine_env", "spectral",
)
BEAT_DIM = 13


@dataclass
class FeatureConfig:
    sample_rate: int = 24000
    frame_ms: int = 10
    representations: Tuple[str, ...] = DEFAULT_REPRESENTATIONS
    n_fft_fine: int = 512
    n_mels_fine: int = 64
    n_fft_mid: int = 2048
    n_mels_mid: int = 96
    n_fft_pitch: int = 8192
    pitch_fmin: float = 27.5
    pitch_octaves: int = 8
    pitch_bins_per_octave: int = 12
    onset_bands: int = 8
    hpss_kernel: int = 17
    hpss_groups: int = 16
    fmin: float = 30.0
    fmax: float = 11000.0
    block_frames: int = 6000
    external: Tuple[str, ...] = ()      # optional neural encoders, see external_reps.py
    version: str = FEATURE_VERSION

    @property
    def hop(self) -> int:
        h = self.sample_rate * self.frame_ms / 1000.0
        if abs(h - round(h)) > 1e-9:
            raise ValueError("sample_rate * frame_ms must be a multiple of 1000")
        return int(round(h))

    def dims(self) -> Dict[str, int]:
        table = {
            "mel_fine": self.n_mels_fine,
            "mel_mid": self.n_mels_mid,
            "pitch": self.pitch_octaves * self.pitch_bins_per_octave,
            "chroma": 12,
            "onset": self.onset_bands + 2,
            "hpss": 2 * self.hpss_groups + 2,
            "fine_env": 2 * self.frame_ms,
            "spectral": 8,
        }
        dims = {name: table[name] for name in self.representations}
        if self.external:
            from .external_reps import external_dim
            for name in self.external:
                dims[name] = external_dim(name)
        return dims

    def total_dim(self) -> int:
        return int(sum(self.dims().values()))

    def slices(self) -> Dict[str, Tuple[int, int]]:
        out, pos = {}, 0
        for name, d in self.dims().items():
            out[name] = (pos, pos + d)
            pos += d
        return out

    def to_dict(self) -> Dict[str, object]:
        d = asdict(self)
        d["representations"] = list(self.representations)
        d["external"] = list(self.external)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, object]) -> "FeatureConfig":
        d = dict(d)
        d["representations"] = tuple(d.get("representations", DEFAULT_REPRESENTATIONS))  # type: ignore[arg-type]
        d["external"] = tuple(d.get("external", ()))  # type: ignore[arg-type]
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)  # type: ignore[arg-type]


def num_frames_for_samples(num_samples: int, cfg: FeatureConfig) -> int:
    return max(1, int(math.ceil(num_samples / cfg.hop)))


# ----------------------------------------------------------------------------------------------
# Audio loading / degradation
# ----------------------------------------------------------------------------------------------

def _find_ffmpeg(custom: Optional[str] = None) -> Optional[str]:
    try:
        from osu_mapper.dataset.feature_extractor import find_ffmpeg
        return find_ffmpeg(custom)
    except Exception:
        return shutil.which("ffmpeg")


def load_audio(
    source: str | Path | bytes,
    sample_rate: int = 24000,
    ext: Optional[str] = None,
    ffmpeg_path: Optional[str] = None,
    extra_filters: Optional[str] = None,
) -> np.ndarray:
    """Decodes any audio file (or in-memory bytes) to mono float32 at ``sample_rate`` via FFmpeg."""
    ffmpeg = _find_ffmpeg(ffmpeg_path)
    if ffmpeg is None:
        if isinstance(source, (str, Path)) and str(source).lower().endswith(".wav") and not extra_filters:
            return _load_wav_fallback(Path(source), sample_rate)
        raise FileNotFoundError("FFmpeg not found: put ffmpeg(.exe) into the 'ffmpeg/' folder or add it to PATH.")
    tmp_in = None
    try:
        if isinstance(source, (bytes, bytearray)):
            suffix = ext or ".bin"
            fd, tmp_in = tempfile.mkstemp(suffix=suffix)
            with os.fdopen(fd, "wb") as fh:
                fh.write(source)
            in_path = tmp_in
        else:
            in_path = str(source)
        cmd = [ffmpeg, "-v", "error", "-nostdin", "-i", in_path, "-ac", "1", "-ar", str(sample_rate)]
        if extra_filters:
            cmd += ["-af", extra_filters]
        cmd += ["-f", "f32le", "-"]
        res = subprocess.run(cmd, capture_output=True)
        if res.returncode != 0:
            raise RuntimeError(f"FFmpeg decoding failed: {res.stderr.decode('utf-8', errors='ignore')[:500]}")
        return np.frombuffer(res.stdout, dtype=np.float32).copy()
    finally:
        if tmp_in and os.path.exists(tmp_in):
            os.unlink(tmp_in)


def _load_wav_fallback(path: Path, sample_rate: int) -> np.ndarray:
    import wave
    with wave.open(str(path), "rb") as wf:
        ch, width, sr, n = wf.getnchannels(), wf.getsampwidth(), wf.getframerate(), wf.getnframes()
        raw = wf.readframes(n)
    if width == 2:
        x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 4:
        x = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    elif width == 1:
        x = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise ValueError(f"unsupported WAV sample width {width} (install FFmpeg)")
    x = x.reshape(-1, ch).mean(axis=1)
    if sr != sample_rate:
        n_out = int(round(len(x) * sample_rate / sr))
        x = np.interp(np.linspace(0, len(x) - 1, n_out), np.arange(len(x)), x).astype(np.float32)
    return x


QUALITY_VARIANTS: Dict[str, Tuple[str, Optional[str]]] = {
    # name: (ffmpeg encode args for a lossy round trip, extra decode filter)
    "mp3_48k": ("-codec:a libmp3lame -b:a 48k", None),
    "lowpass_6k": ("", "lowpass=f=6000,highpass=f=60"),
}


def degrade_audio(wave: np.ndarray, sample_rate: int, variant: str, ffmpeg_path: Optional[str] = None) -> np.ndarray:
    """Re-encodes audio through a lossy chain so the network sees different sound qualities."""
    if variant not in QUALITY_VARIANTS:
        raise ValueError(f"unknown quality variant '{variant}', choose from {sorted(QUALITY_VARIANTS)}")
    ffmpeg = _find_ffmpeg(ffmpeg_path)
    if ffmpeg is None:
        raise FileNotFoundError("FFmpeg is required for quality variants")
    enc_args, filt = QUALITY_VARIANTS[variant]
    raw = np.ascontiguousarray(wave, dtype=np.float32).tobytes()
    base = [ffmpeg, "-v", "error", "-nostdin", "-f", "f32le", "-ar", str(sample_rate), "-ac", "1", "-i", "-"]
    if enc_args:
        enc = subprocess.run(base + enc_args.split() + ["-f", "mp3", "-"], input=raw, capture_output=True)
        if enc.returncode != 0:
            raise RuntimeError(enc.stderr.decode("utf-8", errors="ignore")[:300])
        return load_audio(enc.stdout, sample_rate, ext=".mp3", ffmpeg_path=ffmpeg, extra_filters=filt)
    cmd = base + ["-af", filt or "anull", "-f", "f32le", "-"]
    res = subprocess.run(cmd, input=raw, capture_output=True)
    if res.returncode != 0:
        raise RuntimeError(res.stderr.decode("utf-8", errors="ignore")[:300])
    out = np.frombuffer(res.stdout, dtype=np.float32).copy()
    return out[: len(wave)] if len(out) >= len(wave) else np.pad(out, (0, len(wave) - len(out)))


# ----------------------------------------------------------------------------------------------
# Filterbanks
# ----------------------------------------------------------------------------------------------

def _hz_to_mel(f: np.ndarray | float) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=np.float64) / 700.0)


def _mel_to_hz(m: np.ndarray | float) -> np.ndarray:
    return 700.0 * (10.0 ** (np.asarray(m, dtype=np.float64) / 2595.0) - 1.0)


def _triangular_bank(centers_hz: np.ndarray, edges_lo: np.ndarray, edges_hi: np.ndarray, n_fft: int, sr: int) -> np.ndarray:
    freqs = np.linspace(0.0, sr / 2.0, n_fft // 2 + 1)
    bank = np.zeros((len(centers_hz), freqs.size), dtype=np.float32)
    for i, (lo, c, hi) in enumerate(zip(edges_lo, centers_hz, edges_hi)):
        up = (freqs - lo) / max(c - lo, 1e-9)
        down = (hi - freqs) / max(hi - c, 1e-9)
        w = np.clip(np.minimum(up, down), 0.0, None)
        if w.sum() <= 0:  # narrower than one FFT bin: use the nearest bin
            w[np.argmin(np.abs(freqs - c))] = 1.0
        bank[i] = w / w.sum()
    return bank


_BANK_CACHE: Dict[Tuple, np.ndarray] = {}


def mel_bank(n_mels: int, n_fft: int, sr: int, fmin: float, fmax: float) -> np.ndarray:
    key = ("mel", n_mels, n_fft, sr, fmin, fmax)
    if key not in _BANK_CACHE:
        mels = np.linspace(_hz_to_mel(fmin), _hz_to_mel(min(fmax, sr / 2.0)), n_mels + 2)
        hz = _mel_to_hz(mels)
        _BANK_CACHE[key] = _triangular_bank(hz[1:-1], hz[:-2], hz[2:], n_fft, sr)
    return _BANK_CACHE[key]


def pitch_bank(cfg: FeatureConfig) -> np.ndarray:
    key = ("pitch", cfg.n_fft_pitch, cfg.sample_rate, cfg.pitch_fmin, cfg.pitch_octaves, cfg.pitch_bins_per_octave)
    if key not in _BANK_CACHE:
        n = cfg.pitch_octaves * cfg.pitch_bins_per_octave
        k = np.arange(-1, n + 1)
        f = cfg.pitch_fmin * 2.0 ** (k / cfg.pitch_bins_per_octave)
        f = np.minimum(f, cfg.sample_rate / 2.0 - 1.0)
        _BANK_CACHE[key] = _triangular_bank(f[1:-1], f[:-2], f[2:], cfg.n_fft_pitch, cfg.sample_rate)
    return _BANK_CACHE[key]


# ----------------------------------------------------------------------------------------------
# Feature computation
# ----------------------------------------------------------------------------------------------

def _slice_padded(x: "torch.Tensor", start: int, length: int) -> "torch.Tensor":
    out = torch.zeros(length, dtype=x.dtype, device=x.device)
    s0, s1 = max(0, start), min(x.numel(), start + length)
    if s1 > s0:
        out[s0 - start: s1 - start] = x[s0:s1]
    return out


def _frame_power(x: "torch.Tensor", f0: int, f1: int, n_fft: int, hop: int) -> "torch.Tensor":
    """Power spectrum of frames f0..f1-1, frame f centred at sample f*hop + hop/2. Returns [T, bins]."""
    start = f0 * hop + hop // 2 - n_fft // 2
    seg = _slice_padded(x, start, (f1 - f0 - 1) * hop + n_fft)
    win = torch.hann_window(n_fft, dtype=torch.float32, device=x.device)
    spec = torch.stft(seg, n_fft=n_fft, hop_length=hop, win_length=n_fft, window=win, center=False, return_complex=True)
    return (spec.real ** 2 + spec.imag ** 2).transpose(0, 1).contiguous()


def _median_filter(x: "torch.Tensor", k: int, dim: int) -> "torch.Tensor":
    pad = k // 2
    xt = x.transpose(dim, -1)
    xp = F.pad(xt.unsqueeze(0), (pad, pad), mode="replicate").squeeze(0)
    return xp.unfold(-1, k, 1).median(dim=-1).values.transpose(dim, -1)


def _group_mean(x: "torch.Tensor", groups: int) -> "torch.Tensor":
    T, D = x.shape
    edges = np.linspace(0, D, groups + 1).round().astype(int)
    return torch.stack([x[:, edges[i]:max(edges[i + 1], edges[i] + 1)].mean(dim=1) for i in range(groups)], dim=1)


def _compute_block(x: "torch.Tensor", f0: int, f1: int, cfg: FeatureConfig) -> Dict[str, "torch.Tensor"]:
    hop, sr = cfg.hop, cfg.sample_rate
    reps = set(cfg.representations)
    out: Dict[str, torch.Tensor] = {}
    eps = 1e-10

    need_fine = reps & {"mel_fine", "onset"}
    need_mid = reps & {"mel_mid", "hpss", "spectral"}
    need_pitch = reps & {"pitch", "chroma"}

    if need_fine:
        P = _frame_power(x, f0, f1, cfg.n_fft_fine, hop)
        mel = torch.from_numpy(mel_bank(cfg.n_mels_fine, cfg.n_fft_fine, sr, cfg.fmin, cfg.fmax)).to(x.device)
        L = torch.log(P @ mel.T + eps)
        if "mel_fine" in reps:
            out["mel_fine"] = L
        if "onset" in reps:
            prev = torch.cat([L[:1], L[:-1]], dim=0)
            flux = F.relu(L - prev)
            bands = _group_mean(flux, cfg.onset_bands)
            total = flux.mean(dim=1, keepdim=True)
            prev_max = F.max_pool1d(prev.unsqueeze(0), kernel_size=3, stride=1, padding=1).squeeze(0)
            superflux = F.relu(L - prev_max).mean(dim=1, keepdim=True)
            out["onset"] = torch.cat([torch.log1p(bands), torch.log1p(total), torch.log1p(superflux)], dim=1)

    if need_mid:
        P = _frame_power(x, f0, f1, cfg.n_fft_mid, hop)
        mel = torch.from_numpy(mel_bank(cfg.n_mels_mid, cfg.n_fft_mid, sr, cfg.fmin, cfg.fmax)).to(x.device)
        M = P @ mel.T
        if "mel_mid" in reps:
            out["mel_mid"] = torch.log(M + eps)
        if "hpss" in reps:
            k = cfg.hpss_kernel
            Hm = _median_filter(M, k, dim=0)
            Pm = _median_filter(M, k, dim=1)
            denom = Hm ** 2 + Pm ** 2 + eps
            perc = M * (Pm ** 2 / denom)
            harm = M * (Hm ** 2 / denom)
            out["hpss"] = torch.cat([
                torch.log(_group_mean(perc, cfg.hpss_groups) + eps),
                torch.log(_group_mean(harm, cfg.hpss_groups) + eps),
                torch.log(perc.sum(dim=1, keepdim=True) + eps),
                torch.log(harm.sum(dim=1, keepdim=True) + eps),
            ], dim=1)
        if "spectral" in reps:
            freqs = torch.linspace(0.0, sr / 2.0, P.shape[1], device=x.device)
            tot = P.sum(dim=1) + eps
            centroid = (P * freqs).sum(dim=1) / tot
            bandwidth = torch.sqrt(((freqs[None, :] - centroid[:, None]) ** 2 * P).sum(dim=1) / tot)
            cum = torch.cumsum(P, dim=1)
            roll_idx = torch.searchsorted(cum, (0.85 * cum[:, -1:]).contiguous()).clamp(max=P.shape[1] - 1).squeeze(1)
            rolloff = freqs[roll_idx]
            flatness = torch.exp(torch.log(P + eps).mean(dim=1)) / (P.mean(dim=1) + eps)
            low = P[:, freqs < 200.0].sum(dim=1) + eps
            high = P[:, freqs > 2000.0].sum(dim=1) + eps
            logP = torch.log(P + eps)
            prevP = torch.cat([logP[:1], logP[:-1]], dim=0)
            flux_mid = F.relu(logP - prevP).mean(dim=1)
            seg = _slice_padded(x, f0 * hop, (f1 - f0) * hop).view(f1 - f0, hop)
            zcr = (torch.sign(seg[:, 1:]) != torch.sign(seg[:, :-1])).float().mean(dim=1)
            out["spectral"] = torch.stack([
                torch.log(tot), torch.log1p(centroid), torch.log1p(bandwidth), torch.log1p(rolloff),
                torch.log(flatness + eps), zcr, torch.log(low) - torch.log(high), flux_mid,
            ], dim=1)

    if need_pitch:
        P = _frame_power(x, f0, f1, cfg.n_fft_pitch, hop)
        bank = torch.from_numpy(pitch_bank(cfg)).to(x.device)
        Q = P @ bank.T
        if "pitch" in reps:
            out["pitch"] = torch.log(Q + eps)
        if "chroma" in reps:
            n = cfg.pitch_bins_per_octave
            C = Q.view(Q.shape[0], cfg.pitch_octaves, n).sum(dim=1)
            if n != 12:
                C = F.interpolate(C.unsqueeze(1), size=12, mode="linear", align_corners=False).squeeze(1)
            out["chroma"] = C / (C.sum(dim=1, keepdim=True) + eps)

    if "fine_env" in reps:
        sub = cfg.frame_ms
        step = hop // sub
        seg = _slice_padded(x, f0 * hop - 1, (f1 - f0) * hop + 1)
        y = seg[1:] - 0.97 * seg[:-1]
        e = torch.log((y.view(-1, step) ** 2).mean(dim=1) + eps)
        d = torch.cat([e[:1] * 0.0, e[1:] - e[:-1]])
        out["fine_env"] = torch.cat([e.view(f1 - f0, sub), d.view(f1 - f0, sub)], dim=1)

    return out


def compute_stacked_features(
    wave: np.ndarray,
    cfg: Optional[FeatureConfig] = None,
    normalize: bool = True,
    device: str = "cpu",
) -> np.ndarray:
    """All configured representations concatenated into one [F, D] float32 array (columns: ``cfg.slices()``).

    Filled block by block and normalized in place, so a 60-minute track needs ~0.5 GB for the result.
    ``device="cuda"`` runs the STFTs on the GPU (much faster for big datasets).
    """
    if torch is None:
        raise ImportError("PyTorch is required for feature extraction")
    cfg = cfg or FeatureConfig()
    dev = torch.device("cuda" if (device == "auto" and torch.cuda.is_available()) else ("cpu" if device == "auto" else device))
    x = torch.as_tensor(np.ascontiguousarray(wave, dtype=np.float32)).flatten().to(dev)
    total = num_frames_for_samples(x.numel(), cfg)
    slices = cfg.slices()
    out = np.zeros((total, cfg.total_dim()), dtype=np.float32)
    builtin = [name for name in slices if name in cfg.representations]
    margin = max(2 * cfg.hpss_kernel, 8)
    with torch.no_grad():
        for b0 in range(0, total, cfg.block_frames):
            b1 = min(total, b0 + cfg.block_frames)
            f0, f1 = max(0, b0 - margin), min(total, b1 + margin)
            block = _compute_block(x, f0, f1, cfg)
            for name in builtin:
                a, b = slices[name]
                out[b0:b1, a:b] = block[name][b0 - f0: b1 - f0].float().cpu().numpy()
            del block
    del x
    if cfg.external:
        from .external_reps import compute_external
        for name in cfg.external:
            a, b = slices[name]
            out[:, a:b] = compute_external(name, np.asarray(wave, dtype=np.float32), cfg.sample_rate, total, cfg.frame_ms)
    if normalize:
        normalize_inplace(out)
    return out


def compute_features(
    wave: np.ndarray,
    cfg: Optional[FeatureConfig] = None,
    normalize: bool = True,
    device: str = "cpu",
) -> Dict[str, np.ndarray]:
    """Same as :func:`compute_stacked_features` but as a dict name -> [F, D_r] (views, no copy)."""
    cfg = cfg or FeatureConfig()
    stacked = compute_stacked_features(wave, cfg, normalize, device)
    return {name: stacked[:, a:b] for name, (a, b) in cfg.slices().items()}


def normalize_inplace(x: np.ndarray, clip: float = 8.0, block: int = 65536) -> np.ndarray:
    """Per-song, per-column standardization without full-size temporaries."""
    n = max(1, x.shape[0])
    s1 = np.zeros(x.shape[1], dtype=np.float64)
    s2 = np.zeros(x.shape[1], dtype=np.float64)
    for i in range(0, x.shape[0], block):
        b = x[i:i + block].astype(np.float64)
        s1 += b.sum(axis=0)
        s2 += (b * b).sum(axis=0)
    mean = s1 / n
    std = np.sqrt(np.maximum(s2 / n - mean * mean, 0.0))
    inv = (1.0 / (std + 1e-5)).astype(np.float32)
    mean32 = mean.astype(np.float32)
    for i in range(0, x.shape[0], block):
        seg = x[i:i + block]
        seg -= mean32
        seg *= inv
        np.clip(seg, -clip, clip, out=seg)
    return x


def normalize_per_song(x: np.ndarray, clip: float = 8.0) -> np.ndarray:
    return normalize_inplace(np.array(x, dtype=np.float32, copy=True), clip)


def stack_features(reps: Dict[str, np.ndarray], cfg: FeatureConfig) -> np.ndarray:
    return np.concatenate([reps[name] for name in cfg.dims()], axis=1).astype(np.float32)


def coarse_features(stacked: np.ndarray, frames_per_step: int = 100) -> np.ndarray:
    """1 s mean-pooled song summary used by the length-agnostic global context branch."""
    Fr, D = stacked.shape
    n = int(math.ceil(Fr / frames_per_step))
    pad = n * frames_per_step - Fr
    x = np.pad(stacked.astype(np.float32), ((0, pad), (0, 0)))
    counts = np.full(n, frames_per_step, dtype=np.float32)
    counts[-1] = frames_per_step - pad
    return (x.reshape(n, frames_per_step, D).sum(axis=1) / counts[:, None]).astype(np.float32)


# ----------------------------------------------------------------------------------------------
# Beat-grid features (from red lines)
# ----------------------------------------------------------------------------------------------

def beat_features(
    timing_points: Sequence[RawTimingPoint],
    start_frame: int,
    num_frames: int,
    frame_ms: int = 10,
) -> np.ndarray:
    """[num_frames, 13]: beat/bar phase (sin, cos), subdivision alignment 1/2..1/16, log tempo, available flag."""
    out = np.zeros((num_frames, BEAT_DIM), dtype=np.float32)
    reds = sorted((tp for tp in timing_points if tp.uninherited and tp.beat_len > 0), key=lambda t: t.time)
    if not reds:
        return out
    rt = np.array([tp.time for tp in reds], dtype=np.float64)
    bl = np.array([tp.beat_len for tp in reds], dtype=np.float64)
    meter = np.array([max(1, tp.meter) for tp in reds], dtype=np.float64)
    t = (np.arange(start_frame, start_frame + num_frames, dtype=np.float64) + 0.5) * frame_ms
    idx = np.clip(np.searchsorted(rt, t, side="right") - 1, 0, len(reds) - 1)
    beats = (t - rt[idx]) / bl[idx]
    phase = np.mod(beats, 1.0)
    bar = np.mod(beats / meter[idx], 1.0)
    two_pi = 2.0 * np.pi
    cols = [np.sin(two_pi * phase), np.cos(two_pi * phase), np.sin(two_pi * bar), np.cos(two_pi * bar)]
    cols += [np.cos(two_pi * d * phase) for d in (2, 3, 4, 6, 8, 12, 16)]
    cols.append(np.clip(np.log2((60000.0 / bl[idx]) / 120.0), -2.0, 2.0))
    cols.append(np.ones_like(t))
    out[:] = np.stack(cols, axis=1).astype(np.float32)
    return out


# ----------------------------------------------------------------------------------------------
# External representations (pre-computed by extract_external_features.py)
# ----------------------------------------------------------------------------------------------

def load_external_rep(path: str | Path, src_rate_hz: float, num_frames: int, frame_ms: int = 10, categorical: bool = False) -> np.ndarray:
    """Loads a [T_src, D] array sampled at ``src_rate_hz`` and resamples it to the 10 ms frame grid."""
    arr = np.load(path)
    if arr.ndim == 1:
        arr = arr[:, None]
    t_dst = (np.arange(num_frames) + 0.5) * frame_ms / 1000.0
    pos = t_dst * src_rate_hz - 0.5
    if categorical:
        idx = np.clip(np.round(pos).astype(np.int64), 0, arr.shape[0] - 1)
        return arr[idx]
    i0 = np.clip(np.floor(pos).astype(np.int64), 0, arr.shape[0] - 1)
    i1 = np.clip(i0 + 1, 0, arr.shape[0] - 1)
    w = np.clip(pos - i0, 0.0, 1.0)[:, None]
    return ((1.0 - w) * arr[i0] + w * arr[i1]).astype(np.float32)


# ----------------------------------------------------------------------------------------------
# Fast tempo estimate (only used to write a red line when no template/BPM is given)
# ----------------------------------------------------------------------------------------------

def estimate_tempo(
    reps: Dict[str, np.ndarray],
    frame_ms: int = 10,
    bpm_range: Tuple[float, float] = (70.0, 280.0),
    prior_bpm: float = 150.0,
) -> Tuple[float, float]:
    """Autocorrelation tempo (with a log-tempo prior against octave errors) + 1 ms comb-filter phase.

    Returns (bpm, offset_ms). Only used to write a red line when no template / manual BPM is given.
    """
    if "onset" in reps:
        env = reps["onset"][:, -2].astype(np.float64)
    else:
        first = next(iter(reps.values()))
        env = np.maximum(0.0, np.diff(first.mean(axis=1), prepend=first[0].mean()))
    env = env - env.mean()
    n = env.size
    if n < 400:
        return 120.0, 0.0
    spec = np.fft.rfft(env, n=2 * n)
    ac = np.fft.irfft(np.abs(spec) ** 2)[:n]
    fps = 1000.0 / frame_ms

    def _ac(L: float) -> float:
        i = int(L)
        if i + 1 >= n:
            return 0.0
        return (1 - (L - i)) * ac[i] + (L - i) * ac[i + 1]

    best_bpm, best_score = 120.0, -np.inf
    for bpm in np.arange(bpm_range[0], bpm_range[1] + 1e-9, 0.25):
        lag = 60.0 * fps / bpm
        score = _ac(lag) + 0.5 * _ac(2 * lag) + 0.25 * _ac(4 * lag)
        score *= math.exp(-0.5 * (math.log2(bpm / prior_bpm) / 0.9) ** 2)
        if score > best_score:
            best_score, best_bpm = score, float(bpm)

    beat_ms = 60000.0 / best_bpm
    if "fine_env" in reps:  # 1 ms resolution onset envelope: positive part of the energy derivative
        sub = reps["fine_env"].shape[1] // 2
        fine = np.maximum(0.0, reps["fine_env"][:, sub:].reshape(-1).astype(np.float64))
        fine = np.convolve(fine, np.ones(3) / 3.0, mode="same")
        step_ms = 1.0
    else:
        fine = np.maximum(0.0, env)
        step_ms = float(frame_ms)
    m = fine.size
    k = np.arange(0, int(m * step_ms / beat_ms))
    best_phase, best_val = 0.0, -np.inf
    for phase in np.arange(0.0, beat_ms, step_ms):
        idx = np.clip(np.round((phase + k * beat_ms) / step_ms).astype(np.int64), 0, m - 1)
        val = fine[idx].mean()
        if val > best_val:
            best_val, best_phase = val, float(phase)
    return round(best_bpm, 2), round(best_phase, 1)
