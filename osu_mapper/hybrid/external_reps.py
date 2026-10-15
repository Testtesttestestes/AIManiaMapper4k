"""
Optional neural "tokenizers" on top of the built-in DSP representations (Suno/MusicGen-style idea:
several complementary encoders of the same audio).

    encodec24     EnCodec 24 kHz encoder latents (the continuous vectors that EnCodec quantizes into
                  codec tokens), 128-d @ 75 Hz                    - pip install encodec
    mert95m       MERT-v1-95M self-supervised music embeddings (mean of all layers), 768-d @ 75 Hz
                                                                  - pip install transformers
    demucs_stems  htdemucs source separation; 32-band log-mel of drums/bass/other/vocals, 128-d @ 100 Hz
                                                                  - pip install demucs

Each adapter downloads its weights on first use (internet needed once). Enable them when preparing the
dataset (``prepare_hybrid_dataset.py --external encodec24 demucs_stems``); the list is stored in the
feature config of the dataset and of every checkpoint, so generation computes the same inputs.
Representations are computed in overlapping chunks, resampled to the 10 ms frame grid and
normalized per song like the built-in ones. Custom encoders can be added with ``register_external``.

These adapters could not be exercised in the development sandbox (model hosting was blocked there);
the plumbing itself is covered by tests with a stand-in encoder.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List

import numpy as np


@dataclass
class ExternalRep:
    name: str
    dim: int
    rate_hz: float
    fn: Callable[[np.ndarray, int], np.ndarray]   # (mono wave, sample_rate) -> [T, dim] at rate_hz
    chunk_s: float = 30.0
    overlap_s: float = 1.0
    requires: str = ""


EXTERNAL_REGISTRY: Dict[str, ExternalRep] = {}
_MODEL_CACHE: Dict[str, object] = {}
_DEVICE = "auto"


def set_external_device(device: str) -> None:
    """Device for the neural encoders: "auto" (CUDA if available), "cuda", "cuda:1", "cpu"."""
    global _DEVICE
    _DEVICE = str(device)


def external_device():
    import torch

    if _DEVICE in ("auto", "", "None"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(_DEVICE)


def _fast_fp16(dev) -> bool:
    """fp16 autocast only on GPUs with fast half precision (tensor cores, compute capability >= 7.0).

    Pascal cards (GTX 10xx) run fp16 matmuls many times slower than fp32, so they stay in fp32.
    """
    import torch

    if dev.type != "cuda":
        return False
    return torch.cuda.get_device_capability(dev)[0] >= 7


def register_external(rep: ExternalRep) -> None:
    EXTERNAL_REGISTRY[rep.name] = rep


def external_dim(name: str) -> int:
    if name not in EXTERNAL_REGISTRY:
        raise KeyError(f"unknown external representation '{name}'. Known: {sorted(EXTERNAL_REGISTRY)}")
    return EXTERNAL_REGISTRY[name].dim


INSTALL_HINTS: Dict[str, str] = {
    "encodec24": "pip install encodec",
    "mert95m": "pip install transformers",
    "demucs_stems": "pip install demucs",
}


def preflight_external(names, sample_rate: int = 24000, log: Callable[[str], None] = print, device: str = None) -> None:
    """Imports every requested encoder and runs it once on 1 s of silence BEFORE any real work.

    This is where the weights are downloaded on first use (Hugging Face / torch hub cache), so a missing
    package or a blocked download stops the run immediately with an explanation instead of failing per song.
    """
    if device is not None:
        set_external_device(device)
    log(f"[external] encoders run on: {external_device()}")
    for name in names:
        if name not in EXTERNAL_REGISTRY:
            raise SystemExit(f"[external] unknown encoder '{name}'. Available: {', '.join(sorted(EXTERNAL_REGISTRY))}")
        rep = EXTERNAL_REGISTRY[name]
        log(f"[external] {name}: loading model (the first run downloads the weights, this can take a few minutes)...")
        try:
            out = np.asarray(rep.fn(np.zeros(sample_rate, dtype=np.float32), sample_rate))
        except ImportError as err:
            hint = INSTALL_HINTS.get(name, f"pip install {rep.requires}")
            raise SystemExit(f"[external] {name}: missing Python package ({err}).\n  Install it with: {hint}")
        except Exception as err:
            raise SystemExit(f"[external] {name}: could not load/run the model: {type(err).__name__}: {err}\n"
                             f"  Check the internet connection (weights are downloaded on first use) and the package version.")
        if out.ndim != 2 or out.shape[1] != rep.dim:
            raise SystemExit(f"[external] {name}: unexpected output shape {out.shape}, expected [T, {rep.dim}]")
        log(f"[external] {name}: OK ({rep.dim}-d @ {rep.rate_hz:g} Hz)")


def compute_external(name: str, wave: np.ndarray, sample_rate: int, num_frames: int, frame_ms: int = 10) -> np.ndarray:
    """Runs an external encoder chunk-wise over the whole track and resamples it to ``num_frames``."""
    rep = EXTERNAL_REGISTRY[name]
    n = wave.size
    chunk = int(rep.chunk_s * sample_rate)
    ov = int(rep.overlap_s * sample_rate)
    pieces: List[np.ndarray] = []
    times: List[np.ndarray] = []
    start = 0
    while start < n:
        a = max(0, start - ov)
        b = min(n, start + chunk + ov)
        emb = np.asarray(rep.fn(wave[a:b], sample_rate), dtype=np.float32)
        if emb.ndim != 2 or emb.shape[1] != rep.dim:
            raise ValueError(f"{name}: encoder returned shape {emb.shape}, expected [T, {rep.dim}]")
        t = a / sample_rate + (np.arange(emb.shape[0]) + 0.5) / rep.rate_hz
        keep = (t >= start / sample_rate) & (t < min(n, start + chunk) / sample_rate)
        pieces.append(emb[keep])
        times.append(t[keep])
        start += chunk
    emb = np.concatenate(pieces) if pieces else np.zeros((0, rep.dim), np.float32)
    t_src = np.concatenate(times) if times else np.zeros(0)
    if emb.shape[0] == 0:
        return np.zeros((num_frames, rep.dim), np.float32)
    t_dst = (np.arange(num_frames) + 0.5) * frame_ms / 1000.0
    idx = np.clip(np.searchsorted(t_src, t_dst), 1, max(1, emb.shape[0] - 1))
    if emb.shape[0] == 1:
        return np.repeat(emb, num_frames, axis=0)
    t0, t1 = t_src[idx - 1], t_src[idx]
    w = np.clip((t_dst - t0) / np.maximum(t1 - t0, 1e-9), 0.0, 1.0)[:, None]
    return ((1.0 - w) * emb[idx - 1] + w * emb[idx]).astype(np.float32)


# ----------------------------------------------------------------------------------------------
# Built-in adapters (lazy imports; weights are downloaded by the respective libraries)
# ----------------------------------------------------------------------------------------------

def _resample(wave: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return wave
    import torch
    import torch.nn.functional as F

    x = torch.from_numpy(np.ascontiguousarray(wave, dtype=np.float32)).view(1, 1, -1)
    n_out = int(round(wave.size * sr_out / sr_in))
    return F.interpolate(x, size=n_out, mode="linear", align_corners=False).view(-1).numpy()


def _encodec(wave: np.ndarray, sr: int) -> np.ndarray:
    import torch
    from encodec import EncodecModel  # type: ignore

    model = _MODEL_CACHE.get("encodec24")
    if model is None:
        model = EncodecModel.encodec_model_24khz()
        model.eval()
        _MODEL_CACHE["encodec24"] = model
    dev = external_device()
    model.to(dev)
    x = torch.from_numpy(_resample(wave, sr, 24000)).view(1, 1, -1).to(dev)
    with torch.no_grad():
        z = model.encoder(x)  # [1, 128, T] @ 75 Hz
    return z[0].transpose(0, 1).cpu().numpy()


def _mert(wave: np.ndarray, sr: int) -> np.ndarray:
    import os

    # MERT ships only pytorch_model.bin; transformers then starts a background thread that downloads a second,
    # safetensors copy of the same weights (378 MB) "for next time" and keeps the process alive until it is done.
    # The weights in use are already loaded, so that download is pure waste.
    os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import torch
    from transformers import AutoModel, Wav2Vec2FeatureExtractor  # type: ignore

    pair = _MODEL_CACHE.get("mert95m")
    if pair is None:
        model = AutoModel.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True).eval()
        proc = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True)
        pair = (model, proc)
        _MODEL_CACHE["mert95m"] = pair
    model, proc = pair  # type: ignore[misc]
    dev = external_device()
    model.to(dev)
    x = _resample(wave, sr, 24000)
    inputs = {k: v.to(dev) for k, v in proc(x, sampling_rate=24000, return_tensors="pt").items()}
    with torch.no_grad(), torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=_fast_fp16(dev)):
        out = model(**inputs, output_hidden_states=True)
    return torch.stack(out.hidden_states).float().mean(dim=0)[0].cpu().numpy()


def _demucs(wave: np.ndarray, sr: int) -> np.ndarray:
    import torch
    from demucs.apply import apply_model  # type: ignore
    from demucs.pretrained import get_model  # type: ignore

    from .audio_features import FeatureConfig, _frame_power, mel_bank

    model = _MODEL_CACHE.get("demucs")
    if model is None:
        model = get_model("htdemucs")
        model.eval()
        _MODEL_CACHE["demucs"] = model
    dev = external_device()
    model.to(dev)
    x = torch.from_numpy(_resample(wave, sr, model.samplerate)).view(1, 1, -1).expand(1, 2, -1).contiguous()
    with torch.no_grad():
        stems = apply_model(model, x, split=True, overlap=0.1, device=dev)[0].mean(dim=1).cpu()  # [4, T]
    cfg = FeatureConfig()
    hop = cfg.hop
    feats = []
    for s in stems:
        mono = torch.from_numpy(_resample(s.cpu().numpy(), model.samplerate, cfg.sample_rate))
        frames = max(1, int(np.ceil(mono.numel() / hop)))
        P = _frame_power(mono, 0, frames, 1024, hop)
        mel = torch.from_numpy(mel_bank(32, 1024, cfg.sample_rate, 30.0, 11000.0))
        feats.append(torch.log(P @ mel.T + 1e-10).numpy())
    return np.concatenate(feats, axis=1)


register_external(ExternalRep("encodec24", 128, 75.0, _encodec, chunk_s=30.0, overlap_s=1.0, requires="encodec"))
register_external(ExternalRep("mert95m", 768, 75.0, _mert, chunk_s=10.0, overlap_s=1.0, requires="transformers"))
register_external(ExternalRep("demucs_stems", 128, 100.0, _demucs, chunk_s=60.0, overlap_s=2.0, requires="demucs"))
