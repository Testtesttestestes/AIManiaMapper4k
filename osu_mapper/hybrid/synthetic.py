"""
Synthetic songs with exactly aligned 4K maps, for smoke tests and pre-launch checks.

Every song is a drum + lead arrangement rendered to audio together with maps whose notes are a
deterministic function of what is audible (kick/snare/hat/lead onsets, lead pitch -> lane, long
lead tones -> long notes, a few humanized off-grid notes, optional BPM change). A network that
learns the mapping must get timing exact to the millisecond, so these songs are a quick way to
check that the whole pipeline (ffmpeg -> features -> solver -> training -> inference -> render) works
before spending GPU hours on the real dataset.
"""

from __future__ import annotations

import io
import wave
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .beatmap import ManiaMap, Note, RawTimingPoint, map_to_osu_text

PENTATONIC = [0, 2, 4, 7, 9]


@dataclass
class SynthEvent:
    time_ms: int
    kind: str          # kick / snare / hat / lead
    pitch: int = 0     # MIDI pitch for lead
    dur_ms: int = 0


def _env(n: int, sr: int, decay_s: float) -> np.ndarray:
    return np.exp(-np.arange(n) / (decay_s * sr)).astype(np.float32)


def _instrument(kind: str, sr: int, rng: np.random.Generator, pitch: int = 60, dur_ms: int = 200) -> np.ndarray:
    if kind == "kick":
        n = int(0.18 * sr)
        t = np.arange(n) / sr
        freq = 45.0 + 80.0 * np.exp(-t / 0.03)
        phase = 2 * np.pi * np.cumsum(freq) / sr
        return 0.9 * np.sin(phase).astype(np.float32) * _env(n, sr, 0.06)
    if kind == "snare":
        n = int(0.15 * sr)
        noise = rng.standard_normal(n).astype(np.float32)
        noise = noise - np.convolve(noise, np.ones(8) / 8.0, mode="same")
        tone = np.sin(2 * np.pi * 190.0 * np.arange(n) / sr).astype(np.float32)
        return (0.45 * noise + 0.3 * tone) * _env(n, sr, 0.035)
    if kind == "hat":
        n = int(0.04 * sr)
        noise = np.diff(rng.standard_normal(n + 1)).astype(np.float32)
        return 0.18 * noise * _env(n, sr, 0.008)
    if kind == "lead":
        n = int(dur_ms * sr / 1000)
        t = np.arange(n) / sr
        f0 = 440.0 * 2 ** ((pitch - 69) / 12.0)
        sig = np.sin(2 * np.pi * f0 * t) + 0.35 * np.sin(4 * np.pi * f0 * t) + 0.15 * np.sin(6 * np.pi * f0 * t)
        attack = np.minimum(1.0, t / 0.004)
        release = np.minimum(1.0, (t[-1] - t + 1e-9) / 0.03)
        return (0.22 * sig * attack * release * (0.6 + 0.4 * np.exp(-t / 0.3))).astype(np.float32)
    raise ValueError(kind)


def synth_song(seed: int, duration_s: float = 60.0, sr: int = 24000, bpm_change: Optional[bool] = None) -> Tuple[np.ndarray, List[RawTimingPoint], List[SynthEvent]]:
    rng = np.random.default_rng(seed)
    bpm = float(rng.integers(110, 191))
    offset = float(rng.integers(150, 900))
    change = bool(rng.random() < 0.3) if bpm_change is None else bpm_change
    total_ms = duration_s * 1000.0
    reds = [RawTimingPoint(offset, 60000.0 / bpm, 4, 0, 0, 70, True, 0)]
    events: List[SynthEvent] = []
    t_bar = offset
    bar = 0
    level = 2
    base_pitch = int(rng.integers(55, 64))
    while t_bar < total_ms - 500:
        if change and bar == 16:
            bpm = float(np.clip(bpm + rng.choice([-30, 30]), 100, 220))
            reds.append(RawTimingPoint(round(t_bar, 3), 60000.0 / bpm, 4, 0, 0, 70, True, 0))
        beat = 60000.0 / bpm
        if bar % 8 == 0:
            level = int(rng.choice([0, 1, 2, 2, 3, 3]))
        for step in range(16):  # 1/4-beat steps
            t = t_bar + step * beat / 4.0
            if t >= total_ms - 300:
                break
            ti = int(round(t))
            on_half = step % 2 == 0
            if level >= 1 and step in (0, 8) or (level >= 2 and step == 10 and rng.random() < 0.5):
                events.append(SynthEvent(ti, "kick"))
            if level >= 1 and step in (4, 12):
                events.append(SynthEvent(ti, "snare"))
            if (level >= 2 and on_half) or (level >= 3 and not on_half and rng.random() < 0.7):
                events.append(SynthEvent(ti, "hat"))
            if level >= 1 and (on_half or level >= 3) and rng.random() < (0.35 if level < 3 else 0.5):
                human = int(rng.integers(-3, 4)) if rng.random() < 0.15 else 0
                pitch = base_pitch + 12 * int(rng.integers(0, 2)) + PENTATONIC[int(rng.integers(0, 5))]
                dur = float(rng.choice([0.25, 0.5, 1.0, 1.5, 2.0], p=[0.35, 0.3, 0.2, 0.1, 0.05])) * beat
                events.append(SynthEvent(ti + human, "lead", pitch, int(round(dur))))
        t_bar += 4 * beat
        bar += 1

    n = int(duration_s * sr)
    audio = np.zeros(n + sr, dtype=np.float32)
    for ev in events:
        snd = _instrument(ev.kind, sr, rng, ev.pitch, max(60, ev.dur_ms))
        i = int(round(ev.time_ms * sr / 1000.0))
        audio[i: i + snd.size] += snd[: max(0, audio.size - i)]
    audio = audio[:n]
    audio += 0.002 * rng.standard_normal(n).astype(np.float32)
    peak = float(np.abs(audio).max()) or 1.0
    return (0.9 * audio / peak).astype(np.float32), reds, sorted(events, key=lambda e: (e.time_ms, e.kind))


def map_events(events: Sequence[SynthEvent], difficulty: str, beat_ms: float) -> List[Note]:
    """Deterministic mapping rules: what you hear decides where the note goes."""
    by_time: Dict[int, List[SynthEvent]] = {}
    for ev in events:
        by_time.setdefault(ev.time_ms, []).append(ev)
    notes: List[Note] = []
    busy_until = [-1] * 4
    last_start = [-10 ** 9] * 4
    hat_flip = 0
    for t in sorted(by_time):
        lanes: List[Tuple[int, Optional[int]]] = []
        for ev in by_time[t]:
            if ev.kind == "kick":
                lanes.append((0, None))
            elif ev.kind == "snare":
                lanes.append((3, None))
            elif ev.kind == "hat" and difficulty == "hard":
                lanes.append((1 + hat_flip, None))
                hat_flip ^= 1
            elif ev.kind == "lead":
                lane = 1 if ev.pitch < 62 else (2 if ev.pitch < 69 else 3)
                is_long = ev.dur_ms >= 0.9 * beat_ms and difficulty == "hard"
                lanes.append((lane, t + ev.dur_ms - 20 if is_long else None))
        if difficulty == "easy":
            lanes = lanes[:1]
        used = set()
        for lane, end in lanes:
            cand = [lane] + [c for c in (lane + 1, lane - 1, lane + 2, lane - 2, lane + 3, lane - 3) if 0 <= c < 4]
            pick = next((c for c in cand if c not in used and busy_until[c] < t and t - last_start[c] >= 40), None)
            if pick is None:
                continue
            used.add(pick)
            last_start[pick] = t
            if end is not None and end - t >= 60:
                notes.append(Note(t, pick, end, True))
                busy_until[pick] = end
            else:
                notes.append(Note(t, pick, t, False))
    # LNs must end before the next note on their lane
    out: List[Note] = []
    by_lane: Dict[int, List[Note]] = {}
    for nt in sorted(notes):
        by_lane.setdefault(nt.col, []).append(nt)
    for lane_notes in by_lane.values():
        for i, nt in enumerate(lane_notes):
            if nt.ln and i + 1 < len(lane_notes) and nt.end >= lane_notes[i + 1].start:
                end = lane_notes[i + 1].start - 20
                nt = Note(nt.start, nt.col, end, True) if end - nt.start >= 60 else Note(nt.start, nt.col, nt.start, False)
            out.append(nt)
    return sorted(out)


def wav_bytes(audio: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())
    return buf.getvalue()


def write_synthetic_osz(path: Path, seed: int, duration_s: float = 60.0, sr: int = 24000, difficulties: Sequence[str] = ("easy", "hard")) -> List[ManiaMap]:
    audio, reds, events = synth_song(seed, duration_s, sr)
    maps: List[ManiaMap] = []
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("audio.wav", wav_bytes(audio, sr))
        for diff in difficulties:
            notes = map_events(events, diff, reds[0].beat_len)
            m = ManiaMap(
                notes=notes,
                timing_points=list(reds),
                metadata={"Title": f"Synthetic {seed}", "Artist": "Synth", "Creator": "synthetic", "Version": diff, "AudioFilename": "audio.wav"},
            )
            z.writestr(f"Synth - Synthetic {seed} (synthetic) [{diff}].osu", map_to_osu_text(m))
            maps.append(m)
    return maps


def make_synthetic_dataset(folder: Path, n_songs: int, seed: int = 0, duration_s: float = 60.0) -> List[Path]:
    folder = Path(folder)
    paths = []
    for i in range(n_songs):
        p = folder / f"{900000 + seed * 1000 + i} Synth - Synthetic {seed * 1000 + i}.osz"
        write_synthetic_osz(p, seed * 1000 + i, duration_s)
        paths.append(p)
    return paths
