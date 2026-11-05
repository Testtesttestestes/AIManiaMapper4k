"""
Sanity checks that keep broken (audio, map) pairs out of the training set.

A map is only useful for training if its audio is the music the notes were placed on. Known ways
this breaks in real beatmap archives:

* ``AudioFilename: virtual`` (or empty / missing file): keysounded maps. The music is assembled at
  play time from hundreds of per-note samples (``l_003.ogg`` ...) attached to the hit objects and the
  storyboard; there is NO song file. The old resolver silently took the largest audio file of the
  archive instead - a one-second piano sample under a four-minute map (the "Piano Beatmap Set" case).
* keysounded maps that also name a real file: most hit objects carry their own sample file name, the
  named file is often just an accompaniment / silence.
* audio much shorter than the map, or silent where the notes are (wrong file, broken encode).
* audio that has nothing to do with the notes (another song of the set, an offset of 50+ ms): the notes
  do not sit on audio attacks any better than copies of themselves shifted by 90..400 ms.

Archive-level checks (:func:`audio_reference_problem`) need only the .osu text; audio-level checks
(:func:`audio_alignment_report`) run on the computed feature matrix, so they cost nothing extra.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .beatmap import Note, _split_sections

AUDIO_EXTS = (".mp3", ".ogg", ".wav", ".flac", ".m4a", ".opus")
VIRTUAL_AUDIO_NAMES = {"", "virtual", "none", "null"}

# thresholds (deliberately loose: a real map passes all of them with a large margin)
MAX_KEYSOUND_SHARE = 0.5        # hit objects with their own sample file
MIN_AUDIO_COVERAGE = 0.9        # share of note times that lie inside the decoded audio
MAX_SILENT_NOTE_SHARE = 0.35    # share of notes placed where the audio is (near) silent
MIN_ALIGNMENT_Z = 3.0           # attack strength at notes vs the notes shifted by 90..400 ms (z-score)


def keysound_share(osu_text: str) -> float:
    """Share of hit objects that name their own sample file (keysounds)."""
    lines = _split_sections(osu_text).get("HitObjects", [])
    total = with_file = 0
    for line in lines:
        parts = line.split(",")
        if len(parts) < 5:
            continue
        total += 1
        sample = parts[-1].split(":")
        if len(sample) >= 5 and sample[-1].strip():
            name = sample[-1].strip().lower()
            if name.endswith(AUDIO_EXTS):
                with_file += 1
    return with_file / total if total else 0.0


def resolve_audio_strict(wanted: str, names: Sequence[str]) -> Optional[str]:
    """Exact (case/path-insensitive) match of AudioFilename among the archive entries; no guessing."""
    w = (wanted or "").strip().lower().replace("\\", "/")
    if w in VIRTUAL_AUDIO_NAMES:
        return None
    for n in names:
        nl = n.lower().replace("\\", "/")
        if nl == w or nl.split("/")[-1] == w.split("/")[-1]:
            return n
    return None


def audio_reference_problem(osu_text: str, audio_filename: str, names: Sequence[str], sizes: Dict[str, int]) -> Tuple[Optional[str], Optional[str]]:
    """(audio entry, problem). Exactly one of them is None.

    The audio file is taken only if the .osu names it. A missing name is accepted only when the archive
    has exactly one audio file that is clearly a song (not a keysound sample).
    """
    wanted = (audio_filename or "").strip()
    if wanted.lower() in VIRTUAL_AUDIO_NAMES and wanted:
        return None, f"AudioFilename is '{wanted}' (keysounded map without a song file)"
    share = keysound_share(osu_text)
    if share > MAX_KEYSOUND_SHARE:
        return None, f"{100 * share:.0f}% of hit objects carry their own keysound sample (music is not in one audio file)"
    entry = resolve_audio_strict(wanted, names)
    if entry is not None:
        return entry, None
    audio = [n for n in names if n.lower().endswith(AUDIO_EXTS)]
    if len(audio) == 1 and sizes.get(audio[0], 0) >= 200_000:
        return audio[0], None
    if not audio:
        return None, "no audio file in archive"
    return None, (f"AudioFilename '{wanted}' not found among {len(audio)} audio files "
                  "(not guessing: a wrong file would teach the network nonsense)")


def _onset_curve(feats: np.ndarray, slices: Dict[str, Tuple[int, int]]) -> Optional[np.ndarray]:
    if "onset" in slices:
        a, b = slices["onset"]
        return np.asarray(feats[:, b - 2:b], dtype=np.float32).mean(axis=1)
    if "mel_fine" in slices:
        a, b = slices["mel_fine"]
        m = np.asarray(feats[:, a:b], dtype=np.float32).mean(axis=1)
        return np.maximum(0.0, np.diff(m, prepend=m[:1]))
    return None


def _loudness(feats: np.ndarray, slices: Dict[str, Tuple[int, int]]) -> Optional[np.ndarray]:
    if "spectral" in slices:
        a, _ = slices["spectral"]
        return np.asarray(feats[:, a], dtype=np.float32)
    if "mel_mid" in slices:
        a, b = slices["mel_mid"]
        return np.asarray(feats[:, a:b], dtype=np.float32).mean(axis=1)
    return None


def onset_lag_z(starts_ms: np.ndarray, feats: np.ndarray, slices: Dict[str, Tuple[int, int]], frame_ms: int = 10) -> Optional[float]:
    """How much better the notes sit on audio attacks than the same notes shifted by 90..400 ms.

    Uses the 1 ms energy-derivative envelope (``fine_env``). Real (audio, map) pairs score 3.7..9 on the
    dataset, a map put on another song's audio scores 1.5..3.2: a random shift of a rhythmic pattern stays
    "on some beat", but only the true alignment has a sharp peak within +-50 ms.
    """
    starts_ms = np.asarray(starts_ms, dtype=np.int64)
    if starts_ms.size < 16:
        return None
    if "fine_env" in slices:
        a, b = slices["fine_env"]
        half = (b - a) // 2
        env = np.maximum(0.0, np.asarray(feats[:, a + half:b], dtype=np.float32)).reshape(-1)
        step = frame_ms / float(half)
    else:
        onset = _onset_curve(feats, slices)
        if onset is None:
            return None
        env = np.maximum(0.0, (onset - onset.mean()) / (onset.std() + 1e-6))
        step = float(frame_ms)
    env = np.convolve(env, np.ones(5) / 5.0, mode="same") if step <= 1.0 else env
    M = env.size
    pos = np.round(starts_ms / step).astype(np.int64)
    pos = pos[(pos >= 0) & (pos < M)]
    if pos.size < 16:
        return None
    lags = np.arange(-400, 401, 2)
    score = np.array([env[np.clip(pos + int(round(l / step)), 0, M - 1)].mean() for l in lags])
    near = score[np.abs(lags) <= 50].max()
    far = score[np.abs(lags) >= 90]
    return float((near - np.median(far)) / (far.std() + 1e-6))


def audio_alignment_report(
    notes: Sequence[Note],
    feats: np.ndarray,
    slices: Dict[str, Tuple[int, int]],
    frame_ms: int = 10,
    duration_ms: Optional[int] = None,
    seed: int = 0,  # kept for API stability (the check is deterministic)
) -> Dict[str, float]:
    """Numbers that tell whether the audio is the music of this map (see module docstring)."""
    F = int(feats.shape[0])
    dur = int(duration_ms if duration_ms is not None else F * frame_ms)
    starts = np.array(sorted({n.start for n in notes}), dtype=np.int64)
    rep: Dict[str, float] = {"events": float(starts.size), "audio_sec": dur / 1000.0}
    if starts.size == 0:
        return rep
    rep["map_sec"] = float(max(n.end for n in notes)) / 1000.0
    rep["coverage"] = float(((starts >= 0) & (starts <= dur + 500)).mean())
    inside = starts[(starts >= 0) & (starts < F * frame_ms)]
    fr = np.clip(inside // frame_ms, 0, F - 1)

    loud = _loudness(feats, slices)
    if loud is not None and fr.size:
        # log-power, z-normalized per song: "silent" = far below the typical level of this song's music
        # (the z scale itself depends on how much silence the file has, so the margin is taken from the
        # spread of the loud half: median - 4 x (p90 - median), at least 0.3)
        med = float(np.median(loud))
        thr = med - max(0.3, 4.0 * (float(np.percentile(loud, 90)) - med))
        win = np.maximum.reduce([loud[np.clip(fr + d, 0, F - 1)] for d in (-3, -2, -1, 0, 1, 2, 3)])
        rep["silent_note_share"] = float((win <= thr).mean())

    z = onset_lag_z(inside, feats, slices, frame_ms)
    if z is not None:
        rep["alignment_z"] = z
    return rep


def alignment_problems(rep: Dict[str, float]) -> List[str]:
    """Human-readable reasons to exclude a map, empty if the pair looks healthy."""
    out: List[str] = []
    if rep.get("events", 0) == 0:
        return ["no notes"]
    if rep.get("coverage", 1.0) < MIN_AUDIO_COVERAGE:
        out.append(f"only {100 * rep['coverage']:.0f}% of the notes lie inside the audio "
                   f"(audio {rep.get('audio_sec', 0):.1f}s, map {rep.get('map_sec', 0):.1f}s)")
    if rep.get("silent_note_share", 0.0) > MAX_SILENT_NOTE_SHARE:
        out.append(f"{100 * rep['silent_note_share']:.0f}% of the notes are on silence")
    if "alignment_z" in rep and rep.get("events", 0) >= 50 and rep["alignment_z"] < MIN_ALIGNMENT_Z:
        out.append(f"notes are not on audio onsets (z={rep['alignment_z']:.1f}, real maps score 3.7+) - "
                   "wrong audio file or an offset of 50+ ms")
    return out
