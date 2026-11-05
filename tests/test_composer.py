"""
Tests of the composer (planned maps at an exact star rating), the learned pattern model and the dataset
audio checks.
"""

import dataclasses
import tempfile
import unittest
from pathlib import Path

import numpy as np

from osu_mapper.hybrid.beatmap import Note, RawTimingPoint
from osu_mapper.hybrid.composer import (
    LANES_OF,
    ComposerConfig,
    Composer,
    analyze_song,
    compose,
    segment_sections,
    star_rating,
    stair_next,
)
from osu_mapper.hybrid.data_checks import (
    alignment_problems,
    audio_alignment_report,
    audio_reference_problem,
    keysound_share,
)
from osu_mapper.hybrid.patterns import (
    PatternModel,
    density_correlation,
    density_graph,
    density_profile,
    ln_quality,
    ln_similarity,
)
from osu_mapper.hybrid.synthetic import map_events, synth_song

try:
    import torch  # noqa: F401  (feature extraction needs torch)
    from osu_mapper.hybrid.audio_features import FeatureConfig, compute_stacked_features
    TORCH = True
except ImportError:  # pragma: no cover
    TORCH = False


_CACHE = {}


def song(seed: int, seconds: float = 40.0, repeat: bool = False):
    """(features, slices, timing points, duration ms, events) of a synthetic song (cached)."""
    key = (seed, seconds, repeat)
    if key not in _CACHE:
        wave, reds, events = synth_song(seed, seconds, bpm_change=False)
        if repeat:  # the first 8 bars once more right after them (songs repeat on bar lines)
            sr = 24000
            off = reds[0].time
            span = 8 * 4 * reds[0].beat_len
            a, b = int(round(off * sr / 1000)), int(round((off + span) * sr / 1000))
            wave = np.concatenate([wave[:b], wave[a:b]])
            events = [e for e in events if e.time_ms < off + span]
            events = events + [dataclasses.replace(e, time_ms=int(round(e.time_ms + span))) for e in events if e.time_ms >= off]
        cfg = FeatureConfig()
        feats = compute_stacked_features(wave, cfg, device="cpu")
        _CACHE[key] = (feats, cfg.slices(), reds, int(len(wave) * 1000 / cfg.sample_rate), events)
    return _CACHE[key]


def pattern_model_from_synthetic() -> PatternModel:
    model = PatternModel.empty()
    for seed in range(4):
        wave, reds, events = synth_song(100 + seed, 30.0, bpm_change=False)
        for diff in ("easy", "hard"):
            notes = map_events(events, diff, reds[0].beat_len)
            model.add_map(notes, reds, star_rating(notes, reds))
    return model


class TestStairs(unittest.TestCase):
    def test_accelerating_stair(self):
        # 1 2 3 4 1 23 4 12 34 123 4 12 341: the stair keeps moving while chords grow
        sizes = [1, 1, 1, 1, 1, 2, 1, 2, 2, 3, 1, 2, 3]
        m, out = 1, ["1"]
        for k in sizes[1:]:
            m = stair_next(m, k, 1)
            out.append("".join(str(c + 1) for c in LANES_OF[m]))
        self.assertEqual(out, ["1", "2", "3", "4", "1", "23", "4", "12", "34", "123", "4", "12", "134"])

    def test_stair_down(self):
        self.assertEqual(stair_next(0b0001, 1, -1), 0b1000)
        self.assertEqual(stair_next(0b1000, 2, -1), 0b0110)


class TestDataChecks(unittest.TestCase):
    KEYSOUND = "[General]\nAudioFilename: virtual\nMode: 3\n[HitObjects]\n" + "\n".join(
        f"64,192,{1000 + 100 * i},1,0,0:0:0:0:l_0{i % 9}.ogg" for i in range(40))

    def test_virtual_audio_is_rejected(self):
        audio, problem = audio_reference_problem(self.KEYSOUND, "virtual", ["l_003.ogg", "l_004.ogg"], {"l_003.ogg": 190000, "l_004.ogg": 1000})
        self.assertIsNone(audio)
        self.assertIn("virtual", problem)

    def test_keysounds_are_detected(self):
        self.assertGreater(keysound_share(self.KEYSOUND), 0.9)
        audio, problem = audio_reference_problem(self.KEYSOUND, "song.mp3", ["song.mp3", "l_003.ogg"], {"song.mp3": 4_000_000})
        self.assertIsNone(audio)
        self.assertIn("keysound", problem)

    def test_missing_file_is_not_guessed(self):
        text = "[HitObjects]\n64,192,1000,1,0,0:0:0:0:\n"
        audio, problem = audio_reference_problem(text, "song.mp3", ["other.mp3", "hit.ogg"], {"other.mp3": 4_000_000, "hit.ogg": 9000})
        self.assertIsNone(audio)
        audio, problem = audio_reference_problem(text, "Song.MP3", ["song.mp3"], {"song.mp3": 4_000_000})
        self.assertEqual(audio, "song.mp3")

    @unittest.skipUnless(TORCH, "torch is required for audio features")
    def test_wrong_audio_is_detected(self):
        feats, slices, reds, dur, events = song(11)
        notes = map_events(events, "hard", reds[0].beat_len)
        right = audio_alignment_report(notes, feats, slices, 10, dur)
        self.assertEqual(alignment_problems(right), [], right)
        other, oslices, _, odur, _ = song(12)
        wrong = audio_alignment_report(notes, other, oslices, 10, odur)
        self.assertLess(wrong["alignment_z"], right["alignment_z"])
        self.assertTrue(any("onsets" in p for p in alignment_problems(wrong)), wrong)
        short = audio_alignment_report([Note(t, 0, t) for t in range(1000, 120000, 500)], feats, slices, 10, dur)
        self.assertTrue(any("inside the audio" in p for p in alignment_problems(short)), short)


class TestPatternModel(unittest.TestCase):
    def test_counts_and_queries(self):
        model = pattern_model_from_synthetic()
        self.assertGreater(model.maps.sum(), 0)
        sr = float(np.flatnonzero(model.maps)[0]) + 0.5
        for k in (1, 2, 3):
            lp = model.lane_logprob(sr, 120.0, 0b0001, 0b0010, k)
            p = np.exp(lp[np.isfinite(lp)])
            self.assertAlmostEqual(float(p.sum()), 1.0, places=5)
            for m in range(16):
                if bin(m).count("1") != k:
                    self.assertEqual(lp[m], -np.inf)
        self.assertAlmostEqual(float(model.chord_marginal(sr).sum()), 1.0, places=6)
        self.assertIsNotNone(model.repeat_floor(sr))
        self.assertGreaterEqual(model.max_chord(sr), 1)

    def test_mirror_symmetry(self):
        model = PatternModel.empty()
        notes = [Note(1000 + 150 * i, i % 4, 1000 + 150 * i) for i in range(64)]  # a roll 1234 1234 ...
        model.add_map(notes, [RawTimingPoint(1000.0, 600.0)], 3.5)
        lp = model.lane_logprob(3.5, 150, 0b0001, 0b1000, 1)
        lp_m = model.lane_logprob(3.5, 150, 0b1000, 0b0001, 1)
        self.assertAlmostEqual(float(lp[0b0010]), float(lp_m[0b0100]), places=6)  # mirrored roll is as likely
        self.assertGreater(lp[0b0010], lp[0b0100])                               # and the roll continues

    def test_thin_bins_borrow_from_neighbours(self):
        model = PatternModel.empty()
        model.maps[:] = 0
        model.maps[[3, 4, 9]] = [200, 150, 2]  # 9-10*: two maps only
        w = dict(model._sr_weights(9.5))
        self.assertAlmostEqual(sum(w.values()), 1.0)
        self.assertAlmostEqual(w[9], 0.2)       # 2 maps of the 10 needed to trust a bin
        self.assertAlmostEqual(w[4], 0.8)       # the rest from the nearest well-filled bin
        self.assertEqual(model._sr_weights(3.5), [(3, 1.0)])

    def test_stagger_is_not_rhythm(self):
        model = PatternModel.empty()
        notes = []
        for i in range(64):  # 1/4 at 150 BPM (100 ms) with every chord split 5 ms apart
            t = 1000 + 100 * i
            notes += [Note(t, 0, t), Note(t + 5, 2, t + 5)]
        model.add_map(notes, [RawTimingPoint(1000.0, 400.0)], 4.5)
        self.assertGreaterEqual(model.event_gap_floor(4.5), 90.0)

    def test_style_model_dominates(self):
        base = pattern_model_from_synthetic()
        style = PatternModel.empty()
        jacks = []
        for i in range(200):  # a chordjack map: the same [12] chord again and again
            t = 1000 + 150 * i
            jacks += [Note(t, 0, t), Note(t, 1, t)]
        from osu_mapper.hybrid.patterns import style_ln_share_at, style_model_from_maps
        style = style_model_from_maps([(jacks, [RawTimingPoint(1000.0, 600.0)], 3.5)])
        mixed = base.with_style(style, 20.0)
        self.assertGreater(mixed.fast_repeat_share(3.5), 0.8)            # the style's jack rate becomes the target
        lp_base = base.lane_logprob(3.5, 150, 0b0011, 0b0011, 2)
        lp_mix = mixed.lane_logprob(3.5, 150, 0b0011, 0b0011, 2)
        self.assertGreater(lp_mix[0b0011], lp_base[0b0011])          # the chord jack became typical
        self.assertEqual(int(np.argmax(lp_mix)), 0b0011)
        self.assertEqual(style_ln_share_at(mixed, 3.5), 0.0)
        self.assertAlmostEqual(float(mixed.chord_marginal(3.5)[1]), 1.0, delta=0.25)

    def test_save_load(self):
        model = pattern_model_from_synthetic()
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "patterns.npz"
            model.save(p)
            back = PatternModel.load(p)
        np.testing.assert_array_equal(back.lane, model.lane)
        np.testing.assert_array_equal(back.repeat_gap, model.repeat_gap)

    def test_density_graph(self):
        notes = [Note(t, 0, t) for t in range(0, 10000, 100)] + [Note(t, 1, t) for t in range(5000, 6000, 50)]
        prof = density_profile(notes, 10000, 1000)
        self.assertEqual(prof.size, 10)
        self.assertGreater(prof[5], prof[0])
        self.assertAlmostEqual(density_correlation(notes, notes, 10000), 1.0)
        self.assertEqual(len(density_graph(notes, 10000, width=20)), 20)


@unittest.skipUnless(TORCH, "torch is required for audio features")
class TestComposer(unittest.TestCase):
    def _check_playable(self, notes):
        by_lane = {}
        for n in notes:
            by_lane.setdefault(n.col, []).append(n)
        for lane in by_lane.values():
            lane.sort()
            for a, b in zip(lane[:-1], lane[1:]):
                self.assertGreater(b.start, a.end if a.ln else a.start, f"overlap on lane {a.col}: {a} {b}")

    def test_hits_requested_star_rating_over_the_range(self):
        feats, slices, reds, dur, _ = song(21)
        an = analyze_song(feats, slices, reds, dur)
        secs = segment_sections(an)
        model = pattern_model_from_synthetic()
        prev_notes = 0
        for target in (1.5, 3.0, 4.5):
            res = compose(an, target, ComposerConfig(variants=1, sr_tolerance=0.1), sections=[dataclasses.replace(s) for s in secs],
                          log=lambda *_: None, patterns=model)
            self.assertLessEqual(abs(res.star_rating - target), 0.15, f"target {target}: got {res.star_rating}")
            self.assertGreater(len(res.notes), prev_notes)  # harder = more notes, not an averaged map
            prev_notes = len(res.notes)
            self._check_playable(res.notes)

    def test_without_pattern_model(self):
        feats, slices, reds, dur, _ = song(21)
        an = analyze_song(feats, slices, reds, dur)
        res = compose(an, 3.0, ComposerConfig(variants=1, sr_tolerance=0.1), log=lambda *_: None)
        self.assertLessEqual(abs(res.star_rating - 3.0), 0.15)
        self._check_playable(res.notes)

    def test_every_sound_has_a_candidate(self):
        # grid ticks down to 1/16 plus attack peaks between ticks: a note can be put on any audible event,
        # including the humanized (+-3 ms off-grid) leads of the synthetic songs
        feats, slices, reds, dur, events = song(21)
        an = analyze_song(feats, slices, reds, dur)
        self.assertLessEqual(abs(an.peak_lag), 12)
        comp = Composer(an, segment_sections(an), ComposerConfig(), None, lambda *_: None, target_sr=3.0)
        heard = np.array(sorted({e.time_ms for e in events}))
        cand = np.sort(comp.t)
        j = np.clip(np.searchsorted(cand, heard), 1, cand.size - 1)
        d = np.minimum(np.abs(cand[j] - heard), np.abs(cand[j - 1] - heard))
        self.assertGreater(float((d <= 5).mean()), 0.95)

    def test_notes_land_on_attacks(self):
        feats, slices, reds, dur, events = song(21)
        an = analyze_song(feats, slices, reds, dur)
        res = compose(an, 3.5, ComposerConfig(variants=1, sr_tolerance=0.1), log=lambda *_: None)
        heard = np.array(sorted({e.time_ms for e in events}))
        starts = np.array(sorted({n.start for n in res.notes}))
        dist = np.abs(heard[np.clip(np.searchsorted(heard, starts), 0, heard.size - 1)] - starts)
        dist = np.minimum(dist, np.abs(heard[np.clip(np.searchsorted(heard, starts) - 1, 0, heard.size - 1)] - starts))
        on_heard = float((dist <= 10).mean())
        # share of all grid ticks that carry a sound: what random placement on the grid would reach
        ticks = an.ticks
        td = np.abs(heard[np.clip(np.searchsorted(heard, ticks), 0, heard.size - 1)] - ticks)
        td = np.minimum(td, np.abs(heard[np.clip(np.searchsorted(heard, ticks) - 1, 0, heard.size - 1)] - ticks))
        chance = float((td <= 10).mean())
        self.assertGreater(on_heard, 3.0 * chance, f"notes on sounds {on_heard:.2f}, random grid ticks {chance:.2f}")

    def test_repeated_music_is_mapped_the_same_way(self):
        feats, slices, reds, dur, _ = song(31, 24.0, repeat=True)
        an = analyze_song(feats, slices, reds, dur)
        secs = segment_sections(an)
        self.assertTrue(any(s.repeat_src_ms is not None for s in secs), [(s.start_ms, s.similarity) for s in secs])
        res = compose(an, 3.0, ComposerConfig(variants=1, sr_tolerance=0.1, mirror_repeats=False), sections=secs, log=lambda *_: None)
        off, span = reds[0].time, int(round(8 * 4 * reds[0].beat_len))
        first = {(n.start, n.col) for n in res.notes if off + 1000 <= n.start < off + span - 1000}
        second = {(n.start - span, n.col) for n in res.notes if off + span + 1000 <= n.start < off + 2 * span - 1000}
        # the repeat starts a non-integer number of ms later (ticks round differently): match within 2 ms
        same = sum(1 for (t, c) in second if any((t + d, c) in first for d in (-2, -1, 0, 1, 2))) / max(1, min(len(first), len(second)))
        self.assertGreater(same, 0.5, f"only {same:.2f} of the repeated passage is mapped like the original")


class TestLongNotes(unittest.TestCase):
    def test_ln_quality_report(self):
        beat = 500.0
        tps = [RawTimingPoint(0.0, beat)]
        notes = [Note(0, 0, 250, True), Note(250, 2, 250), Note(500, 1, 1000, True), Note(750, 3, 750), Note(1000, 2, 1000)]
        q = ln_quality(notes, tps)
        self.assertAlmostEqual(float(q["share"][0]), 0.4)
        self.assertAlmostEqual(float(q["release"][0]), 1.0)           # both holds end on another press
        self.assertAlmostEqual(float(q["length"].sum()), 1.0)
        self.assertEqual(float(q["span"][0]), 0.5)                    # first hold covers no chord, second covers one
        sim = ln_similarity(q, q)
        self.assertTrue(all(abs(v - 1.0) < 1e-6 for v in sim.values()))

    @unittest.skipUnless(TORCH, "torch is required for audio features")
    def test_ln_style_and_share(self):
        feats, slices, reds, dur, events = song(21)
        model = PatternModel.empty()
        for seed in range(4):  # synthetic maps: lead notes longer than a beat are long notes
            _, r, ev = synth_song(100 + seed, 30.0, bpm_change=False)
            notes = map_events(ev, "hard", r[0].beat_len)
            model.add_map(notes, r, star_rating(notes, r))
        self.assertTrue(model.has_ln_model())
        an = analyze_song(feats, slices, reds, dur)
        for share in (0.0, 0.2, 0.45):
            res = compose(an, 3.0, ComposerConfig(variants=1, sr_tolerance=0.15, ln_ratio=share), log=lambda *_: None, patterns=model)
            got = float(ln_quality(res.notes, reds)["share"][0])
            self.assertLess(abs(got - share), 0.08, f"asked {share}, got {got:.2f}")
            if share > 0:
                q = ln_quality(res.notes, reds)
                self.assertLess(float(q["release"][2]), 0.05)        # releases sit on the grid
            by_lane = {}
            for n in res.notes:
                by_lane.setdefault(n.col, []).append(n)
            for lane in by_lane.values():
                lane.sort()
                for a, b in zip(lane[:-1], lane[1:]):
                    self.assertGreater(b.start, a.end if a.ln else a.start)


class TestConsole(unittest.TestCase):
    def test_parse_stars(self):
        from mapper_console import parse_stars
        self.assertEqual(parse_stars("4"), [4.0])
        self.assertEqual(parse_stars("2 4 6"), [2.0, 4.0, 6.0])
        self.assertEqual(parse_stars("2,5 4"), [2.5, 4.0])      # Russian decimal comma
        self.assertEqual(parse_stars("2,4,6"), [2.0, 4.0, 6.0])  # several commas = a list
        self.assertEqual(parse_stars("2-5"), [2.0, 3.0, 4.0, 5.0])
        self.assertEqual(parse_stars("5 3 3"), [3.0, 5.0])

    def test_clean_dragged_path(self):
        from mapper_console import clean_path
        self.assertEqual(clean_path('"C:\\osu\\Songs\\a b.osz"'), "C:\\osu\\Songs\\a b.osz")
        self.assertEqual(clean_path("& 'C:\\a b.osz'"), "C:\\a b.osz")  # PowerShell drag and drop


class TestPitchShift(unittest.TestCase):
    def test_chroma_rotates(self):
        from osu_mapper.hybrid.dataset import pitch_shift_features

        slices = {"chroma": (0, 12), "pitch": (12, 24)}
        x = np.tile(np.arange(24, dtype=np.float32), (3, 1))
        y = pitch_shift_features(x, slices, 2)
        np.testing.assert_array_equal(y[0, :12], np.roll(np.arange(12), 2))
        self.assertEqual(float(y[0, 14]), 12.0)  # pitch bin 2 now holds the old bin 0
        np.testing.assert_array_equal(pitch_shift_features(x, slices, 0), x)


if __name__ == "__main__":
    unittest.main()


class TestNetworkFit(unittest.TestCase):
    def _outputs(self, shift_ms: int = 0, seed: int = 3):
        """Graded 'network' outputs for a synthetic hard map: every real event gets a probability in 0.2..1."""
        from osu_mapper.hybrid.renderer import oracle_outputs
        from osu_mapper.hybrid.solver import solve_map
        from osu_mapper.hybrid.beatmap import ManiaMap
        wave, reds, events = synth_song(41, 40.0, bpm_change=False)
        notes = map_events(events, "hard", reds[0].beat_len)
        if shift_ms:
            notes = [Note(n.start + shift_ms, n.col, n.end + shift_ms, n.ln) for n in notes]
        track = solve_map(ManiaMap(notes=notes, timing_points=reds))
        out = oracle_outputs(track)
        rng = np.random.default_rng(seed)
        ev = np.flatnonzero(out.time_probs[:, 0] < 0.5)
        p = rng.uniform(0.2, 1.0, ev.size)
        out.time_probs[ev] *= p[:, None]
        out.time_probs[ev, 0] = 1.0 - p
        return out, reds, notes

    def test_hedge_relaxed_only_when_rating_unreachable(self):
        from osu_mapper.hybrid.network_fit import fit_network_map
        from osu_mapper.hybrid.renderer import DecodeKnobs, HeadOutputs, decode_outputs
        F = 3000
        tp = np.zeros((F, 11), np.float32)
        tp[:, 0] = 1.0
        for i in range(120):                      # pairs of weak notes 50 ms apart: the hedge merges each pair
            for f in (200 + 20 * i, 205 + 20 * i):
                tp[f, 0], tp[f, 1] = 0.45, 0.55
        lanes = np.zeros((F, 25), np.float32)
        out = HeadOutputs(tp, lanes, np.zeros((F, 4), np.int64), np.full((F, 4), -9.0, np.float32), np.zeros((F, 4), np.int64),
                          np.zeros((F, 4), np.float32))
        reds = [RawTimingPoint(0.0, 400.0)]
        merged = decode_outputs(out, DecodeKnobs(onset_threshold=0.15), reds)
        kept = decode_outputs(out, DecodeKnobs(onset_threshold=0.15, hedge_ms=40, hedge_sum=1.0), reds)
        self.assertLess(len(merged), len(kept))
        sr_m, sr_k = star_rating(merged, reds), star_rating(kept, reds)
        easy = fit_network_map(out, DecodeKnobs(), reds, sr_m, min_threshold=0.15, fix_lag=False)
        self.assertEqual(len(easy.notes), len(merged))          # reachable: hedge untouched
        hard = fit_network_map(out, DecodeKnobs(), reds, sr_k, min_threshold=0.15, fix_lag=False)
        self.assertGreater(len(hard.notes), len(merged))        # unreachable with the hedge: pairs kept apart
        self.assertLess(abs(hard.star_rating - sr_k), abs(sr_m - sr_k))

    def test_fit_reaches_rating_without_new_notes(self):
        from osu_mapper.hybrid.network_fit import fit_network_map
        from osu_mapper.hybrid.renderer import DecodeKnobs, decode_outputs
        out, reds, notes = self._outputs()
        full = decode_outputs(out, DecodeKnobs(onset_threshold=0.15), reds)
        sr_full = star_rating(full, reds)
        target = round(sr_full - 0.6, 2)
        fit = fit_network_map(out, DecodeKnobs(), reds, target, min_threshold=0.15)
        self.assertTrue(fit.reached, (fit.star_rating, target))
        self.assertLessEqual(abs(fit.star_rating - target), 0.05)
        self.assertGreaterEqual(fit.threshold, 0.15)
        self.assertTrue({n.start for n in fit.notes} <= {n.start for n in full})  # only moments the network heard
        too_hard = fit_network_map(out, DecodeKnobs(), reds, sr_full + 2.0, min_threshold=0.15)
        self.assertFalse(too_hard.reached)                                        # says so instead of filling silence
        self.assertLessEqual(too_hard.star_rating, sr_full + 0.05)

    def test_network_lag(self):
        from osu_mapper.hybrid.network_fit import network_lag
        out, reds, _ = self._outputs(shift_ms=6)
        self.assertEqual(network_lag(out, reds), 6)

    def test_nested_chord_sizes(self):
        from osu_mapper.hybrid.composer import _nested_sizes
        rng = np.random.default_rng(0)
        u = rng.random((1000, 3))
        r = np.tile([0.3, 0.07, 0.01], (1000, 1))
        sz = _nested_sizes(u, r)
        self.assertAlmostEqual(float((sz >= 2).mean()), 0.3, delta=0.05)
        self.assertAlmostEqual(float((sz >= 3).mean()), 0.07, delta=0.01)
        self.assertAlmostEqual(float((sz >= 4).mean()), 0.01, delta=0.005)
        self.assertTrue(np.all(sz[sz >= 3] >= 3) and np.all((u[:, 0] < 0.3)[sz >= 3]))  # triples only among 2+ chords
