"""Tests of the motif library (osu_mapper/hybrid/motifs.py) and the zone comparison helpers (compare_zones.py)."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from osu_mapper.hybrid.beatmap import Note, RawTimingPoint
from osu_mapper.hybrid.motifs import (
    R_REST,
    RHYTHM_NAMES,
    Corpus,
    MotifLibrary,
    build_library,
    chord_seq,
    mirror_tokens,
    name_motif,
    rhythm_class,
    self_repeats,
    tokenize,
)

BEAT = 400.0  # 150 BPM
TP = [RawTimingPoint(0.0, BEAT, 4, 0, 0, 70, True, 0)]


def notes_from(cols, step_beats=0.25, start=0.0, ln_beats=None):
    out = []
    t = start
    for c in cols:
        for col in (c if isinstance(c, (tuple, list)) else (c,)):
            if ln_beats:
                out.append(Note(int(round(t)), col, int(round(t + ln_beats * BEAT)), True))
            else:
                out.append(Note(int(round(t)), col, int(round(t)), False))
        t += step_beats * BEAT
    return out


def random_cols(rng, n):
    return [int(x) for x in rng.integers(0, 4, n)]


class TokenTests(unittest.TestCase):
    def test_rhythm_classes(self):
        g = np.array([0.25, 1 / 3, 0.5, 1 / 12, 10.0, 0.2])
        r = rhythm_class(g, np.full(g.size, BEAT))
        self.assertEqual([RHYTHM_NAMES[x] for x in r[:4]], ["1/4", "1/3", "1/2", "1/12"])
        self.assertEqual(r[4], R_REST)
        self.assertEqual(RHYTHM_NAMES[r[5]], "off")  # 1/5 is not a mapping divisor of the list

    def test_mirror_is_an_involution(self):
        cs = chord_seq(notes_from([0, 1, (2, 3), 3, (0, 2)]), TP)
        tok = tokenize(cs)
        for lv, (t, tm) in tok.items():
            if tm is None:
                continue
            np.testing.assert_array_equal(mirror_tokens(t, lv), tm)
            np.testing.assert_array_equal(mirror_tokens(tm, lv), t)

    def test_flow_sees_growing_holds_and_slowing_gaps(self):
        # an LN stair whose gaps and holds grow by 30 % every chord (the song slows down)
        notes, t, gap = [], 0.0, 200.0
        for i in range(8):
            notes.append(Note(int(t), i % 4, int(t + 0.8 * gap), True))
            t += gap
            gap *= 1.3
        cs = chord_seq(notes, TP)
        flow = tokenize(cs)["flow"][0]
        self.assertEqual(name_motif("flow", flow[2:]), "expanding holds")


class MiningTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.motif = [0, 1, 2, 3, (0, 1), 2, 3, (2, 3), 1, 0]   # a stair that grows into chords
        self.corpus = Corpus()
        for s in range(5):  # five sets, motif in each (mirrored in two of them), random filler around it
            for d in range(2):
                filler = random_cols(rng, 30)
                motif = self.motif if s % 2 == 0 else [tuple(3 - x for x in c) if isinstance(c, tuple) else 3 - c for c in self.motif]
                cols = filler[:15] + motif + filler[15:]
                self.corpus.add(f"set{s}/diff{d}", 1000 + s, notes_from(cols), TP, 2.5 + d)
        self.lib = build_library(self.corpus, min_sets=4, max_len=16, log=lambda *_: None)

    def test_motif_found_once_for_both_orientations(self):
        cs = chord_seq(notes_from(self.motif), TP)
        lanes = tokenize(cs)["lanes"][0]
        t = self.lib.tables["lanes"]
        k = t.find(lanes.tolist())
        self.assertGreaterEqual(k, 0)
        self.assertEqual(int(t.sets[k]), 5)
        self.assertEqual(int(t.nmaps[k]), 10)
        self.assertEqual(int(t.count[k]), 10)
        mirrored = mirror_tokens(lanes, "lanes").tolist()
        self.assertEqual(t.find(mirrored), k)

    def test_random_filler_is_not_a_motif(self):
        t = self.lib.tables["lanes"]
        self.assertTrue(all(int(L) <= len(self.motif) + 1 for L in t.lengths))

    def test_save_load_and_prediction(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "m.npz"
            self.lib.save(p)
            lib = MotifLibrary.load(p)
        self.assertEqual(lib.n_sets, 5)
        self.assertEqual(len(lib.tables["lanes"]), len(self.lib.tables["lanes"]))
        # after 0 1 2 3 [12] 2 3 the library knows the next chord is [34]
        ctx = [1, 2, 4, 8, 3, 4, 8]
        cand, p = lib.next_distribution("lanes", ctx, sr=3.0)
        self.assertEqual(int(cand[np.argmax(p)]), 12)

    def test_report_runs(self):
        import mine_patterns

        text = mine_patterns.report(self.lib)
        self.assertIn("Level `lanes`", text)


class FormTests(unittest.TestCase):
    def test_lz_parse_finds_exact_mirror_and_variation(self):
        rng = np.random.default_rng(1)
        phrase = random_cols(rng, 16)
        other = random_cols(rng, 16)
        varied = list(phrase)
        for i in (3, 9, 12):
            varied[i] = (varied[i] + 1) % 4
        cols = phrase + other + phrase + [3 - c for c in phrase] + varied
        cs = chord_seq(notes_from(cols), TP)
        tok = tokenize(cs)
        # the variation keeps the rhythm: make the rhythm non-uniform so it is a real rhythm copy
        cp = self_repeats(cs, tok)
        kinds = [c.kind for c in cp]
        self.assertIn("exact", kinds)
        self.assertIn("mirror", kinds)
        ex = [c for c in cp if c.kind == "exact"][0]
        self.assertEqual((ex.start, ex.src), (33, 1))  # chord 0 has the "rest" rhythm, the copy starts with chord 1
        self.assertGreaterEqual(ex.length, 15)


class ZoneTests(unittest.TestCase):
    def test_break_periods_and_rest_spans(self):
        from compare_zones import Diff, break_periods, rest_spans
        from osu_mapper.hybrid.beatmap import ManiaMap

        text = "[Events]\n//Break Periods\n2,12020,20820\n\n[TimingPoints]\n0,400,4,2,0,70,1,0\n"
        self.assertEqual(break_periods(text), [(12020, 20820)])
        m = ManiaMap(notes=notes_from([0, 1, 2, 3] * 4) + notes_from([0, 1, 2, 3] * 4, start=30000), timing_points=TP)
        spans, end = rest_spans([Diff("x", text, m, 3.0, [], break_periods(text))])
        self.assertIn((12020, 20820), spans)
        self.assertTrue(any(a < 15000 < b for a, b in spans))  # the silent gap between the two parts
        self.assertGreater(end, 30000)


if __name__ == "__main__":
    unittest.main()
