"""Tests of the v5 network: tokens, model (cache = full pass), sampling grammar, critic negatives, hard mining."""

import random
import unittest

import numpy as np
import torch

from osu_mapper.hybrid.beatmap import Note, RawTimingPoint


def random_notes(seed: int, n: int = 300):
    rng = random.Random(seed)
    out = []
    t = 200
    held = {c: -1 for c in range(4)}
    for _ in range(n):
        t += rng.choice([1, 3, 10, 62, 125, 250, 500])
        for c in rng.sample(range(4), rng.choice([1, 1, 2, 3])):
            if held[c] + 30 > t:
                continue
            if rng.random() < 0.2:
                L = rng.choice([1, 9, 30, 480, 4999, 12345])
                out.append(Note(t, c, t + L, True))
                held[c] = t + L
            else:
                out.append(Note(t, c, t, False))
                held[c] = t
    return sorted(out)


class TestTokens(unittest.TestCase):
    def test_roundtrip_and_grammar(self):
        from osu_mapper.v5.tokens import GrammarState, Vocab, decode, encode
        v = Vocab()
        for seed in range(5):
            notes = random_notes(seed)
            got = []
            end = notes[-1].start
            for s in range(0, end + 1, v.window_ms):
                toks, ev, clipped = encode(notes, s, v)
                g = GrammarState(v)
                for i, t in enumerate(toks[1:], 1):
                    self.assertTrue(any(lo <= t < hi for lo, hi in g.ranges(toks[:i])), f"token {t} at {i}")
                    g.push(t)
                got += decode(toks, s, v)
            # long notes over the cap (10 s) are clipped, the rest is exact
            exp = [n if not (n.ln and n.end - n.start >= v.len_steps * 10) else Note(n.start, n.col, n.start + v.len_steps * 10 - 1, True)
                   for n in notes]
            self.assertEqual(sorted(got), sorted(exp))

    def test_grammar_min_time(self):
        from osu_mapper.v5.tokens import SOS, GrammarState, Vocab
        v = Vocab()
        g = GrammarState(v, min_time=5000)
        (lo, hi), _eos = g.ranges([SOS])
        self.assertEqual(lo, v.time0 + 5000)


class TestModel(unittest.TestCase):
    def _cfg(self):
        from osu_mapper.v5.model import V5Config
        return V5Config(groups={"a": 6, "b": 4}, d_model=32, enc_layers=1, dec_layers=2, n_heads=4, window_frames=128,
                        window_ms=1280, max_tokens=512)

    def _batch(self, cfg, notes):
        from osu_mapper.v5.data import collate, PHASE_NONE
        from osu_mapper.v5.tokens import encode
        v = cfg.vocab()
        toks, _, _ = encode(notes, 0, v)
        toks = torch.tensor(toks)
        is_t = torch.from_numpy(v.is_time(toks.numpy()).astype(np.int64))
        t_in = torch.where(is_t > 0, toks - v.time0, torch.full_like(toks, -1))
        cur, fr = -1, []
        for x in t_in.tolist():
            cur = x if x >= 0 else cur
            fr.append(cur // 10 if cur >= 0 else -1)
        item = {"feats": torch.randn(128, 10), "beat": torch.randn(128, 13), "tokens": toks,
                "phase": torch.full_like(toks, PHASE_NONE), "tok_frame": torch.tensor(fr), "is_time": is_t,
                "cause": torch.full((128,), -1), "sr": torch.tensor([4.0]), "style": torch.zeros(11),
                "style_mask": torch.zeros(11)}
        return collate([item])

    def test_cache_equals_full_pass(self):
        from osu_mapper.v5.model import V5Net
        cfg = self._cfg()
        torch.manual_seed(0)
        m = V5Net(cfg).eval()
        notes = [n for n in random_notes(1, 80) if n.start < 1280 and not (n.ln and n.end - n.start > 9000)]
        b = self._batch(cfg, notes)
        with torch.no_grad():
            full = m.full_logprobs(m(b)["parts"])[0]
            enc = m.encode(b["feats"], b["beat"])
            cond = m.condition(b["sr"], b["style"], b["style_mask"])
            mem = m.memory_kv(enc["memory"])
            caches = [dict() for _ in m.dec]
            P = 5
            outs = [m.decode(m.embed_tokens(b["tokens"][:, :P], b["phase"][:, :P], b["tok_frame"][:, :P], enc["memory"], cond, 0),
                             mem, caches, tok_frame=b["tok_frame"][:, :P])[0]]
            for i in range(P, b["tokens"].shape[1]):
                sl = slice(i, i + 1)
                outs.append(m.decode(m.embed_tokens(b["tokens"][:, sl], b["phase"][:, sl], b["tok_frame"][:, sl], enc["memory"], cond, i),
                                     mem, caches, tok_frame=b["tok_frame"][:, sl])[0])
            inc = torch.cat(outs, 0)
        fin = torch.isfinite(full)
        self.assertTrue(torch.equal(fin, torch.isfinite(inc)))
        self.assertLess(float((full[fin] - inc[fin]).abs().max()), 1e-4)

    def test_loss_and_sampling(self):
        from osu_mapper.v5.generate import sample_window
        from osu_mapper.v5.model import V5Net, v5_losses
        from osu_mapper.v5.tokens import decode
        cfg = self._cfg()
        m = V5Net(cfg)
        notes = [n for n in random_notes(2, 60) if n.start < 1280 and not n.ln]
        b = self._batch(cfg, notes)
        m.train()
        loss, logs = v5_losses(m(b), b, window_ms=cfg.window_ms)
        loss.backward()
        self.assertTrue(np.isfinite(logs["total"]))
        m.eval()
        tps = [RawTimingPoint(0.0, 500.0, 4, 2, 1, 60, True, 0)]
        seqs = sample_window(m, b["feats"][0], b["beat"][0], 4.0, np.zeros(11), np.zeros(11), [1], 600, 0, tps, n=3,
                             cfg_scale=1.5, max_new=60)
        for s in seqs:
            got = decode(s, 0, cfg.vocab())
            self.assertTrue(all(n.start >= 600 for n in got))          # min_time respected
            ts = [n.start for n in got]
            self.assertEqual(ts, sorted(ts))


class TestCriticAndMining(unittest.TestCase):
    def test_blandify_collapses_bursts(self):
        from osu_mapper.hybrid.metrics import bursts
        from osu_mapper.v5.critic import blandify, map_frames
        tps = [RawTimingPoint(0.0, 500.0, 4, 2, 1, 60, True, 0)]
        rice = [Note(1000 + i * 125, i % 4, 1000 + i * 125, False) for i in range(16)]
        b = blandify(rice, tps)
        self.assertEqual(len(bursts(np.array([n.start for n in rice]), tps)), 1)
        self.assertEqual(len(bursts(np.array([n.start for n in b]), tps)), 0)
        self.assertTrue(all(n.start % 250 == 0 for n in b))
        x = map_frames(rice, 0, 400)
        self.assertEqual(int(x[:, :4].sum()), 16)

    def test_hard_weights(self):
        from osu_mapper.v5.data import V5Dataset
        ds = V5Dataset.__new__(V5Dataset)
        ds.base = np.array([0.25, 0.25, 0.25, 0.25])
        ds.hard_share = 0.5
        ds.weights = ds.base.copy()
        ds.set_hard_weights(np.array([1.0, 1.0, 1.0, 4.0]))
        self.assertGreater(ds.weights[3], ds.weights[0])
        self.assertAlmostEqual(float(ds.weights.sum()), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()


class TestDistancePrior(unittest.TestCase):
    def test_prior_prefers_usual_gaps(self):
        from osu_mapper.v5.model import V5Config, V5Net
        m = V5Net(V5Config(groups={"a": 4}, d_model=16, enc_layers=1, dec_layers=1, n_heads=2, window_frames=64,
                           window_ms=640, max_tokens=64))
        m.init_distance_prior([12] * 50 + [25] * 20 + [3] * 2)
        w = m.dist_bias.weight[:, 0]
        self.assertEqual(int(w[1:].argmax()) , 12)        # index = distance + 1, weights[1:] start at distance 0
        self.assertLess(float(w[0]), float(w[1:].min()))  # the past is least likely


class TestSectionsRegimesExplainMeta(unittest.TestCase):
    def _song(self):
        # 64 bars of 2 s: quiet intro, loud singing part (with kiai), loud instrumental part, quiet end
        F, slices = 12800, {"mel_mid": (0, 8), "chroma": (8, 20), "onset": (20, 30), "spectral": (30, 38), "demucs_stems": (38, 166)}
        rng = np.random.default_rng(0)
        x = rng.normal(0, 0.1, (F, 166)).astype(np.float32)
        loud = np.zeros(F)
        loud[3200:6400] = 2.0
        loud[6400:9600] = 2.0
        x[:, 30] += loud
        x[:, 0:8] += loud[:, None]
        x[3200:6400, 38 + 96:38 + 128] += 2.0          # vocals sing in the first loud part
        x[6400:9600, 38:38 + 64] += 2.0                # drums + bass in the second
        tps = [RawTimingPoint(0.0, 500.0, 4, 2, 1, 60, True, 0), RawTimingPoint(32000.0, -100.0, 4, 2, 1, 60, False, 1),
               RawTimingPoint(64000.0, -100.0, 4, 2, 1, 60, False, 0)]
        return x, slices, tps

    def test_sections_and_kiai(self):
        from osu_mapper.v5.sections import (find_sections, fit_highlight, highlight_samples, kiai_frames, kiai_timing_points,
                                            section_features, SECTION_DIM)
        x, slices, tps = self._song()
        secs = find_sections(x, slices, tps)
        self.assertGreaterEqual(len(secs), 3)
        labels = {s.label for s in secs if 3200 <= (s.start + s.end) / 2 < 6400}
        self.assertTrue(labels & {"chorus", "verse"})             # singing
        labels2 = {s.label for s in secs if 6400 <= (s.start + s.end) / 2 < 9600}
        self.assertIn("drop", labels2)                            # loud, no singing
        k = kiai_frames(tps, x.shape[0])
        self.assertTrue(k[4000] and not k[8000])
        w = fit_highlight(highlight_samples(x, slices, tps) * 3)
        secs_w = find_sections(x, slices, tps, highlight=w)
        self.assertTrue(any(s.stats["p_high"] >= 0.5 for s in secs_w if 3200 <= s.start < 6400))
        f = section_features(secs_w, x.shape[0])
        self.assertEqual(f.shape, (x.shape[0], SECTION_DIM))
        plain = [tps[0]]
        out = kiai_timing_points(secs_w, plain)
        self.assertTrue(any(tp.effects & 1 for tp in out))

    def test_regime_growth(self):
        from osu_mapper.v5.model import RegimeModule
        r = RegimeModule(8, 6, 2, 4)
        x = torch.randn(1, 40, 8)
        seg = torch.tensor([[0] * 20 + [1] * 20])
        r.eval()
        _, z = r(x, seg)
        self.assertTrue(int(z.max()) < 2)                          # only active regimes
        k = r.grow(0)
        self.assertEqual(k, 2)
        self.assertEqual(int(r.active), 3)
        self.assertTrue(torch.allclose(r.up[2], r.up[0]))
        r.train()
        y, _ = r(x, seg)
        y.sum().backward()
        self.assertIsNotNone(r.emb.grad)

    def test_explain_and_meta(self):
        from osu_mapper.v5.explain import SOURCES, explain_events
        from osu_mapper.v5.meta import BurstMeta, MetaConfig, burst_inputs, span_has_burst
        from osu_mapper.v5.model import V5Config, V5Net
        from osu_mapper.v5.tokens import encode
        cfg = V5Config(groups={"a": 6, "stem_vocals": 4, "attacks": 4}, d_model=32, enc_layers=1, dec_layers=1, n_heads=4,
                       window_frames=128, window_ms=1280, max_tokens=256, regimes=0)
        m = V5Net(cfg)
        notes = [Note(100 + 125 * i, i % 4, 100 + 125 * i, False) for i in range(8)]
        toks, _, _ = encode(notes, 0, cfg.vocab())
        rows = explain_events(m, np.random.randn(128, 14).astype(np.float32), np.random.randn(128, 13).astype(np.float32),
                              toks, 4.0, 0, events=[1, 2], steps=2)
        self.assertEqual(len(rows), 2)
        self.assertAlmostEqual(sum(rows[0][s] for s in SOURCES), 1.0, places=5)
        tps = [RawTimingPoint(0.0, 500.0, 4, 2, 1, 60, True, 0)]
        fr, sh = burst_inputs(np.random.randn(500, 10).astype(np.float32), np.zeros((500, 13), np.float32), (100, 975),
                              notes, tps, 4.0, list(range(10)))
        net = BurstMeta(MetaConfig(feat_dim=10, cols=list(range(10))))
        self.assertEqual(net(torch.from_numpy(fr.astype(np.float32))[None], torch.from_numpy(sh)[None]).shape, (1,))
        self.assertTrue(span_has_burst((0, 1000), [(400, 1200)]))
        self.assertFalse(span_has_burst((0, 1000), [(900, 1200)]))


class TestLongNotes(unittest.TestCase):
    def test_ln_counts_in_expressiveness(self):
        from osu_mapper.hybrid.metrics import expressive_report, expressive_score
        tps = [RawTimingPoint(0.0, 500.0, 4, 2, 1, 60, True, 0)]
        ln_map = [Note(i * 500, i % 4, i * 500 + 1000, True) for i in range(40)]
        taps = [Note(n.start, n.col, n.start, False) for n in ln_map]
        good = expressive_report(ln_map, ln_map, tps)
        bad = expressive_report(taps, ln_map, tps)               # same rhythm, the long notes lost
        self.assertAlmostEqual(good["ln_start_f1"], 1.0)
        self.assertEqual(bad["ln_share_pred"], 0.0)
        self.assertLess(expressive_score(bad), expressive_score(good) - 0.3)
        # a map without long notes is judged by its rhythm only
        r = expressive_report(taps, taps, tps)
        rhythm = 0.4 * r["burst_f1"] + 0.3 * (1 - r["subdiv_tv"]) + 0.3 * max(0.0, r["bar_density_corr"])
        self.assertAlmostEqual(expressive_score(r), rhythm)

    def test_critic_ln_damage_and_meta_keeps_ln_bursts(self):
        import random as _r
        from osu_mapper.v5.critic import corrupt
        from osu_mapper.v5.meta import MetaFilter, BurstMeta, MetaConfig
        notes = [Note(1000 + i * 125, i % 4, 1000 + i * 125 + 400, True) for i in range(16)]
        rng = _r.Random(0)
        kinds = set()
        for _ in range(40):
            out = corrupt(notes, rng)
            kinds.add((sum(n.ln for n in out), len(out)))
            for c in range(4):
                lane = sorted((n for n in out if n.col == c), key=lambda n: n.start)
                for a, b in zip(lane, lane[1:]):
                    self.assertGreaterEqual(b.start, (a.end if a.ln else a.start) + 30)
        self.assertGreater(len(kinds), 2)
        tps = [RawTimingPoint(0.0, 500.0, 4, 2, 1, 60, True, 0)]
        net = BurstMeta(MetaConfig(feat_dim=4, cols=[0, 1, 2, 3]))
        with torch.no_grad():
            net.head[-1].bias.fill_(-20.0)                         # rejects everything
        mf = MetaFilter(net, torch.device("cpu"), 0.35)
        out, rep = mf.apply(np.zeros((500, 4), np.float32), np.zeros((500, 13), np.float32), notes, tps, 5.0)
        self.assertEqual(rep["ln_bursts_kept"], rep["bursts"])
        self.assertEqual(sorted(out), sorted(notes))


class TestFactorizedTime(unittest.TestCase):
    def test_loss_equals_full_cross_entropy(self):
        """The factorized loss is the cross-entropy of the full distribution (which sums to 1)."""
        from osu_mapper.v5.model import V5Config, V5Net, v5_losses
        from osu_mapper.v5.tokens import encode
        from osu_mapper.v5.data import collate, PHASE_NONE
        cfg = V5Config(groups={"a": 6}, d_model=32, enc_layers=1, dec_layers=1, n_heads=4, window_frames=128,
                       window_ms=1280, max_tokens=256, regimes=0)
        m = V5Net(cfg).eval()
        v = cfg.vocab()
        notes = [Note(100 + 97 * i, i % 4, 100 + 97 * i + (300 if i % 3 == 0 else 0), i % 3 == 0) for i in range(10)]
        toks = torch.tensor(encode(notes, 0, v)[0])
        is_t = torch.from_numpy(v.is_time(toks.numpy()).astype(np.int64))
        fr, cur = [], -1
        for t in toks.tolist():
            cur = t - v.time0 if v.time0 <= t < v.mask0 else cur
            fr.append(cur // 10 if cur >= 0 else -1)
        b = collate([{"feats": torch.randn(128, 6), "beat": torch.randn(128, 13), "tokens": toks,
                      "phase": torch.full_like(toks, PHASE_NONE), "tok_frame": torch.tensor(fr), "is_time": is_t,
                      "cause": torch.full((128,), -1), "sr": torch.tensor([4.0]), "style": torch.zeros(11),
                      "style_mask": torch.zeros(11)}])
        with torch.no_grad():
            out = m(b)
            full = m.full_logprobs(out["parts"])[0, :-1]
            self.assertTrue(torch.allclose(full.exp().sum(-1), torch.ones(full.shape[0]), atol=1e-4))
            ref = -full.gather(-1, toks[1:].unsqueeze(-1)).squeeze(-1).mean()
            _, logs = v5_losses(out, b, label_smoothing=0.0, window_ms=cfg.window_ms)
        self.assertAlmostEqual(logs["tokens"], float(ref), places=4)


class TestAutocast(unittest.TestCase):
    def test_training_step_under_bf16_autocast(self):
        """The GPU trains with bf16 autocast: every module (regimes included) must accept mixed dtypes."""
        from osu_mapper.v5.model import V5Config, V5Net, v5_losses
        from osu_mapper.v5.data import collate, PHASE_NONE
        from osu_mapper.v5.sections import SECTION_DIM
        from osu_mapper.v5.tokens import encode
        cfg = V5Config(groups={"a": 6, "section": SECTION_DIM}, d_model=32, enc_layers=2, dec_layers=1, n_heads=4,
                       window_frames=128, window_ms=1280, max_tokens=256, regimes=6, regimes_active=3)
        m = V5Net(cfg).train()
        v = cfg.vocab()
        notes = [Note(100 + 97 * i, i % 4, 100 + 97 * i, False) for i in range(10)]
        toks = torch.tensor(encode(notes, 0, v)[0])
        is_t = torch.from_numpy(v.is_time(toks.numpy()).astype(np.int64))
        fr, cur = [], -1
        for t in toks.tolist():
            cur = t - v.time0 if v.time0 <= t < v.mask0 else cur
            fr.append(cur // 10 if cur >= 0 else -1)
        feats = torch.randn(128, 6 + SECTION_DIM)
        item = {"feats": feats, "beat": torch.randn(128, 13), "tokens": toks, "phase": torch.full_like(toks, PHASE_NONE),
                "tok_frame": torch.tensor(fr), "is_time": is_t, "cause": torch.full((128,), -1), "sr": torch.tensor([4.0]),
                "style": torch.zeros(11), "style_mask": torch.zeros(11)}
        b = collate([item, item])
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            out = m(b)
            loss, logs = v5_losses(out, b, window_ms=cfg.window_ms)
        loss.backward()
        self.assertTrue(np.isfinite(logs["total"]))


class TestLnShare(unittest.TestCase):
    def test_fit_predict(self):
        from osu_mapper.v5.lnstyle import fit, predict, song_descriptor
        rng = np.random.default_rng(0)
        samples = []
        for _ in range(60):
            x = rng.normal(0, 1, 15)                 # 4 flux + 4 stem flux + 4 attack rates + bpm, stars, bias
            x[-1] = 1.0
            samples.append((x, float(np.clip(0.3 + 0.1 * x[0], 0, 1))))
        w = fit(samples, l2=1e-3)
        errs = [abs(predict(w, x) - y) for x, y in samples]
        self.assertLess(float(np.mean(errs)), 0.02)
        feats = rng.normal(0, 1, (3000, 166)).astype(np.float32)
        sl = {"mel_mid": (0, 8), "pitch": (8, 20), "chroma": (20, 32), "hpss": (32, 38), "demucs_stems": (38, 166)}
        d = song_descriptor(feats, sl, [RawTimingPoint(0.0, 400.0, 4, 2, 1, 60, True, 0)], 4.5)
        self.assertEqual(d.size, 15)
        self.assertTrue(0.0 <= predict(w, d) <= 1.0)


class TestDraftPasses(unittest.TestCase):
    def test_draft_and_passes(self):
        import random as _r
        from osu_mapper.v5.data import draft_frames, make_draft
        from osu_mapper.v5.generate import generate_passes, style_of_notes
        from osu_mapper.v5.model import V5Config, V5Net
        tps = [RawTimingPoint(0.0, 500.0, 4, 2, 1, 60, True, 0)]
        notes = [Note(250 * i, i % 4, 250 * i + (600 if i % 5 == 0 else 0), i % 5 == 0) for i in range(40)]
        d = make_draft(notes, tps, _r.Random(1))
        self.assertNotEqual(sorted(d), sorted(notes))
        fr = draft_frames(d, 0, 200)
        self.assertEqual(fr.shape[1], 10)
        self.assertTrue((fr[:, -1] == 1).all())
        st, sm = style_of_notes(notes, np.zeros(11, np.float32), np.zeros(11, np.float32), 10000)
        from osu_mapper.hybrid.solver import STYLE_KEYS
        self.assertAlmostEqual(float(st[STYLE_KEYS.index("ln_ratio")]), 8 / 40, places=5)
        cfg = V5Config(groups={"a": 6}, d_model=32, enc_layers=1, dec_layers=1, n_heads=4, window_frames=128,
                       window_ms=1280, max_tokens=256, regimes=0, draft=True)
        m = V5Net(cfg).eval()
        out = generate_passes(m, np.random.randn(400, 6).astype(np.float32), np.zeros((400, 13), np.float32), tps, 4.0,
                              passes=2)
        for c in range(4):
            lane = sorted((n for n in out if n.col == c), key=lambda n: n.start)
            for a, b in zip(lane, lane[1:]):
                self.assertGreaterEqual(b.start, (a.end if a.ln else a.start) + 30)

    def test_resume_adds_draft_input(self):
        from osu_mapper.v5.model import V5Config, V5Net
        cfg = V5Config(groups={"a": 6}, d_model=32, enc_layers=1, dec_layers=1, n_heads=4, window_frames=128,
                       window_ms=1280, max_tokens=256, regimes=0)
        old = V5Net(cfg).eval()
        cfg2 = V5Config(**{**cfg.to_dict(), "draft": True})
        new = V5Net(cfg2).eval()
        missing, unexpected = new.load_state_dict(old.state_dict(), strict=False)
        self.assertEqual(unexpected, [])
        self.assertTrue(all(k.startswith("draft_in.") for k in missing))
        f, b = torch.randn(1, 128, 6), torch.randn(1, 128, 13)
        with torch.no_grad():
            a = old.encode(f, b)["memory"]
            c = new.encode(f, b, torch.randn(1, 128, 10))["memory"]
        self.assertTrue(torch.allclose(a, c, atol=1e-5))        # the zero-initialized draft input changes nothing yet
