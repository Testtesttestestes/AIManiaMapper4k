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
            full = m(b)["logits"][0]
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
        loss, logs = v5_losses(m(b), b)
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
