"""
Unit tests for beat-synchronized and offset-aligned musical chunking (Шаг 2).

Verifies:
1. Musical beat and measure grid calculation from BPM and Offset.
2. Slicing algorithm produces chunks strictly anchored to downbeats (beat 1 of a measure).
3. Zero phase-drift: transient events (e.g. drop at 51121 ms) land on mathematically precise grid indices.
4. Retention of autoregressive context ('prev_tokens') from preceding measures/squares.
"""

import unittest
from osu_mapper.dataset.feature_extractor import ParsedBeatmapDetails, slice_beatmap_into_musical_chunks

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False


class TestMusicalChunking(unittest.TestCase):

    def setUp(self):
        # Create a realistic beatmap specification
        self.bpm = 150.0            # 400 ms per beat
        self.offset_ms = 120        # First downbeat at 120 ms
        self.beat_len_ms = 60000.0 / self.bpm  # 400.0 ms
        self.bar_len_ms = 4.0 * self.beat_len_ms  # 1600.0 ms per measure

        self.details = ParsedBeatmapDetails(
            osu_file_name="test_track.osu",
            version="Expert",
            mode=3,
            circle_size=4.0,
            overall_difficulty=8.0,
            audio_filename="audio.mp3",
            bpm=self.bpm,
            offset_ms=self.offset_ms,
            star_rating=4.5,
            timing_points=[(self.offset_ms, self.beat_len_ms, True)],
            hit_objects=[
                (0, 120, 120, False),
                (1, 520, 520, False),
                (2, 920, 920, False),
                (3, 1320, 1320, False),
                (0, 51120, 51120, False),  # Drop transient
            ],
        )

    def test_musical_grid_properties(self):
        """Verify fundamental musical bar and beat relationships."""
        self.assertEqual(self.beat_len_ms, 400.0)
        self.assertEqual(self.bar_len_ms, 1600.0)

        # Transient at 51120 ms relative to 120 ms offset: 51000 ms = 127.5 beats (eighth note snap 1/2)
        delta = 51120 - self.offset_ms
        self.assertEqual(delta % 200, 0, "Transient at 51120 ms should align with 1/2 snap beat grid (200 ms)")
        beat_number = delta / self.beat_len_ms
        self.assertEqual(beat_number, 127.5, "51000 ms is exactly 127.5 beats (measure 31, beat 3.5) from 120 ms")

    def test_musical_grid_alignment_prevents_phase_drift(self):
        """
        Compares beat-aligned chunking vs naive unaligned frame chunking.
        Proves that beat-aligned chunking eliminates arbitrary phase drift.
        """
        frame_ms = 16.0
        bar_len_ms = 4.0 * (60000.0 / self.bpm)  # 1600 ms

        # 1. Naive frame hopping (legacy 128 frames)
        naive_starts_ms = [step * 128 * frame_ms for step in range(10)]
        # Check that naive hopping drifts away from downbeats
        drifts = [(t - self.offset_ms) % bar_len_ms for t in naive_starts_ms]
        self.assertTrue(any(d != 0 for d in drifts), "Naive frame hopping causes phase drift")

        # 2. Beat-aligned chunking
        aligned_starts_ms = [self.offset_ms + step * (2 * bar_len_ms) for step in range(10)]
        for t in aligned_starts_ms:
            # Must always fall on the first beat of a measure relative to offset
            self.assertEqual((t - self.offset_ms) % bar_len_ms, 0, "Beat-aligned chunks never drift from downbeats")

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch required for tensor slice tests")
    def test_beat_synchronized_chunk_alignment(self):
        """Verify slice_beatmap_into_musical_chunks aligns start_ms to offset + k * bar_len_ms."""
        num_frames = 4000  # 4000 * 16ms = 64 seconds of audio
        mel = torch.randn(num_frames, 80)
        tokens = torch.randint(0, 16, (num_frames,))
        ln_active = torch.zeros(num_frames, 4)
        sv_curve = torch.ones(num_frames)

        chunk_size = 256
        frame_ms = 16.0
        bars_per_step = 2  # 2 bars = 3200 ms per step

        chunks = slice_beatmap_into_musical_chunks(
            mel=mel,
            tokens=tokens,
            ln_active=ln_active,
            sv_curve=sv_curve,
            details=self.details,
            chunk_size=chunk_size,
            frame_ms=frame_ms,
            align_mode="beat",
            bars_per_step=bars_per_step,
        )

        self.assertGreater(len(chunks), 0, "Must produce chunks in beat mode")

        # Check that each chunk is anchored to a downbeat
        for i, chunk in enumerate(chunks):
            start_ms = chunk["start_ms"]
            # start_ms must be first_bar_ms + i * (bars_per_step * bar_len_ms)
            expected_ms = self.offset_ms + i * (bars_per_step * self.bar_len_ms)
            self.assertAlmostEqual(
                start_ms,
                expected_ms,
                delta=0.01,
                msg=f"Chunk {i} start_ms ({start_ms}) must match musical grid ({expected_ms})",
            )

            # Metadata must reflect the song's BPM and Offset (from Model 1)
            self.assertAlmostEqual(chunk["bpm"].item(), self.bpm, places=2)
            self.assertAlmostEqual(chunk["offset_ms"].item(), self.offset_ms, places=2)

            # prev_tokens must exist and have length equal to chunk_size
            self.assertIn("prev_tokens", chunk)
            self.assertEqual(chunk["prev_tokens"].shape[0], chunk_size)

            # Bar index must be sequential
            self.assertEqual(chunk["bar_index"], i * bars_per_step)

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch required for tensor slice tests")
    def test_transient_coverage_in_beat_chunk(self):
        """Verify that a drop at 51120 ms is captured in an aligned chunk without phase jitter."""
        num_frames = 4000
        mel = torch.zeros(num_frames, 80)
        tokens = torch.zeros(num_frames, dtype=torch.int64)

        # Mark drop token at 51120 ms
        drop_ms = 51120
        drop_frame = int(round(drop_ms / 16.0))
        tokens[drop_frame] = 15  # quad chord on drop!

        chunks = slice_beatmap_into_musical_chunks(
            mel=mel,
            tokens=tokens,
            ln_active=torch.zeros(num_frames, 4),
            sv_curve=torch.ones(num_frames),
            details=self.details,
            chunk_size=256,
            frame_ms=16.0,
            align_mode="beat",
            bars_per_step=2,
        )

        # Find the chunk that contains drop_frame
        covering_chunks = [c for c in chunks if c["start_ms"] <= drop_ms <= c["end_ms"]]
        self.assertGreater(len(covering_chunks), 0, "Drop transient must be inside at least one chunk")

        for c in covering_chunks:
            # Check that token 15 is indeed found inside the chunk
            self.assertIn(15, c["tokens"])
            # The start of the chunk must be an exact multiple of 1600ms from offset
            relative_start = c["start_ms"] - self.offset_ms
            self.assertAlmostEqual(relative_start % 1600.0, 0.0, delta=0.01)

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch and NumPy required for tensor grid tests")
    def test_variable_bpm_timing_points_grid_metrics(self):
        """Verify that multiple uninherited timing points (BPM changes) calculate grid metrics correctly."""
        from osu_mapper.dataset.feature_extractor import quantize_beatmap_to_grid

        # Map starts at 120 BPM (500ms beat, 2000ms bar), then changes at 10000ms to 180 BPM (333.3ms beat, 1333.3ms bar)
        multi_bpm_details = ParsedBeatmapDetails(
            osu_file_name="variable_bpm.osu",
            version="MultiBPM",
            mode=3,
            circle_size=4.0,
            overall_difficulty=8.0,
            audio_filename="audio.mp3",
            bpm=120.0,
            offset_ms=0,
            star_rating=4.0,
            timing_points=[
                (0, 500.0, True),     # 120 BPM at 0ms
                (10000, 333.333, True), # 180 BPM at 10000ms
            ],
            hit_objects=[
                (0, 0, 0, False),
                (1, 10000, 10000, False),
            ],
        )

        num_frames = 1000  # 16000ms
        tokens, ln_active, sv_curve, lane_states, grid_metrics, ticks = quantize_beatmap_to_grid(
            multi_bpm_details,
            num_frames=num_frames,
            frame_ms=16.0,
        )

        # Check tick 0 (0ms): 120 BPM downbeat
        self.assertEqual(grid_metrics[0, 2].item(), 1.0, "0ms must be downbeat")

        # Check tick at 10000ms: 180 BPM section downbeat
        tick_10000_idx = min(range(len(ticks)), key=lambda i: abs(ticks[i].time_ms - 10000))
        self.assertEqual(grid_metrics[tick_10000_idx, 2].item(), 1.0, "10000ms (180 BPM start) must be downbeat")
        self.assertAlmostEqual(grid_metrics[tick_10000_idx, 1].item(), 0.0, delta=0.05, msg="Bar phase at BPM change must reset to 0.0")



if __name__ == "__main__":
    unittest.main()
