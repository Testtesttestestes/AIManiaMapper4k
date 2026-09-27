"""
Unit tests for Star Rating scaling, density, stream preservation, and pattern generation.
"""

import unittest
from osu_mapper.dataset.difficulty_calculator import (
    calculate_osu_mania_star_rating,
)
from osu_mapper.pipeline.pattern_generator import (
    ManiaHitObject,
    generate_patterns,
    get_star_rating_params,
    token_to_keys,
    compute_audio_sustain_duration,
)


class TestStarRatingAndTiming(unittest.TestCase):
    def test_star_rating_parameter_scaling(self):
        """Verify that difficulty parameters scale strictly across star ratings."""
        p_1 = get_star_rating_params(1.0)
        p_2 = get_star_rating_params(2.5)
        p_4 = get_star_rating_params(4.5)
        p_7 = get_star_rating_params(7.5)

        # 1.0* Easy map parameters
        self.assertEqual(p_1["max_chord_size"], 1, "1.0* maps must be strictly singles (no chords)")
        self.assertEqual(p_1["subdiv_1_4_prob"], 0.0, "1.0* maps must not contain 1/4 note streams")
        self.assertLessEqual(p_1["target_nps"], 1.5)

        # 2.5* Normal map parameters
        self.assertLessEqual(p_2["max_chord_size"], 2)
        self.assertGreater(p_2["target_nps"], p_1["target_nps"])

        # 4.5* Hard map parameters
        self.assertGreater(p_4["subdiv_1_4_prob"], 0.3)
        self.assertGreater(p_4["target_nps"], p_2["target_nps"])

        # 7.5* Expert map parameters
        self.assertEqual(p_7["max_chord_size"], 4)
        self.assertGreater(p_7["subdiv_1_8_prob"], 0.2)
        self.assertGreater(p_7["target_nps"], p_4["target_nps"])

    def test_algorithmic_density_difference_between_1_and_5_stars(self):
        """Verify that 1.0* generates significantly fewer notes than 5.0* on the same audio onsets."""
        bpm = 180.0
        offset_ms = 1000
        duration_ms = 30000  # 30 seconds
        fake_onsets = [t for t in range(1000, 30000, 83)]  # Dense transient onsets every 1/4 beat

        # Generate 1.0* map
        objects_1_star = generate_patterns(
            bpm=bpm,
            offset_ms=offset_ms,
            duration_ms=duration_ms,
            star_rating=1.0,
            onsets_ms=fake_onsets,
            seed=42,
        )

        # Generate 5.0* map
        objects_5_star = generate_patterns(
            bpm=bpm,
            offset_ms=offset_ms,
            duration_ms=duration_ms,
            star_rating=5.0,
            onsets_ms=fake_onsets,
            seed=42,
        )

        nps_1 = len(objects_1_star) / (duration_ms / 1000.0)
        nps_5 = len(objects_5_star) / (duration_ms / 1000.0)

        print(f"\n[Test Star Rating] 1.0* Note Count: {len(objects_1_star)} ({nps_1:.2f} NPS)")
        print(f"[Test Star Rating] 5.0* Note Count: {len(objects_5_star)} ({nps_5:.2f} NPS)")

        self.assertLess(len(objects_1_star), len(objects_5_star) * 0.45, "1.0* must have significantly fewer notes than 5.0*")
        self.assertLessEqual(nps_1, 2.0, "1.0* NPS should stay around beginner level (<= 2.0 NPS)")
        self.assertGreater(nps_5, 4.0, "5.0* NPS should provide dense rhythm")

        # Verify no chords in 1.0* map
        time_to_notes = {}
        for obj in objects_1_star:
            time_to_notes.setdefault(obj.start_time, []).append(obj)
        for t, notes in time_to_notes.items():
            self.assertEqual(len(notes), 1, f"Found chord of size {len(notes)} at {t}ms in 1.0* map!")

    def test_audio_sustain_duration_quantization(self):
        """Verify that LN hold duration is quantized to exact musical beat fractions."""
        beat_len_ms = 500.0  # 120 BPM
        sustain_1 = compute_audio_sustain_duration([], 0, beat_len_ms)
        # 120 BPM -> beat_len=500ms -> 1/2 beat is 250ms, 1/1 beat is 500ms
        self.assertIn(sustain_1, [250, 375, 500, 750, 1000])

    def test_token_to_keys_bijection(self):
        """Test token decoding for 4K columns."""
        self.assertEqual(token_to_keys(0), [])
        self.assertEqual(token_to_keys(1), [0])
        self.assertEqual(token_to_keys(2), [1])
        self.assertEqual(token_to_keys(3), [0, 1])
        self.assertEqual(token_to_keys(15), [0, 1, 2, 3])

    def test_osu_mania_strain_calculator(self):
        """Test the strain-based difficulty calculation for mania patterns."""
        from osu_mapper.dataset.difficulty_calculator import ensure_rosu_pp_installed
        if not ensure_rosu_pp_installed():
            self.skipTest("rosu-pp-py is not installed in this environment")

        # 1. Very sparse map (slow 1 note per sec)
        sparse_objects = [(i % 4, i * 1000, i * 1000, False) for i in range(30)]
        sr_sparse = calculate_osu_mania_star_rating(hit_objects=sparse_objects)
        self.assertLess(sr_sparse, 2.0, "Sparse map must have low SR")

        # 2. Dense fast stream map (1/4 streams at 180 BPM = 83ms intervals)
        stream_objects = [(i % 4, int(i * 83.3), int(i * 83.3), False) for i in range(200)]
        sr_stream = calculate_osu_mania_star_rating(hit_objects=stream_objects)
        self.assertGreater(sr_stream, 3.0, "Dense 180 BPM streams must have substantial SR")
        self.assertGreater(sr_stream, sr_sparse, "Streams must be harder than sparse notes")


    def test_multi_bpm_grid_switching_in_algorithmic_generator(self):
        """Verify that generate_patterns switches rhythm grid seamlessly when BPM changes."""
        from osu_mapper.pipeline.timing import TimingPoint

        # Section 1: 0ms to 10000ms at 120 BPM (500ms beat length)
        # Section 2: 10000ms to 20000ms at 240 BPM (250ms beat length)
        # Section 3: 20000ms to 30000ms at 180 BPM (333.33ms beat length)
        tp1 = TimingPoint(time=0, beat_length=500.0, uninherited=1)
        tp2 = TimingPoint(time=10000, beat_length=250.0, uninherited=1)
        tp3 = TimingPoint(time=20000, beat_length=333.33, uninherited=1)

        tps = [tp1, tp2, tp3]
        objects = generate_patterns(
            bpm=120.0,
            offset_ms=0,
            duration_ms=30000,
            star_rating=3.0,
            timing_points=tps,
            seed=42,
        )

        sec1_notes = [o.start_time for o in objects if o.start_time < 10000]
        sec2_notes = [o.start_time for o in objects if 10000 <= o.start_time < 20000]
        sec3_notes = [o.start_time for o in objects if 20000 <= o.start_time < 30000]

        # Check section 1 notes are aligned to 120 BPM grid (multiples of 125ms = 1/4 beat)
        for t in sec1_notes:
            self.assertEqual(t % 125, 0, f"Section 1 note at {t}ms not aligned to 120 BPM 1/4 grid!")

        # Check section 2 notes are aligned to 240 BPM grid (multiples of 62.5ms = 1/4 beat relative to 10000ms)
        for t in sec2_notes:
            rel_t = t - 10000
            rem = rel_t % 62.5
            self.assertTrue(rem < 1.0 or rem > 61.5, f"Section 2 note at {t}ms (rel {rel_t}ms) not aligned to 240 BPM grid!")

        # Check section 3 notes are aligned to 20000ms offset
        self.assertTrue(any(abs(t - 20000) <= 2 for t in sec3_notes), "Must have notes anchored at section 3 timing point 20000ms")


    def test_kiai_time_and_effects_preserved(self):
        """Verify that green lines with Kiai Time (effects=1) and other attributes are preserved."""
        from generate_beatmap import parse_template_osu_text
        from osu_mapper.pipeline.timing import TimingPoint, parse_timing_point_line

        sample_osu = """[General]
AudioFilename: audio.mp3
Mode: 3

[Metadata]
Title:Test Song
Artist:Test Artist
Version:Normal

[TimingPoints]
0,500.0,4,1,0,70,1,0
15000,-100,4,1,0,85,0,1
30000,-100,4,1,0,70,0,0
45000,-50,4,2,1,90,0,9
"""
        meta, tps = parse_template_osu_text(sample_osu)
        self.assertEqual(len(tps), 4)

        # TP 0: Red line, no kiai
        self.assertEqual(tps[0].uninherited, 1)
        self.assertEqual(tps[0].effects, 0)
        self.assertFalse(tps[0].is_kiai)

        # TP 1: Green line with Kiai enabled (effects = 1)
        self.assertEqual(tps[1].uninherited, 0)
        self.assertEqual(tps[1].effects, 1)
        self.assertTrue(tps[1].is_kiai)
        self.assertEqual(tps[1].volume, 85)
        self.assertIn("15000,-100,4,1,0,85,0,1", tps[1].to_osu_line())

        # TP 2: Green line with Kiai disabled (effects = 0)
        self.assertEqual(tps[2].uninherited, 0)
        self.assertEqual(tps[2].effects, 0)
        self.assertFalse(tps[2].is_kiai)

        # TP 3: Green line with Kiai + omit bar line (effects = 9 = 1 + 8)
        self.assertEqual(tps[3].uninherited, 0)
        self.assertEqual(tps[3].effects, 9)
        self.assertTrue(tps[3].is_kiai)
        self.assertTrue(tps[3].omit_first_bar_line)
        self.assertEqual(tps[3].volume, 90)


if __name__ == "__main__":
    unittest.main()
