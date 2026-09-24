"""
Unit tests for Star Rating scaling, density, stream preservation, and pattern generation.
"""

import unittest
from osu_mapper.dataset.difficulty_calculator import (
    calculate_osu_mania_star_rating,
    OsuManiaStrainCalculator,
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
        # 1. Very sparse map (slow 1 note per sec)
        sparse_objects = [(i % 4, i * 1000, i * 1000, False) for i in range(30)]
        sr_sparse = calculate_osu_mania_star_rating(hit_objects=sparse_objects)
        self.assertLess(sr_sparse, 2.0, "Sparse map must have low SR")

        # 2. Dense fast stream map (1/4 streams at 180 BPM = 83ms intervals)
        stream_objects = [(i % 4, int(i * 83.3), int(i * 83.3), False) for i in range(200)]
        sr_stream = calculate_osu_mania_star_rating(hit_objects=stream_objects)
        self.assertGreater(sr_stream, 3.0, "Dense 180 BPM streams must have substantial SR")
        self.assertGreater(sr_stream, sr_sparse, "Streams must be harder than sparse notes")


if __name__ == "__main__":
    unittest.main()
