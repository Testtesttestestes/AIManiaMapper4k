"""
Unit test for absolute audio alignment and transient synchronization.
Requirement from Section 5 of TЗ:
- Input: Synthetic .wav with a sharp click impulse at 51121 ms.
- Test verifies that the generated HitObject has a timing delta <= +-2 ms from 51121 ms.
"""

import os
import shutil
import tempfile
import unittest

from osu_mapper.pipeline.audio import analyze_audio_rhythm, create_synthetic_click_wav
from osu_mapper.pipeline.pattern_generator import generate_patterns


class TestAudioAlignment(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="osu_align_test_")
        self.wav_path = os.path.join(self.test_dir, "test_click_51121.wav")

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_transient_alignment_at_51121ms(self):
        target_timestamp_ms = 51121
        duration_sec = 55.0  # long enough to encompass 51.121s

        # Generate regular clicks + one sharp click exactly at target_timestamp_ms
        regular_clicks = [int(i * 500) for i in range(int(duration_sec * 2))]
        all_clicks = sorted(list(set(regular_clicks + [target_timestamp_ms])))

        create_synthetic_click_wav(
            output_path=self.wav_path,
            duration_sec=duration_sec,
            click_times_ms=all_clicks,
            sample_rate=44100,
        )

        # Run audio analysis
        analysis = analyze_audio_rhythm(self.wav_path)
        self.assertGreater(len(analysis.onsets_ms), 0, "Should detect transient onsets")

        # Verify onset detector found the target click within 2 ms
        closest_onset = min(analysis.onsets_ms, key=lambda o: abs(o - target_timestamp_ms))
        onset_delta = abs(closest_onset - target_timestamp_ms)
        self.assertLessEqual(
            onset_delta,
            2,
            f"Onset detection delta {onset_delta} ms exceeded +-2 ms limit (detected {closest_onset} vs target {target_timestamp_ms})",
        )

        # Generate hit objects with Star Rating = 4.0
        hit_objects = generate_patterns(
            bpm=analysis.bpm,
            offset_ms=analysis.offset_ms,
            duration_ms=analysis.duration_ms,
            star_rating=4.0,
            onsets_ms=analysis.onsets_ms,
            seed=42,
        )

        self.assertGreater(len(hit_objects), 0, "Hit objects must be generated")

        # Find closest generated hit object start time to target_timestamp_ms
        closest_obj = min(hit_objects, key=lambda ho: abs(ho.start_time - target_timestamp_ms))
        delta = abs(closest_obj.start_time - target_timestamp_ms)

        print(f"\n[Test Audio Alignment] Target: {target_timestamp_ms} ms | HitObject: {closest_obj.start_time} ms | Delta: {delta} ms")
        self.assertLessEqual(
            delta,
            2,
            f"HitObject alignment delta {delta} ms exceeded strict requirement of +-2 ms",
        )


if __name__ == "__main__":
    unittest.main()
