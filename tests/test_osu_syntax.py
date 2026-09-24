"""
Unit test for .osu v14 file format syntax and .osz archive compliance.
Verifies structure of:
- [General] (Mode: 3)
- [Metadata]
- [Difficulty] (CircleSize: 4)
- [TimingPoints]
- [HitObjects]
"""

import math
import os
import shutil
import tempfile
import unittest
import zipfile

from osu_mapper.pipeline import generate_beatmap_pipeline
from osu_mapper.pipeline.audio import create_synthetic_click_wav


class TestOsuSyntax(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="osu_syntax_test_")
        self.wav_path = os.path.join(self.test_dir, "syntax_audio.wav")
        self.osz_path = os.path.join(self.test_dir, "test_output.osz")
        create_synthetic_click_wav(self.wav_path, duration_sec=10.0, bpm=150.0)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_osu_v14_syntax_and_sections(self):
        osu_content, osz_path, analysis = generate_beatmap_pipeline(
            wav_path=self.wav_path,
            output_osz_path=self.osz_path,
            star_rating=4.5,
            title="Syntax Verification Track",
            artist="Test Artist",
            version="AI 4K Insane",
            creator="AIMapperTester",
        )

        first_osu = osu_content[0] if isinstance(osu_content, list) else osu_content
        lines = first_osu.splitlines()

        # 1. Header check
        self.assertEqual(lines[0].strip(), "osu file format v14", "Header must be 'osu file format v14'")

        # 2. Section parsing
        sections = {}
        current_section = None
        for line in lines:
            line_str = line.strip()
            if line_str.startswith("[") and line_str.endswith("]"):
                current_section = line_str[1:-1].strip()
                sections[current_section] = []
            elif current_section:
                sections[current_section].append(line_str)

        required_sections = ["General", "Metadata", "Difficulty", "TimingPoints", "HitObjects"]
        for req in required_sections:
            self.assertIn(req, sections, f"Section [{req}] must be present in .osu file")

        # 3. [General] checks
        general_lines = sections["General"]
        gen_dict = dict(item.split(":", 1) for item in general_lines if ":" in item)
        self.assertEqual(gen_dict.get("Mode", "").strip(), "3", "Mode must be 3 (osu!mania)")
        self.assertTrue(len(gen_dict.get("AudioFilename", "").strip()) > 0, "AudioFilename must be specified")

        # 4. [Metadata] checks
        meta_lines = sections["Metadata"]
        meta_dict = dict(item.split(":", 1) for item in meta_lines if ":" in item)
        self.assertEqual(meta_dict.get("Title", "").strip(), "Syntax Verification Track")
        self.assertEqual(meta_dict.get("Artist", "").strip(), "Test Artist")
        self.assertEqual(meta_dict.get("Version", "").strip(), "AI 4K Insane")
        self.assertEqual(meta_dict.get("Creator", "").strip(), "AIMapperTester")

        # 5. [Difficulty] checks
        diff_lines = sections["Difficulty"]
        diff_dict = dict(item.split(":", 1) for item in diff_lines if ":" in item)
        self.assertEqual(diff_dict.get("CircleSize", "").strip(), "4", "CircleSize must be 4 for mania 4K")
        self.assertIn("OverallDifficulty", diff_dict)
        self.assertIn("HPDrainRate", diff_dict)

        # 6. [TimingPoints] checks
        timing_lines = [l for l in sections["TimingPoints"] if l and not l.startswith("//")]
        self.assertGreater(len(timing_lines), 0, "Must contain at least one timing point")

        has_uninherited = False
        for line in timing_lines:
            parts = line.split(",")
            self.assertGreaterEqual(len(parts), 7, f"TimingPoint line invalid: {line}")
            t_ms = float(parts[0])
            beat_len = float(parts[1])
            uninherited = int(parts[6])
            if uninherited == 1:
                has_uninherited = True
                self.assertGreater(beat_len, 0, "Uninherited timing point must have positive beat length")

        self.assertTrue(has_uninherited, "At least one uninherited (red line) timing point required")

        # 7. [HitObjects] checks
        hit_lines = [l for l in sections["HitObjects"] if l and not l.startswith("//")]
        self.assertGreater(len(hit_lines), 0, "Must contain hit objects")

        valid_4k_x = {64, 192, 320, 448}
        found_columns = set()

        for line in hit_lines:
            parts = line.split(",")
            self.assertGreaterEqual(len(parts), 5, f"HitObject line invalid: {line}")
            x = int(parts[0])
            y = int(parts[1])
            t = int(parts[2])
            obj_type = int(parts[3])

            self.assertIn(x, valid_4k_x, f"HitObject X coordinate {x} is not a valid 4K column coordinate")
            col = min(3, max(0, int(math.floor(x * 4 / 512))))
            found_columns.add(col)

            # Check LN
            if bool(obj_type & 128):
                extras = parts[5].split(":")
                end_time = int(extras[0])
                self.assertGreater(end_time, t, f"LN end time {end_time} must be > start time {t}")

        # Check all 4 columns are utilized across the map
        self.assertEqual(found_columns, {0, 1, 2, 3}, "All 4 keys (columns 0..3) should be utilized")

        # 8. [Package .osz] checks
        self.assertTrue(os.path.exists(osz_path), ".osz archive must exist")
        with zipfile.ZipFile(osz_path, "r") as z:
            names = z.namelist()
            osu_files = [n for n in names if n.endswith(".osu")]
            audio_files = [n for n in names if n.endswith(".wav")]
            self.assertEqual(len(osu_files), 1, "Archive must contain exactly one .osu file")
            self.assertEqual(len(audio_files), 1, "Archive must contain audio file")


if __name__ == "__main__":
    unittest.main()
