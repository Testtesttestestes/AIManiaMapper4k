"""
Unit test for dataset extractor parsing real .osz archives in 'sample beatmaps/'.
"""

import csv
import json
import os
import shutil
import tempfile
import unittest

from pathlib import Path
from osu_mapper.dataset.extractor import DatasetExtractor
from osu_mapper.dataset.source_resolver import resolve_beatmap_sources


class TestExtractor(unittest.TestCase):

    def setUp(self):
        self.sample_dir = "sample beatmaps"
        self.output_dir = tempfile.mkdtemp(prefix="osu_extractor_test_")

    def tearDown(self):
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)

    def test_source_resolver_pointer_files(self):
        """Verify that text pointer files pointing to external folders resolve correctly."""
        temp_external_dir = tempfile.mkdtemp(prefix="external_osu_")
        temp_sample_dir = tempfile.mkdtemp(prefix="sample_pointer_")

        try:
            # Create a fake .osz in external directory
            fake_osz = Path(temp_external_dir) / "test_song.osz"
            fake_osz.write_text("fake osz content")

            # Create a pointer .txt in sample dir pointing to external dir
            pointer_file = Path(temp_sample_dir) / "sources.txt"
            pointer_file.write_text(f"# External source path\n{temp_external_dir}\n")

            osz_files, folders = resolve_beatmap_sources(temp_sample_dir)
            self.assertEqual(len(osz_files), 1, "Should find the .osz via the text pointer file")
            self.assertEqual(osz_files[0].resolve(), fake_osz.resolve())

        finally:
            shutil.rmtree(temp_external_dir, ignore_errors=True)
            shutil.rmtree(temp_sample_dir, ignore_errors=True)

    def test_extract_real_sample_beatmaps(self):
        if not os.path.exists(self.sample_dir) or not list(Path(self.sample_dir).glob("*.osz")):
            self.skipTest("No .osz archives found in 'sample beatmaps/'")

        extractor = DatasetExtractor(sample_dir=self.sample_dir, output_dir=self.output_dir)
        beatmaps = extractor.extract_all()

        # Check that beatmaps were extracted
        self.assertGreater(len(beatmaps), 0, "Should extract at least one 4K mania beatmap")
        print(f"\n[Test Extractor] Extracted {len(beatmaps)} valid mania 4K difficulties from sample beatmaps.")

        for b in beatmaps:
            self.assertEqual(b.mode, 3, "Mode must be 3 (mania)")
            self.assertEqual(round(b.circle_size), 4, "CircleSize must be 4 (4K)")
            self.assertGreater(b.bpm, 0, "BPM must be positive")
            self.assertGreater(b.total_notes, 0, "Total notes must be positive")
            self.assertGreaterEqual(b.ln_ratio, 0.0, "LN ratio must be >= 0")
            self.assertLessEqual(b.ln_ratio, 1.0, "LN ratio must be <= 1")
            self.assertGreater(b.duration_sec, 0, "Duration must be positive")
            self.assertGreater(b.avg_nps, 0, "NPS must be positive")

            # Check chord counts sum
            total_chords = b.single_count + b.jump_count + b.hand_count + b.quad_count
            self.assertGreater(total_chords, 0, "Total chords must be > 0")

            # Column distribution length
            self.assertEqual(len(b.column_distribution), 4, "Column distribution must have 4 keys")

        # Save and verify manifest files
        json_path, csv_path = extractor.save_manifests(beatmaps)
        self.assertTrue(json_path.exists(), "manifest.json must exist")
        self.assertTrue(csv_path.exists(), "manifest.csv must exist")

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.assertEqual(len(data), len(beatmaps))

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            self.assertEqual(len(rows), len(beatmaps))


if __name__ == "__main__":
    unittest.main()
