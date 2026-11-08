"""Dataset cleanup (osu_mapper/dataset/cleaner.py): only 4K mania difficulties and the audio they name stay."""

import tempfile
import unittest
import zipfile
from pathlib import Path

from osu_mapper.dataset.cleaner import apply_plan, plan_dataset


def osu_text(keys: int = 4, mode: int = 3, audio: str = "audio.mp3", shift: int = 0, n: int = 40) -> str:
    cols = [int((c + 0.5) * 512 / keys) for c in range(keys)]
    objs = "\n".join(f"{cols[i % keys]},192,{1000 + 250 * i + shift},1,0,0:0:0:0:" for i in range(n))
    return (f"osu file format v14\n\n[General]\nAudioFilename: {audio}\nMode: {mode}\n\n[Metadata]\nTitle:t\nArtist:a\n"
            f"Version:v{keys}{shift}\n\n[Difficulty]\nCircleSize:{keys}\nOverallDifficulty:8\n\n[Events]\n0,0,\"bg.jpg\",0,0\n"
            f"Video,0,\"video.mp4\"\n\n[TimingPoints]\n1000,500,4,2,0,70,1,0\n\n[HitObjects]\n{objs}\n")


def make_osz(path: Path, files: dict) -> None:
    with zipfile.ZipFile(path, "w") as z:
        for name, data in files.items():
            z.writestr(name, data)


class CleanerTests(unittest.TestCase):
    def test_keeps_only_training_files_and_drops_clones(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            song = b"\xff\xfb" + b"0" * 300_000
            make_osz(root / "100 Artist - Song.osz", {
                "a [easy].osu": osu_text(shift=0), "a [hard].osu": osu_text(shift=7),
                "a [7k].osu": osu_text(keys=7), "a [std].osu": osu_text(mode=0),
                "audio.mp3": song, "bg.jpg": b"j" * 5000, "video.mp4": b"v" * 50_000,
                "normal-hitclap.wav": b"w" * 3000, "Artist - Song.osb": "[Events]\n", "skin.ini": "x"})
            make_osz(root / "Artist - Song (reupload).osz", {"x.osu": osu_text(shift=0), "audio.mp3": song})
            make_osz(root / "200 Other - Only7k.osz", {"b.osu": osu_text(keys=7), "audio.mp3": song})
            rep = plan_dataset(root, log=lambda *_: None)
            plans = {p.path.name: p for p in rep.plans}
            main = plans["100 Artist - Song.osz"]
            self.assertEqual(sorted(main.keep_files), ["a [easy].osu", "a [hard].osu", "audio.mp3"])
            kinds = {k for _, _, k in main.drop_files}
            self.assertTrue({"video", "image", "hitsound/sample", "storyboard", "osu", "other"} <= kinds)
            self.assertTrue(plans["Artist - Song (reupload).osz"].delete_whole)          # clone of the set-ID archive
            self.assertIn("clone", plans["Artist - Song (reupload).osz"].diffs[0].reason)
            self.assertTrue(plans["200 Other - Only7k.osz"].delete_whole)               # no 4K difficulty at all
            # dry run changed nothing
            self.assertEqual(len(list(root.glob("*.osz"))), 3)
            apply_plan(rep, log=lambda *_: None)
            self.assertEqual([p.name for p in root.glob("*.osz")], ["100 Artist - Song.osz"])
            with zipfile.ZipFile(root / "100 Artist - Song.osz") as z:
                self.assertEqual(sorted(z.namelist()), ["a [easy].osu", "a [hard].osu", "audio.mp3"])
                self.assertEqual(z.read("audio.mp3"), song)
            again = plan_dataset(root, log=lambda *_: None)                                 # idempotent
            self.assertFalse(again.plans[0].drop_files)

    def test_keysounded_map_without_song_is_removed(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            make_osz(root / "1 ks.osz", {"k.osu": osu_text(audio="virtual"), "s1.wav": b"a" * 400_000})
            rep = plan_dataset(root, log=lambda *_: None)
            self.assertTrue(rep.plans[0].delete_whole)
            self.assertTrue(rep.plans[0].diffs[0].reason.startswith("audio"))


if __name__ == "__main__":
    unittest.main()
