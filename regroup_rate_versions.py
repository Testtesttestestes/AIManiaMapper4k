#!/usr/bin/env python3
"""
Groups rate versions of a song (sped up / slowed down copies with their own audio) into one piece of music, found by
the charts' CONTENT - whatever their names say - so the train / validation split never puts a song on one side and
its rate version on the other (validation would contain music the network trained on at another tempo).

    python regroup_rate_versions.py --data dataset_hybrid           # fixes index.json (backup: index.json.bak)
    python regroup_rate_versions.py --data dataset_hybrid --dry_run

prepare_hybrid_dataset.py runs it at the end; run it by hand for a dataset prepared before. Note: it changes the
validation songs of a NEW training; resuming an older checkpoint after it reshuffles its validation set.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="dataset_hybrid")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()
    from osu_mapper.hybrid.rate_groups import regroup_index

    root = Path(args.data)
    path = root / "index.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    before = len({m.get("piece") for m in data["maps"]})
    changed = regroup_index(data, root)
    after = len({m.get("piece") for m in data["maps"]})
    print(f"[regroup] {changed} maps moved into the piece of their rate version; pieces {before} -> {after}")
    if changed and not args.dry_run:
        shutil.copy2(path, path.with_suffix(".json.bak"))
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)
        print(f"[regroup] {path} updated (backup {path.with_suffix('.json.bak')})")


if __name__ == "__main__":
    main()
