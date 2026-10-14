#!/usr/bin/env python3
"""
Explicit parameters of the universal map algorithm: export them, look at them, render a map from them.

    # list the difficulties of an archive
    python hybrid_params.py list "sample beatmaps/894666 Zekk - Vespera.osz"

    # 1) map -> parameters (JSON file you can read and edit)
    python hybrid_params.py export "sample beatmaps/894666 Zekk - Vespera.osz" --diff Sirius -o vespera_sirius.params.json

    # 2) parameters -> map. Reads ONLY the JSON (no access to the original), runs the algorithm, writes .osu
    python hybrid_params.py render vespera_sirius.params.json -o vespera_sirius_rendered.osu

    # 3) the whole experiment for one difficulty or the whole dataset: export to a file, render from that file
    #    in a separate step, compare the rendered .osu with the original .osu object by object
    python hybrid_params.py verify "sample beatmaps/894666 Zekk - Vespera.osz" --diff Sirius
    python hybrid_params.py verify "sample beatmaps"

What is compared: lane, start ms, end ms and tap/long-note of every hit object, and all 8 fields of every
timing point. Not part of the parameters (not gameplay): hitsounds/samples, storyboard, background.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


from osu_mapper.hybrid.beatmap import ManiaMap, Note, RawTimingPoint, map_to_osu_text, parse_osu_mania
from osu_mapper.hybrid.params import LANE_CODE_NAMES, MAX_MEMBERS, ParamTrack, mask_lanes
from osu_mapper.hybrid.renderer import render_track
from osu_mapper.hybrid.solver import solve_map
from osu_mapper.hybrid.sources import LFS_HINT, is_lfs_pointer

PARAMS_FORMAT = "hybrid-params-json-v1"
CODE_BY_NAME = {name: i for i, name in enumerate(LANE_CODE_NAMES)}


# ----------------------------------------------------------------------------------------------
# Reading maps
# ----------------------------------------------------------------------------------------------

def read_osu_files(path: Path) -> List[Tuple[str, str]]:
    """(name, text) of every .osu in a .osz / folder / single .osu."""
    if is_lfs_pointer(path):
        sys.exit(f"'{path}' is a Git LFS pointer.\n{LFS_HINT}")
    out: List[Tuple[str, str]] = []
    if path.suffix.lower() == ".osz":
        with zipfile.ZipFile(path) as z:
            for n in z.namelist():
                if n.lower().endswith(".osu"):
                    out.append((n, z.read(n).decode("utf-8-sig", errors="replace")))
    elif path.suffix.lower() == ".osu":
        out.append((path.name, path.read_text(encoding="utf-8-sig", errors="replace")))
    else:
        for p in sorted(path.glob("*.osu")):
            out.append((p.name, p.read_text(encoding="utf-8-sig", errors="replace")))
    return out


def pick_map(path: Path, diff: Optional[str]) -> Tuple[str, ManiaMap]:
    cands = [(n, parse_osu_mania(t)) for n, t in read_osu_files(path)]
    cands = [(n, m) for n, m in cands if m is not None]
    if not cands:
        sys.exit(f"no osu!mania 4K difficulty in '{path}'")
    if diff:
        sel = [(n, m) for n, m in cands if diff.lower() in (m.metadata.get("Version", "") + " " + n).lower()]
        if not sel:
            sys.exit(f"no difficulty matching '{diff}'. Available: " + ", ".join(m.metadata.get("Version", n) for n, m in cands))
        cands = sel
    if len(cands) > 1:
        sys.exit("several 4K difficulties, choose one with --diff: " + ", ".join(m.metadata.get("Version", n) for n, m in cands))
    return cands[0]


# ----------------------------------------------------------------------------------------------
# Parameters <-> JSON
# ----------------------------------------------------------------------------------------------

def track_to_json(track: ParamTrack, m: ManiaMap, source: str) -> Dict[str, Any]:
    events = []
    for f in track.event_frames():
        k = len(mask_lanes(int(track.event_mask[f])))
        events.append({
            "frame": int(f),
            "offset_ms": int(track.time_class[f]) - 1,
            "lane_code": LANE_CODE_NAMES[int(track.lane_code[f])],
            "stagger_ms": [int(x) for x in track.stagger[f, :k]],
            "ln_ms": [int(track.ln_len[f, i]) if track.ln_flag[f, i] else 0 for i in range(k)],
        })
    n_notes = sum(len(e["ln_ms"]) for e in events)
    return {
        "format": PARAMS_FORMAT,
        "source": source,
        "frame_ms": track.frame_ms,
        "num_frames": track.num_frames,
        "metadata": dict(m.metadata),
        "overall_difficulty": m.overall_difficulty,
        "hp_drain": m.hp_drain,
        "timing_points": [list(tp.as_tuple()) for tp in track.timing_points],
        "events": events,
        "overflow": [list(n.as_tuple()) for n in track.overflow],
        "parameter_count": {
            "frames": track.num_frames,
            "events": len(events),
            "notes": n_notes + len(track.overflow),
            "event_parameters (offset + lane_code per event)": 2 * len(events),
            "note_parameters (stagger + ln length per note)": 2 * n_notes,
            "timing_point_values": 8 * len(track.timing_points),
            "total_informative": 2 * len(events) + 2 * n_notes + 8 * len(track.timing_points) + 4 * len(track.overflow),
            "dense_tensor_values (what the network predicts: 14 per frame)": 14 * track.num_frames,
        },
        "legend": {
            "event": "a note group starting in 10 ms frame `frame`; start time = frame*10 + offset_ms (+ stagger_ms per note)",
            "lane_code": "SAME=jack, ALT=as two events ago, SHIFT_R/L, ROT_R/L, SHIFT_R2/L2, MIRROR, COMPLEMENT relative to the previous chords; Mxxxx = explicit lanes (1 = pressed, left->right)",
            "stagger_ms / ln_ms": "per note of the chord, notes ordered by lane left->right; ln_ms 0 = tap, otherwise long-note length",
            "overflow": "objects the dense language cannot express, copied verbatim [start, lane, end, is_ln]",
        },
    }


def json_to_map(data: Dict[str, Any]) -> ManiaMap:
    """The algorithm side: builds the parameter track from the JSON alone and renders it."""
    if data.get("format") != PARAMS_FORMAT:
        raise ValueError(f"not a parameter file (format={data.get('format')})")
    track = ParamTrack.empty(int(data["num_frames"]), int(data["frame_ms"]))
    for e in data["events"]:
        f = int(e["frame"])
        track.time_class[f] = int(e["offset_ms"]) + 1
        track.lane_code[f] = CODE_BY_NAME[e["lane_code"]]
        for i, (st, ln) in enumerate(zip(e["stagger_ms"], e["ln_ms"])):
            if i >= MAX_MEMBERS:
                raise ValueError(f"frame {f}: more than {MAX_MEMBERS} notes")
            track.stagger[f, i] = int(st)
            track.ln_flag[f, i] = int(ln) > 0
            track.ln_len[f, i] = int(ln)
    track.overflow = [Note(int(a), int(b), int(c), bool(d)) for a, b, c, d in data["overflow"]]
    notes = render_track(track, check_masks=False)
    tps = [RawTimingPoint(float(t[0]), float(t[1]), int(t[2]), int(t[3]), int(t[4]), int(t[5]), bool(t[6]), int(t[7])) for t in data["timing_points"]]
    return ManiaMap(notes=notes, timing_points=tps, overall_difficulty=float(data["overall_difficulty"]),
                    hp_drain=float(data["hp_drain"]), metadata=dict(data["metadata"]))


def compare(original: ManiaMap, rendered: ManiaMap) -> Tuple[bool, str]:
    a, b = original.canonical_notes(), rendered.canonical_notes()
    ta = [tp.as_tuple() for tp in original.timing_points]
    tb = [tp.as_tuple() for tp in rendered.timing_points]
    if a == b and ta == tb:
        return True, f"IDENTICAL: {len(a)} objects, {len(ta)} timing points"
    sa, sb = set(a), set(b)
    return False, (f"DIFFERENT: {len(sa - sb)} objects missing, {len(sb - sa)} extra, "
                   f"timing points {'equal' if ta == tb else 'differ'}; first missing {sorted(sa - sb)[:3]}")


# ----------------------------------------------------------------------------------------------
# Commands
# ----------------------------------------------------------------------------------------------

def cmd_list(args: argparse.Namespace) -> None:
    for n, t in read_osu_files(Path(args.path)):
        m = parse_osu_mania(t)
        tag = f"4K, {len(m.notes)} objects" if m else "not 4K mania (not handled)"
        print(f"  [{(m.metadata.get('Version') if m else '') or '-'}]  {n}  ({tag})")


def cmd_export(args: argparse.Namespace) -> None:
    name, m = pick_map(Path(args.path), args.diff)
    data = track_to_json(solve_map(m), m, f"{args.path} :: {name}")
    out = Path(args.output or (Path(name).stem + ".params.json"))
    out.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
    pc = data["parameter_count"]
    print(f"[export] {name} -> {out}")
    for k, v in pc.items():
        print(f"   {k:<62} {v:>10,}")


def cmd_render(args: argparse.Namespace) -> None:
    data = json.loads(Path(args.params).read_text(encoding="utf-8"))
    m = json_to_map(data)
    out = Path(args.output or Path(args.params).name.replace(".params.json", "") + ".rendered.osu")
    out.write_text(map_to_osu_text(m), encoding="utf-8")
    print(f"[render] {args.params} -> {out} ({len(m.notes)} objects, {len(m.timing_points)} timing points)")


def verify_one(name: str, m: ManiaMap, source: str, work: Path) -> Tuple[bool, str, Dict[str, Any]]:
    pfile = work / "map.params.json"
    data = track_to_json(solve_map(m), m, source)
    pfile.write_text(json.dumps(data), encoding="utf-8")
    del data                                                    # nothing from step 1 survives except the file
    rendered = json_to_map(json.loads(pfile.read_text(encoding="utf-8")))
    ofile = work / "rendered.osu"
    ofile.write_text(map_to_osu_text(rendered), encoding="utf-8")
    reparsed = parse_osu_mania(ofile.read_text(encoding="utf-8"))  # compare what a player would load
    ok, msg = compare(m, reparsed)
    pc = json.loads(pfile.read_text(encoding="utf-8"))["parameter_count"]
    return ok, msg, pc


def cmd_verify(args: argparse.Namespace) -> None:
    path = Path(args.path)
    work = Path(tempfile.mkdtemp(prefix="hybrid_params_"))
    if path.is_dir() and not args.diff:
        archives = sorted(path.glob("*.osz"))
        ok_n = total = 0
        objects = params = dense = 0
        for osz in archives:
            for n, t in read_osu_files(osz):
                m = parse_osu_mania(t)
                if m is None:
                    continue
                ok, msg, pc = verify_one(n, m, f"{osz.name} :: {n}", work)
                total += 1
                ok_n += ok
                objects += pc["notes"]
                params += pc["total_informative"]
                dense += pc["dense_tensor_values (what the network predicts: 14 per frame)"]
                if not ok or args.verbose:
                    print(f"  {'OK  ' if ok else 'FAIL'} {osz.name} :: {n}: {msg}")
        print(f"\n{ok_n} / {total} difficulties re-created identically from their parameter files")
        print(f"objects {objects:,} | informative parameters {params:,} ({params / max(1, objects):.2f} per object) | dense values {dense:,}")
        sys.exit(0 if ok_n == total else 1)
    name, m = pick_map(path, args.diff)
    ok, msg, pc = verify_one(name, m, f"{path} :: {name}", work)
    print(f"[verify] {name}\n  step 1: map -> parameters -> {work / 'map.params.json'}\n"
          f"  step 2: parameters file -> algorithm -> {work / 'rendered.osu'}\n  step 3: compare with the original: {msg}")
    for k, v in pc.items():
        print(f"   {k:<62} {v:>10,}")
    sys.exit(0 if ok else 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("list", help="list difficulties of an .osz / folder / .osu")
    p.add_argument("path")
    p.set_defaults(fn=cmd_list)
    p = sub.add_parser("export", help="map -> parameters JSON")
    p.add_argument("path")
    p.add_argument("--diff", default=None, help="part of the difficulty name")
    p.add_argument("-o", "--output", default=None)
    p.set_defaults(fn=cmd_export)
    p = sub.add_parser("render", help="parameters JSON -> .osu (algorithm only)")
    p.add_argument("params")
    p.add_argument("-o", "--output", default=None)
    p.set_defaults(fn=cmd_render)
    p = sub.add_parser("verify", help="export -> render from the file -> compare (one difficulty or a whole folder)")
    p.add_argument("path")
    p.add_argument("--diff", default=None)
    p.add_argument("-v", "--verbose", action="store_true")
    p.set_defaults(fn=cmd_verify)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
