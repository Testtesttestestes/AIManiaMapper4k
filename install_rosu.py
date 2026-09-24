#!/usr/bin/env python3
"""
Installer script to install the bundled rosu_pp_py wheel for the current Python environment.
"""

import glob
import os
import platform
import subprocess
import sys


def install():
    repo_root = os.path.dirname(os.path.abspath(__file__))
    wheels_dir = os.path.join(repo_root, "vendor", "wheels")

    if not os.path.isdir(wheels_dir):
        print(f"❌ Error: {wheels_dir} directory not found.")
        sys.exit(1)

    py_ver_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    is_windows = sys.platform.startswith("win") or os.name == "nt"
    is_linux = sys.platform.startswith("linux")
    is_mac = sys.platform == "darwin"
    is_64bit = sys.maxsize > 2**32
    machine = platform.machine().lower()

    wheel_files = glob.glob(os.path.join(wheels_dir, "*.whl"))
    candidate_wheels = []

    for w in wheel_files:
        basename = os.path.basename(w).lower()
        if py_ver_tag not in basename and "none" not in basename:
            continue

        if is_windows and "win_amd64" in basename and is_64bit:
            candidate_wheels.append((10, w))
        elif is_linux:
            if "aarch64" in basename and ("aarch64" in machine or "arm64" in machine):
                candidate_wheels.append((10, w))
            elif "x86_64" in basename and is_64bit and "aarch64" not in machine:
                candidate_wheels.append((10, w))
        elif is_mac:
            if "arm64" in basename and ("arm64" in machine or "aarch64" in machine):
                candidate_wheels.append((10, w))
            elif "x86_64" in basename:
                candidate_wheels.append((8, w))

    if not candidate_wheels:
        print(
            f"⚠️ No exact matching binary wheel found for {sys.platform} / Python {sys.version_info.major}.{sys.version_info.minor}."
        )
        print("Falling back to automatic vendor loader extraction...")
        from osu_mapper.dataset.vendor_loader import ensure_rosu_pp_loaded

        mod = ensure_rosu_pp_loaded()
        if mod is not None:
            print("✅ Auto-extracted vendor wheel loaded successfully!")
            return
        else:
            print("❌ Failed to load vendor wheel.")
            sys.exit(1)

    candidate_wheels.sort(key=lambda x: x[0], reverse=True)
    best_wheel = candidate_wheels[0][1]

    print(f"📦 Selected prebuilt wheel: {os.path.basename(best_wheel)}")
    print(f"Installing via pip into Python ({sys.executable})...")

    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--force-reinstall",
        "--no-index",
        "--find-links",
        wheels_dir,
        best_wheel,
    ]
    try:
        subprocess.check_call(cmd)
        print("\n✅ rosu_pp_py installed successfully!")
    except Exception as e:
        print(
            f"\n⚠️ Pip install returned error ({e}), extracting directly to vendor/installed..."
        )
        from osu_mapper.dataset.vendor_loader import ensure_rosu_pp_loaded

        mod = ensure_rosu_pp_loaded()
        if mod is not None:
            print("✅ Extracted to vendor/installed and verified working!")
        else:
            print("❌ Failed.")


if __name__ == "__main__":
    install()
