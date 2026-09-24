"""
Automatic vendor loader for bundled `rosu_pp_py` prebuilt wheels.
Allows instant, zero-setup binary execution across Windows, Linux, and macOS.
"""

import glob
import os
import platform
import sys
import zipfile


def ensure_rosu_pp_loaded():
    """
    Attempts to import rosu_pp_py.
    If not found in environment, checks `vendor/wheels` for the matching OS/Python wheel,
    extracts it to `vendor/installed`, and dynamically injects it into `sys.path`.
    """
    try:
        import rosu_pp_py  # type: ignore

        return rosu_pp_py
    except ImportError:
        pass

    # Determine paths
    repo_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..")
    )
    wheels_dir = os.path.join(repo_root, "vendor", "wheels")
    installed_dir = os.path.join(repo_root, "vendor", "installed")

    if installed_dir not in sys.path:
        sys.path.insert(0, installed_dir)

    try:
        import rosu_pp_py  # type: ignore

        return rosu_pp_py
    except ImportError:
        pass

    if not os.path.isdir(wheels_dir):
        return None

    # Detect current platform tags
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

        # Match Python version
        if py_ver_tag not in basename and "none" not in basename:
            continue

        # Match OS & Architecture
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
        # Relaxed matching (any wheel for this python version)
        for w in wheel_files:
            basename = os.path.basename(w).lower()
            if py_ver_tag in basename:
                candidate_wheels.append((1, w))

    if candidate_wheels:
        candidate_wheels.sort(key=lambda x: x[0], reverse=True)
        chosen_wheel = candidate_wheels[0][1]

        os.makedirs(installed_dir, exist_ok=True)
        try:
            with zipfile.ZipFile(chosen_wheel, "r") as z:
                z.extractall(installed_dir)

            if installed_dir not in sys.path:
                sys.path.insert(0, installed_dir)

            import rosu_pp_py  # type: ignore

            return rosu_pp_py
        except Exception:
            return None

    return None
