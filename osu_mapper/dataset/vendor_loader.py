from __future__ import annotations

import importlib
import subprocess
import sys


def ensure_rosu_pp() -> bool:
    """Tries to import rosu_pp_py or auto-install it if missing."""
    try:
        importlib.import_module("rosu_pp_py")
        return True
    except ImportError:
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "rosu-pp-py>=0.10.0"],
                capture_output=True,
                check=True,
            )
            importlib.invalidate_caches()
            importlib.import_module("rosu_pp_py")
            return True
        except Exception:
            return False
