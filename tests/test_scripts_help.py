"""Every script parses its arguments and imports only what the package provides."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = sorted((ROOT / "scripts").rglob("*.py"))


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.relative_to(ROOT).as_posix())
def test_help_exits_cleanly_from_another_directory(script, tmp_path):
    env = {**os.environ, "NS3_DIR": ""}
    result = subprocess.run([sys.executable, str(script), "--help"], cwd=tmp_path,
                            capture_output=True, text=True, env=env, timeout=120)
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout
