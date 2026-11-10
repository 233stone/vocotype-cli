"""Tests for CLI startup config handling (no audio hardware needed)."""

import subprocess
import sys
from pathlib import Path

MAIN_PY = Path(__file__).resolve().parents[1] / "main.py"


def _run_main(*args):
    return subprocess.run(
        [sys.executable, str(MAIN_PY), *args],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=str(MAIN_PY.parent),
    )


def test_missing_config_file_exits_cleanly():
    """A typo'd --config path must print a one-line error, not a traceback."""
    result = _run_main("--config", "does-not-exist.json")

    assert result.returncode == 2
    assert "Traceback" not in result.stderr
    assert "does-not-exist.json" in result.stderr


def test_malformed_config_file_exits_cleanly(tmp_path):
    """A --config file with invalid JSON must print a one-line error."""
    bad = tmp_path / "bad.json"
    bad.write_text("{not valid json", encoding="utf-8")

    result = _run_main("--config", str(bad))

    assert result.returncode == 2
    assert "Traceback" not in result.stderr
