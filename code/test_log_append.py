"""Regression tests for append-only, cross-process transcript logging."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

from transcript_log import append_entry


def _writer_command(log_path: Path, line: str) -> list[str]:
    return [sys.executable, str(Path(__file__).resolve()), "--write", str(log_path), line]


def test_separate_processes_append_in_order(log_path: Path) -> None:
    subprocess.run(_writer_command(log_path, "first process line"), check=True)
    subprocess.run(_writer_command(log_path, "second process line"), check=True)
    assert log_path.read_text(encoding="utf-8").splitlines() == [
        "first process line",
        "second process line",
    ]


def test_concurrent_processes_preserve_complete_lines(log_path: Path) -> None:
    expected = [f"concurrent writer {number}" for number in range(20)]
    writers = [subprocess.Popen(_writer_command(log_path, line)) for line in expected]
    assert all(writer.wait() == 0 for writer in writers)

    actual = log_path.read_text(encoding="utf-8").splitlines()
    assert len(actual) == len(expected)
    assert set(actual) == set(expected)
    assert all(line.startswith("concurrent writer ") for line in actual)


def main() -> None:
    if len(sys.argv) == 4 and sys.argv[1] == "--write":
        append_entry(Path(sys.argv[2]), sys.argv[3])
        return

    # Keep subprocess test data inside the repository: the managed sandbox
    # permits every test process to access this location.
    # Use the OS temporary location.  A subprocess can access this directory
    # under both the Windows desktop sandbox and a normal user installation.
    with tempfile.TemporaryDirectory() as directory:
        temp_dir = Path(directory)
        test_separate_processes_append_in_order(temp_dir / "sequential.log")
        test_concurrent_processes_preserve_complete_lines(temp_dir / "concurrent.log")
    print("[PASS] append-only transcript logging survives sequential and concurrent writers")


if __name__ == "__main__":
    main()
