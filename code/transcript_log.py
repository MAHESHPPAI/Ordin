"""Safe append-only writing for the shared challenge transcript.

``append_entry`` serializes each write with a sidecar lock file, then opens the
transcript in append mode for the duration of that write.  It is suitable for
separate stage processes as well as concurrent retries on Windows and POSIX.
"""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


@contextmanager
def _exclusive_lock(lock_path: Path, timeout_seconds: float = 10.0) -> Iterator[None]:
    """Take an advisory lock on a stable one-byte sidecar lock file."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "a+", encoding="utf-8") as lock_file:
        lock_file.seek(0)
        if lock_file.tell() == 0 and lock_path.stat().st_size == 0:
            lock_file.write("0")
            lock_file.flush()

        deadline = time.monotonic() + timeout_seconds
        while True:
            try:
                if os.name == "nt":
                    import msvcrt

                    lock_file.seek(0)
                    msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Timed out waiting for transcript lock: {lock_path}")
                time.sleep(0.01)

        try:
            yield
        finally:
            lock_file.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def append_entry(path: Path, entry: str) -> None:
    """Atomically append one complete, newline-terminated transcript entry."""
    if not entry:
        return

    path = Path(path)
    text = entry if entry.endswith("\n") else f"{entry}\n"
    with _exclusive_lock(path.with_name(f"{path.name}.lock")):
        with open(path, "a", encoding="utf-8", newline="\n") as transcript:
            transcript.write(text)
            transcript.flush()
            os.fsync(transcript.fileno())
