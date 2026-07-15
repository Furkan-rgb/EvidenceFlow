"""Small durability primitives shared by policy-index readers and writers.

The SQLite database is the policy index's canonical generation.  Its embedded
manifest moves with the vectors in one atomic ``os.replace``.  The JSON file is
only a human-readable mirror, so readers may regenerate it from the database.
The advisory lock serializes that mirror repair with the writer's short commit
section; embedding work happens outside the lock.
"""

from __future__ import annotations

import fcntl
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def policy_index_commit_lock(index_path: Path) -> Iterator[None]:
    """Serialize index publication and derived-manifest repair across processes."""

    lock_path = index_path.with_name(f".{index_path.name}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_CREAT | os.O_RDWR
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    descriptor = os.open(lock_path, flags, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def write_text_durably(path: Path, content: str) -> None:
    """Write and fsync a file that is not yet visible at its final path."""

    with path.open("w", encoding="utf-8") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def fsync_file(path: Path) -> None:
    """Flush a completed file before it becomes the canonical generation."""

    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def fsync_directory(path: Path) -> None:
    """Flush directory entries after an atomic replacement."""

    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
