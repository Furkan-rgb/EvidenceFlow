"""Review-scoped local artifact storage with traversal-safe identifiers."""

from __future__ import annotations

import asyncio
import os
import re
from pathlib import Path

from app.errors import DocumentNotFoundError

_SAFE_ID = re.compile(r"^[a-zA-Z0-9_-]+$")


class LocalArtifactStore:
    def __init__(self, uploads_dir: Path, exports_dir: Path) -> None:
        self._uploads_dir = uploads_dir.resolve()
        self._exports_dir = exports_dir.resolve()
        self._uploads_dir.mkdir(parents=True, exist_ok=True)
        self._exports_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _validate_id(value: str) -> None:
        if not _SAFE_ID.fullmatch(value):
            raise ValueError("Artifact identifiers contain unsupported characters")

    def upload_artifact_id(self, review_id: str, document_id: str) -> str:
        self._validate_id(review_id)
        self._validate_id(document_id)
        return f"{review_id}/{document_id}.pdf"

    def _resolve(self, base: Path, artifact_id: str) -> Path:
        candidate = (base / artifact_id).resolve()
        if base not in candidate.parents:
            raise ValueError("Artifact path escapes its configured root")
        return candidate

    async def save_upload_bytes(
        self, review_id: str, document_id: str, content: bytes
    ) -> str:
        artifact_id = self.upload_artifact_id(review_id, document_id)
        target = self._resolve(self._uploads_dir, artifact_id)
        await asyncio.to_thread(self._atomic_write, target, content)
        return artifact_id

    async def read_upload(self, artifact_id: str) -> bytes:
        target = self._resolve(self._uploads_dir, artifact_id)
        if not target.is_file():
            raise DocumentNotFoundError("The requested document was not found")
        return await asyncio.to_thread(target.read_bytes)

    async def upload_path(self, artifact_id: str) -> Path:
        target = self._resolve(self._uploads_dir, artifact_id)
        if not target.is_file():
            raise DocumentNotFoundError("The requested document was not found")
        return target

    async def save_export(self, review_id: str, name: str, content: bytes) -> str:
        self._validate_id(review_id)
        if Path(name).name != name or not name:
            raise ValueError("Invalid export name")
        artifact_id = f"{review_id}/{name}"
        target = self._resolve(self._exports_dir, artifact_id)
        await asyncio.to_thread(self._atomic_write, target, content)
        return artifact_id

    @staticmethod
    def _atomic_write(target: Path, content: bytes) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(f"{target.suffix}.tmp")
        with temporary.open("wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
