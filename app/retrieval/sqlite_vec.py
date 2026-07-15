"""Policy-specific semantic search over a local sqlite-vec index."""

from __future__ import annotations

import asyncio
import contextlib
import os
import sqlite3
import threading
import uuid
from pathlib import Path

import sqlite_vec

from app.domain import PolicyEvidence
from app.errors import EmbeddingIndexMismatchError, PolicyIndexMissingError
from app.ports import EmbeddingProvider
from app.retrieval.atomic import (
    fsync_directory,
    policy_index_commit_lock,
    write_text_durably,
)
from app.retrieval.chunking import load_policy_corpus
from app.retrieval.manifest import (
    PolicyIndexManifest,
    assert_manifest_compatible,
    load_manifest,
    manifest_json,
)


class SqliteVecPolicyRetriever:
    """Return domain evidence rather than leaking a generic vector-store API."""

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        *,
        dimensions: int,
        index_path: Path,
        manifest_path: Path,
        model_digest: str | None = None,
        policies_dir: Path | None = None,
        target_characters: int = 1000,
        max_characters: int = 1200,
        overlap_characters: int = 150,
    ) -> None:
        if not index_path.is_file():
            raise PolicyIndexMissingError(
                "The policy index is missing; run `python -m app.cli rebuild-policy-index`.",
                details={"index_path": str(index_path)},
            )
        connection: sqlite3.Connection | None = None
        try:
            # Keep this exact database generation open for the retriever's
            # lifetime.  A concurrent successful rebuild may replace the path,
            # but searches continue against the already validated old inode
            # instead of querying a new DB under an old manifest.
            with policy_index_commit_lock(index_path):
                connection = _connect(index_path)
                manifest = self._verify_database(connection, index_path)
                _ensure_manifest_mirror(manifest_path, manifest)

            provider_digest = getattr(embedding_provider, "model_digest", None)
            expected_digest = model_digest if model_digest is not None else provider_digest
            assert_manifest_compatible(
                manifest,
                provider=embedding_provider.provider,
                model=embedding_provider.model,
                dimensions=dimensions,
                model_digest=expected_digest,
            )
            if policies_dir is not None:
                corpus = load_policy_corpus(
                    policies_dir,
                    target_characters=target_characters,
                    max_characters=max_characters,
                    overlap_characters=overlap_characters,
                )
                mismatches: dict[str, dict[str, object]] = {}
                expected = {
                    "corpus_sha256": corpus.sha256,
                    "document_count": corpus.document_count,
                    "chunk_count": len(corpus.chunks),
                    "chunk_target_characters": target_characters,
                    "chunk_max_characters": max_characters,
                    "chunk_overlap_characters": overlap_characters,
                }
                for field_name, expected_value in expected.items():
                    actual_value = getattr(manifest, field_name)
                    if actual_value != expected_value:
                        mismatches[field_name] = {
                            "expected": expected_value,
                            "actual": actual_value,
                        }
                if mismatches:
                    raise EmbeddingIndexMismatchError(
                        "The policy corpus or chunk configuration changed; "
                        "rebuild the policy index.",
                        details={"mismatches": mismatches},
                    )
        except Exception:
            if connection is not None:
                connection.close()
            raise
        assert connection is not None
        self._embedding_provider = embedding_provider
        self._dimensions = dimensions
        self._connection = connection
        self._connection_lock = threading.Lock()
        self.manifest = manifest

    async def search(self, query: str, *, limit: int = 5) -> list[PolicyEvidence]:
        if not query.strip():
            return []
        if limit < 1:
            raise ValueError("limit must be at least 1")
        vector = await self._embedding_provider.embed_query(query)
        if len(vector) != self._dimensions:
            raise EmbeddingIndexMismatchError(
                "The query embedding dimensions do not match the policy index; rebuild the index.",
                details={"expected": self._dimensions, "actual": len(vector)},
            )
        return await asyncio.to_thread(self._search_sync, vector, limit)

    def _search_sync(self, vector: list[float], limit: int) -> list[PolicyEvidence]:
        with self._connection_lock:
            rows = self._connection.execute(
                """
                SELECT
                    chunks.evidence_id,
                    chunks.policy_id,
                    chunks.title,
                    chunks.section_id,
                    chunks.text,
                    chunks.source_path,
                    neighbors.distance
                FROM policy_vectors AS neighbors
                JOIN policy_chunks AS chunks ON chunks.rowid = neighbors.rowid
                WHERE neighbors.embedding MATCH ? AND k = ?
                ORDER BY neighbors.distance, chunks.evidence_id
                """,
                (sqlite_vec.serialize_float32(vector), limit),
            ).fetchall()
        return [
            PolicyEvidence(
                evidence_id=row[0],
                policy_id=row[1],
                title=row[2],
                section_id=row[3],
                text=row[4],
                source_path=row[5],
                score=max(0.0, min(1.0, 1.0 - float(row[6]))),
            )
            for row in rows
        ]

    def close(self) -> None:
        """Release the pinned, read-only database generation."""

        with self._connection_lock:
            self._connection.close()

    def __del__(self) -> None:
        connection = getattr(self, "_connection", None)
        if connection is not None:
            with contextlib.suppress(Exception):
                connection.close()

    @staticmethod
    def _verify_database(
        connection: sqlite3.Connection, index_path: Path
    ) -> PolicyIndexManifest:
        try:
            row = connection.execute(
                """
                SELECT build_id, manifest_json
                FROM index_metadata
                WHERE singleton = 1
                """
            ).fetchone()
            chunk_count = connection.execute(
                "SELECT COUNT(*) FROM policy_chunks"
            ).fetchone()
            vector_count = connection.execute(
                "SELECT COUNT(*) FROM policy_vectors"
            ).fetchone()
        except sqlite3.DatabaseError as exc:
            raise PolicyIndexMissingError(
                "The policy index is invalid; rebuild the policy index.",
                details={"index_path": str(index_path)},
            ) from exc
        if row is None:
            raise EmbeddingIndexMismatchError(
                "The policy index metadata is missing; rebuild the policy index."
            )

        try:
            embedded_manifest = PolicyIndexManifest.model_validate_json(row[1])
        except (TypeError, ValueError) as exc:
            raise EmbeddingIndexMismatchError(
                "The policy index contains invalid embedded metadata; rebuild the policy index."
            ) from exc

        if row[0] != embedded_manifest.build_id:
            raise EmbeddingIndexMismatchError(
                "The policy index metadata identifies different builds; rebuild the policy index.",
                details={
                    "mismatches": {
                        "build_id": {
                            "expected": embedded_manifest.build_id,
                            "actual": row[0],
                        }
                    }
                },
            )

        counts = {
            "policy_chunks": None if chunk_count is None else int(chunk_count[0]),
            "policy_vectors": None if vector_count is None else int(vector_count[0]),
        }
        count_mismatches = {
            table: {"expected": embedded_manifest.chunk_count, "actual": actual}
            for table, actual in counts.items()
            if actual != embedded_manifest.chunk_count
        }
        if count_mismatches:
            raise EmbeddingIndexMismatchError(
                "The policy index row counts do not match its manifest; rebuild the policy index.",
                details={"mismatches": count_mismatches},
            )
        return embedded_manifest


def _connect(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(
        f"file:{path}?mode=ro", uri=True, check_same_thread=False
    )
    try:
        connection.enable_load_extension(True)
        sqlite_vec.load(connection)
        connection.enable_load_extension(False)
        return connection
    except Exception:
        connection.close()
        raise


def _ensure_manifest_mirror(
    manifest_path: Path, canonical_manifest: PolicyIndexManifest
) -> None:
    """Atomically repair the derived JSON mirror from canonical DB metadata."""

    try:
        mirror = load_manifest(manifest_path)
    except PolicyIndexMissingError:
        mirror = None
    if mirror is not None and mirror.model_dump(mode="json") == canonical_manifest.model_dump(
        mode="json"
    ):
        return

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest_path.with_name(
        f".{manifest_path.name}.{canonical_manifest.build_id}.{uuid.uuid4().hex}.tmp"
    )
    try:
        write_text_durably(temporary, manifest_json(canonical_manifest))
        os.replace(temporary, manifest_path)
        fsync_directory(manifest_path.parent)
        repaired = load_manifest(manifest_path)
        if repaired.model_dump(mode="json") != canonical_manifest.model_dump(mode="json"):
            raise PolicyIndexMissingError(
                "The policy index manifest mirror could not be repaired.",
                details={"manifest_path": str(manifest_path)},
            )
    except OSError as exc:
        raise PolicyIndexMissingError(
            "The policy index manifest mirror could not be repaired.",
            details={"manifest_path": str(manifest_path)},
        ) from exc
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()
