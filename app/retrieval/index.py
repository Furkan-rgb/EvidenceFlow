"""Atomic construction of the local sqlite-vec policy index.

The fully written SQLite file, including its embedded manifest, is the
canonical generation and the sole commit point.  The JSON manifest is a
derived, human-readable mirror that readers can repair from the database.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path

import sqlite_vec

from app.ports import EmbeddingProvider
from app.retrieval.atomic import (
    fsync_directory,
    fsync_file,
    policy_index_commit_lock,
    write_text_durably,
)
from app.retrieval.chunking import PolicyChunk, load_policy_corpus
from app.retrieval.manifest import PolicyIndexManifest, manifest_json

logger = logging.getLogger(__name__)


class PolicyIndexBuilder:
    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        *,
        dimensions: int,
        model_digest: str | None = None,
        target_characters: int = 1000,
        max_characters: int = 1200,
        overlap_characters: int = 150,
        batch_size: int = 32,
    ) -> None:
        if dimensions < 1:
            raise ValueError("dimensions must be positive")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self._embedding_provider = embedding_provider
        self._dimensions = dimensions
        provider_digest = getattr(embedding_provider, "model_digest", None)
        self._model_digest = model_digest if model_digest is not None else provider_digest
        self._target_characters = target_characters
        self._max_characters = max_characters
        self._overlap_characters = overlap_characters
        self._batch_size = batch_size

    async def rebuild(
        self,
        *,
        policies_dir: Path,
        index_path: Path,
        manifest_path: Path,
    ) -> PolicyIndexManifest:
        corpus = load_policy_corpus(
            policies_dir,
            target_characters=self._target_characters,
            max_characters=self._max_characters,
            overlap_characters=self._overlap_characters,
        )
        vectors: list[list[float]] = []
        for start in range(0, len(corpus.chunks), self._batch_size):
            batch = corpus.chunks[start : start + self._batch_size]
            vectors.extend(
                await self._embedding_provider.embed_documents([chunk.text for chunk in batch])
            )
        self._validate_vectors(vectors, expected_count=len(corpus.chunks))

        manifest = PolicyIndexManifest(
            build_id=str(uuid.uuid4()),
            provider=self._embedding_provider.provider,
            model=self._embedding_provider.model,
            model_digest=self._model_digest,
            dimensions=self._dimensions,
            chunk_target_characters=self._target_characters,
            chunk_max_characters=self._max_characters,
            chunk_overlap_characters=self._overlap_characters,
            corpus_sha256=corpus.sha256,
            document_count=corpus.document_count,
            chunk_count=len(corpus.chunks),
            created_at=datetime.now(UTC),
        )

        index_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        index_tmp = index_path.with_name(f".{index_path.name}.{manifest.build_id}.tmp")
        manifest_tmp = manifest_path.with_name(
            f".{manifest_path.name}.{manifest.build_id}.tmp"
        )
        try:
            await asyncio.to_thread(
                _write_index,
                index_tmp,
                corpus.chunks,
                vectors,
                manifest,
                self._dimensions,
            )
            await asyncio.to_thread(fsync_file, index_tmp)
            await asyncio.to_thread(_validate_index_file, index_tmp, manifest)
            write_text_durably(manifest_tmp, manifest_json(manifest))

            # Publication is deliberately short.  Once the complete database
            # is replaced, the new generation is committed.  Publishing the
            # sidecar is best effort because it is only a derived mirror; a
            # reader will atomically regenerate it while holding the same lock.
            with policy_index_commit_lock(index_path):
                os.replace(index_tmp, index_path)
                fsync_directory(index_path.parent)
                try:
                    os.replace(manifest_tmp, manifest_path)
                    fsync_directory(manifest_path.parent)
                except OSError as exc:
                    logger.warning(
                        "Policy index committed, but its derived manifest mirror "
                        "could not be published; the next reader will repair it: %s",
                        exc,
                    )
        finally:
            with contextlib.suppress(FileNotFoundError):
                index_tmp.unlink()
            with contextlib.suppress(FileNotFoundError):
                manifest_tmp.unlink()
        return manifest

    def _validate_vectors(
        self, vectors: list[list[float]], *, expected_count: int
    ) -> None:
        if len(vectors) != expected_count:
            raise ValueError(
                f"Embedding provider returned {len(vectors)} vectors for {expected_count} chunks"
            )
        for index, vector in enumerate(vectors):
            if len(vector) != self._dimensions:
                raise ValueError(
                    f"Embedding {index} has {len(vector)} dimensions; "
                    f"expected {self._dimensions}"
                )


async def rebuild_policy_index(
    embedding_provider: EmbeddingProvider,
    *,
    dimensions: int,
    policies_dir: Path,
    index_path: Path,
    manifest_path: Path,
    model_digest: str | None = None,
) -> PolicyIndexManifest:
    return await PolicyIndexBuilder(
        embedding_provider,
        dimensions=dimensions,
        model_digest=model_digest,
    ).rebuild(
        policies_dir=policies_dir,
        index_path=index_path,
        manifest_path=manifest_path,
    )


def _write_index(
    path: Path,
    chunks: list[PolicyChunk],
    vectors: list[list[float]],
    manifest: PolicyIndexManifest,
    dimensions: int,
) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.enable_load_extension(True)
        sqlite_vec.load(connection)
        connection.enable_load_extension(False)
        connection.executescript(
            f"""
            PRAGMA journal_mode=DELETE;
            PRAGMA synchronous=FULL;
            CREATE TABLE policy_chunks (
                rowid INTEGER PRIMARY KEY,
                evidence_id TEXT NOT NULL UNIQUE,
                policy_id TEXT NOT NULL,
                title TEXT NOT NULL,
                section_id TEXT NOT NULL,
                text TEXT NOT NULL,
                source_path TEXT NOT NULL
            );
            CREATE VIRTUAL TABLE policy_vectors USING vec0(
                embedding float[{dimensions}] distance_metric=cosine
            );
            CREATE TABLE index_metadata (
                singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                build_id TEXT NOT NULL,
                manifest_json TEXT NOT NULL
            );
            """
        )
        for rowid, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True), start=1):
            connection.execute(
                """
                INSERT INTO policy_chunks(
                    rowid, evidence_id, policy_id, title, section_id, text, source_path
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    rowid,
                    chunk.evidence_id,
                    chunk.policy_id,
                    chunk.title,
                    chunk.section_id,
                    chunk.text,
                    chunk.source_path,
                ),
            )
            connection.execute(
                "INSERT INTO policy_vectors(rowid, embedding) VALUES (?, ?)",
                (rowid, sqlite_vec.serialize_float32(vector)),
            )
        connection.execute(
            "INSERT INTO index_metadata(singleton, build_id, manifest_json) VALUES (1, ?, ?)",
            (manifest.build_id, manifest_json(manifest)),
        )
        connection.commit()
    finally:
        connection.close()


def _validate_index_file(path: Path, manifest: PolicyIndexManifest) -> None:
    """Refuse to publish a partial or internally inconsistent temporary index."""

    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        connection.enable_load_extension(True)
        sqlite_vec.load(connection)
        connection.enable_load_extension(False)
        integrity = connection.execute("PRAGMA quick_check").fetchone()
        metadata = connection.execute(
            "SELECT build_id, manifest_json FROM index_metadata WHERE singleton = 1"
        ).fetchone()
        chunks = connection.execute("SELECT COUNT(*) FROM policy_chunks").fetchone()
        vectors = connection.execute("SELECT COUNT(*) FROM policy_vectors").fetchone()
    finally:
        connection.close()

    expected_json = manifest_json(manifest)
    if (
        integrity != ("ok",)
        or metadata != (manifest.build_id, expected_json)
        or chunks != (manifest.chunk_count,)
        or vectors != (manifest.chunk_count,)
    ):
        raise RuntimeError("Refusing to publish an incomplete policy index")
