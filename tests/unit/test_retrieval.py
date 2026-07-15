from __future__ import annotations

import json
import shutil
import sqlite3
from pathlib import Path

import pytest
import sqlite_vec

from app.ai.fakes import DeterministicEmbeddingProvider
from app.domain import PolicyEvidence
from app.errors import EmbeddingIndexMismatchError
from app.retrieval.chunking import load_policy_corpus, split_text
from app.retrieval.index import PolicyIndexBuilder
from app.retrieval.manifest import assert_manifest_compatible, load_manifest
from app.retrieval.sqlite_vec import SqliteVecPolicyRetriever

POLICIES_DIR = Path(__file__).parents[2] / "policies"


def test_policy_chunks_preserve_stable_metadata() -> None:
    corpus = load_policy_corpus(POLICIES_DIR)

    assert corpus.document_count == 6
    assert len(corpus.chunks) == 24
    financial = next(
        chunk
        for chunk in corpus.chunks
        if chunk.evidence_id == "EFP-FINANCIAL:2.2:chunk-0"
    )
    assert financial.policy_id == "EFP-FINANCIAL"
    assert financial.title == "Financial Document Requirements"
    assert financial.section_id == "2.2"
    assert financial.source_path == "financial-document-requirements.md"
    assert "Revenue comparison" in financial.text


def test_chunker_respects_maximum_and_overlap() -> None:
    text = " ".join(f"token-{index}" for index in range(500))

    chunks = split_text(
        text,
        target_characters=200,
        max_characters=240,
        overlap_characters=30,
    )

    assert len(chunks) > 2
    assert all(len(chunk) <= 240 for chunk in chunks)


@pytest.mark.asyncio
async def test_policy_index_and_typed_retrieval(tmp_path: Path) -> None:
    provider = DeterministicEmbeddingProvider(dimensions=16)
    index_path = tmp_path / "policy.db"
    manifest_path = tmp_path / "manifest.json"

    manifest = await PolicyIndexBuilder(provider, dimensions=16).rebuild(
        policies_dir=POLICIES_DIR,
        index_path=index_path,
        manifest_path=manifest_path,
    )
    retriever = SqliteVecPolicyRetriever(
        provider,
        dimensions=16,
        index_path=index_path,
        manifest_path=manifest_path,
    )
    results = await retriever.search("registration number conflict", limit=4)

    assert manifest.document_count == 6
    assert manifest.chunk_count == 24
    assert len(results) == 4
    assert all(isinstance(item, PolicyEvidence) for item in results)
    assert all(item.evidence_id and item.section_id and item.source_path for item in results)
    assert [item.score for item in results] == sorted(
        (item.score for item in results), reverse=True
    )


@pytest.mark.asyncio
async def test_manifest_model_mismatch_is_refused(tmp_path: Path) -> None:
    provider = DeterministicEmbeddingProvider(dimensions=8)
    index_path = tmp_path / "policy.db"
    manifest_path = tmp_path / "manifest.json"
    await PolicyIndexBuilder(provider, dimensions=8).rebuild(
        policies_dir=POLICIES_DIR,
        index_path=index_path,
        manifest_path=manifest_path,
    )
    changed_provider = DeterministicEmbeddingProvider(dimensions=8)
    changed_provider.model = "changed-model"

    with pytest.raises(EmbeddingIndexMismatchError, match="rebuild"):
        SqliteVecPolicyRetriever(
            changed_provider,
            dimensions=8,
            index_path=index_path,
            manifest_path=manifest_path,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("field_name", "changed_value"),
    [
        ("model", "tampered-model"),
        ("model_digest", "tampered-digest"),
        ("dimensions", 9),
    ],
)
async def test_sidecar_manifest_tampering_is_repaired_from_canonical_database(
    tmp_path: Path,
    field_name: str,
    changed_value: str | int,
) -> None:
    provider = DeterministicEmbeddingProvider(dimensions=8)
    index_path = tmp_path / "policy.db"
    manifest_path = tmp_path / "manifest.json"
    original = await PolicyIndexBuilder(provider, dimensions=8).rebuild(
        policies_dir=POLICIES_DIR,
        index_path=index_path,
        manifest_path=manifest_path,
    )
    sidecar = json.loads(manifest_path.read_text(encoding="utf-8"))
    sidecar[field_name] = changed_value
    manifest_path.write_text(json.dumps(sidecar), encoding="utf-8")

    assert sidecar["build_id"] == original.build_id
    retriever = SqliteVecPolicyRetriever(
        provider,
        dimensions=8,
        index_path=index_path,
        manifest_path=manifest_path,
    )

    assert retriever.manifest == original
    assert load_manifest(manifest_path) == original


@pytest.mark.asyncio
@pytest.mark.parametrize("broken_mirror", [None, "not-json"])
async def test_missing_or_invalid_manifest_mirror_is_repaired(
    tmp_path: Path, broken_mirror: str | None
) -> None:
    provider = DeterministicEmbeddingProvider(dimensions=8)
    index_path = tmp_path / "policy.db"
    manifest_path = tmp_path / "manifest.json"
    original = await PolicyIndexBuilder(provider, dimensions=8).rebuild(
        policies_dir=POLICIES_DIR,
        index_path=index_path,
        manifest_path=manifest_path,
    )
    if broken_mirror is None:
        manifest_path.unlink()
    else:
        manifest_path.write_text(broken_mirror, encoding="utf-8")

    retriever = SqliteVecPolicyRetriever(
        provider,
        dimensions=8,
        index_path=index_path,
        manifest_path=manifest_path,
    )

    assert retriever.manifest == original
    assert load_manifest(manifest_path) == original


@pytest.mark.asyncio
async def test_failure_before_canonical_replace_preserves_previous_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = DeterministicEmbeddingProvider(dimensions=8)
    policies = tmp_path / "policies"
    shutil.copytree(POLICIES_DIR, policies)
    index_path = tmp_path / "policy.db"
    manifest_path = tmp_path / "manifest.json"
    original = await PolicyIndexBuilder(provider, dimensions=8).rebuild(
        policies_dir=policies,
        index_path=index_path,
        manifest_path=manifest_path,
    )
    original_index = index_path.read_bytes()
    original_mirror = manifest_path.read_bytes()
    policy = policies / "manual-review-policy.md"
    policy.write_text(
        policy.read_text(encoding="utf-8") + "\nUncommitted policy edit.\n",
        encoding="utf-8",
    )

    real_replace = __import__("os").replace

    def fail_canonical_replace(source: Path, destination: Path) -> None:
        if Path(destination) == index_path:
            raise OSError("simulated failure before the commit point")
        real_replace(source, destination)

    with monkeypatch.context() as patch:
        patch.setattr("app.retrieval.index.os.replace", fail_canonical_replace)
        with pytest.raises(OSError, match="before the commit point"):
            await PolicyIndexBuilder(provider, dimensions=8).rebuild(
                policies_dir=policies,
                index_path=index_path,
                manifest_path=manifest_path,
            )

    assert index_path.read_bytes() == original_index
    assert manifest_path.read_bytes() == original_mirror
    retriever = SqliteVecPolicyRetriever(
        provider,
        dimensions=8,
        index_path=index_path,
        manifest_path=manifest_path,
    )
    assert retriever.manifest == original


@pytest.mark.asyncio
async def test_crash_after_canonical_replace_is_healed_from_embedded_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class SimulatedCrash(BaseException):
        pass

    provider = DeterministicEmbeddingProvider(dimensions=8)
    policies = tmp_path / "policies"
    shutil.copytree(POLICIES_DIR, policies)
    index_path = tmp_path / "policy.db"
    manifest_path = tmp_path / "manifest.json"
    original = await PolicyIndexBuilder(provider, dimensions=8).rebuild(
        policies_dir=policies,
        index_path=index_path,
        manifest_path=manifest_path,
    )
    policy = policies / "manual-review-policy.md"
    policy.write_text(
        policy.read_text(encoding="utf-8") + "\nCommitted policy edit.\n",
        encoding="utf-8",
    )
    real_replace = __import__("os").replace

    def crash_before_mirror_replace(source: Path, destination: Path) -> None:
        if Path(destination) == manifest_path:
            raise SimulatedCrash
        real_replace(source, destination)

    with monkeypatch.context() as patch:
        patch.setattr("app.retrieval.index.os.replace", crash_before_mirror_replace)
        with pytest.raises(SimulatedCrash):
            await PolicyIndexBuilder(provider, dimensions=8).rebuild(
                policies_dir=policies,
                index_path=index_path,
                manifest_path=manifest_path,
            )

    # The stale mirror still names the old generation, but a new reader uses
    # the complete embedded manifest and repairs the mirror before accepting it.
    assert load_manifest(manifest_path) == original
    retriever = SqliteVecPolicyRetriever(
        provider,
        dimensions=8,
        index_path=index_path,
        manifest_path=manifest_path,
        policies_dir=policies,
    )
    assert retriever.manifest.build_id != original.build_id
    assert load_manifest(manifest_path) == retriever.manifest


@pytest.mark.asyncio
async def test_open_retriever_keeps_its_validated_generation_during_rebuild(
    tmp_path: Path,
) -> None:
    provider = DeterministicEmbeddingProvider(dimensions=8)
    policies = tmp_path / "policies"
    shutil.copytree(POLICIES_DIR, policies)
    index_path = tmp_path / "policy.db"
    manifest_path = tmp_path / "manifest.json"
    original = await PolicyIndexBuilder(provider, dimensions=8).rebuild(
        policies_dir=policies,
        index_path=index_path,
        manifest_path=manifest_path,
    )
    old_reader = SqliteVecPolicyRetriever(
        provider,
        dimensions=8,
        index_path=index_path,
        manifest_path=manifest_path,
    )
    policy = policies / "manual-review-policy.md"
    marker = "Atomic generation marker."
    policy.write_text(
        policy.read_text(encoding="utf-8") + f"\n{marker}\n",
        encoding="utf-8",
    )
    rebuilt = await PolicyIndexBuilder(provider, dimensions=8).rebuild(
        policies_dir=policies,
        index_path=index_path,
        manifest_path=manifest_path,
    )
    new_reader = SqliteVecPolicyRetriever(
        provider,
        dimensions=8,
        index_path=index_path,
        manifest_path=manifest_path,
    )

    old_results = await old_reader.search("manual review", limit=original.chunk_count)
    new_results = await new_reader.search("manual review", limit=rebuilt.chunk_count)
    assert old_reader.manifest.build_id == original.build_id
    assert new_reader.manifest.build_id == rebuilt.build_id
    assert all(marker not in result.text for result in old_results)
    assert any(marker in result.text for result in new_results)


@pytest.mark.asyncio
async def test_invalid_embedded_manifest_is_refused_even_with_valid_sidecar(
    tmp_path: Path,
) -> None:
    provider = DeterministicEmbeddingProvider(dimensions=8)
    index_path = tmp_path / "policy.db"
    manifest_path = tmp_path / "manifest.json"
    await PolicyIndexBuilder(provider, dimensions=8).rebuild(
        policies_dir=POLICIES_DIR,
        index_path=index_path,
        manifest_path=manifest_path,
    )
    connection = sqlite3.connect(index_path)
    try:
        connection.execute("UPDATE index_metadata SET manifest_json = '{}' WHERE singleton = 1")
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(EmbeddingIndexMismatchError, match="invalid embedded metadata"):
        SqliteVecPolicyRetriever(
            provider,
            dimensions=8,
            index_path=index_path,
            manifest_path=manifest_path,
        )


@pytest.mark.asyncio
async def test_missing_policy_vector_is_refused(tmp_path: Path) -> None:
    provider = DeterministicEmbeddingProvider(dimensions=8)
    index_path = tmp_path / "policy.db"
    manifest_path = tmp_path / "manifest.json"
    await PolicyIndexBuilder(provider, dimensions=8).rebuild(
        policies_dir=POLICIES_DIR,
        index_path=index_path,
        manifest_path=manifest_path,
    )

    connection = sqlite3.connect(index_path)
    try:
        connection.enable_load_extension(True)
        sqlite_vec.load(connection)
        connection.enable_load_extension(False)
        connection.execute("DELETE FROM policy_vectors WHERE rowid = 1")
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(EmbeddingIndexMismatchError, match="row counts") as exc_info:
        SqliteVecPolicyRetriever(
            provider,
            dimensions=8,
            index_path=index_path,
            manifest_path=manifest_path,
        )

    assert exc_info.value.details["mismatches"]["policy_vectors"] == {
        "expected": 24,
        "actual": 23,
    }


@pytest.mark.asyncio
async def test_manifest_records_embedding_and_corpus_identity(tmp_path: Path) -> None:
    provider = DeterministicEmbeddingProvider(dimensions=12)
    index_path = tmp_path / "policy.db"
    manifest_path = tmp_path / "manifest.json"
    await PolicyIndexBuilder(
        provider, dimensions=12, model_digest="fake-digest"
    ).rebuild(
        policies_dir=POLICIES_DIR,
        index_path=index_path,
        manifest_path=manifest_path,
    )

    manifest = load_manifest(manifest_path)
    assert len(manifest.corpus_sha256) == 64
    assert manifest.preprocessing_profile == "markdown-sections-v1"
    assert_manifest_compatible(
        manifest,
        provider="fake",
        model="deterministic-hash-v1",
        dimensions=12,
        model_digest="fake-digest",
    )


@pytest.mark.asyncio
async def test_stale_policy_corpus_is_refused(tmp_path: Path) -> None:
    provider = DeterministicEmbeddingProvider(dimensions=8)
    policies = tmp_path / "policies"
    shutil.copytree(POLICIES_DIR, policies)
    index_path = tmp_path / "policy.db"
    manifest_path = tmp_path / "manifest.json"
    await PolicyIndexBuilder(provider, dimensions=8).rebuild(
        policies_dir=policies,
        index_path=index_path,
        manifest_path=manifest_path,
    )
    policy = policies / "manual-review-policy.md"
    policy.write_text(
        policy.read_text(encoding="utf-8") + "\nA newly edited policy line.\n",
        encoding="utf-8",
    )

    with pytest.raises(EmbeddingIndexMismatchError, match="corpus"):
        SqliteVecPolicyRetriever(
            provider,
            dimensions=8,
            index_path=index_path,
            manifest_path=manifest_path,
            policies_dir=policies,
        )
