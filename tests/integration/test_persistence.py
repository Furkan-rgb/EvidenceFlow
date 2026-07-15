from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from app.errors import ReviewNotResumableError
from app.persistence import LocalArtifactStore, SQLiteReviewRepository


def document_row(artifact_id: str = "review_1/document_1.pdf") -> dict[str, object]:
    return {
        "document_id": "document_1",
        "filename": "application.pdf",
        "artifact_id": artifact_id,
        "sha256": "abc123",
        "size_bytes": 42,
        "content_type": "application/pdf",
    }


@pytest.mark.asyncio
async def test_repository_migrates_creates_and_claims_job(tmp_path: Path) -> None:
    repository = SQLiteReviewRepository(tmp_path / "evidenceflow.db")
    await repository.migrate()
    await repository.create("review_1", "thread_1", [document_row()])

    review = await repository.get("review_1")
    assert review is not None
    assert review["status"] == "processing"
    assert review["documents"][0]["document_id"] == "document_1"

    job = await repository.claim_next_job()
    assert job is not None
    assert job["kind"] == "start"
    await repository.finish_job(str(job["job_id"]))


@pytest.mark.asyncio
async def test_resume_is_atomic_and_accepts_only_pending_items(tmp_path: Path) -> None:
    repository = SQLiteReviewRepository(tmp_path / "evidenceflow.db")
    await repository.migrate()
    await repository.create("review_1", "thread_1", [document_row()])
    first_job = await repository.claim_next_job()
    assert first_job is not None
    await repository.finish_job(str(first_job["job_id"]))
    item = {
        "review_item_id": "item_1",
        "type": "low_confidence_field",
        "field_id": "document_1:employee_count",
    }
    await repository.save_review_items("review_1", [item])
    decision = {
        "decision_id": "decision_1",
        "review_item_id": "item_1",
        "action": "approve",
        "actor": "local_reviewer",
        "decided_at": "2026-01-02T03:04:05+00:00",
    }

    results = await asyncio.gather(
        repository.begin_resume("review_1", [decision]),
        repository.begin_resume("review_1", [decision]),
        return_exceptions=True,
    )
    assert sum(isinstance(result, str) for result in results) == 1
    assert sum(isinstance(result, ReviewNotResumableError) for result in results) == 1
    async with repository.connect() as connection, connection.execute(
        "SELECT decided_at FROM review_decisions WHERE decision_id = 'decision_1'"
    ) as cursor:
        row = await cursor.fetchone()
    assert row is not None
    assert row[0] == "2026-01-02T03:04:05+00:00"


@pytest.mark.asyncio
async def test_review_item_identity_and_decisions_are_scoped_to_review(
    tmp_path: Path,
) -> None:
    repository = SQLiteReviewRepository(tmp_path / "evidenceflow.db")
    await repository.migrate()
    await repository.create("review_1", "thread_1", [document_row()])
    second_document = document_row("review_2/document_2.pdf")
    second_document["document_id"] = "document_2"
    await repository.create("review_2", "thread_2", [second_document])

    shared_item = {
        "review_item_id": "shared_item",
        "type": "low_confidence_field",
        "field_id": "employee_count",
    }
    await repository.save_review_items("review_1", [shared_item])
    await repository.save_review_items("review_2", [shared_item])

    for review_id in ("review_1", "review_2"):
        job_id = await repository.begin_resume(
            review_id,
            [
                {
                    "review_item_id": "shared_item",
                    "action": "approve",
                    "actor": "test_reviewer",
                }
            ],
        )
        assert isinstance(job_id, str)

    async with repository.connect() as connection:
        async with connection.execute(
            "SELECT review_id, state FROM review_items "
            "WHERE review_item_id = 'shared_item' ORDER BY review_id"
        ) as cursor:
            items = [tuple(row) for row in await cursor.fetchall()]
        async with connection.execute(
            "SELECT review_id, review_item_id FROM review_decisions "
            "WHERE review_item_id = 'shared_item' ORDER BY review_id"
        ) as cursor:
            decisions = [tuple(row) for row in await cursor.fetchall()]
        async with connection.execute(
            "SELECT review_id, resume_count FROM reviews ORDER BY review_id"
        ) as cursor:
            resume_counts = [tuple(row) for row in await cursor.fetchall()]

    assert items == [("review_1", "decided"), ("review_2", "decided")]
    assert decisions == [
        ("review_1", "shared_item"),
        ("review_2", "shared_item"),
    ]
    assert resume_counts == [("review_1", 1), ("review_2", 1)]


@pytest.mark.asyncio
async def test_local_artifacts_are_scoped_and_atomic(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "uploads", tmp_path / "exports")
    artifact_id = await store.save_upload_bytes(
        "review_1", "document_1", b"%PDF-1.7\nsynthetic"
    )
    assert artifact_id == "review_1/document_1.pdf"
    assert await store.read_upload(artifact_id) == b"%PDF-1.7\nsynthetic"

    with pytest.raises(ValueError, match="unsupported characters"):
        await store.save_upload_bytes("../outside", "document_1", b"bad")
