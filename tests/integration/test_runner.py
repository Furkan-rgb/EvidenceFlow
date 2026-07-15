from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from langgraph.types import Command

from app.domain import ReportStatus, ReviewReport
from app.errors import ReviewNotResumableError
from app.observability import NoOpTracer
from app.persistence import SQLiteReviewRepository
from app.runner import WorkflowRunner


def document_row() -> dict[str, object]:
    return {
        "document_id": "document_1",
        "filename": "application.pdf",
        "artifact_id": "review_1/document_1.pdf",
        "sha256": "abc123",
        "size_bytes": 42,
        "content_type": "application/pdf",
    }


def report_payload() -> dict[str, Any]:
    return ReviewReport(
        company_name="Acme BV",
        status=ReportStatus.COMPLETE,
        executive_summary="The deterministic review completed successfully.",
        sections=[],
    ).model_dump(mode="json")


def completed_snapshot() -> dict[str, Any]:
    return {
        "review_id": "review_1",
        "status": "completed",
        "report": report_payload(),
        "report_markdown": "# EvidenceFlow review report\n",
    }


class FakeGraph:
    def __init__(
        self,
        *,
        output: dict[str, Any] | None = None,
        interrupts: tuple[object, ...] = (),
        checkpoint_values: dict[str, Any] | None = None,
        checkpoint_interrupts: tuple[object, ...] = (),
        error: Exception | None = None,
    ) -> None:
        self.output = output or completed_snapshot()
        self.interrupts = interrupts
        self.checkpoint_values = checkpoint_values or {}
        self.checkpoint_interrupts = checkpoint_interrupts
        self.error = error
        self.invocations: list[tuple[Any, dict[str, Any], str]] = []
        self.state_reads: list[dict[str, Any]] = []

    async def ainvoke(
        self, graph_input: Any, configuration: dict[str, Any], *, version: str
    ) -> SimpleNamespace:
        self.invocations.append((graph_input, configuration, version))
        if self.error is not None:
            raise self.error
        return SimpleNamespace(value=self.output, interrupts=self.interrupts)

    async def aget_state(self, configuration: dict[str, Any]) -> SimpleNamespace:
        self.state_reads.append(configuration)
        return SimpleNamespace(
            values=self.checkpoint_values,
            interrupts=self.checkpoint_interrupts,
        )


async def repository(tmp_path: Path) -> SQLiteReviewRepository:
    result = SQLiteReviewRepository(tmp_path / "evidenceflow.db")
    await result.migrate()
    await result.create("review_1", "thread_1", [document_row()])
    return result


async def job_rows(repository: SQLiteReviewRepository) -> list[dict[str, Any]]:
    async with repository.connect() as connection, connection.execute(
        "SELECT job_id, kind, status, attempts, payload_json "
        "FROM workflow_jobs ORDER BY created_at, job_id"
    ) as cursor:
        return [dict(row) for row in await cursor.fetchall()]


async def prepare_running_resume(
    repository: SQLiteReviewRepository,
) -> tuple[str, dict[str, Any]]:
    start = await repository.claim_next_job()
    assert start is not None
    await repository.finish_job(str(start["job_id"]))
    item = {
        "review_item_id": "item_1",
        "type": "low_confidence_field",
        "document_id": "document_1",
        "field_id": "document_1:employee_count",
    }
    await repository.save_review_items("review_1", [item])
    resume_job_id = await repository.begin_resume(
        "review_1",
        [
            {
                "review_item_id": "item_1",
                "action": "correct",
                "value": 42,
                "actor": "test_reviewer",
            }
        ],
    )
    claimed = await repository.claim_next_job()
    assert claimed is not None
    assert claimed["job_id"] == resume_job_id
    assert claimed["kind"] == "resume"
    return resume_job_id, claimed["payload"]["decisions"][0]


@pytest.mark.asyncio
async def test_initial_start_persists_interrupt_and_pending_review(
    tmp_path: Path,
) -> None:
    repo = await repository(tmp_path)
    pending_item = {
        "review_item_id": "item_1",
        "type": "low_confidence_classification",
        "document_id": "document_1",
    }
    interrupted = {
        "review_id": "review_1",
        "status": "needs_review",
        "pending_review_items": [pending_item],
        "review_stage": "classification",
    }
    graph = FakeGraph(output=interrupted, interrupts=(object(),))
    runner = WorkflowRunner(repo, graph, NoOpTracer())

    assert await runner.drain_once() is True

    graph_input, config, version = graph.invocations[0]
    assert graph_input["review_id"] == "review_1"
    assert graph_input["thread_id"] == "thread_1"
    assert graph_input["uploaded_documents"] == [document_row()]
    assert graph_input["review_decisions"] == []
    assert config["configurable"]["thread_id"] == "thread_1"
    assert version == "v2"
    persisted = await repo.get("review_1")
    assert persisted is not None
    assert persisted["status"] == "needs_review"
    assert persisted["pending_reviews"] == [pending_item]
    assert persisted["snapshot"] == interrupted
    assert (await job_rows(repo))[0]["status"] == "completed"


@pytest.mark.asyncio
async def test_completed_graph_persists_validated_report(tmp_path: Path) -> None:
    repo = await repository(tmp_path)
    graph = FakeGraph()
    runner = WorkflowRunner(repo, graph, NoOpTracer())

    assert await runner.drain_once() is True

    persisted = await repo.get("review_1")
    assert persisted is not None
    assert persisted["status"] == "completed"
    assert persisted["report_status"] == "complete"
    assert persisted["report"] == report_payload()
    assert persisted["report_markdown"] == "# EvidenceFlow review report\n"
    assert persisted["snapshot"] == completed_snapshot()
    assert (await job_rows(repo))[0]["status"] == "completed"


@pytest.mark.asyncio
@pytest.mark.parametrize("checkpoint_interrupted", [True, False])
async def test_recovery_replays_persisted_resume_decisions_only_at_interrupt(
    tmp_path: Path, checkpoint_interrupted: bool
) -> None:
    repo = await repository(tmp_path)
    resume_job_id, persisted_decision = await prepare_running_resume(repo)

    assert await repo.recover_jobs() == 1
    recovered_row = next(
        row for row in await job_rows(repo) if row["job_id"] == resume_job_id
    )
    assert recovered_row["kind"] == "recover"
    assert recovered_row["status"] == "queued"
    assert json.loads(recovered_row["payload_json"])["decisions"] == [
        persisted_decision
    ]

    graph = FakeGraph(
        checkpoint_values={
            "review_id": "review_1",
            "status": "processing",
            "pending_review_items": [
                {
                    "review_item_id": "item_1",
                    "type": "low_confidence_field",
                }
            ],
        },
        checkpoint_interrupts=(object(),) if checkpoint_interrupted else (),
    )
    runner = WorkflowRunner(repo, graph, NoOpTracer())
    assert await runner.drain_once() is True

    graph_input, config, version = graph.invocations[0]
    if checkpoint_interrupted:
        assert isinstance(graph_input, Command)
        assert graph_input.resume == {"decisions": [persisted_decision]}
    else:
        assert graph_input is None
    assert graph.state_reads == [config]
    assert config["configurable"]["thread_id"] == "thread_1"
    assert version == "v2"
    final_row = next(row for row in await job_rows(repo) if row["job_id"] == resume_job_id)
    assert final_row["status"] == "completed"


@pytest.mark.asyncio
async def test_recovery_reemits_new_field_interrupt_after_classification_resume_crash(
    tmp_path: Path,
) -> None:
    repo = await repository(tmp_path)
    start = await repo.claim_next_job()
    assert start is not None
    await repo.finish_job(str(start["job_id"]))

    classification_item = {
        "review_item_id": "classification_item",
        "type": "low_confidence_classification",
        "document_id": "document_1",
    }
    await repo.save_review_items("review_1", [classification_item])
    resume_job_id = await repo.begin_resume(
        "review_1",
        [
            {
                "review_item_id": "classification_item",
                "action": "approve",
                "actor": "test_reviewer",
            }
        ],
    )
    running = await repo.claim_next_job()
    assert running is not None
    assert running["job_id"] == resume_job_id

    field_item = {
        "review_item_id": "field_item",
        "type": "low_confidence_field",
        "document_id": "document_1",
        "field_id": "document_1:employee_count",
    }
    assert await repo.recover_jobs() == 1
    graph = FakeGraph(
        output={
            "review_id": "review_1",
            "status": "needs_review",
            "pending_review_items": [field_item],
            "review_stage": "field",
        },
        interrupts=(object(),),
        checkpoint_values={
            "review_id": "review_1",
            "status": "needs_review",
            "pending_review_items": [field_item],
        },
        checkpoint_interrupts=(object(),),
    )
    runner = WorkflowRunner(repo, graph, NoOpTracer())

    assert await runner.drain_once() is True
    graph_input, _, _ = graph.invocations[0]
    assert graph_input is None

    persisted = await repo.get("review_1")
    assert persisted is not None
    assert persisted["status"] == "needs_review"
    assert persisted["pending_reviews"] == [field_item]
    assert persisted["resume_count"] == 1
    recovered_row = next(
        row for row in await job_rows(repo) if row["job_id"] == resume_job_id
    )
    assert recovered_row["status"] == "completed"

    field_decision = {
        "review_item_id": "field_item",
        "action": "correct",
        "value": 42,
        "actor": "test_reviewer",
    }
    assert isinstance(await repo.begin_resume("review_1", [field_decision]), str)
    with pytest.raises(ReviewNotResumableError):
        await repo.begin_resume("review_1", [field_decision])


@pytest.mark.asyncio
async def test_unexpected_workflow_failure_is_redacted_and_job_fails(
    tmp_path: Path,
) -> None:
    repo = await repository(tmp_path)
    graph = FakeGraph(error=RuntimeError("secret contents at /private/document.pdf"))
    runner = WorkflowRunner(repo, graph, NoOpTracer())

    assert await runner.drain_once() is True

    persisted = await repo.get("review_1")
    assert persisted is not None
    assert persisted["status"] == "failed"
    assert persisted["error"] == {
        "code": "workflow_failed",
        "message": "The review workflow could not be completed.",
        "details": {},
        "retryable": False,
    }
    assert "secret" not in json.dumps(persisted["error"])
    assert (await job_rows(repo))[0]["status"] == "failed"
