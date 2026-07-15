"""Durable single-worker execution loop for review graph jobs."""

from __future__ import annotations

import asyncio
import logging
from contextlib import suppress
from datetime import UTC, datetime
from typing import Any

from langgraph.types import Command

from app.domain import ReviewReport, UploadedDocument
from app.errors import EvidenceFlowError
from app.observability import Tracer
from app.ports import ReviewRepository

logger = logging.getLogger(__name__)


class WorkflowRunner:
    """Claims SQLite jobs serially and advances the matching graph thread."""

    def __init__(
        self,
        repository: ReviewRepository,
        graph: Any,
        tracer: Tracer,
        *,
        log_sensitive_content: bool = False,
    ) -> None:
        self._repository = repository
        self._graph = graph
        self._tracer = tracer
        self._log_sensitive_content = log_sensitive_content
        self._wake_event = asyncio.Event()
        self._stop_event = asyncio.Event()
        self._task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        if self._task is not None:
            return
        await self._repository.recover_jobs()
        self._task = asyncio.create_task(self._run(), name="evidenceflow-worker")
        self.wake()

    async def stop(self) -> None:
        self._stop_event.set()
        self.wake()
        if self._task is not None:
            self._task.cancel()
            with suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    def wake(self) -> None:
        self._wake_event.set()

    async def drain_once(self) -> bool:
        """Execute one queued job; exposed for deterministic integration tests."""

        job = await self._repository.claim_next_job()
        if job is None:
            return False
        await self._execute(job)
        return True

    async def _run(self) -> None:
        while not self._stop_event.is_set():
            processed = await self.drain_once()
            if processed:
                continue
            self._wake_event.clear()
            with suppress(TimeoutError):
                await asyncio.wait_for(self._wake_event.wait(), timeout=1.0)

    async def _execute(self, job: dict[str, Any]) -> None:
        review_id = str(job["review_id"])
        job_id = str(job["job_id"])
        review = await self._repository.get(review_id)
        if review is None:
            await self._repository.finish_job(job_id, failed=True)
            return
        configuration = {
            "configurable": {"thread_id": str(review["thread_id"])},
            "recursion_limit": 100,
        }
        try:
            with self._tracer.span(
                "workflow.execute",
                span_type="CHAIN",
                attributes={
                    "review_id": review_id,
                    "thread_id": str(review["thread_id"]),
                    "job_kind": str(job["kind"]),
                    "document_count": len(review["documents"]),
                    "resume_count": int(review["resume_count"]),
                },
            ) as span:
                graph_input = await self._graph_input(job, review, configuration)
                output = await self._graph.ainvoke(
                    graph_input, configuration, version="v2"
                )
                snapshot = dict(output.value)
                if span is not None:
                    span.set_outputs(
                        self._execution_summary(
                            review, snapshot, interrupted=bool(output.interrupts)
                        )
                    )
            await self._persist_result(review_id, snapshot, bool(output.interrupts))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if self._log_sensitive_content:
                logger.exception("Review workflow failed for %s", review_id)
            else:
                logger.error(
                    "Review workflow failed for %s (error_type=%s)",
                    review_id,
                    type(exc).__name__,
                )
            error = self._safe_error(exc)
            await self._repository.update_status(
                review_id, "failed", error=error
            )
            await self._repository.finish_job(job_id, failed=True)
            return
        await self._repository.finish_job(job_id)

    async def _graph_input(
        self,
        job: dict[str, Any],
        review: dict[str, Any],
        configuration: dict[str, Any],
    ) -> Any:
        kind = str(job["kind"])
        if kind == "resume":
            return Command(resume={"decisions": job["payload"]["decisions"]})
        if kind == "recover":
            existing = await self._graph.aget_state(configuration)
            if existing.values:
                decisions = job.get("payload", {}).get("decisions")
                pending = existing.values.get("pending_review_items")
                if (
                    existing.interrupts
                    and decisions
                    and self._decision_ids_match_pending(decisions, pending)
                ):
                    return Command(resume={"decisions": decisions})
                return None
        documents = [
            UploadedDocument.model_validate(document).model_dump(mode="json")
            for document in review["documents"]
        ]
        return {
            "review_id": review["review_id"],
            "thread_id": review["thread_id"],
            "uploaded_documents": documents,
            "review_decisions": [],
            "review_decision_audits": [],
            "resolved_review_items": [],
            "reviewed_item_ids": [],
            "pending_review_items": [],
            "status": "processing",
            "error": None,
        }

    @staticmethod
    def _decision_ids_match_pending(decisions: object, pending: object) -> bool:
        """Return whether a recovered resume still targets the checkpoint's interrupt."""

        if not isinstance(decisions, list) or not isinstance(pending, list):
            return False

        def identifiers(items: list[object]) -> list[str] | None:
            result: list[str] = []
            for item in items:
                if not isinstance(item, dict):
                    return None
                identifier = item.get("review_item_id")
                if not isinstance(identifier, str) or not identifier:
                    return None
                result.append(identifier)
            return result

        decision_ids = identifiers(decisions)
        pending_ids = identifiers(pending)
        if not decision_ids or not pending_ids:
            return False
        return (
            len(decision_ids) == len(set(decision_ids))
            and len(pending_ids) == len(set(pending_ids))
            and set(decision_ids) == set(pending_ids)
        )

    async def _persist_result(
        self, review_id: str, snapshot: dict[str, Any], interrupted: bool
    ) -> None:
        await self._repository.save_snapshot(review_id, snapshot)
        if interrupted:
            items = list(snapshot.get("pending_review_items") or [])
            if not items:
                raise RuntimeError("Interrupted graph did not expose pending review items")
            await self._repository.save_review_items(review_id, items)
            return
        report_payload = snapshot.get("report")
        if snapshot.get("status") != "completed" or not isinstance(
            report_payload, dict
        ):
            raise RuntimeError("Graph finished without a validated report")
        report = ReviewReport.model_validate(report_payload)
        await self._repository.save_report(
            review_id, report, str(snapshot.get("report_markdown") or "")
        )

    @staticmethod
    def _safe_error(error: Exception) -> dict[str, Any]:
        if isinstance(error, EvidenceFlowError):
            return {
                "code": error.code,
                "message": error.message,
                "details": error.details,
                "retryable": error.retryable,
            }
        return {
            "code": "workflow_failed",
            "message": "The review workflow could not be completed.",
            "details": {},
            "retryable": False,
        }

    @staticmethod
    def _execution_summary(
        review: dict[str, Any], snapshot: dict[str, Any], *, interrupted: bool
    ) -> dict[str, Any]:
        findings = list(snapshot.get("findings") or [])
        pending = list(snapshot.get("pending_review_items") or [])
        resolved = list(snapshot.get("resolved_review_items") or [])
        review_items = [*pending, *resolved]
        extractions = list(snapshot.get("extraction_results") or [])
        created_at = datetime.fromisoformat(str(review["created_at"]))
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=UTC)
        return {
            "status": snapshot.get("status", "processing"),
            "interrupted": interrupted,
            "document_count": len(review["documents"]),
            "classification_count": len(snapshot.get("classifications") or []),
            "extraction_count": len(extractions),
            "extracted_field_count": sum(
                len(item.get("fields") or []) for item in extractions
            ),
            "finding_count": len(findings),
            "missing_document_count": sum(
                item.get("type") == "missing_document" for item in findings
            ),
            "conflict_count": sum(
                item.get("type") == "field_conflict" for item in findings
            ),
            "review_item_count": len(review_items),
            "low_confidence_field_count": sum(
                item.get("type") == "low_confidence_field" for item in review_items
            ),
            "resume_count": int(review["resume_count"]),
            "total_review_latency_seconds": max(
                0.0, (datetime.now(UTC) - created_at).total_seconds()
            ),
        }
