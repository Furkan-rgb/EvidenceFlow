"""JSON-serialisable graph state persisted by LangGraph checkpoints."""

from __future__ import annotations

from typing import Any, Literal, TypedDict

ReviewStage = Literal["classification", "field", "conflict"]


class ReviewState(TypedDict, total=False):
    review_id: str
    thread_id: str
    uploaded_documents: list[dict[str, Any]]
    processed_documents: list[dict[str, Any]]
    classifications: list[dict[str, Any]]
    extraction_results: list[dict[str, Any]]
    effective_fields: list[dict[str, Any]]
    findings: list[dict[str, Any]]
    pending_review_items: list[dict[str, Any]]
    resolved_review_items: list[dict[str, Any]]
    review_decisions: list[dict[str, Any]]
    review_decision_audits: list[dict[str, Any]]
    submitted_decisions: list[dict[str, Any]]
    reviewed_item_ids: list[str]
    unresolved_finding_ids: list[str]
    review_stage: ReviewStage
    policy_evidence: list[dict[str, Any]]
    verified_review: dict[str, Any]
    report: dict[str, Any]
    report_markdown: str
    status: Literal["processing", "needs_review", "completed", "failed"]
    summary: dict[str, int]
    error: dict[str, Any] | None
