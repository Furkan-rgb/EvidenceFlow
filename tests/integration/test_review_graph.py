from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import pytest
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from app.domain import (
    DocumentClassification,
    DocumentType,
    EvidenceReference,
    ExtractedField,
    ExtractionResult,
    PageContent,
    PolicyEvidence,
    ProcessedDocument,
    ProcessorMetadata,
    ReportSection,
    ReviewReport,
    UploadedDocument,
    VerifiedReview,
)
from app.graph import WorkflowDependencies, build_review_graph
from app.review import load_review_rules

Scenario = Literal["complete", "classification", "field", "conflict", "missing"]
pytestmark = pytest.mark.e2e


class FakeProcessor:
    async def process(self, document: UploadedDocument) -> ProcessedDocument:
        return ProcessedDocument(
            document_id=document.document_id,
            filename=document.filename,
            pages=[PageContent(page_number=1, text=f"Synthetic {document.filename} content")],
            processor_metadata=ProcessorMetadata(processor="fake", version="1"),
        )


class FakeClassifier:
    def __init__(self, scenario: Scenario) -> None:
        self.scenario = scenario

    async def classify(self, document: ProcessedDocument) -> DocumentClassification:
        document_type = {
            "application": DocumentType.APPLICATION_FORM,
            "extract": DocumentType.COMPANY_EXTRACT,
            "financial": DocumentType.FINANCIAL_STATEMENT,
        }[document.document_id]
        confidence = 0.98
        if self.scenario == "classification" and document.document_id == "application":
            document_type = DocumentType.UNKNOWN
            confidence = 0.55
        return DocumentClassification(
            document_id=document.document_id,
            document_type=document_type,
            confidence=confidence,
            reasoning_summary="Deterministic fake classification.",
        )


class FakeExtractor:
    def __init__(self, scenario: Scenario) -> None:
        self.scenario = scenario

    @staticmethod
    def _field(
        document_id: str,
        name: str,
        value: str | int | float | bool | None,
        *,
        confidence: float = 0.98,
    ) -> ExtractedField:
        return ExtractedField(
            field_id=f"{document_id}:{name}",
            document_id=document_id,
            field_name=name,
            value=value,
            confidence=confidence,
            evidence=(
                []
                if value is None
                else [
                    EvidenceReference(
                        document_id=document_id,
                        page_number=1,
                        source_text=f"{name}: {value}",
                    )
                ]
            ),
        )

    async def extract(
        self, document: ProcessedDocument, document_type: DocumentType
    ) -> ExtractionResult:
        if document_type is DocumentType.APPLICATION_FORM:
            registration = "NL00123"
            confidence = 0.98
            if self.scenario in {"field", "conflict"}:
                registration = "NL00999"
            if self.scenario == "field":
                confidence = 0.60
            fields = [
                self._field(document.document_id, "company_name", "Acme B.V."),
                self._field(
                    document.document_id,
                    "registration_number",
                    registration,
                    confidence=confidence,
                ),
                self._field(document.document_id, "annual_revenue_eur", 1_000_000),
                self._field(document.document_id, "employee_count", 42),
            ]
        elif document_type is DocumentType.COMPANY_EXTRACT:
            fields = [
                self._field(document.document_id, "company_name", "ACME BV"),
                self._field(document.document_id, "registration_number", "NL00123"),
                self._field(document.document_id, "incorporation_date", "2020-01-01"),
            ]
        elif document_type is DocumentType.FINANCIAL_STATEMENT:
            fields = [
                self._field(document.document_id, "company_name", "Acme, B.V."),
                self._field(document.document_id, "annual_revenue_eur", 1_010_000),
                self._field(document.document_id, "reporting_year", 2025),
                self._field(document.document_id, "employee_count", 42),
            ]
        else:
            fields = []
        return ExtractionResult(
            document_id=document.document_id,
            document_type=document_type,
            fields=fields,
        )


class FakeRetriever:
    async def search(self, query: str, *, limit: int = 5) -> list[PolicyEvidence]:
        del query, limit
        return [
            PolicyEvidence(
                evidence_id="policy-manual-review-2.1",
                policy_id="manual-review-policy",
                title="Manual Review Policy",
                section_id="2.1",
                text="Material inconsistencies require deterministic resolution.",
                score=0.9,
                source_path="policies/manual-review-policy.md",
            )
        ]


class FakeReportComposer:
    async def compose(
        self, review: VerifiedReview, policy_evidence: list[PolicyEvidence]
    ) -> ReviewReport:
        return ReviewReport(
            company_name=review.company_name,
            status=review.status,
            executive_summary="A deterministic synthetic review was completed.",
            sections=[
                ReportSection(
                    title="Verified findings",
                    summary="All listed findings came from deterministic validation.",
                    finding_ids=[finding.finding_id for finding in review.findings],
                    policy_evidence_ids=[
                        evidence.evidence_id for evidence in policy_evidence[:1]
                    ],
                )
            ],
        )


def dependencies(scenario: Scenario) -> WorkflowDependencies:
    return WorkflowDependencies(
        processor=FakeProcessor(),
        classifier=FakeClassifier(scenario),
        extractor=FakeExtractor(scenario),
        retriever=FakeRetriever(),
        report_composer=FakeReportComposer(),
        rules=load_review_rules(),
    )


def initial_state(
    review_id: str,
    thread_id: str,
    document_ids: tuple[str, ...] = ("application", "extract", "financial"),
) -> dict[str, Any]:
    documents = [
        UploadedDocument(
            document_id=document_id,
            filename=f"{document_id}.pdf",
            artifact_id=f"{review_id}/{document_id}.pdf",
            size_bytes=100,
        ).model_dump(mode="json")
        for document_id in document_ids
    ]
    return {
        "review_id": review_id,
        "thread_id": thread_id,
        "uploaded_documents": documents,
        "status": "processing",
        "review_decisions": [],
        "reviewed_item_ids": [],
    }


async def interrupt_then_restart(
    database: Path,
    scenario: Scenario,
    decision: dict[str, object],
) -> tuple[dict[str, Any], dict[str, Any]]:
    thread_id = f"thread-{scenario}"
    config = {"configurable": {"thread_id": thread_id}}
    async with AsyncSqliteSaver.from_conn_string(str(database)) as saver:
        graph = build_review_graph(dependencies(scenario), checkpointer=saver)
        interrupted = await graph.ainvoke(
            initial_state(f"review-{scenario}", thread_id), config=config
        )
    assert interrupted["status"] == "needs_review"
    assert interrupted["pending_review_items"]

    # Reopen both the SQLite connection and compiled graph to prove that resume
    # depends on persisted state, not an in-memory graph object.
    async with AsyncSqliteSaver.from_conn_string(str(database)) as saver:
        restarted = build_review_graph(dependencies(scenario), checkpointer=saver)
        completed = await restarted.ainvoke(
            Command(resume={"decisions": [decision]}), config=config
        )
    return interrupted, completed


@pytest.mark.asyncio
async def test_complete_package_reaches_validated_report() -> None:
    graph = build_review_graph(dependencies("complete"))
    result = await graph.ainvoke(initial_state("review-complete", "thread-complete"))

    assert result["status"] == "completed"
    assert result["report"]["status"] == "complete"
    assert result["summary"] == {
        "document_count": 3,
        "classified_count": 3,
        "extracted_field_count": 11,
        "finding_count": 0,
        "pending_review_count": 0,
        "decision_count": 0,
        "policy_evidence_count": 1,
    }


@pytest.mark.asyncio
async def test_missing_required_document_reaches_incomplete_report() -> None:
    graph = build_review_graph(dependencies("missing"))
    result = await graph.ainvoke(
        initial_state(
            "review-missing",
            "thread-missing",
            ("application", "extract"),
        )
    )

    assert result["status"] == "completed"
    assert result["report"]["status"] == "incomplete"
    assert [finding["finding_id"] for finding in result["findings"]] == [
        "finding-missing-financial-statement"
    ]


@pytest.mark.asyncio
async def test_classification_interrupt_survives_sqlite_restart(tmp_path: Path) -> None:
    item_id = "review-classification-application"
    interrupted, completed = await interrupt_then_restart(
        tmp_path / "classification-checkpoints.db",
        "classification",
        {
            "review_item_id": item_id,
            "action": "correct",
            "value": "application_form",
        },
    )

    assert interrupted["review_stage"] == "classification"
    classification = next(
        item
        for item in completed["classifications"]
        if item["document_id"] == "application"
    )
    assert classification["document_type"] == "unknown"
    assert classification["effective_document_type"] == "application_form"
    assert completed["status"] == "completed"


@pytest.mark.asyncio
async def test_field_interrupt_correction_is_revalidated_after_restart(
    tmp_path: Path,
) -> None:
    item_id = "review-field-application:registration_number"
    interrupted, completed = await interrupt_then_restart(
        tmp_path / "field-checkpoints.db",
        "field",
        {
            "review_item_id": item_id,
            "action": "correct",
            "value": "NL00123",
        },
    )

    assert interrupted["review_stage"] == "field"
    registration = next(
        item
        for item in completed["effective_fields"]
        if item["field_id"] == "application:registration_number"
    )
    assert registration["original_value"] == "NL00999"
    assert registration["effective_value"] == "NL00123"
    assert completed["review_decision_audits"][0]["original_values"] == [
        "NL00999"
    ]
    assert completed["review_decision_audits"][0]["effective_value"] == "NL00123"
    assert completed["resolved_review_items"][0]["status"] == "resolved"
    assert completed["findings"] == []
    assert completed["report"]["status"] == "complete"


@pytest.mark.asyncio
async def test_conflict_selection_is_revalidated_after_restart(tmp_path: Path) -> None:
    item_id = "review-conflict-finding-registration-number-conflict"
    interrupted, completed = await interrupt_then_restart(
        tmp_path / "conflict-checkpoints.db",
        "conflict",
        {
            "review_item_id": item_id,
            "action": "select_value",
            "selected_field_id": "extract:registration_number",
        },
    )

    assert interrupted["review_stage"] == "conflict"
    values = [
        item
        for item in completed["effective_fields"]
        if item["field_name"] == "registration_number"
    ]
    assert {item["original_value"] for item in values} == {"NL00999", "NL00123"}
    assert {item["effective_value"] for item in values} == {"NL00123"}
    assert completed["findings"] == []
    assert completed["report"]["status"] == "complete"


@pytest.mark.asyncio
async def test_unresolved_conflict_remains_in_final_report(tmp_path: Path) -> None:
    _, completed = await interrupt_then_restart(
        tmp_path / "unresolved-checkpoints.db",
        "conflict",
        {
            "review_item_id": "review-conflict-finding-registration-number-conflict",
            "action": "mark_unresolved",
        },
    )

    assert [finding["finding_id"] for finding in completed["findings"]] == [
        "finding-registration-number-conflict"
    ]
    assert completed["report"]["status"] == "needs_follow_up"
