"""Durable, capability-oriented EvidenceFlow review graph."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, cast

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from app.ai.usage import consume_usage_metadata
from app.domain import (
    DocumentClassification,
    EffectiveFieldValue,
    ExtractionResult,
    Finding,
    FindingType,
    PolicyEvidence,
    ProcessedDocument,
    ReviewDecision,
    ReviewDecisionAudit,
    ReviewItem,
    UploadedDocument,
)
from app.graph.state import ReviewStage, ReviewState
from app.observability import NoOpTracer, Tracer
from app.ports import (
    DocumentClassifier,
    DocumentProcessor,
    FieldExtractor,
    PolicyRetriever,
    ReportComposer,
)
from app.review import (
    ReviewRules,
    apply_review_decisions,
    build_effective_fields,
    build_verified_review,
    create_classification_review_items,
    create_conflict_review_items,
    create_field_review_items,
    finalize_report,
    render_report_markdown,
    validate_required_documents,
    validate_required_fields,
    validate_review_package,
)


@dataclass(frozen=True, slots=True)
class WorkflowDependencies:
    """Task capabilities injected into graph construction, never graph state."""

    processor: DocumentProcessor
    classifier: DocumentClassifier
    extractor: FieldExtractor
    retriever: PolicyRetriever
    report_composer: ReportComposer
    rules: ReviewRules
    tracer: Tracer = field(default_factory=NoOpTracer)
    task_metadata: dict[str, dict[str, str | int | float | bool]] = field(
        default_factory=dict
    )


def _dump(value: Any) -> dict[str, Any]:
    return cast(dict[str, Any], value.model_dump(mode="json"))


def _load_many(model: type[Any], values: Sequence[dict[str, Any]]) -> list[Any]:
    return [model.model_validate(value) for value in values]


async def _gather_traced(
    values: Sequence[Any],
    operation: Callable[[Any], Awaitable[Any]],
    *,
    tracer: Tracer,
    span_name: str,
    span_type: str,
    identity: Callable[[Any], str],
    attributes: Mapping[str, Any] | None = None,
    result_summary: Callable[[Any], dict[str, Any]] | None = None,
) -> list[Any]:
    async def invoke(value: Any) -> Any:
        with tracer.span(
            span_name,
            span_type=span_type,
            attributes={"document_id": identity(value), **dict(attributes or {})},
        ) as span:
            result = await operation(value)
            if span is not None:
                outputs = {"document_id": identity(value), "success": True}
                if result_summary is not None:
                    outputs.update(result_summary(result))
                outputs.update(consume_usage_metadata())
                span.set_outputs(outputs)
            return result

    return list(await asyncio.gather(*(invoke(value) for value in values)))


def _decision_payload(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        value = value.get("decisions")
    if not isinstance(value, list):
        raise ValueError("A resumed review must contain a decisions list")
    if not all(isinstance(item, dict) for item in value):
        raise ValueError("Every resumed review decision must be an object")
    return cast(list[dict[str, Any]], value)


def _route_pending(state: ReviewState) -> Literal["human_review", "continue"]:
    return "human_review" if state.get("pending_review_items") else "continue"


def _route_review_stage(
    state: ReviewState,
) -> Literal["apply_classification", "apply_field", "apply_conflict"]:
    stage = state["review_stage"]
    return cast(
        Literal["apply_classification", "apply_field", "apply_conflict"],
        f"apply_{stage}",
    )


def _policy_queries(findings: Sequence[Finding]) -> list[str]:
    queries = [
        "required onboarding documents, reliable evidence, and manual review requirements"
    ]
    for finding in findings:
        if finding.type is FindingType.MISSING_DOCUMENT:
            document_type = (
                finding.document_type.value if finding.document_type else "unknown"
            )
            queries.append(
                f"required document missing {document_type} onboarding evidence"
            )
        elif finding.type is FindingType.MISSING_REQUIRED_FIELD:
            document_type = (
                finding.document_type.value if finding.document_type else "unknown"
            )
            queries.append(
                f"incomplete {document_type} missing {finding.field_name}"
            )
        elif finding.type is FindingType.FIELD_CONFLICT:
            queries.append(
                f"conflicting {finding.field_name} evidence and manual review resolution"
            )
    return list(dict.fromkeys(queries))


def build_review_graph(
    dependencies: WorkflowDependencies,
    *,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
) -> Any:
    """Build the V1 graph with durable interrupt points and injected capabilities."""

    async def process_documents(state: ReviewState) -> dict[str, Any]:
        uploaded = _load_many(UploadedDocument, state["uploaded_documents"])
        processed = await _gather_traced(
            uploaded,
            dependencies.processor.process,
            tracer=dependencies.tracer,
            span_name="document.process",
            span_type="PARSER",
            identity=lambda item: cast(UploadedDocument, item).document_id,
            attributes={
                "review_id": state["review_id"],
                "thread_id": state["thread_id"],
                "processor": "pymupdf",
            },
            result_summary=lambda result: {
                "page_count": len(cast(ProcessedDocument, result).pages)
            },
        )
        return {"processed_documents": [_dump(item) for item in processed]}

    async def classify_documents(state: ReviewState) -> dict[str, Any]:
        processed = _load_many(ProcessedDocument, state["processed_documents"])
        classifications = await _gather_traced(
            processed,
            dependencies.classifier.classify,
            tracer=dependencies.tracer,
            span_name="ai.classification",
            span_type="LLM",
            identity=lambda item: cast(ProcessedDocument, item).document_id,
            attributes={
                "review_id": state["review_id"],
                "thread_id": state["thread_id"],
                **dependencies.task_metadata.get("classification", {}),
            },
            result_summary=lambda result: {
                "document_type": cast(
                    DocumentClassification, result
                ).document_type.value,
                "confidence": cast(DocumentClassification, result).confidence,
            },
        )
        return {"classifications": [_dump(item) for item in classifications]}

    async def prepare_classification_review(state: ReviewState) -> dict[str, Any]:
        classifications = _load_many(
            DocumentClassification, state["classifications"]
        )
        items = create_classification_review_items(
            classifications, dependencies.rules
        )
        return {
            "pending_review_items": [_dump(item) for item in items],
            "review_stage": "classification",
            "status": "needs_review" if items else "processing",
        }

    async def human_review(state: ReviewState) -> dict[str, Any]:
        resumed = interrupt(
            {
                "review_id": state["review_id"],
                "stage": state["review_stage"],
                "review_items": state["pending_review_items"],
            }
        )
        return {
            "submitted_decisions": _decision_payload(resumed),
            "status": "processing",
        }

    def apply_decisions(
        state: ReviewState, expected_stage: ReviewStage
    ) -> dict[str, Any]:
        if state["review_stage"] != expected_stage:
            raise ValueError(
                f"Review stage {state['review_stage']} cannot be applied as {expected_stage}"
            )
        classifications = _load_many(
            DocumentClassification, state.get("classifications", [])
        )
        effective_fields = _load_many(
            EffectiveFieldValue, state.get("effective_fields", [])
        )
        items = _load_many(ReviewItem, state["pending_review_items"])
        decisions = _load_many(ReviewDecision, state["submitted_decisions"])
        application = apply_review_decisions(
            classifications, effective_fields, items, decisions
        )
        previous = _load_many(ReviewDecision, state.get("review_decisions", []))
        previous_audits = _load_many(
            ReviewDecisionAudit, state.get("review_decision_audits", [])
        )
        previously_resolved = _load_many(
            ReviewItem, state.get("resolved_review_items", [])
        )
        reviewed_ids = [*state.get("reviewed_item_ids", [])]
        reviewed_ids.extend(item.review_item_id for item in items)
        return {
            "classifications": [_dump(item) for item in application.classifications],
            "effective_fields": [_dump(item) for item in application.effective_fields],
            "review_decisions": [_dump(item) for item in [*previous, *decisions]],
            "review_decision_audits": [
                _dump(item)
                for item in [*previous_audits, *application.audit_records]
            ],
            "resolved_review_items": [
                _dump(item)
                for item in [*previously_resolved, *application.review_items]
            ],
            "reviewed_item_ids": list(dict.fromkeys(reviewed_ids)),
            "pending_review_items": [],
            "submitted_decisions": [],
        }

    async def apply_classification(state: ReviewState) -> dict[str, Any]:
        return apply_decisions(state, "classification")

    async def extract_fields(state: ReviewState) -> dict[str, Any]:
        processed = _load_many(ProcessedDocument, state["processed_documents"])
        classifications = _load_many(
            DocumentClassification, state["classifications"]
        )
        by_document = {item.document_id: item for item in classifications}

        async def extract(document: ProcessedDocument) -> ExtractionResult:
            classification = by_document[document.document_id]
            return await dependencies.extractor.extract(
                document, classification.resolved_document_type
            )

        extractions = await _gather_traced(
            processed,
            extract,
            tracer=dependencies.tracer,
            span_name="ai.extraction",
            span_type="LLM",
            identity=lambda item: cast(ProcessedDocument, item).document_id,
            attributes={
                "review_id": state["review_id"],
                "thread_id": state["thread_id"],
                **dependencies.task_metadata.get("extraction", {}),
            },
            result_summary=lambda result: {
                "field_count": len(cast(ExtractionResult, result).fields)
            },
        )
        return {"extraction_results": [_dump(item) for item in extractions]}

    async def normalize_and_check_completeness(
        state: ReviewState,
    ) -> dict[str, Any]:
        classifications = _load_many(
            DocumentClassification, state["classifications"]
        )
        extractions = _load_many(ExtractionResult, state["extraction_results"])
        effective = build_effective_fields(extractions)
        findings = [
            *validate_required_documents(classifications, dependencies.rules),
            *validate_required_fields(extractions, dependencies.rules, effective),
        ]
        return {
            "effective_fields": [_dump(item) for item in effective],
            "findings": [_dump(item) for item in findings],
        }

    async def prepare_field_review(state: ReviewState) -> dict[str, Any]:
        effective = _load_many(EffectiveFieldValue, state["effective_fields"])
        items = create_field_review_items(effective, dependencies.rules)
        return {
            "pending_review_items": [_dump(item) for item in items],
            "review_stage": "field",
            "status": "needs_review" if items else "processing",
        }

    async def apply_field(state: ReviewState) -> dict[str, Any]:
        return apply_decisions(state, "field")

    async def revalidate_completeness(state: ReviewState) -> dict[str, Any]:
        classifications = _load_many(
            DocumentClassification, state["classifications"]
        )
        extractions = _load_many(ExtractionResult, state["extraction_results"])
        effective = _load_many(EffectiveFieldValue, state["effective_fields"])
        findings = [
            *validate_required_documents(classifications, dependencies.rules),
            *validate_required_fields(extractions, dependencies.rules, effective),
        ]
        return {"findings": [_dump(item) for item in findings]}

    async def cross_check(state: ReviewState) -> dict[str, Any]:
        classifications = _load_many(
            DocumentClassification, state["classifications"]
        )
        extractions = _load_many(ExtractionResult, state["extraction_results"])
        effective = _load_many(EffectiveFieldValue, state["effective_fields"])
        findings = validate_review_package(
            classifications, extractions, effective, dependencies.rules
        )
        return {"findings": [_dump(item) for item in findings]}

    async def prepare_conflict_review(state: ReviewState) -> dict[str, Any]:
        findings = _load_many(Finding, state["findings"])
        effective = _load_many(EffectiveFieldValue, state["effective_fields"])
        items = create_conflict_review_items(findings, effective)
        return {
            "pending_review_items": [_dump(item) for item in items],
            "review_stage": "conflict",
            "status": "needs_review" if items else "processing",
        }

    async def apply_conflict(state: ReviewState) -> dict[str, Any]:
        return apply_decisions(state, "conflict")

    async def revalidate_after_conflict(state: ReviewState) -> dict[str, Any]:
        classifications = _load_many(
            DocumentClassification, state["classifications"]
        )
        extractions = _load_many(ExtractionResult, state["extraction_results"])
        effective = _load_many(EffectiveFieldValue, state["effective_fields"])
        findings = validate_review_package(
            classifications, extractions, effective, dependencies.rules
        )
        return {"findings": [_dump(item) for item in findings]}

    async def retrieve_policy_evidence(state: ReviewState) -> dict[str, Any]:
        findings = _load_many(Finding, state.get("findings", []))
        queries = _policy_queries(findings)

        async def search(query: str) -> list[PolicyEvidence]:
            with dependencies.tracer.span(
                "policy.retrieve",
                span_type="RETRIEVER",
                attributes={
                    "review_id": state["review_id"],
                    "thread_id": state["thread_id"],
                    "finding_count": len(findings),
                    **dependencies.task_metadata.get("embeddings", {}),
                },
            ) as span:
                evidence = await dependencies.retriever.search(query, limit=3)
                if span is not None:
                    span.set_outputs({"evidence_count": len(evidence)})
                return evidence

        batches = await asyncio.gather(*(search(query) for query in queries))
        best: dict[str, PolicyEvidence] = {}
        for evidence in (item for batch in batches for item in batch):
            current = best.get(evidence.evidence_id)
            if current is None or evidence.score > current.score:
                best[evidence.evidence_id] = evidence
        selected = sorted(
            best.values(), key=lambda item: (-item.score, item.evidence_id)
        )[:8]
        return {"policy_evidence": [_dump(item) for item in selected]}

    async def compose_report(state: ReviewState) -> dict[str, Any]:
        classifications = _load_many(
            DocumentClassification, state["classifications"]
        )
        extractions = _load_many(ExtractionResult, state["extraction_results"])
        effective = _load_many(EffectiveFieldValue, state["effective_fields"])
        findings = _load_many(Finding, state.get("findings", []))
        decisions = _load_many(ReviewDecision, state.get("review_decisions", []))
        evidence = _load_many(PolicyEvidence, state.get("policy_evidence", []))
        verified = build_verified_review(
            review_id=state["review_id"],
            classifications=classifications,
            extractions=extractions,
            effective_fields=effective,
            findings=findings,
            review_decisions=decisions,
        )
        with dependencies.tracer.span(
            "ai.report",
            span_type="LLM",
            attributes={
                "finding_count": len(findings),
                "policy_evidence_count": len(evidence),
                "review_id": state["review_id"],
                "thread_id": state["thread_id"],
                **dependencies.task_metadata.get("reporting", {}),
            },
        ) as span:
            proposed = await dependencies.report_composer.compose(verified, evidence)
            report = finalize_report(proposed, verified, evidence)
            if span is not None:
                span.set_outputs(
                    {
                        "status": report.status.value,
                        "section_count": len(report.sections),
                        **consume_usage_metadata(),
                    }
                )
        summary = {
            "document_count": len(classifications),
            "classified_count": len(classifications),
            "extracted_field_count": sum(
                len(extraction.fields) for extraction in extractions
            ),
            "finding_count": len(findings),
            "pending_review_count": 0,
            "decision_count": len(decisions),
            "policy_evidence_count": len(evidence),
        }
        return {
            "verified_review": _dump(verified),
            "report": _dump(report),
            "report_markdown": render_report_markdown(report),
            "pending_review_items": [],
            "status": "completed",
            "summary": summary,
        }

    graph = StateGraph(ReviewState)
    graph.add_node("process_documents", process_documents)
    graph.add_node("classify_documents", classify_documents)
    graph.add_node("prepare_classification_review", prepare_classification_review)
    graph.add_node("human_review", human_review)
    graph.add_node("apply_classification", apply_classification)
    graph.add_node("extract_fields", extract_fields)
    graph.add_node("normalize_and_check_completeness", normalize_and_check_completeness)
    graph.add_node("prepare_field_review", prepare_field_review)
    graph.add_node("apply_field", apply_field)
    graph.add_node("revalidate_completeness", revalidate_completeness)
    graph.add_node("cross_check", cross_check)
    graph.add_node("prepare_conflict_review", prepare_conflict_review)
    graph.add_node("apply_conflict", apply_conflict)
    graph.add_node("revalidate_after_conflict", revalidate_after_conflict)
    graph.add_node("retrieve_policy_evidence", retrieve_policy_evidence)
    graph.add_node("compose_report", compose_report)

    graph.add_edge(START, "process_documents")
    graph.add_edge("process_documents", "classify_documents")
    graph.add_edge("classify_documents", "prepare_classification_review")
    graph.add_conditional_edges(
        "prepare_classification_review",
        _route_pending,
        {"human_review": "human_review", "continue": "extract_fields"},
    )
    graph.add_conditional_edges(
        "human_review",
        _route_review_stage,
        {
            "apply_classification": "apply_classification",
            "apply_field": "apply_field",
            "apply_conflict": "apply_conflict",
        },
    )
    graph.add_edge("apply_classification", "extract_fields")
    graph.add_edge("extract_fields", "normalize_and_check_completeness")
    graph.add_edge("normalize_and_check_completeness", "prepare_field_review")
    graph.add_conditional_edges(
        "prepare_field_review",
        _route_pending,
        {"human_review": "human_review", "continue": "cross_check"},
    )
    graph.add_edge("apply_field", "revalidate_completeness")
    graph.add_edge("revalidate_completeness", "cross_check")
    graph.add_edge("cross_check", "prepare_conflict_review")
    graph.add_conditional_edges(
        "prepare_conflict_review",
        _route_pending,
        {"human_review": "human_review", "continue": "retrieve_policy_evidence"},
    )
    graph.add_edge("apply_conflict", "revalidate_after_conflict")
    graph.add_edge("revalidate_after_conflict", "retrieve_policy_evidence")
    graph.add_edge("retrieve_policy_evidence", "compose_report")
    graph.add_edge("compose_report", END)
    return graph.compile(checkpointer=checkpointer, name="evidenceflow-v1")
