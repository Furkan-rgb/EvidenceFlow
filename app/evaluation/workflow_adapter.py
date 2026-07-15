"""Real EvidenceFlow workflow adapter used by the synthetic evaluation suite."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from uuid import uuid4

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from app.ai import LLMDocumentClassifier, LLMFieldExtractor, LLMReportComposer
from app.ai.config import ModelsConfig
from app.ai.models import create_chat_model
from app.bootstrap import model_task_metadata
from app.documents import PyMuPDFDocumentProcessor
from app.graph import WorkflowDependencies, build_review_graph
from app.observability import Tracer
from app.ports import PolicyRetriever
from app.review import ReviewRules


class EvaluationArtifactReader:
    """Read-only mapping from evaluation artifact IDs to generated PDFs."""

    def __init__(self) -> None:
        self._paths: dict[str, Path] = {}

    def register(self, artifact_id: str, path: Path) -> None:
        self._paths[artifact_id] = path

    async def read_upload(self, artifact_id: str) -> bytes:
        try:
            path = self._paths[artifact_id]
        except KeyError as error:
            raise FileNotFoundError(f"Unknown evaluation artifact: {artifact_id}") from error
        return await asyncio.to_thread(path.read_bytes)


class WorkflowEvaluationAdapter:
    """Run actual configured capabilities and simulate only required reviewer input."""

    def __init__(
        self,
        *,
        models: ModelsConfig,
        rules: ReviewRules,
        retriever: PolicyRetriever,
        tracer: Tracer,
        max_pages: int = 50,
    ) -> None:
        self._reader = EvaluationArtifactReader()
        self._tracer = tracer
        self._graph = build_review_graph(
            WorkflowDependencies(
                processor=PyMuPDFDocumentProcessor(
                    self._reader, max_pages=max_pages
                ),
                classifier=LLMDocumentClassifier(
                    create_chat_model(models.classification)
                ),
                extractor=LLMFieldExtractor(create_chat_model(models.extraction)),
                retriever=retriever,
                report_composer=LLMReportComposer(
                    create_chat_model(models.reporting)
                ),
                rules=rules,
                tracer=tracer,
                task_metadata=model_task_metadata(models),
            ),
            checkpointer=InMemorySaver(),
        )

    async def evaluate_bundle(
        self, bundle_path: Path, ground_truth: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        attributes = {
            "bundle_id": str(ground_truth.get("bundle_id", bundle_path.name)),
            "category": str(ground_truth.get("category", "unknown")),
            "document_count": len(ground_truth.get("documents", ())),
        }
        with self._tracer.span(
            "evaluation.bundle",
            span_type="CHAIN",
            attributes=attributes,
        ) as span:
            result = await self._evaluate_bundle(bundle_path, ground_truth)
            if span is not None:
                span.set_outputs(
                    {
                        "finding_count": len(result.get("findings", ())),
                        "review_item_count": len(result.get("review_items", ())),
                        "report_available": result.get("report") is not None,
                    }
                )
            return result

    async def _evaluate_bundle(
        self, bundle_path: Path, ground_truth: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        expected_documents = {
            str(item["file_name"]): item
            for item in ground_truth.get("documents", ())
        }
        documents: list[dict[str, Any]] = []
        filenames_by_id: dict[str, str] = {}
        for index, pdf_path in enumerate(
            sorted((bundle_path / "documents").glob("*.pdf")), start=1
        ):
            document_id = f"eval-document-{index}-{pdf_path.stem}"
            artifact_id = f"{bundle_path.name}/{pdf_path.name}"
            self._reader.register(artifact_id, pdf_path)
            filenames_by_id[document_id] = pdf_path.name
            documents.append(
                {
                    "document_id": document_id,
                    "filename": pdf_path.name,
                    "artifact_id": artifact_id,
                    "content_type": "application/pdf",
                    "size_bytes": pdf_path.stat().st_size,
                }
            )

        review_id = f"evaluation-{bundle_path.name}-{uuid4().hex}"
        thread_id = f"thread-{review_id}"
        configuration = {"configurable": {"thread_id": thread_id}}
        output = await self._graph.ainvoke(
            {
                "review_id": review_id,
                "thread_id": thread_id,
                "uploaded_documents": documents,
                "review_decisions": [],
                "review_decision_audits": [],
                "resolved_review_items": [],
                "reviewed_item_ids": [],
                "pending_review_items": [],
                "status": "processing",
            },
            configuration,
            version="v2",
        )
        observed_review_items: list[dict[str, Any]] = []
        observed_findings: dict[str, dict[str, Any]] = {}
        for _ in range(6):
            state = dict(output.value)
            for finding in state.get("findings", []):
                observed_findings[str(finding["finding_id"])] = finding
            if not output.interrupts:
                break
            pending = list(state.get("pending_review_items") or [])
            observed_review_items.extend(pending)
            decisions = self._review_decisions(
                pending,
                filenames_by_id=filenames_by_id,
                expected_documents=expected_documents,
                configured=ground_truth.get("reviewer_decisions", ()),
            )
            output = await self._graph.ainvoke(
                Command(resume={"decisions": decisions}),
                configuration,
                version="v2",
            )
        else:
            raise RuntimeError("Evaluation exceeded the expected human-review stages")

        if output.interrupts:
            raise RuntimeError("Evaluation ended with an unresolved graph interrupt")
        state = dict(output.value)
        for finding in state.get("findings", []):
            observed_findings[str(finding["finding_id"])] = finding
        if not self._tracer.healthy:
            raise RuntimeError("MLflow tracing became unavailable during evaluation")
        classifications = {
            str(item["document_id"]): item for item in state["classifications"]
        }
        extractions = {
            str(item["document_id"]): item for item in state["extraction_results"]
        }
        actual_documents: list[dict[str, Any]] = []
        for document_id, filename in filenames_by_id.items():
            classification = classifications[document_id]
            extraction = extractions[document_id]
            actual_documents.append(
                {
                    "file_name": filename,
                    # Model-quality metrics use original AI output; reviewer
                    # corrections remain visible in workflow/audit metrics.
                    "document_type": classification["document_type"],
                    "fields": extraction.get("fields", []),
                }
            )
        return {
            "documents": actual_documents,
            "findings": list(observed_findings.values()),
            "review_items": observed_review_items,
            "review_required": bool(observed_review_items),
            "review_routing_reasons": self._review_routing_reasons(
                observed_review_items
            ),
            "policy_evidence": state.get("policy_evidence", []),
            "retrieved_policy_evidence_ids": [
                item["evidence_id"] for item in state.get("policy_evidence", [])
            ],
            "report": state.get("report"),
        }

    @staticmethod
    def _review_routing_reasons(
        review_items: Sequence[Mapping[str, Any]],
    ) -> list[str]:
        reasons: set[str] = set()
        for item in review_items:
            item_type = str(item.get("type", ""))
            field_name = item.get("field_name")
            if item_type == "low_confidence_classification":
                reasons.add("low_confidence:classification")
            elif item_type == "low_confidence_field" and field_name is not None:
                reasons.add(f"low_confidence:{field_name}")
            elif item_type == "field_conflict" and field_name is not None:
                reasons.add(f"conflict:{field_name}")
        return sorted(reasons)

    @staticmethod
    def _review_decisions(
        pending: Sequence[Mapping[str, Any]],
        *,
        filenames_by_id: Mapping[str, str],
        expected_documents: Mapping[str, Mapping[str, Any]],
        configured: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        configured_by_field = {
            str(item["field"]): item for item in configured if item.get("field")
        }
        decisions: list[dict[str, Any]] = []
        for item in pending:
            item_id = str(item["review_item_id"])
            item_type = str(item["type"])
            if item_type == "low_confidence_classification":
                filename = filenames_by_id[str(item["document_id"])]
                expected = str(expected_documents[filename]["document_type"])
                if item.get("proposed_document_type") == expected:
                    decisions.append(
                        {"review_item_id": item_id, "action": "approve"}
                    )
                else:
                    decisions.append(
                        {
                            "review_item_id": item_id,
                            "action": "correct",
                            "value": expected,
                        }
                    )
                continue
            if item_type == "low_confidence_field":
                field_name = str(item["field_name"])
                configured_decision = configured_by_field.get(field_name)
                if configured_decision is not None:
                    decision = {
                        "review_item_id": item_id,
                        "action": configured_decision["action"],
                    }
                    if "value" in configured_decision:
                        decision["value"] = configured_decision["value"]
                    decisions.append(decision)
                    continue
                filename = filenames_by_id[str(item["document_id"])]
                expected_field = expected_documents[filename].get(
                    "expected_fields", {}
                ).get(field_name)
                expected_value = (
                    expected_field.get("value")
                    if isinstance(expected_field, Mapping)
                    else None
                )
                if expected_value is not None and item.get("extracted_value") != expected_value:
                    decisions.append(
                        {
                            "review_item_id": item_id,
                            "action": "correct",
                            "value": expected_value,
                        }
                    )
                else:
                    decisions.append(
                        {"review_item_id": item_id, "action": "approve"}
                    )
                continue
            decisions.append(
                {"review_item_id": item_id, "action": "mark_unresolved"}
            )
        return decisions


class PolicyRetrieverEvaluationAdapter:
    def __init__(
        self,
        retriever: PolicyRetriever,
        *,
        tracer: Tracer,
        task_metadata: Mapping[str, str | int | float | bool],
    ) -> None:
        self._retriever = retriever
        self._tracer = tracer
        self._task_metadata = dict(task_metadata)

    async def retrieve_policy_ids(self, query: str, *, k: int) -> Sequence[str]:
        started = time.perf_counter()
        attributes = {**self._task_metadata, "scope": "evaluation_benchmark", "limit": k}
        with self._tracer.span(
            "policy.retrieve",
            span_type="RETRIEVER",
            attributes=attributes,
        ) as span:
            evidence = await self._retriever.search(query, limit=k)
            if span is not None:
                span.set_outputs(
                    {
                        "result_count": len(evidence),
                        "latency_seconds": time.perf_counter() - started,
                    }
                )
        return [item.evidence_id for item in evidence]
