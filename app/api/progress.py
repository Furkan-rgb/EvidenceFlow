"""Stable API projection of durable LangGraph workflow progress."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from app.api.schemas import (
    WorkflowProgress,
    WorkflowProgressStep,
    WorkflowStepId,
)

WORKFLOW_STEPS: tuple[WorkflowStepId, ...] = (
    "document_processing",
    "classification",
    "extraction",
    "completeness",
    "cross_check",
    "policy_retrieval",
    "report_composition",
)

_NODE_TO_STEP: dict[str, WorkflowStepId] = {
    "process_documents": "document_processing",
    "classify_documents": "classification",
    "prepare_classification_review": "classification",
    "apply_classification": "classification",
    "extract_fields": "extraction",
    "normalize_and_check_completeness": "completeness",
    "prepare_field_review": "completeness",
    "apply_field": "completeness",
    "revalidate_completeness": "completeness",
    "cross_check": "cross_check",
    "prepare_conflict_review": "cross_check",
    "apply_conflict": "cross_check",
    "revalidate_after_conflict": "cross_check",
    "retrieve_policy_evidence": "policy_retrieval",
    "compose_report": "report_composition",
}

_REVIEW_STAGE_TO_STEP: dict[str, WorkflowStepId] = {
    "classification": "classification",
    "field": "completeness",
    "conflict": "cross_check",
}


def _step_for_node(node: str, state: Mapping[str, Any]) -> WorkflowStepId | None:
    if node == "human_review":
        return _REVIEW_STAGE_TO_STEP.get(str(state.get("review_stage") or ""))
    return _NODE_TO_STEP.get(node)


def _infer_current_step(state: Mapping[str, Any]) -> WorkflowStepId:
    """Use the furthest durable state when no live checkpoint can be read."""

    review_step = _REVIEW_STAGE_TO_STEP.get(str(state.get("review_stage") or ""))
    if state.get("status") == "completed" or "report" in state:
        return "report_composition"
    if "policy_evidence" in state:
        return "report_composition"
    if review_step == "cross_check":
        return "cross_check"
    if "effective_fields" in state:
        return review_step or "completeness"
    if "extraction_results" in state:
        return "completeness"
    if review_step == "classification":
        return "classification"
    if "classifications" in state:
        return "extraction"
    if "processed_documents" in state:
        return "classification"
    return "document_processing"


def project_workflow_progress(
    review_status: str,
    state: Mapping[str, Any],
    next_nodes: Sequence[str] = (),
) -> WorkflowProgress:
    """Return completed/current/upcoming stages for one review poll response."""

    if review_status == "completed":
        return WorkflowProgress(
            current_step_id=None,
            steps=[
                WorkflowProgressStep(id=step_id, status="completed")
                for step_id in WORKFLOW_STEPS
            ],
        )

    current_step: WorkflowStepId | None = None
    if review_status == "needs_review":
        current_step = _REVIEW_STAGE_TO_STEP.get(
            str(state.get("review_stage") or "")
        )
    if current_step is None:
        for node in next_nodes:
            current_step = _step_for_node(node, state)
            if current_step is not None:
                break
    if current_step is None:
        # This also keeps report composition current during the short interval
        # between graph completion and the repository's completed transition.
        current_step = _infer_current_step(state)

    current_index = WORKFLOW_STEPS.index(current_step)
    steps = [
        WorkflowProgressStep(
            id=step_id,
            status=(
                "completed"
                if index < current_index
                else "current"
                if index == current_index
                else "upcoming"
            ),
        )
        for index, step_id in enumerate(WORKFLOW_STEPS)
    ]
    return WorkflowProgress(current_step_id=current_step, steps=steps)
