from __future__ import annotations

import pytest

from app.api.progress import WORKFLOW_STEPS, project_workflow_progress
from app.api.schemas import WorkflowProgress, WorkflowStepId


def assert_current_step(
    current_step_id: WorkflowStepId, progress: WorkflowProgress
) -> None:
    assert progress.current_step_id == current_step_id
    steps = progress.steps
    current_index = WORKFLOW_STEPS.index(current_step_id)
    assert [step.id for step in steps] == list(WORKFLOW_STEPS)
    assert [step.status for step in steps] == [
        (
            "completed"
            if index < current_index
            else "current"
            if index == current_index
            else "upcoming"
        )
        for index in range(len(WORKFLOW_STEPS))
    ]


@pytest.mark.parametrize(
    ("node", "step_id"),
    [
        ("process_documents", "document_processing"),
        ("classify_documents", "classification"),
        ("prepare_classification_review", "classification"),
        ("apply_classification", "classification"),
        ("extract_fields", "extraction"),
        ("normalize_and_check_completeness", "completeness"),
        ("prepare_field_review", "completeness"),
        ("apply_field", "completeness"),
        ("revalidate_completeness", "completeness"),
        ("cross_check", "cross_check"),
        ("prepare_conflict_review", "cross_check"),
        ("apply_conflict", "cross_check"),
        ("revalidate_after_conflict", "cross_check"),
        ("retrieve_policy_evidence", "policy_retrieval"),
        ("compose_report", "report_composition"),
    ],
)
def test_projects_every_internal_node_to_a_stable_public_step(
    node: str, step_id: WorkflowStepId
) -> None:
    progress = project_workflow_progress("processing", {}, (node,))

    assert_current_step(step_id, progress)


@pytest.mark.parametrize(
    ("review_stage", "step_id"),
    [
        ("classification", "classification"),
        ("field", "completeness"),
        ("conflict", "cross_check"),
    ],
)
def test_human_review_projects_to_the_stage_that_requires_a_decision(
    review_stage: str, step_id: WorkflowStepId
) -> None:
    state = {"review_stage": review_stage}

    processing = project_workflow_progress(
        "processing", state, ("human_review",)
    )
    interrupted = project_workflow_progress("needs_review", state)

    assert_current_step(step_id, processing)
    assert_current_step(step_id, interrupted)


def test_fresh_review_starts_with_document_processing() -> None:
    progress = project_workflow_progress("processing", {})

    assert_current_step("document_processing", progress)


def test_processing_finalization_keeps_report_composition_current() -> None:
    progress = project_workflow_progress(
        "processing", {"status": "completed", "report": {}}
    )

    assert_current_step("report_composition", progress)


def test_completed_review_has_no_current_or_upcoming_steps() -> None:
    progress = project_workflow_progress(
        "completed", {"review_stage": "classification"}, ("human_review",)
    )

    assert progress.current_step_id is None
    assert [step.id for step in progress.steps] == list(WORKFLOW_STEPS)
    assert {step.status for step in progress.steps} == {"completed"}
