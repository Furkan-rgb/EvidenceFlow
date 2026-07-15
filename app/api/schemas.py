"""Public API request and response contracts."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

WorkflowStepId = Literal[
    "document_processing",
    "classification",
    "extraction",
    "completeness",
    "cross_check",
    "policy_retrieval",
    "report_composition",
]
WorkflowStepStatus = Literal["completed", "current", "upcoming"]


class WorkflowProgressStep(BaseModel):
    """One stable, user-facing stage in the review workflow."""

    model_config = ConfigDict(extra="forbid")

    id: WorkflowStepId
    status: WorkflowStepStatus


class WorkflowProgress(BaseModel):
    """Current workflow position without exposing internal graph node names."""

    model_config = ConfigDict(extra="forbid")

    current_step_id: WorkflowStepId | None
    steps: list[WorkflowProgressStep] = Field(min_length=7, max_length=7)

    @model_validator(mode="after")
    def validate_current_step(self) -> WorkflowProgress:
        current = [step.id for step in self.steps if step.status == "current"]
        expected = [] if self.current_step_id is None else [self.current_step_id]
        if current != expected:
            raise ValueError("current_step_id must identify the one current workflow step")
        return self


class ReviewDecisionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    review_item_id: str = Field(min_length=1)
    action: Literal["approve", "correct", "select_value", "mark_unresolved"]
    value: str | int | float | bool | None = None
    selected_field_id: str | None = None

    @model_validator(mode="after")
    def validate_action_shape(self) -> ReviewDecisionInput:
        if self.action == "correct":
            if self.value is None:
                raise ValueError("A corrected value is required")
            if self.selected_field_id is not None:
                raise ValueError("correct does not accept selected_field_id")
        if self.action == "select_value":
            if not self.selected_field_id:
                raise ValueError("selected_field_id is required for select_value")
            if self.value is not None:
                raise ValueError("select_value does not accept value")
        if self.action in {"approve", "mark_unresolved"} and (
            self.value is not None or self.selected_field_id is not None
        ):
            raise ValueError(f"{self.action} does not accept value or selected_field_id")
        return self


class ResumeReviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decisions: list[ReviewDecisionInput] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_items(self) -> ResumeReviewRequest:
        item_ids = [item.review_item_id for item in self.decisions]
        if len(item_ids) != len(set(item_ids)):
            raise ValueError("Only one decision may be supplied per review item")
        return self


class CreateReviewResponse(BaseModel):
    review_id: str
    thread_id: str
    status: Literal["processing"] = "processing"


class ResumeReviewResponse(BaseModel):
    review_id: str
    status: Literal["processing"] = "processing"


class ErrorBody(BaseModel):
    code: str
    message: str
    details: Any = None
    request_id: str


class ErrorResponse(BaseModel):
    error: ErrorBody
