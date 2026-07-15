"""Human-review and verified-review contracts."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from uuid import uuid4

from pydantic import Field, model_validator

from app.domain.base import DomainModel
from app.domain.documents import DocumentType
from app.domain.extractions import (
    DocumentClassification,
    EffectiveFieldValue,
    EvidenceReference,
    ExtractionResult,
    FieldValue,
    NormalizedValue,
)
from app.domain.findings import Finding
from app.domain.reports import ReportStatus


class ReviewItemType(StrEnum):
    LOW_CONFIDENCE_CLASSIFICATION = "low_confidence_classification"
    LOW_CONFIDENCE_FIELD = "low_confidence_field"
    FIELD_CONFLICT = "field_conflict"


class ReviewItemStatus(StrEnum):
    PENDING = "pending"
    RESOLVED = "resolved"


class ReviewAction(StrEnum):
    APPROVE = "approve"
    CORRECT = "correct"
    SELECT_VALUE = "select_value"
    MARK_UNRESOLVED = "mark_unresolved"


class ExpectedValueType(StrEnum):
    STRING = "string"
    INTEGER = "integer"
    NUMBER = "number"
    BOOLEAN = "boolean"
    DOCUMENT_TYPE = "document_type"


class ConflictCandidate(DomainModel):
    field_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    document_type: DocumentType
    value: FieldValue
    normalized_value: NormalizedValue
    evidence: list[EvidenceReference] = Field(default_factory=list)


class ReviewItem(DomainModel):
    """One conditional review shape, validated according to its item type."""

    review_item_id: str = Field(min_length=1)
    type: ReviewItemType
    status: ReviewItemStatus = ReviewItemStatus.PENDING
    document_id: str | None = None
    proposed_document_type: DocumentType | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    reasoning_summary: str | None = None
    field_id: str | None = None
    field_name: str | None = None
    extracted_value: FieldValue | None = None
    evidence: list[EvidenceReference] = Field(default_factory=list)
    expected_value_type: ExpectedValueType | None = None
    finding_id: str | None = None
    candidates: list[ConflictCandidate] = Field(default_factory=list)
    resolved_by_decision_id: str | None = None

    @model_validator(mode="after")
    def validate_shape(self) -> ReviewItem:
        if self.status is ReviewItemStatus.PENDING and self.resolved_by_decision_id:
            raise ValueError("pending review items cannot reference a decision")
        if self.status is ReviewItemStatus.RESOLVED and not self.resolved_by_decision_id:
            raise ValueError("resolved review items require a decision ID")

        if self.type is ReviewItemType.LOW_CONFIDENCE_CLASSIFICATION:
            if (
                not self.document_id
                or self.proposed_document_type is None
                or self.confidence is None
                or not self.reasoning_summary
            ):
                raise ValueError(
                    "classification review requires document, proposed type, "
                    "confidence, and reasoning"
                )
            if self.expected_value_type not in (
                None,
                ExpectedValueType.DOCUMENT_TYPE,
            ):
                raise ValueError("classification review has document_type value kind")
        elif self.type is ReviewItemType.LOW_CONFIDENCE_FIELD:
            if (
                not self.document_id
                or not self.field_id
                or not self.field_name
                or self.extracted_value is None
                or self.confidence is None
                or self.expected_value_type is None
            ):
                raise ValueError(
                    "field review requires field details, a non-null value, and confidence"
                )
        elif self.type is ReviewItemType.FIELD_CONFLICT:
            if (
                not self.finding_id
                or not self.field_name
                or self.expected_value_type is None
                or len(self.candidates) < 2
            ):
                raise ValueError(
                    "conflict review requires a finding, field, type, and candidates"
                )
            candidate_ids = [candidate.field_id for candidate in self.candidates]
            if len(candidate_ids) != len(set(candidate_ids)):
                raise ValueError("conflict candidate field IDs must be unique")
        return self

    @property
    def allowed_actions(self) -> frozenset[ReviewAction]:
        if self.type is ReviewItemType.LOW_CONFIDENCE_CLASSIFICATION:
            return frozenset({ReviewAction.APPROVE, ReviewAction.CORRECT})
        if self.type is ReviewItemType.LOW_CONFIDENCE_FIELD:
            return frozenset({ReviewAction.APPROVE, ReviewAction.CORRECT})
        return frozenset(
            {
                ReviewAction.SELECT_VALUE,
                ReviewAction.CORRECT,
                ReviewAction.MARK_UNRESOLVED,
            }
        )


class ReviewDecision(DomainModel):
    decision_id: str = Field(
        default_factory=lambda: f"decision-{uuid4().hex}", min_length=1
    )
    review_item_id: str = Field(min_length=1)
    action: ReviewAction
    value: FieldValue | None = None
    selected_field_id: str | None = None
    decided_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @model_validator(mode="after")
    def validate_action_payload(self) -> ReviewDecision:
        if self.decided_at.tzinfo is None or self.decided_at.utcoffset() is None:
            raise ValueError("decided_at must include timezone information")
        if self.action is ReviewAction.CORRECT:
            if self.value is None or self.selected_field_id is not None:
                raise ValueError("correct requires value and forbids selected_field_id")
        elif self.action is ReviewAction.SELECT_VALUE:
            if not self.selected_field_id or self.value is not None:
                raise ValueError(
                    "select_value requires selected_field_id and forbids value"
                )
        elif self.value is not None or self.selected_field_id is not None:
            raise ValueError(f"{self.action.value} does not accept a value selection")
        return self


class ReviewDecisionAudit(DomainModel):
    """Immutable data suitable for repository audit persistence."""

    decision: ReviewDecision
    review_item_type: ReviewItemType
    original_values: list[FieldValue] = Field(default_factory=list)
    effective_value: FieldValue | None = None


class VerifiedReview(DomainModel):
    """The only review input accepted by report composition."""

    review_id: str = Field(min_length=1)
    company_name: str | None = None
    status: ReportStatus
    classifications: list[DocumentClassification] = Field(default_factory=list)
    extractions: list[ExtractionResult] = Field(default_factory=list)
    effective_fields: list[EffectiveFieldValue] = Field(default_factory=list)
    findings: list[Finding] = Field(default_factory=list)
    review_decisions: list[ReviewDecision] = Field(default_factory=list)
    pending_review_items: list[ReviewItem] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_unique_ids(self) -> VerifiedReview:
        for label, values in (
            ("classification document", [item.document_id for item in self.classifications]),
            ("extraction document", [item.document_id for item in self.extractions]),
            ("effective field", [item.field_id for item in self.effective_fields]),
            ("finding", [item.finding_id for item in self.findings]),
            ("review decision", [item.decision_id for item in self.review_decisions]),
            (
                "pending review item",
                [item.review_item_id for item in self.pending_review_items],
            ),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"{label} IDs must be unique")
        if any(
            item.status is not ReviewItemStatus.PENDING
            for item in self.pending_review_items
        ):
            raise ValueError("pending_review_items may contain only pending items")
        return self

    @property
    def extraction_results(self) -> list[ExtractionResult]:
        """Compatibility spelling for workflow state and capability code."""

        return self.extractions

    @property
    def decisions(self) -> list[ReviewDecision]:
        return self.review_decisions
