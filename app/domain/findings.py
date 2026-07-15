"""Deterministic finding contracts."""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field, model_validator

from app.domain.base import DomainModel
from app.domain.documents import DocumentType
from app.domain.extractions import FieldValue, NormalizedValue


class FindingType(StrEnum):
    MISSING_DOCUMENT = "missing_document"
    MISSING_REQUIRED_FIELD = "missing_required_field"
    FIELD_CONFLICT = "field_conflict"


class FindingSeverity(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class FindingEvidence(DomainModel):
    """A compact, serialisable view of a conflicting field value."""

    field_id: str | None = None
    document_id: str = Field(min_length=1)
    document_type: DocumentType | None = None
    value: FieldValue | None = None
    normalized_value: NormalizedValue | None = None
    page_number: int | None = Field(default=None, ge=1)
    source_text: str | None = Field(default=None, max_length=1_000)


class Finding(DomainModel):
    """A fact produced by deterministic validation, never by report generation."""

    finding_id: str = Field(min_length=1)
    type: FindingType
    severity: FindingSeverity
    message: str = Field(min_length=1, max_length=2_000)
    field_name: str | None = None
    document_id: str | None = None
    document_type: DocumentType | None = None
    evidence: list[FindingEvidence] = Field(default_factory=list)
    resolved: bool = False
    resolution: str | None = None

    @model_validator(mode="after")
    def validate_shape(self) -> Finding:
        if self.type is FindingType.MISSING_DOCUMENT:
            if self.document_type is None:
                raise ValueError("missing-document findings require document_type")
        elif self.type is FindingType.MISSING_REQUIRED_FIELD:
            if self.document_id is None or self.document_type is None or not self.field_name:
                raise ValueError(
                    "missing-field findings require document_id, document_type, and field_name"
                )
        elif self.type is FindingType.FIELD_CONFLICT and (
            not self.field_name or len(self.evidence) < 2
        ):
            raise ValueError(
                "field-conflict findings require field_name and at least two values"
            )
        if self.resolved and not self.resolution:
            raise ValueError("resolved findings require a resolution")
        if not self.resolved and self.resolution is not None:
            raise ValueError("unresolved findings cannot include a resolution")
        return self
