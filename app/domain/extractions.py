"""Classification, extraction, provenance, and effective-value contracts."""

from __future__ import annotations

from decimal import Decimal
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import Field, model_validator

from app.domain.base import DomainModel
from app.domain.documents import DocumentType

type FieldValue = str | int | float | bool
type NormalizedValue = str | int | Decimal | bool


class EvidenceReference(DomainModel):
    """An exact source span supporting an extracted value."""

    document_id: str = Field(min_length=1)
    page_number: int = Field(ge=1)
    source_text: str = Field(min_length=1, max_length=1_000)


class DocumentClassification(DomainModel):
    """Raw classifier output plus a separately stored human override, if any."""

    document_id: str = Field(min_length=1)
    document_type: DocumentType
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning_summary: str = Field(min_length=1, max_length=1_000)
    effective_document_type: DocumentType | None = None
    was_human_reviewed: bool = False
    review_decision_id: str | None = None

    @model_validator(mode="after")
    def validate_review_metadata(self) -> DocumentClassification:
        if self.was_human_reviewed and self.effective_document_type is None:
            raise ValueError(
                "effective_document_type is required after classification review"
            )
        if self.was_human_reviewed and self.review_decision_id is None:
            raise ValueError("review_decision_id is required after classification review")
        if not self.was_human_reviewed and (
            self.effective_document_type is not None
            or self.review_decision_id is not None
        ):
            raise ValueError("classification review metadata requires was_human_reviewed")
        return self

    @property
    def resolved_document_type(self) -> DocumentType:
        """The type downstream processing must use without changing raw output."""

        return self.effective_document_type or self.document_type


class ExtractedField(DomainModel):
    """One model-extracted field with immutable raw value and provenance."""

    field_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    field_name: str = Field(min_length=1)
    value: FieldValue | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[EvidenceReference] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_evidence(self) -> ExtractedField:
        if self.value is not None and not self.evidence:
            raise ValueError("a non-null extracted value requires evidence")
        if any(item.document_id != self.document_id for item in self.evidence):
            raise ValueError("field evidence must belong to the field document")
        return self


class ClarificationStatement(DomainModel):
    """A relevant statement extracted from supporting correspondence."""

    statement_id: str = Field(min_length=1)
    topic: str = Field(min_length=1, max_length=200)
    text: str = Field(min_length=1, max_length=2_000)
    value: FieldValue | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[EvidenceReference] = Field(min_length=1)


FIELDS_BY_DOCUMENT_TYPE: dict[DocumentType, frozenset[str]] = {
    DocumentType.APPLICATION_FORM: frozenset(
        {
            "company_name",
            "registration_number",
            "annual_revenue_eur",
            "employee_count",
        }
    ),
    DocumentType.COMPANY_EXTRACT: frozenset(
        {"company_name", "registration_number", "incorporation_date"}
    ),
    DocumentType.FINANCIAL_STATEMENT: frozenset(
        {"company_name", "annual_revenue_eur", "reporting_year", "employee_count"}
    ),
    DocumentType.SUPPORTING_CORRESPONDENCE: frozenset({"company_name"}),
    DocumentType.UNKNOWN: frozenset(),
}


class ExtractionResult(DomainModel):
    """Fields extracted from one classified document."""

    document_id: str = Field(min_length=1)
    document_type: DocumentType
    fields: list[ExtractedField] = Field(default_factory=list)
    clarification_statements: list[ClarificationStatement] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_result(self) -> ExtractionResult:
        field_ids = [field.field_id for field in self.fields]
        if len(field_ids) != len(set(field_ids)):
            raise ValueError("field_id values must be unique within an extraction")
        field_names = [field.field_name for field in self.fields]
        if len(field_names) != len(set(field_names)):
            raise ValueError("field_name values must be unique within an extraction")
        statement_ids = [
            statement.statement_id for statement in self.clarification_statements
        ]
        if len(statement_ids) != len(set(statement_ids)):
            raise ValueError("statement_id values must be unique within an extraction")

        allowed = FIELDS_BY_DOCUMENT_TYPE[self.document_type]
        for field in self.fields:
            if field.document_id != self.document_id:
                raise ValueError("extracted fields must belong to the result document")
            if field.field_name not in allowed:
                raise ValueError(
                    f"field {field.field_name!r} is not valid for {self.document_type.value}"
                )

        if (
            self.clarification_statements
            and self.document_type is not DocumentType.SUPPORTING_CORRESPONDENCE
        ):
            raise ValueError(
                "clarification statements are only valid for supporting correspondence"
            )
        for statement in self.clarification_statements:
            if any(
                evidence.document_id != self.document_id
                for evidence in statement.evidence
            ):
                raise ValueError(
                    "clarification evidence must belong to the result document"
                )
        return self


class ApplicationFormExtraction(ExtractionResult):
    document_type: Literal[DocumentType.APPLICATION_FORM] = (
        DocumentType.APPLICATION_FORM
    )


class CompanyExtractExtraction(ExtractionResult):
    document_type: Literal[DocumentType.COMPANY_EXTRACT] = DocumentType.COMPANY_EXTRACT


class FinancialStatementExtraction(ExtractionResult):
    document_type: Literal[DocumentType.FINANCIAL_STATEMENT] = (
        DocumentType.FINANCIAL_STATEMENT
    )


class SupportingCorrespondenceExtraction(ExtractionResult):
    document_type: Literal[DocumentType.SUPPORTING_CORRESPONDENCE] = (
        DocumentType.SUPPORTING_CORRESPONDENCE
    )


class UnknownDocumentExtraction(ExtractionResult):
    document_type: Literal[DocumentType.UNKNOWN] = DocumentType.UNKNOWN


type TypedExtractionResult = Annotated[
    ApplicationFormExtraction
    | CompanyExtractExtraction
    | FinancialStatementExtraction
    | SupportingCorrespondenceExtraction
    | UnknownDocumentExtraction,
    Field(discriminator="document_type"),
]


class EffectiveValueSource(StrEnum):
    EXTRACTED = "extracted"
    APPROVED = "approved"
    CORRECTED = "corrected"
    SELECTED = "selected"


class EffectiveFieldValue(DomainModel):
    """A normalized working value that never replaces the original extraction."""

    field_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    document_type: DocumentType
    field_name: str = Field(min_length=1)
    original_value: FieldValue | None = None
    normalized_original_value: NormalizedValue | None = None
    effective_value: FieldValue | None = None
    normalized_value: NormalizedValue | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[EvidenceReference] = Field(default_factory=list)
    value_source: EffectiveValueSource = EffectiveValueSource.EXTRACTED
    review_decision_id: str | None = None

    @model_validator(mode="after")
    def validate_override_metadata(self) -> EffectiveFieldValue:
        if self.value_source is EffectiveValueSource.EXTRACTED:
            if self.review_decision_id is not None:
                raise ValueError("raw extracted values cannot have a review decision")
        elif self.review_decision_id is None:
            raise ValueError("reviewed effective values require review_decision_id")
        return self
