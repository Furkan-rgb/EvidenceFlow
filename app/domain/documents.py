"""Document contracts used by upload, processing, and AI capabilities."""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field, model_validator

from app.domain.base import DomainModel


class DocumentType(StrEnum):
    """The complete set of document types supported by EvidenceFlow V1."""

    APPLICATION_FORM = "application_form"
    COMPANY_EXTRACT = "company_extract"
    FINANCIAL_STATEMENT = "financial_statement"
    SUPPORTING_CORRESPONDENCE = "supporting_correspondence"
    UNKNOWN = "unknown"


class UploadedDocument(DomainModel):
    """Safe reference to an uploaded artifact; never contains a filesystem path."""

    document_id: str = Field(min_length=1)
    filename: str = Field(min_length=1)
    artifact_id: str = Field(min_length=1)
    content_type: str = "application/pdf"
    size_bytes: int = Field(ge=0)
    sha256: str | None = Field(default=None, min_length=1)


class PageContent(DomainModel):
    """Text extracted from one user-facing, 1-based PDF page."""

    page_number: int = Field(ge=1)
    text: str


class ProcessorMetadata(DomainModel):
    """Identifies the deterministic document processor implementation."""

    processor: str = Field(min_length=1)
    version: str | None = None


class ProcessedDocument(DomainModel):
    """Provider-neutral text representation consumed by AI capabilities."""

    document_id: str = Field(min_length=1)
    filename: str = Field(min_length=1)
    pages: list[PageContent] = Field(min_length=1)
    processor_metadata: ProcessorMetadata

    @model_validator(mode="after")
    def validate_page_order(self) -> ProcessedDocument:
        numbers = [page.page_number for page in self.pages]
        if numbers != sorted(numbers):
            raise ValueError("pages must be ordered by page_number")
        if len(numbers) != len(set(numbers)):
            raise ValueError("page_number values must be unique")
        return self
