from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.domain import (
    DocumentClassification,
    DocumentType,
    EvidenceReference,
    ExtractedField,
    ExtractionResult,
    PageContent,
    ProcessedDocument,
    ProcessorMetadata,
)


def test_page_numbers_and_confidence_are_validated() -> None:
    with pytest.raises(ValidationError, match="greater than or equal to 1"):
        PageContent(page_number=0, text="text")
    with pytest.raises(ValidationError, match="less than or equal to 1"):
        DocumentClassification(
            document_id="document-1",
            document_type=DocumentType.APPLICATION_FORM,
            confidence=1.01,
            reasoning_summary="Looks like a form.",
        )


def test_processed_document_requires_unique_ordered_pages() -> None:
    metadata = ProcessorMetadata(processor="test")
    with pytest.raises(ValidationError, match="ordered"):
        ProcessedDocument(
            document_id="document-1",
            filename="document.pdf",
            pages=[
                PageContent(page_number=2, text="second"),
                PageContent(page_number=1, text="first"),
            ],
            processor_metadata=metadata,
        )
    with pytest.raises(ValidationError, match="unique"):
        ProcessedDocument(
            document_id="document-1",
            filename="document.pdf",
            pages=[
                PageContent(page_number=1, text="first"),
                PageContent(page_number=1, text="again"),
            ],
            processor_metadata=metadata,
        )


def test_non_null_field_requires_same_document_evidence() -> None:
    with pytest.raises(ValidationError, match="requires evidence"):
        ExtractedField(
            field_id="document-1:company_name",
            document_id="document-1",
            field_name="company_name",
            value="Acme BV",
            confidence=0.9,
        )
    with pytest.raises(ValidationError, match="field document"):
        ExtractedField(
            field_id="document-1:company_name",
            document_id="document-1",
            field_name="company_name",
            value="Acme BV",
            confidence=0.9,
            evidence=[
                EvidenceReference(
                    document_id="document-2",
                    page_number=1,
                    source_text="Acme BV",
                )
            ],
        )


def test_extraction_rejects_fields_outside_document_schema() -> None:
    field = ExtractedField(
        field_id="extract:annual_revenue_eur",
        document_id="extract",
        field_name="annual_revenue_eur",
        value=10,
        confidence=0.9,
        evidence=[
            EvidenceReference(
                document_id="extract", page_number=1, source_text="Revenue 10"
            )
        ],
    )
    with pytest.raises(ValidationError, match="not valid for company_extract"):
        ExtractionResult(
            document_id="extract",
            document_type=DocumentType.COMPANY_EXTRACT,
            fields=[field],
        )


def test_classification_override_preserves_raw_model_value() -> None:
    classification = DocumentClassification(
        document_id="document-1",
        document_type=DocumentType.UNKNOWN,
        confidence=0.4,
        reasoning_summary="Ambiguous.",
        effective_document_type=DocumentType.APPLICATION_FORM,
        was_human_reviewed=True,
        review_decision_id="decision-1",
    )
    assert classification.document_type is DocumentType.UNKNOWN
    assert classification.resolved_document_type is DocumentType.APPLICATION_FORM
