from __future__ import annotations

from collections.abc import Sequence

from app.domain import (
    DocumentClassification,
    DocumentType,
    EvidenceReference,
    ExtractedField,
    ExtractionResult,
    FindingType,
    ReviewItemType,
)
from app.review import (
    build_effective_fields,
    create_classification_review_items,
    create_conflict_review_items,
    create_field_review_items,
    load_review_rules,
    validate_cross_document_fields,
    validate_required_documents,
    validate_required_fields,
)


def classification(
    document_id: str, document_type: DocumentType, confidence: float = 0.9
) -> DocumentClassification:
    return DocumentClassification(
        document_id=document_id,
        document_type=document_type,
        confidence=confidence,
        reasoning_summary="Synthetic test classification.",
    )


def field(
    document_id: str,
    field_name: str,
    value: str | int | float | bool | None,
    confidence: float = 0.9,
) -> ExtractedField:
    evidence = (
        []
        if value is None
        else [
            EvidenceReference(
                document_id=document_id,
                page_number=1,
                source_text=f"{field_name}: {value}",
            )
        ]
    )
    return ExtractedField(
        field_id=f"{document_id}:{field_name}",
        document_id=document_id,
        field_name=field_name,
        value=value,
        confidence=confidence,
        evidence=evidence,
    )


def extraction(
    document_id: str,
    document_type: DocumentType,
    fields: Sequence[ExtractedField],
) -> ExtractionResult:
    return ExtractionResult(
        document_id=document_id, document_type=document_type, fields=list(fields)
    )


def complete_extractions() -> list[ExtractionResult]:
    return [
        extraction(
            "application",
            DocumentType.APPLICATION_FORM,
            [
                field("application", "company_name", "Acme B.V."),
                field("application", "registration_number", "NL-00123"),
                field("application", "annual_revenue_eur", 99),
                field("application", "employee_count", 42),
            ],
        ),
        extraction(
            "extract",
            DocumentType.COMPANY_EXTRACT,
            [
                field("extract", "company_name", "ACME BV"),
                field("extract", "registration_number", "NL00123"),
                field("extract", "incorporation_date", "2020-01-01"),
            ],
        ),
        extraction(
            "financial",
            DocumentType.FINANCIAL_STATEMENT,
            [
                field("financial", "company_name", "Acme, B.V."),
                field("financial", "annual_revenue_eur", 101),
                field("financial", "reporting_year", 2025),
                field("financial", "employee_count", 42),
            ],
        ),
    ]


def test_required_document_and_field_findings_are_high_severity() -> None:
    rules = load_review_rules()
    classifications = [
        classification("application", DocumentType.APPLICATION_FORM),
        classification("extract", DocumentType.COMPANY_EXTRACT),
    ]
    missing_documents = validate_required_documents(classifications, rules)
    assert [finding.finding_id for finding in missing_documents] == [
        "finding-missing-financial-statement"
    ]

    incomplete = extraction(
        "application",
        DocumentType.APPLICATION_FORM,
        [
            field("application", "company_name", "Acme BV"),
            field("application", "registration_number", None),
        ],
    )
    missing_fields = validate_required_fields([incomplete], rules)
    assert {finding.field_name for finding in missing_fields} == {
        "registration_number",
        "annual_revenue_eur",
        "employee_count",
    }
    assert all(finding.type is FindingType.MISSING_REQUIRED_FIELD for finding in missing_fields)


def test_normalized_and_tolerated_values_do_not_conflict() -> None:
    rules = load_review_rules()
    effective = build_effective_fields(complete_extractions())
    assert validate_cross_document_fields(effective, rules) == []


def test_divergent_duplicate_values_collapse_to_one_grouped_conflict() -> None:
    rules = load_review_rules()
    extractions = complete_extractions()
    extractions.append(
        extraction(
            "application-copy",
            DocumentType.APPLICATION_FORM,
            [
                field("application-copy", "company_name", "Acme BV"),
                field("application-copy", "registration_number", "NL00999"),
                field("application-copy", "annual_revenue_eur", 99),
                field("application-copy", "employee_count", 42),
            ],
        )
    )
    effective = build_effective_fields(extractions)
    findings = validate_cross_document_fields(effective, rules)
    registration_findings = [
        finding for finding in findings if finding.field_name == "registration_number"
    ]
    assert len(registration_findings) == 1
    assert len(registration_findings[0].evidence) == 3

    items = create_conflict_review_items(findings, effective)
    assert len(items) == 1
    assert items[0].type is ReviewItemType.FIELD_CONFLICT
    assert len(items[0].candidates) == 3


def test_confidence_threshold_is_strict_and_null_fields_are_not_routed() -> None:
    rules = load_review_rules()
    classifications = [
        classification("at-threshold", DocumentType.APPLICATION_FORM, 0.70),
        classification("below", DocumentType.APPLICATION_FORM, 0.699),
    ]
    items = create_classification_review_items(classifications, rules)
    assert [item.document_id for item in items] == ["below"]

    result = extraction(
        "financial",
        DocumentType.FINANCIAL_STATEMENT,
        [
            field("financial", "company_name", "Acme BV", 0.75),
            field("financial", "annual_revenue_eur", 100, 0.749),
            field("financial", "reporting_year", None, 0.1),
        ],
    )
    field_items = create_field_review_items(build_effective_fields([result]), rules)
    assert [item.field_name for item in field_items] == ["annual_revenue_eur"]
