from __future__ import annotations

import pytest

from app.domain import (
    DocumentClassification,
    DocumentType,
    EffectiveFieldValue,
    EffectiveValueSource,
    EvidenceReference,
    ExpectedValueType,
    Finding,
    FindingEvidence,
    FindingSeverity,
    FindingType,
    PolicyEvidence,
    ReportSection,
    ReportStatus,
    ReviewAction,
    ReviewDecision,
    ReviewItem,
    ReviewItemType,
    ReviewReport,
)
from app.review import (
    ReportReferenceError,
    ReviewDecisionValidationError,
    apply_review_decisions,
    build_verified_review,
    determine_report_status,
    finalize_report,
    validate_review_decisions,
)


def effective_field(
    field_id: str,
    document_id: str,
    value: str,
) -> EffectiveFieldValue:
    return EffectiveFieldValue(
        field_id=field_id,
        document_id=document_id,
        document_type=DocumentType.COMPANY_EXTRACT,
        field_name="registration_number",
        original_value=value,
        normalized_original_value=value.replace("-", ""),
        effective_value=value,
        normalized_value=value.replace("-", ""),
        confidence=0.95,
        evidence=[
            EvidenceReference(
                document_id=document_id,
                page_number=1,
                source_text=f"Registration {value}",
            )
        ],
    )


def low_confidence_item() -> ReviewItem:
    return ReviewItem(
        review_item_id="review-field-financial:employee_count",
        type=ReviewItemType.LOW_CONFIDENCE_FIELD,
        document_id="financial",
        field_id="financial:employee_count",
        field_name="employee_count",
        extracted_value=42,
        confidence=0.6,
        evidence=[
            EvidenceReference(
                document_id="financial", page_number=1, source_text="42 employees"
            )
        ],
        expected_value_type=ExpectedValueType.INTEGER,
    )


def test_decision_batch_requires_exactly_one_valid_decision_per_item() -> None:
    item = low_confidence_item()
    with pytest.raises(ReviewDecisionValidationError, match="every pending"):
        validate_review_decisions([item], [])
    with pytest.raises(ReviewDecisionValidationError, match="invalid for"):
        validate_review_decisions(
            [item],
            [
                ReviewDecision(
                    review_item_id=item.review_item_id,
                    action=ReviewAction.MARK_UNRESOLVED,
                )
            ],
        )
    with pytest.raises(ReviewDecisionValidationError, match="integer"):
        validate_review_decisions(
            [item],
            [
                ReviewDecision(
                    review_item_id=item.review_item_id,
                    action=ReviewAction.CORRECT,
                    value="forty-two",
                )
            ],
        )


def test_low_confidence_correction_preserves_raw_value() -> None:
    item = low_confidence_item()
    raw = EffectiveFieldValue(
        field_id="financial:employee_count",
        document_id="financial",
        document_type=DocumentType.FINANCIAL_STATEMENT,
        field_name="employee_count",
        original_value=42,
        normalized_original_value=42,
        effective_value=42,
        normalized_value=42,
        confidence=0.6,
        evidence=item.evidence,
    )
    decision = ReviewDecision(
        review_item_id=item.review_item_id,
        action=ReviewAction.CORRECT,
        value=40,
    )
    applied = apply_review_decisions([], [raw], [item], [decision])
    corrected = applied.effective_fields[0]
    assert corrected.original_value == 42
    assert corrected.normalized_original_value == 42
    assert corrected.effective_value == 40
    assert corrected.value_source is EffectiveValueSource.CORRECTED
    assert applied.audit_records[0].original_values == [42]


def test_classification_correction_preserves_raw_type() -> None:
    classification = DocumentClassification(
        document_id="document",
        document_type=DocumentType.UNKNOWN,
        confidence=0.5,
        reasoning_summary="Ambiguous.",
    )
    item = ReviewItem(
        review_item_id="review-classification-document",
        type=ReviewItemType.LOW_CONFIDENCE_CLASSIFICATION,
        document_id="document",
        proposed_document_type=DocumentType.UNKNOWN,
        confidence=0.5,
        reasoning_summary="Ambiguous.",
        expected_value_type=ExpectedValueType.DOCUMENT_TYPE,
    )
    decision = ReviewDecision(
        review_item_id=item.review_item_id,
        action=ReviewAction.CORRECT,
        value=DocumentType.APPLICATION_FORM.value,
    )
    applied = apply_review_decisions([classification], [], [item], [decision])
    updated = applied.classifications[0]
    assert updated.document_type is DocumentType.UNKNOWN
    assert updated.resolved_document_type is DocumentType.APPLICATION_FORM


def test_conflict_selection_changes_effective_values_only() -> None:
    left = effective_field("application:registration", "application", "001-23")
    right = effective_field("extract:registration", "extract", "009-99")
    item = ReviewItem(
        review_item_id="review-conflict-registration",
        type=ReviewItemType.FIELD_CONFLICT,
        field_name="registration_number",
        finding_id="finding-registration-number-conflict",
        expected_value_type=ExpectedValueType.STRING,
        candidates=[
            {
                "field_id": left.field_id,
                "document_id": left.document_id,
                "document_type": left.document_type,
                "value": left.effective_value,
                "normalized_value": left.normalized_value,
                "evidence": left.evidence,
            },
            {
                "field_id": right.field_id,
                "document_id": right.document_id,
                "document_type": right.document_type,
                "value": right.effective_value,
                "normalized_value": right.normalized_value,
                "evidence": right.evidence,
            },
        ],
    )
    decision = ReviewDecision(
        review_item_id=item.review_item_id,
        action=ReviewAction.SELECT_VALUE,
        selected_field_id=left.field_id,
    )
    applied = apply_review_decisions([], [left, right], [item], [decision])
    assert [value.original_value for value in applied.effective_fields] == [
        "001-23",
        "009-99",
    ]
    assert [value.effective_value for value in applied.effective_fields] == [
        "001-23",
        "001-23",
    ]


def test_report_status_precedence_and_reference_guard() -> None:
    conflict = Finding(
        finding_id="finding-conflict",
        type=FindingType.FIELD_CONFLICT,
        severity=FindingSeverity.HIGH,
        message="Values conflict.",
        field_name="registration_number",
        evidence=[
            FindingEvidence(document_id="a", value="1"),
            FindingEvidence(document_id="b", value="2"),
        ],
    )
    missing = Finding(
        finding_id="finding-missing-financial-statement",
        type=FindingType.MISSING_DOCUMENT,
        severity=FindingSeverity.HIGH,
        message="Missing.",
        document_type=DocumentType.FINANCIAL_STATEMENT,
    )
    assert determine_report_status([conflict]) is ReportStatus.NEEDS_FOLLOW_UP
    assert determine_report_status([conflict, missing]) is ReportStatus.INCOMPLETE
    assert determine_report_status([]) is ReportStatus.COMPLETE

    review = build_verified_review(
        review_id="review",
        classifications=[],
        extractions=[],
        effective_fields=[],
        findings=[conflict],
    )
    evidence = PolicyEvidence(
        evidence_id="policy-evidence-1",
        policy_id="manual-review-policy",
        title="Manual Review Policy",
        section_id="2.1",
        text="Conflicts require review.",
        score=0.9,
        source_path="policies/manual-review-policy.md",
    )
    report = ReviewReport(
        company_name="Invented Company",
        status=ReportStatus.COMPLETE,
        executive_summary="A conflict needs follow-up.",
        sections=[
            ReportSection(
                title="Conflict",
                summary="Registration values conflict.",
                finding_ids=[conflict.finding_id],
                policy_evidence_ids=[evidence.evidence_id],
            )
        ],
    )
    finalized = finalize_report(report, review, [evidence])
    assert finalized.company_name is None
    assert finalized.status is ReportStatus.NEEDS_FOLLOW_UP

    invalid = report.model_copy(
        update={
            "sections": [
                ReportSection(
                    title="Invented",
                    summary="Unsupported.",
                    finding_ids=["finding-invented"],
                )
            ]
        }
    )
    with pytest.raises(ReportReferenceError) as exc_info:
        finalize_report(invalid, review, [evidence])
    assert exc_info.value.unknown_finding_ids == {"finding-invented"}
