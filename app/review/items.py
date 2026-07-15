"""Deterministic creation of typed human-review items."""

from __future__ import annotations

from collections.abc import Sequence

from app.domain.extractions import (
    DocumentClassification,
    EffectiveFieldValue,
    EffectiveValueSource,
)
from app.domain.findings import Finding, FindingType
from app.domain.reviews import (
    ConflictCandidate,
    ExpectedValueType,
    ReviewItem,
    ReviewItemType,
)
from app.review.config import ReviewRules

_EXPECTED_VALUE_TYPES: dict[str, ExpectedValueType] = {
    "company_name": ExpectedValueType.STRING,
    "registration_number": ExpectedValueType.STRING,
    "annual_revenue_eur": ExpectedValueType.NUMBER,
    "employee_count": ExpectedValueType.INTEGER,
    "incorporation_date": ExpectedValueType.STRING,
    "reporting_year": ExpectedValueType.INTEGER,
}


def expected_value_type(field_name: str) -> ExpectedValueType:
    return _EXPECTED_VALUE_TYPES.get(field_name, ExpectedValueType.STRING)


def create_classification_review_items(
    classifications: Sequence[DocumentClassification], rules: ReviewRules
) -> list[ReviewItem]:
    """Route only values strictly below the configured threshold."""

    threshold = rules.confidence.classification_review_threshold
    return [
        ReviewItem(
            review_item_id=f"review-classification-{classification.document_id}",
            type=ReviewItemType.LOW_CONFIDENCE_CLASSIFICATION,
            document_id=classification.document_id,
            proposed_document_type=classification.document_type,
            confidence=classification.confidence,
            reasoning_summary=classification.reasoning_summary,
            expected_value_type=ExpectedValueType.DOCUMENT_TYPE,
        )
        for classification in classifications
        if classification.confidence < threshold
        and not classification.was_human_reviewed
    ]


def create_field_review_items(
    effective_fields: Sequence[EffectiveFieldValue], rules: ReviewRules
) -> list[ReviewItem]:
    """Create review items for non-null, unreviewed low-confidence values."""

    threshold = rules.confidence.field_review_threshold
    return [
        ReviewItem(
            review_item_id=f"review-field-{field.field_id}",
            type=ReviewItemType.LOW_CONFIDENCE_FIELD,
            document_id=field.document_id,
            field_id=field.field_id,
            field_name=field.field_name,
            extracted_value=field.effective_value,
            confidence=field.confidence,
            evidence=field.evidence,
            expected_value_type=expected_value_type(field.field_name),
        )
        for field in effective_fields
        if field.effective_value is not None
        and field.confidence < threshold
        and field.value_source is EffectiveValueSource.EXTRACTED
    ]


def create_conflict_review_items(
    findings: Sequence[Finding],
    effective_fields: Sequence[EffectiveFieldValue],
) -> list[ReviewItem]:
    fields_by_id = {field.field_id: field for field in effective_fields}
    items: list[ReviewItem] = []
    for finding in findings:
        if finding.type is not FindingType.FIELD_CONFLICT or finding.resolved:
            continue
        candidates: list[ConflictCandidate] = []
        for evidence in finding.evidence:
            if evidence.field_id is None:
                continue
            field = fields_by_id.get(evidence.field_id)
            if (
                field is None
                or field.effective_value is None
                or field.normalized_value is None
            ):
                continue
            candidates.append(
                ConflictCandidate(
                    field_id=field.field_id,
                    document_id=field.document_id,
                    document_type=field.document_type,
                    value=field.effective_value,
                    normalized_value=field.normalized_value,
                    evidence=field.evidence,
                )
            )
        if len(candidates) < 2:
            raise ValueError(
                f"conflict finding {finding.finding_id} has fewer than two live candidates"
            )
        assert finding.field_name is not None
        items.append(
            ReviewItem(
                review_item_id=f"review-conflict-{finding.finding_id}",
                type=ReviewItemType.FIELD_CONFLICT,
                field_name=finding.field_name,
                expected_value_type=expected_value_type(finding.field_name),
                finding_id=finding.finding_id,
                candidates=candidates,
            )
        )
    return items


def create_review_items(
    classifications: Sequence[DocumentClassification],
    effective_fields: Sequence[EffectiveFieldValue],
    findings: Sequence[Finding],
    rules: ReviewRules,
) -> list[ReviewItem]:
    """Convenience function for state snapshots and model-free evaluation."""

    items = [
        *create_classification_review_items(classifications, rules),
        *create_field_review_items(effective_fields, rules),
        *create_conflict_review_items(findings, effective_fields),
    ]
    ids = [item.review_item_id for item in items]
    if len(ids) != len(set(ids)):
        raise ValueError("review-item IDs must be unique")
    return items


def assess_confidence_and_review(
    classifications: Sequence[DocumentClassification],
    effective_fields: Sequence[EffectiveFieldValue],
    rules: ReviewRules,
) -> list[ReviewItem]:
    return [
        *create_classification_review_items(classifications, rules),
        *create_field_review_items(effective_fields, rules),
    ]
