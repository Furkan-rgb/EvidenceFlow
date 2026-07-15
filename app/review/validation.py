"""Deterministic completeness and cross-document validation."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from decimal import Decimal
from itertools import combinations

from app.domain.extractions import (
    DocumentClassification,
    EffectiveFieldValue,
    EffectiveValueSource,
    ExtractionResult,
    NormalizedValue,
)
from app.domain.findings import (
    Finding,
    FindingEvidence,
    FindingSeverity,
    FindingType,
)
from app.review.config import (
    ExactMatchRule,
    NormalisedMatchRule,
    NumericToleranceRule,
    ReviewRules,
)
from app.review.normalization import (
    normalize_field_value,
    within_symmetric_percent_tolerance,
)


def _slug(value: str) -> str:
    return "".join(character if character.isalnum() else "-" for character in value).strip(
        "-"
    )


def build_effective_fields(
    extractions: Sequence[ExtractionResult],
) -> list[EffectiveFieldValue]:
    """Create normalized working values while preserving every raw value."""

    result: list[EffectiveFieldValue] = []
    seen: set[str] = set()
    for extraction in extractions:
        for field in extraction.fields:
            if field.field_id in seen:
                raise ValueError(f"duplicate field_id across extractions: {field.field_id}")
            seen.add(field.field_id)
            normalized = (
                None
                if field.value is None
                else normalize_field_value(field.field_name, field.value)
            )
            result.append(
                EffectiveFieldValue(
                    field_id=field.field_id,
                    document_id=field.document_id,
                    document_type=extraction.document_type,
                    field_name=field.field_name,
                    original_value=field.value,
                    normalized_original_value=normalized,
                    effective_value=field.value,
                    normalized_value=normalized,
                    confidence=field.confidence,
                    evidence=field.evidence,
                    value_source=EffectiveValueSource.EXTRACTED,
                )
            )
    return result


def normalise_extractions(
    extractions: Sequence[ExtractionResult],
) -> list[EffectiveFieldValue]:
    return build_effective_fields(extractions)


def validate_required_documents(
    classifications: Sequence[DocumentClassification],
    rules: ReviewRules,
) -> list[Finding]:
    present = {item.resolved_document_type for item in classifications}
    return [
        Finding(
            finding_id=f"finding-missing-{document_type.value.replace('_', '-')}",
            type=FindingType.MISSING_DOCUMENT,
            severity=FindingSeverity.HIGH,
            message=f"Required document is missing: {document_type.value}.",
            document_type=document_type,
        )
        for document_type in rules.required_documents
        if document_type not in present
    ]


def validate_required_fields(
    extractions: Sequence[ExtractionResult],
    rules: ReviewRules,
    effective_fields: Sequence[EffectiveFieldValue] | None = None,
) -> list[Finding]:
    """Validate every recognized document, including every duplicate instance."""

    effective_fields = (
        list(effective_fields)
        if effective_fields is not None
        else build_effective_fields(extractions)
    )
    by_document: dict[str, dict[str, list[EffectiveFieldValue]]] = {}
    for field in effective_fields:
        by_document.setdefault(field.document_id, {}).setdefault(
            field.field_name, []
        ).append(field)

    findings: list[Finding] = []
    for extraction in extractions:
        required = rules.required_fields.get(extraction.document_type, [])
        fields = by_document.get(extraction.document_id, {})
        for field_name in required:
            values = fields.get(field_name, [])
            if any(value.effective_value is not None for value in values):
                continue
            findings.append(
                Finding(
                    finding_id=(
                        f"finding-missing-{_slug(extraction.document_id)}-"
                        f"{field_name.replace('_', '-')}"
                    ),
                    type=FindingType.MISSING_REQUIRED_FIELD,
                    severity=FindingSeverity.HIGH,
                    message=(
                        f"Required field {field_name} is missing from "
                        f"{extraction.document_type.value} ({extraction.document_id})."
                    ),
                    field_name=field_name,
                    document_id=extraction.document_id,
                    document_type=extraction.document_type,
                )
            )
    return findings


def _all_equal(values: Sequence[NormalizedValue]) -> bool:
    return all(left == right for left, right in combinations(values, 2))


def _matches_rule(
    values: Sequence[NormalizedValue],
    rule: ExactMatchRule | NormalisedMatchRule | NumericToleranceRule,
) -> bool:
    if isinstance(rule, (ExactMatchRule, NormalisedMatchRule)):
        return _all_equal(values)

    decimals: list[Decimal] = []
    try:
        decimals = [
            value if isinstance(value, Decimal) else Decimal(str(value))
            for value in values
        ]
    except Exception as exc:  # normalized numeric rules should make this unreachable
        raise ValueError(f"non-numeric value for numeric rule {rule.field}") from exc

    if rule.tolerance_percent is not None:
        return all(
            within_symmetric_percent_tolerance(
                left, right, Decimal(str(rule.tolerance_percent))
            )
            for left, right in combinations(decimals, 2)
        )
    assert rule.tolerance_absolute is not None
    tolerance = Decimal(str(rule.tolerance_absolute))
    return all(abs(left - right) <= tolerance for left, right in combinations(decimals, 2))


def _finding_evidence(field: EffectiveFieldValue) -> FindingEvidence:
    evidence = field.evidence[0] if field.evidence else None
    return FindingEvidence(
        field_id=field.field_id,
        document_id=field.document_id,
        document_type=field.document_type,
        value=field.effective_value,
        normalized_value=field.normalized_value,
        page_number=evidence.page_number if evidence else None,
        source_text=evidence.source_text if evidence else None,
    )


def validate_cross_document_fields(
    effective_fields: Sequence[EffectiveFieldValue],
    rules: ReviewRules,
) -> list[Finding]:
    """Create at most one grouped conflict per semantic field."""

    findings: list[Finding] = []
    for rule in rules.comparison_rules:
        candidates = sorted(
            (
                field
                for field in effective_fields
                if field.field_name == rule.field
                and field.document_type in rule.sources
                and field.effective_value is not None
                and field.normalized_value is not None
            ),
            key=lambda field: field.field_id,
        )
        if len(candidates) < 2:
            continue
        if _matches_rule(
            [field.normalized_value for field in candidates if field.normalized_value is not None],
            rule,
        ):
            continue
        findings.append(
            Finding(
                finding_id=f"finding-{rule.field.replace('_', '-')}-conflict",
                type=FindingType.FIELD_CONFLICT,
                severity=FindingSeverity.HIGH,
                message=f"Conflicting values were submitted for {rule.field}.",
                field_name=rule.field,
                evidence=[_finding_evidence(field) for field in candidates],
            )
        )
    return findings


def cross_validate(
    effective_fields: Sequence[EffectiveFieldValue], rules: ReviewRules
) -> list[Finding]:
    return validate_cross_document_fields(effective_fields, rules)


def validate_review_package(
    classifications: Sequence[DocumentClassification],
    extractions: Sequence[ExtractionResult],
    effective_fields: Sequence[EffectiveFieldValue],
    rules: ReviewRules,
) -> list[Finding]:
    """Run all deterministic rules and guard against duplicate finding IDs."""

    findings = [
        *validate_required_documents(classifications, rules),
        *validate_required_fields(extractions, rules, effective_fields),
        *validate_cross_document_fields(effective_fields, rules),
    ]
    ids = [finding.finding_id for finding in findings]
    if len(ids) != len(set(ids)):
        raise ValueError("deterministic validation generated duplicate finding IDs")
    return findings


def effective_values_for_field(
    values: Iterable[EffectiveFieldValue], field_name: str
) -> list[EffectiveFieldValue]:
    return [value for value in values if value.field_name == field_name]
