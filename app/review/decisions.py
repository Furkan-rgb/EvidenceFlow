"""Review-decision validation and immutable effective-value application."""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import Field

from app.domain.base import DomainModel
from app.domain.documents import DocumentType
from app.domain.extractions import (
    DocumentClassification,
    EffectiveFieldValue,
    EffectiveValueSource,
    FieldValue,
)
from app.domain.reviews import (
    ExpectedValueType,
    ReviewAction,
    ReviewDecision,
    ReviewDecisionAudit,
    ReviewItem,
    ReviewItemStatus,
    ReviewItemType,
)
from app.review.normalization import (
    ValueNormalizationError,
    normalize_employee_count,
    normalize_field_value,
    normalize_reporting_year,
    normalize_revenue,
    normalize_text,
)


class ReviewDecisionValidationError(ValueError):
    """A safe validation error suitable for conversion to an API 422/409."""

    def __init__(self, message: str, *, details: dict[str, object] | None = None) -> None:
        super().__init__(message)
        self.details = details or {}


class DecisionApplicationResult(DomainModel):
    classifications: list[DocumentClassification] = Field(default_factory=list)
    effective_fields: list[EffectiveFieldValue] = Field(default_factory=list)
    review_items: list[ReviewItem] = Field(default_factory=list)
    audit_records: list[ReviewDecisionAudit] = Field(default_factory=list)


def _coerce_correction(
    value: FieldValue, expected: ExpectedValueType
) -> FieldValue:
    try:
        if expected is ExpectedValueType.DOCUMENT_TYPE:
            if not isinstance(value, str):
                raise ValueError("document type must be a string")
            return DocumentType(value).value
        if expected is ExpectedValueType.STRING:
            if not isinstance(value, str):
                raise ValueError("corrected value must be a string")
            return normalize_text(value)
        if expected is ExpectedValueType.INTEGER:
            # Reporting years and employee counts share an integer wire type.
            return normalize_employee_count(value)
        if expected is ExpectedValueType.NUMBER:
            number = normalize_revenue(value)
            return int(number) if number == number.to_integral_value() else float(number)
        if expected is ExpectedValueType.BOOLEAN:
            if not isinstance(value, bool):
                raise ValueError("corrected value must be a boolean")
            return value
    except (ValueError, ValueNormalizationError) as exc:
        raise ReviewDecisionValidationError(
            f"invalid corrected value for {expected.value}: {exc}"
        ) from exc
    raise ReviewDecisionValidationError(f"unsupported expected type: {expected.value}")


def _coerce_for_item(item: ReviewItem, decision: ReviewDecision) -> ReviewDecision:
    if decision.action is not ReviewAction.CORRECT:
        return decision
    assert decision.value is not None
    assert item.expected_value_type is not None
    value = _coerce_correction(decision.value, item.expected_value_type)
    if item.field_name == "reporting_year":
        try:
            value = normalize_reporting_year(value)  # type: ignore[arg-type]
        except ValueNormalizationError as exc:
            raise ReviewDecisionValidationError(str(exc)) from exc
    elif item.field_name == "incorporation_date":
        try:
            normalized_date = normalize_field_value(item.field_name, value)
        except ValueNormalizationError as exc:
            raise ReviewDecisionValidationError(str(exc)) from exc
        assert isinstance(normalized_date, str)
        value = normalized_date
    elif item.field_name:
        try:
            normalize_field_value(item.field_name, value)
        except ValueNormalizationError as exc:
            raise ReviewDecisionValidationError(str(exc)) from exc
    return decision.model_copy(update={"value": value})


def validate_review_decisions(
    pending_items: Sequence[ReviewItem],
    decisions: Sequence[ReviewDecision],
    *,
    require_all: bool = True,
) -> list[ReviewDecision]:
    """Validate one non-duplicated, type-compatible decision per pending item."""

    if any(item.status is not ReviewItemStatus.PENDING for item in pending_items):
        raise ReviewDecisionValidationError("only pending items may be decided")

    item_ids = [item.review_item_id for item in pending_items]
    if len(item_ids) != len(set(item_ids)):
        raise ReviewDecisionValidationError("pending review-item IDs are not unique")
    decision_ids = [decision.review_item_id for decision in decisions]
    if len(decision_ids) != len(set(decision_ids)):
        raise ReviewDecisionValidationError(
            "multiple decisions were supplied for the same review item"
        )
    audit_ids = [decision.decision_id for decision in decisions]
    if len(audit_ids) != len(set(audit_ids)):
        raise ReviewDecisionValidationError("review decision IDs must be unique")

    items_by_id = {item.review_item_id: item for item in pending_items}
    unknown = sorted(set(decision_ids) - set(items_by_id))
    if unknown:
        raise ReviewDecisionValidationError(
            "a decision references an unknown or non-pending review item",
            details={"review_item_ids": unknown},
        )
    missing = sorted(set(items_by_id) - set(decision_ids))
    if require_all and missing:
        raise ReviewDecisionValidationError(
            "every pending review item must receive exactly one decision",
            details={"review_item_ids": missing},
        )

    validated: dict[str, ReviewDecision] = {}
    for decision in decisions:
        item = items_by_id[decision.review_item_id]
        if decision.action not in item.allowed_actions:
            raise ReviewDecisionValidationError(
                f"action {decision.action.value} is invalid for {item.type.value}",
                details={"review_item_id": item.review_item_id},
            )
        if decision.action is ReviewAction.SELECT_VALUE:
            candidate_ids = {candidate.field_id for candidate in item.candidates}
            if decision.selected_field_id not in candidate_ids:
                raise ReviewDecisionValidationError(
                    "selected_field_id is not a candidate for this conflict",
                    details={"review_item_id": item.review_item_id},
                )
        validated[item.review_item_id] = _coerce_for_item(item, decision)

    return [
        validated[item.review_item_id]
        for item in pending_items
        if item.review_item_id in validated
    ]


def apply_classification_decisions(
    classifications: Sequence[DocumentClassification],
    items: Sequence[ReviewItem],
    decisions: Sequence[ReviewDecision],
) -> list[DocumentClassification]:
    document_ids = [classification.document_id for classification in classifications]
    if len(document_ids) != len(set(document_ids)):
        raise ReviewDecisionValidationError("classification document IDs must be unique")
    classifications_by_document = {
        classification.document_id: classification for classification in classifications
    }
    items_by_id = {item.review_item_id: item for item in items}
    result = dict(classifications_by_document)
    for decision in decisions:
        item = items_by_id[decision.review_item_id]
        if item.type is not ReviewItemType.LOW_CONFIDENCE_CLASSIFICATION:
            continue
        assert item.document_id is not None
        classification = classifications_by_document.get(item.document_id)
        if classification is None:
            raise ReviewDecisionValidationError(
                "classification review references an absent classification"
            )
        effective_type = classification.document_type
        if decision.action is ReviewAction.CORRECT:
            assert isinstance(decision.value, str)
            effective_type = DocumentType(decision.value)
        result[item.document_id] = classification.model_copy(
            update={
                "effective_document_type": effective_type,
                "was_human_reviewed": True,
                "review_decision_id": decision.decision_id,
            }
        )
    return [result[classification.document_id] for classification in classifications]


def _override_field(
    field: EffectiveFieldValue,
    value: FieldValue,
    source: EffectiveValueSource,
    decision_id: str,
) -> EffectiveFieldValue:
    return field.model_copy(
        update={
            "effective_value": value,
            "normalized_value": normalize_field_value(field.field_name, value),
            "value_source": source,
            "review_decision_id": decision_id,
        }
    )


def apply_field_decisions(
    effective_fields: Sequence[EffectiveFieldValue],
    items: Sequence[ReviewItem],
    decisions: Sequence[ReviewDecision],
) -> list[EffectiveFieldValue]:
    field_ids = [field.field_id for field in effective_fields]
    if len(field_ids) != len(set(field_ids)):
        raise ReviewDecisionValidationError("effective field IDs must be unique")
    fields = {field.field_id: field for field in effective_fields}
    items_by_id = {item.review_item_id: item for item in items}
    for decision in decisions:
        item = items_by_id[decision.review_item_id]
        if item.type is ReviewItemType.LOW_CONFIDENCE_FIELD:
            assert item.field_id is not None
            field = fields.get(item.field_id)
            if field is None:
                raise ReviewDecisionValidationError(
                    "field review references an absent effective field"
                )
            if decision.action is ReviewAction.APPROVE:
                fields[item.field_id] = field.model_copy(
                    update={
                        "value_source": EffectiveValueSource.APPROVED,
                        "review_decision_id": decision.decision_id,
                    }
                )
            else:
                assert decision.value is not None
                fields[item.field_id] = _override_field(
                    field,
                    decision.value,
                    EffectiveValueSource.CORRECTED,
                    decision.decision_id,
                )
        elif item.type is ReviewItemType.FIELD_CONFLICT:
            candidate_ids = [candidate.field_id for candidate in item.candidates]
            if decision.action is ReviewAction.MARK_UNRESOLVED:
                continue
            if decision.action is ReviewAction.SELECT_VALUE:
                assert decision.selected_field_id is not None
                selected = fields[decision.selected_field_id]
                assert selected.effective_value is not None
                value = selected.effective_value
                source = EffectiveValueSource.SELECTED
            else:
                assert decision.value is not None
                value = decision.value
                source = EffectiveValueSource.CORRECTED
            for field_id in candidate_ids:
                fields[field_id] = _override_field(
                    fields[field_id], value, source, decision.decision_id
                )
    return [fields[field.field_id] for field in effective_fields]


def _audit_record(item: ReviewItem, decision: ReviewDecision) -> ReviewDecisionAudit:
    if item.type is ReviewItemType.LOW_CONFIDENCE_CLASSIFICATION:
        assert item.proposed_document_type is not None
        originals: list[FieldValue] = [item.proposed_document_type.value]
    elif item.type is ReviewItemType.LOW_CONFIDENCE_FIELD:
        assert item.extracted_value is not None
        originals = [item.extracted_value]
    else:
        originals = [candidate.value for candidate in item.candidates]

    effective: FieldValue | None = None
    if decision.action is ReviewAction.APPROVE:
        effective = originals[0]
    elif decision.action is ReviewAction.CORRECT:
        effective = decision.value
    elif decision.action is ReviewAction.SELECT_VALUE:
        effective = next(
            candidate.value
            for candidate in item.candidates
            if candidate.field_id == decision.selected_field_id
        )
    return ReviewDecisionAudit(
        decision=decision,
        review_item_type=item.type,
        original_values=originals,
        effective_value=effective,
    )


def apply_review_decisions(
    classifications: Sequence[DocumentClassification],
    effective_fields: Sequence[EffectiveFieldValue],
    pending_items: Sequence[ReviewItem],
    decisions: Sequence[ReviewDecision],
    *,
    require_all: bool = True,
) -> DecisionApplicationResult:
    """Validate and apply a complete interrupt batch without mutating raw values."""

    validated = validate_review_decisions(
        pending_items, decisions, require_all=require_all
    )
    items_by_id = {item.review_item_id: item for item in pending_items}
    resolved_items = [
        items_by_id[decision.review_item_id].model_copy(
            update={
                "status": ReviewItemStatus.RESOLVED,
                "resolved_by_decision_id": decision.decision_id,
            }
        )
        for decision in validated
    ]
    return DecisionApplicationResult(
        classifications=apply_classification_decisions(
            classifications, pending_items, validated
        ),
        effective_fields=apply_field_decisions(
            effective_fields, pending_items, validated
        ),
        review_items=resolved_items,
        audit_records=[
            _audit_record(items_by_id[decision.review_item_id], decision)
            for decision in validated
        ],
    )
