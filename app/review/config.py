"""Typed configuration for deterministic review rules."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import Field, model_validator

from app.domain.base import DomainModel
from app.domain.documents import DocumentType
from app.domain.extractions import FIELDS_BY_DOCUMENT_TYPE


class ReviewRulesConfigError(RuntimeError):
    """Raised when deterministic review configuration cannot be loaded."""


class ConfidenceRules(DomainModel):
    field_review_threshold: float = Field(ge=0.0, le=1.0)
    classification_review_threshold: float = Field(ge=0.0, le=1.0)


class ComparisonRule(DomainModel):
    field: str = Field(min_length=1)
    sources: list[DocumentType] = Field(min_length=2)

    @model_validator(mode="after")
    def validate_sources(self) -> ComparisonRule:
        if len(self.sources) != len(set(self.sources)):
            raise ValueError("comparison-rule sources must be unique")
        if DocumentType.UNKNOWN in self.sources:
            raise ValueError("unknown documents cannot participate in comparisons")
        return self


class ExactMatchRule(ComparisonRule):
    pass


class NormalisedMatchRule(ComparisonRule):
    pass


class NumericToleranceRule(ComparisonRule):
    tolerance_percent: float | None = Field(default=None, ge=0.0)
    tolerance_absolute: float | None = Field(default=None, ge=0.0)

    @model_validator(mode="after")
    def validate_tolerance(self) -> NumericToleranceRule:
        configured = sum(
            value is not None
            for value in (self.tolerance_percent, self.tolerance_absolute)
        )
        if configured != 1:
            raise ValueError(
                "numeric rules must configure exactly one of percent or absolute tolerance"
            )
        return self


class ReviewRules(DomainModel):
    required_documents: list[DocumentType] = Field(min_length=1)
    required_fields: dict[DocumentType, list[str]]
    confidence: ConfidenceRules
    exact_match_rules: list[ExactMatchRule] = Field(default_factory=list)
    normalised_match_rules: list[NormalisedMatchRule] = Field(default_factory=list)
    numeric_tolerance_rules: list[NumericToleranceRule] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_rule_set(self) -> ReviewRules:
        if len(self.required_documents) != len(set(self.required_documents)):
            raise ValueError("required_documents must be unique")
        if DocumentType.UNKNOWN in self.required_documents:
            raise ValueError("unknown cannot be a required document type")
        missing_field_config = set(self.required_documents) - set(self.required_fields)
        if missing_field_config:
            names = ", ".join(sorted(item.value for item in missing_field_config))
            raise ValueError(f"required field configuration is missing for: {names}")
        for document_type, fields in self.required_fields.items():
            if document_type is DocumentType.UNKNOWN:
                raise ValueError("unknown documents cannot have required fields")
            if len(fields) != len(set(fields)):
                raise ValueError(
                    f"required fields for {document_type.value} must be unique"
                )
            unsupported = set(fields) - FIELDS_BY_DOCUMENT_TYPE[document_type]
            if unsupported:
                names = ", ".join(sorted(unsupported))
                raise ValueError(
                    f"unsupported required fields for {document_type.value}: {names}"
                )

        fields = [rule.field for rule in self.comparison_rules]
        if len(fields) != len(set(fields)):
            raise ValueError("a field may have only one cross-document comparison rule")
        for rule in self.comparison_rules:
            invalid_sources = [
                source
                for source in rule.sources
                if rule.field not in FIELDS_BY_DOCUMENT_TYPE[source]
            ]
            if invalid_sources:
                names = ", ".join(source.value for source in invalid_sources)
                raise ValueError(
                    f"field {rule.field} is not valid for comparison sources: {names}"
                )
        return self

    @property
    def comparison_rules(
        self,
    ) -> tuple[ExactMatchRule | NormalisedMatchRule | NumericToleranceRule, ...]:
        return tuple(
            [
                *self.exact_match_rules,
                *self.normalised_match_rules,
                *self.numeric_tolerance_rules,
            ]
        )


DEFAULT_REVIEW_RULES_PATH = (
    Path(__file__).resolve().parents[2] / "config" / "review_rules.yaml"
)


def load_review_rules(path: str | Path | None = None) -> ReviewRules:
    """Load and validate review rules; never silently substitute defaults."""

    rules_path = Path(path) if path is not None else DEFAULT_REVIEW_RULES_PATH
    try:
        with rules_path.open(encoding="utf-8") as stream:
            raw = yaml.safe_load(stream)
    except (OSError, yaml.YAMLError) as exc:
        raise ReviewRulesConfigError(
            f"could not load review rules from {rules_path}: {exc}"
        ) from exc
    if not isinstance(raw, dict):
        raise ReviewRulesConfigError(
            f"review rules at {rules_path} must contain a YAML mapping"
        )
    try:
        return ReviewRules.model_validate(raw)
    except ValueError as exc:
        raise ReviewRulesConfigError(
            f"invalid review rules at {rules_path}: {exc}"
        ) from exc
