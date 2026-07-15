"""Small model-facing schemas converted into provider-neutral domain results."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    page_number: int = Field(ge=1)
    source_text: str = Field(min_length=1, max_length=1_000)


class ModelField(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field_id: str = Field(min_length=1)
    value: str | int | float | bool | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[ModelEvidence] = Field(default_factory=list)


class StringModelField(ModelField):
    value: str | None = None


class NumberModelField(ModelField):
    value: int | float | None = None


class IntegerModelField(ModelField):
    value: int | None = None


class ApplicationFormModelOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    company_name: StringModelField | None
    registration_number: StringModelField | None
    annual_revenue_eur: NumberModelField | None
    employee_count: IntegerModelField | None


class CompanyExtractModelOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    company_name: StringModelField | None
    registration_number: StringModelField | None
    incorporation_date: StringModelField | None


class FinancialStatementModelOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    company_name: StringModelField | None
    annual_revenue_eur: NumberModelField | None
    reporting_year: IntegerModelField | None
    employee_count: IntegerModelField | None


class ModelClarificationStatement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    statement_id: str = Field(min_length=1)
    topic: str = Field(min_length=1, max_length=200)
    text: str | None = Field(default=None, min_length=1, max_length=2_000)
    value: str | int | float | bool | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[ModelEvidence] = Field(min_length=1)


class SupportingCorrespondenceModelOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    company_name: StringModelField | None
    clarification_statements: list[ModelClarificationStatement] = Field(
        default_factory=list
    )


type ModelExtractionOutput = (
    ApplicationFormModelOutput
    | CompanyExtractModelOutput
    | FinancialStatementModelOutput
    | SupportingCorrespondenceModelOutput
)
