"""Model-facing report schema with every structural key required."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ModelReportSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1, max_length=300)
    summary: str = Field(min_length=1, max_length=10_000)
    finding_ids: list[str]
    policy_evidence_ids: list[str]

    @field_validator("finding_ids", "policy_evidence_ids")
    @classmethod
    def validate_unique_ids(cls, values: list[str]) -> list[str]:
        if any(not value.strip() for value in values):
            raise ValueError("reference IDs cannot be blank")
        if len(values) != len(set(values)):
            raise ValueError("reference IDs must be unique within a section")
        return values


class ModelReportNarrative(BaseModel):
    model_config = ConfigDict(extra="forbid")

    executive_summary: str = Field(min_length=1, max_length=10_000)
    sections: list[ModelReportSection]
