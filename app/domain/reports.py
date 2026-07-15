"""Structured report contracts."""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field, field_validator

from app.domain.base import DomainModel


class ReportStatus(StrEnum):
    COMPLETE = "complete"
    NEEDS_FOLLOW_UP = "needs_follow_up"
    INCOMPLETE = "incomplete"


class ReportSection(DomainModel):
    title: str = Field(min_length=1, max_length=300)
    summary: str = Field(min_length=1, max_length=10_000)
    finding_ids: list[str] = Field(default_factory=list)
    policy_evidence_ids: list[str] = Field(default_factory=list)

    @field_validator("finding_ids", "policy_evidence_ids")
    @classmethod
    def validate_unique_references(cls, values: list[str]) -> list[str]:
        if any(not value.strip() for value in values):
            raise ValueError("report reference IDs cannot be blank")
        if len(values) != len(set(values)):
            raise ValueError("report reference IDs must be unique within a section")
        return values


class ReportNarrative(DomainModel):
    """The only report content an LLM is allowed to author."""

    executive_summary: str = Field(min_length=1, max_length=10_000)
    sections: list[ReportSection] = Field(default_factory=list)


class ReviewReport(ReportNarrative):
    """Final report after deterministic identity, status, and ID validation."""

    company_name: str | None = None
    status: ReportStatus
