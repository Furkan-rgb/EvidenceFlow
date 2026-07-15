"""Policy retrieval contracts."""

from __future__ import annotations

from pydantic import Field

from app.domain.base import DomainModel


class PolicyEvidence(DomainModel):
    evidence_id: str = Field(min_length=1)
    policy_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    section_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    score: float = Field(ge=0.0)
    source_path: str = Field(min_length=1)
