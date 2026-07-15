"""Model-facing classification schema without code-owned review metadata."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from app.domain import DocumentType


class ModelDocumentClassification(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_type: DocumentType
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning_summary: str = Field(min_length=1, max_length=1_000)
