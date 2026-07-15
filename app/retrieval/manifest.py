"""Vector-index identity and compatibility validation."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from app.errors import EmbeddingIndexMismatchError, PolicyIndexMissingError


class PolicyIndexManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index_version: int = 1
    build_id: str = Field(min_length=1)
    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    model_digest: str | None = None
    dimensions: int = Field(gt=0)
    preprocessing_profile: str = "markdown-sections-v1"
    chunk_target_characters: int = Field(gt=0)
    chunk_max_characters: int = Field(gt=0)
    chunk_overlap_characters: int = Field(ge=0)
    corpus_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    document_count: int = Field(ge=0)
    chunk_count: int = Field(ge=0)
    created_at: datetime


def load_manifest(path: Path) -> PolicyIndexManifest:
    if not path.is_file():
        raise PolicyIndexMissingError(
            "The policy index manifest is missing; rebuild the policy index.",
            details={"manifest_path": str(path)},
        )
    try:
        return PolicyIndexManifest.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise PolicyIndexMissingError(
            "The policy index manifest is invalid; rebuild the policy index.",
            details={"manifest_path": str(path)},
        ) from exc


def manifest_json(manifest: PolicyIndexManifest) -> str:
    return json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"


def assert_manifest_compatible(
    manifest: PolicyIndexManifest,
    *,
    provider: str,
    model: str,
    dimensions: int,
    model_digest: str | None = None,
    preprocessing_profile: str = "markdown-sections-v1",
) -> None:
    mismatches: dict[str, dict[str, object]] = {}
    expected: dict[str, object] = {
        "provider": provider,
        "model": model,
        "dimensions": dimensions,
        "preprocessing_profile": preprocessing_profile,
    }
    if model_digest is not None:
        expected["model_digest"] = model_digest
    for field_name, expected_value in expected.items():
        actual_value = getattr(manifest, field_name)
        if actual_value != expected_value:
            mismatches[field_name] = {"expected": expected_value, "actual": actual_value}
    if mismatches:
        raise EmbeddingIndexMismatchError(
            "The configured embedding model is incompatible with the policy index; "
            "run `python -m app.cli rebuild-policy-index`.",
            details={"mismatches": mismatches},
        )
