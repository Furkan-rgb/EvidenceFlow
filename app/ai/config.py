"""Typed model registry loaded from ``config/models.yaml``."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

ModelProvider = Literal["ollama", "openai", "azure_openai"]
EmbeddingProviderName = Literal["ollama", "openai", "azure_openai", "huggingface"]


class ChatModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: ModelProvider
    model: str = Field(min_length=1)
    model_digest: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_output_tokens: int = Field(default=2048, gt=0, le=8192)
    timeout_seconds: float = Field(default=180.0, gt=0)
    base_url: str | None = None


class EmbeddingModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: EmbeddingProviderName
    model: str = Field(min_length=1)
    model_digest: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    dimensions: int = Field(gt=0)
    base_url: str | None = None


class ModelsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    classification: ChatModelConfig
    extraction: ChatModelConfig
    reporting: ChatModelConfig
    embeddings: EmbeddingModelConfig


class _ModelsFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    models: ModelsConfig


def load_models_config(
    path: Path,
    *,
    classification_model: str | None = None,
    extraction_model: str | None = None,
    reporting_model: str | None = None,
    embedding_model: str | None = None,
    ollama_base_url: str | None = None,
) -> ModelsConfig:
    """Load and validate the registry, applying explicit environment overrides.

    Environment access remains in application settings/bootstrap. Keeping this
    function explicit makes configuration deterministic in tests and CLIs.
    """

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    parsed = _ModelsFile.model_validate(payload)
    config = parsed.models.model_copy(deep=True)

    overrides = {
        "classification": classification_model,
        "extraction": extraction_model,
        "reporting": reporting_model,
    }
    for task, model_name in overrides.items():
        if model_name:
            task_config = getattr(config, task)
            setattr(
                config,
                task,
                task_config.model_copy(
                    update={
                        "model": model_name,
                        "model_digest": (
                            task_config.model_digest
                            if model_name == task_config.model
                            else None
                        ),
                    }
                ),
            )
    if embedding_model:
        config.embeddings = config.embeddings.model_copy(
            update={
                "model": embedding_model,
                "model_digest": (
                    config.embeddings.model_digest
                    if embedding_model == config.embeddings.model
                    else None
                ),
            }
        )

    if ollama_base_url:
        for task in ("classification", "extraction", "reporting"):
            task_config = getattr(config, task)
            if task_config.provider == "ollama":
                setattr(
                    config,
                    task,
                    task_config.model_copy(update={"base_url": ollama_base_url}),
                )
        if config.embeddings.provider == "ollama":
            config.embeddings = config.embeddings.model_copy(
                update={"base_url": ollama_base_url}
            )
    return config
