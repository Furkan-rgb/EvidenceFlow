"""Provider-neutral model requirements and readiness registry."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from app.ai.config import ChatModelConfig, EmbeddingModelConfig, ModelsConfig
from app.preparation.models import (
    ModelTask,
    PreparationComponent,
    PreparationMode,
    PreparationResult,
    SeverityResolver,
    default_failure_severity,
    failed_result,
)

DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"


@dataclass(frozen=True, slots=True)
class ModelRequirement:
    """One configured task/model identity to verify before an operation."""

    task: ModelTask
    provider: str
    model: str
    expected_digest: str | None
    endpoint: str | None


class ProviderReadinessChecker(Protocol):
    """Extensible readiness boundary for one model provider.

    A cloud implementation owns its credential-presence, authentication,
    connectivity, and model/deployment-access checks. Results must be redacted;
    secret values never belong in messages or structured result fields.
    """

    provider: str

    async def check(
        self,
        requirements: Sequence[ModelRequirement],
        *,
        mode: PreparationMode,
        severity_resolver: SeverityResolver = default_failure_severity,
    ) -> tuple[PreparationResult, ...]: ...


class ProviderCheckerRegistry:
    """Explicit registry; a configured but absent provider fails before startup."""

    def __init__(self, checkers: Sequence[ProviderReadinessChecker] = ()) -> None:
        registered: dict[str, ProviderReadinessChecker] = {}
        for checker in checkers:
            if checker.provider in registered:
                raise ValueError(f"Readiness checker already registered for '{checker.provider}'.")
            registered[checker.provider] = checker
        self._checkers = registered

    def get(self, provider: str) -> ProviderReadinessChecker | None:
        return self._checkers.get(provider)


def model_tasks_for_mode(mode: PreparationMode) -> tuple[ModelTask, ...]:
    """Select only the capabilities used by the requested operation."""

    if mode is PreparationMode.POLICY_INDEX_REBUILD:
        return (ModelTask.EMBEDDINGS,)
    return (
        ModelTask.CLASSIFICATION,
        ModelTask.EXTRACTION,
        ModelTask.REPORTING,
        ModelTask.EMBEDDINGS,
    )


def model_requirements(
    models: ModelsConfig,
    *,
    mode: PreparationMode,
    default_ollama_base_url: str | None = None,
) -> tuple[ModelRequirement, ...]:
    """Translate model config into task-preserving provider requirements."""

    configs: Mapping[ModelTask, ChatModelConfig | EmbeddingModelConfig] = {
        ModelTask.CLASSIFICATION: models.classification,
        ModelTask.EXTRACTION: models.extraction,
        ModelTask.REPORTING: models.reporting,
        ModelTask.EMBEDDINGS: models.embeddings,
    }
    fallback_url = default_ollama_base_url or DEFAULT_OLLAMA_BASE_URL
    requirements: list[ModelRequirement] = []
    for task in model_tasks_for_mode(mode):
        config = configs[task]
        endpoint = config.base_url
        if config.provider == "ollama":
            endpoint = endpoint or fallback_url
        requirements.append(
            ModelRequirement(
                task=task,
                provider=config.provider,
                model=config.model,
                expected_digest=config.model_digest,
                endpoint=endpoint,
            )
        )
    return tuple(requirements)


def provider_not_implemented_results(
    requirements: Sequence[ModelRequirement],
    *,
    mode: PreparationMode,
    severity_resolver: SeverityResolver = default_failure_severity,
) -> tuple[PreparationResult, ...]:
    """Create a task-specific failure for typed providers lacking an adapter."""

    return tuple(
        failed_result(
            mode=mode,
            code="provider_not_implemented",
            component=PreparationComponent.MODEL_PROVIDER,
            message=(
                f"Provider '{requirement.provider}' configured for "
                f"{requirement.task.value} is not implemented."
            ),
            remediation=(
                "Install and register a readiness adapter before selecting this provider."
            ),
            task=requirement.task,
            provider=requirement.provider,
            model=requirement.model,
            severity_resolver=severity_resolver,
        )
        for requirement in requirements
    )
