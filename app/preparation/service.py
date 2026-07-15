"""Preparation orchestration across configured model providers."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from app.ai.config import ModelsConfig
from app.preparation.models import (
    CheckOutcome,
    ModelTask,
    PreparationComponent,
    PreparationMode,
    PreparationReport,
    PreparationResult,
    SeverityResolver,
    default_failure_severity,
    failed_result,
)
from app.preparation.ollama import OllamaInventoryLoader, OllamaReadinessChecker
from app.preparation.providers import (
    ModelRequirement,
    ProviderCheckerRegistry,
    model_requirements,
    provider_not_implemented_results,
)


def default_provider_registry(
    *,
    ollama_inventory_loader: OllamaInventoryLoader | None = None,
    timeout_seconds: float = 2.0,
) -> ProviderCheckerRegistry:
    """Build the V1 registry; future providers plug in at this boundary."""

    checker = (
        OllamaReadinessChecker(timeout_seconds=timeout_seconds)
        if ollama_inventory_loader is None
        else OllamaReadinessChecker(
            inventory_loader=ollama_inventory_loader,
            timeout_seconds=timeout_seconds,
        )
    )
    return ProviderCheckerRegistry((checker,))


async def prepare_model_providers(
    models: ModelsConfig,
    *,
    mode: PreparationMode,
    default_ollama_base_url: str | None = None,
    registry: ProviderCheckerRegistry | None = None,
    ollama_inventory_loader: OllamaInventoryLoader | None = None,
    timeout_seconds: float = 2.0,
    severity_resolver: SeverityResolver = default_failure_severity,
) -> PreparationReport:
    """Verify only the task providers required by ``mode``.

    ``registry`` and ``ollama_inventory_loader`` are mutually exclusive: callers
    either supply a complete provider registry or customize the built-in Ollama
    transport for deterministic tests.
    """

    if registry is not None and ollama_inventory_loader is not None:
        raise ValueError("Pass either registry or ollama_inventory_loader, not both.")
    active_registry = registry or default_provider_registry(
        ollama_inventory_loader=ollama_inventory_loader,
        timeout_seconds=timeout_seconds,
    )
    requirements = model_requirements(
        models,
        mode=mode,
        default_ollama_base_url=default_ollama_base_url,
    )
    by_provider: dict[str, list[ModelRequirement]] = defaultdict(list)
    for requirement in requirements:
        by_provider[requirement.provider].append(requirement)

    results: list[PreparationResult] = []
    for provider, provider_requirements in by_provider.items():
        checker = active_registry.get(provider)
        if checker is None:
            results.extend(
                provider_not_implemented_results(
                    provider_requirements,
                    mode=mode,
                    severity_resolver=severity_resolver,
                )
            )
            continue
        provider_results = await checker.check(
            provider_requirements,
            mode=mode,
            severity_resolver=severity_resolver,
        )
        results.extend(provider_results)
        results.extend(
            _provider_coverage_failures(
                provider_requirements,
                provider_results,
                mode=mode,
                severity_resolver=severity_resolver,
            )
        )
    return PreparationReport.from_results(mode, results)


def _provider_coverage_failures(
    requirements: Sequence[ModelRequirement],
    results: Sequence[PreparationResult],
    *,
    mode: PreparationMode,
    severity_resolver: SeverityResolver,
) -> tuple[PreparationResult, ...]:
    """Refuse incomplete or foreign results from a provider adapter."""

    expected = {requirement.task: requirement for requirement in requirements}
    by_task: dict[ModelTask, list[PreparationResult]] = defaultdict(list)
    foreign_result = False
    for result in results:
        if result.task is None:
            foreign_result = True
            continue
        requirement = expected.get(result.task)
        if requirement is None or result.provider not in {None, requirement.provider}:
            foreign_result = True
            continue
        if result.model not in {None, requirement.model}:
            foreign_result = True
            continue
        by_task[result.task].append(result)

    failures: list[PreparationResult] = []
    for requirement in requirements:
        task_results = by_task[requirement.task]
        if len(task_results) == 1 and task_results[0].outcome is not CheckOutcome.SKIPPED:
            continue
        failures.append(
            failed_result(
                mode=mode,
                code="provider_probe_incomplete",
                component=PreparationComponent.MODEL_PROVIDER,
                message=(
                    f"Provider readiness did not return one conclusive result for "
                    f"{requirement.task.value}."
                ),
                remediation="Correct the provider readiness adapter before startup.",
                task=requirement.task,
                provider=requirement.provider,
                model=requirement.model,
                severity_resolver=severity_resolver,
            )
        )
    if foreign_result:
        failures.append(
            failed_result(
                mode=mode,
                code="provider_probe_invalid",
                component=PreparationComponent.MODEL_PROVIDER,
                message="A provider readiness adapter returned a result for another task.",
                remediation="Correct the provider readiness adapter before startup.",
                severity_resolver=severity_resolver,
            )
        )
    return tuple(failures)
