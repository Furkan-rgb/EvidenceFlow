"""Ollama model-inventory probe with redacted, task-specific diagnostics."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import urlopen

from app.preparation.models import (
    PreparationComponent,
    PreparationMode,
    PreparationResult,
    SeverityResolver,
    default_failure_severity,
    failed_result,
    passed_result,
)
from app.preparation.providers import ModelRequirement

OllamaInventoryLoader = Callable[[str, float], Awaitable[Mapping[str, str]]]


class ProviderProbeError(Exception):
    """Internal typed probe error whose message is never shown to users."""


class ProviderUnreachableError(ProviderProbeError):
    pass


class ProviderMalformedResponseError(ProviderProbeError):
    pass


class ProviderEndpointError(ProviderProbeError):
    pass


class ProviderHTTPError(ProviderProbeError):
    def __init__(self, status_code: int) -> None:
        super().__init__("provider returned an unsuccessful HTTP status")
        self.status_code = status_code


def safe_endpoint_label(endpoint: str | None) -> str:
    """Strip userinfo, paths, query parameters, and fragments from an endpoint."""

    if not endpoint:
        return "configured endpoint"
    try:
        parsed = urlsplit(endpoint)
        if parsed.scheme not in {"http", "https"} or parsed.hostname is None:
            return "configured endpoint"
        host = parsed.hostname
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        if parsed.port is not None:
            host = f"{host}:{parsed.port}"
        return urlunsplit((parsed.scheme, host, "", "", ""))
    except ValueError:
        return "configured endpoint"


def parse_ollama_inventory(payload: object) -> dict[str, str]:
    """Strictly validate ``/api/tags`` without accepting partial inventories."""

    if not isinstance(payload, dict) or not isinstance(payload.get("models"), list):
        raise ProviderMalformedResponseError

    inventory: dict[str, str] = {}
    models: list[Any] = payload["models"]
    for item in models:
        if not isinstance(item, dict):
            raise ProviderMalformedResponseError
        name = item.get("name")
        digest = item.get("digest")
        if not isinstance(name, str) or not name.strip():
            raise ProviderMalformedResponseError
        if not isinstance(digest, str) or not digest.strip():
            raise ProviderMalformedResponseError
        _add_inventory_identity(inventory, name, digest)
        if name.endswith(":latest"):
            _add_inventory_identity(inventory, name.removesuffix(":latest"), digest)
    return inventory


def _add_inventory_identity(inventory: dict[str, str], name: str, digest: str) -> None:
    existing = inventory.get(name)
    if existing is not None and existing != digest:
        raise ProviderMalformedResponseError
    inventory[name] = digest


def _load_ollama_inventory_sync(base_url: str, timeout_seconds: float) -> dict[str, str]:
    try:
        parsed = urlsplit(base_url)
        if parsed.scheme not in {"http", "https"} or parsed.hostname is None:
            raise ProviderEndpointError
        with urlopen(f"{base_url.rstrip('/')}/api/tags", timeout=timeout_seconds) as response:
            status = int(getattr(response, "status", 200))
            if not 200 <= status < 300:
                raise ProviderHTTPError(status)
            try:
                payload = json.load(response)
            except (ValueError, json.JSONDecodeError) as error:
                raise ProviderMalformedResponseError from error
    except HTTPError as error:
        raise ProviderHTTPError(error.code) from error
    except ProviderProbeError:
        raise
    except (OSError, URLError, TimeoutError) as error:
        raise ProviderUnreachableError from error
    except ValueError as error:
        raise ProviderEndpointError from error
    return parse_ollama_inventory(payload)


async def load_ollama_inventory(base_url: str, timeout_seconds: float) -> Mapping[str, str]:
    """Fetch Ollama inventory without blocking the application's event loop."""

    return await asyncio.to_thread(_load_ollama_inventory_sync, base_url, timeout_seconds)


class OllamaReadinessChecker:
    provider = "ollama"

    def __init__(
        self,
        *,
        inventory_loader: OllamaInventoryLoader = load_ollama_inventory,
        timeout_seconds: float = 2.0,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self._inventory_loader = inventory_loader
        self._timeout_seconds = timeout_seconds

    async def check(
        self,
        requirements: Sequence[ModelRequirement],
        *,
        mode: PreparationMode,
        severity_resolver: SeverityResolver = default_failure_severity,
    ) -> tuple[PreparationResult, ...]:
        if any(requirement.provider != self.provider for requirement in requirements):
            raise ValueError("Ollama checker received a requirement for another provider.")

        by_endpoint: dict[str | None, list[ModelRequirement]] = {}
        for requirement in requirements:
            by_endpoint.setdefault(requirement.endpoint, []).append(requirement)

        results: list[PreparationResult] = []
        for endpoint, endpoint_requirements in by_endpoint.items():
            results.extend(
                await self._check_endpoint(
                    endpoint,
                    endpoint_requirements,
                    mode=mode,
                    severity_resolver=severity_resolver,
                )
            )
        return tuple(results)

    async def _check_endpoint(
        self,
        endpoint: str | None,
        requirements: Sequence[ModelRequirement],
        *,
        mode: PreparationMode,
        severity_resolver: SeverityResolver,
    ) -> tuple[PreparationResult, ...]:
        label = safe_endpoint_label(endpoint)
        if endpoint is None:
            return self._endpoint_failures(
                requirements,
                mode=mode,
                severity_resolver=severity_resolver,
                code="provider_endpoint_invalid",
                message="Ollama has no valid configured endpoint.",
                remediation="Set a valid Ollama base URL in models.yaml.",
                endpoint=label,
            )
        try:
            loaded_inventory = await self._inventory_loader(endpoint, self._timeout_seconds)
            inventory = _validate_loaded_inventory(loaded_inventory)
        except asyncio.CancelledError:
            raise
        except ProviderEndpointError:
            return self._endpoint_failures(
                requirements,
                mode=mode,
                severity_resolver=severity_resolver,
                code="provider_endpoint_invalid",
                message="Ollama has no valid configured endpoint.",
                remediation="Set a valid Ollama base URL in models.yaml.",
                endpoint=label,
            )
        except ProviderHTTPError as error:
            return self._endpoint_failures(
                requirements,
                mode=mode,
                severity_resolver=severity_resolver,
                code="provider_http_error",
                message=f"Ollama returned HTTP {error.status_code} at {label}.",
                remediation="Check the provider endpoint and access configuration.",
                endpoint=label,
                http_status=error.status_code,
            )
        except ProviderMalformedResponseError:
            return self._endpoint_failures(
                requirements,
                mode=mode,
                severity_resolver=severity_resolver,
                code="provider_malformed_response",
                message=f"Ollama returned an invalid model inventory at {label}.",
                remediation="Verify that the configured endpoint is a compatible Ollama server.",
                endpoint=label,
            )
        except ProviderUnreachableError:
            return self._endpoint_failures(
                requirements,
                mode=mode,
                severity_resolver=severity_resolver,
                code="provider_unreachable",
                message=f"Ollama could not be reached at {label}.",
                remediation="Start Ollama and verify its configured base URL.",
                endpoint=label,
            )
        except Exception:
            return self._endpoint_failures(
                requirements,
                mode=mode,
                severity_resolver=severity_resolver,
                code="provider_probe_failed",
                message=f"Ollama readiness could not be verified at {label}.",
                remediation="Check the provider configuration and retry preparation.",
                endpoint=label,
            )

        return tuple(
            self._model_result(
                requirement,
                inventory,
                mode=mode,
                severity_resolver=severity_resolver,
                endpoint=label,
            )
            for requirement in requirements
        )

    @staticmethod
    def _endpoint_failures(
        requirements: Sequence[ModelRequirement],
        *,
        mode: PreparationMode,
        severity_resolver: SeverityResolver,
        code: str,
        message: str,
        remediation: str,
        endpoint: str,
        http_status: int | None = None,
    ) -> tuple[PreparationResult, ...]:
        return tuple(
            failed_result(
                mode=mode,
                code=code,
                component=PreparationComponent.MODEL_PROVIDER,
                message=message,
                remediation=remediation,
                task=requirement.task,
                provider=requirement.provider,
                model=requirement.model,
                endpoint=endpoint,
                http_status=http_status,
                severity_resolver=severity_resolver,
            )
            for requirement in requirements
        )

    @staticmethod
    def _model_result(
        requirement: ModelRequirement,
        inventory: Mapping[str, str],
        *,
        mode: PreparationMode,
        severity_resolver: SeverityResolver,
        endpoint: str,
    ) -> PreparationResult:
        observed_digest = _inventory_digest(inventory, requirement.model)
        if observed_digest is None:
            return failed_result(
                mode=mode,
                code="model_missing",
                component=PreparationComponent.MODEL_PROVIDER,
                message=(
                    f"Ollama model '{requirement.model}' required for "
                    f"{requirement.task.value} is not installed."
                ),
                remediation=f"Run `ollama pull {requirement.model}` and retry preparation.",
                task=requirement.task,
                provider=requirement.provider,
                model=requirement.model,
                endpoint=endpoint,
                expected_digest=requirement.expected_digest,
                observed_digest=observed_digest,
                severity_resolver=severity_resolver,
            )
        if (
            requirement.expected_digest is not None
            and observed_digest != requirement.expected_digest
        ):
            return failed_result(
                mode=mode,
                code="model_digest_mismatch",
                component=PreparationComponent.MODEL_PROVIDER,
                message=(
                    f"Ollama model '{requirement.model}' for {requirement.task.value} "
                    "does not match the digest pinned in models.yaml."
                ),
                remediation=(
                    "Install the pinned model build or deliberately update its configured digest."
                ),
                task=requirement.task,
                provider=requirement.provider,
                model=requirement.model,
                endpoint=endpoint,
                expected_digest=requirement.expected_digest,
                observed_digest=observed_digest,
                severity_resolver=severity_resolver,
            )
        return passed_result(
            code="model_ready",
            component=PreparationComponent.MODEL_PROVIDER,
            message=(
                f"Ollama model '{requirement.model}' is ready for "
                f"{requirement.task.value}."
            ),
            task=requirement.task,
            provider=requirement.provider,
            model=requirement.model,
            endpoint=endpoint,
            expected_digest=requirement.expected_digest,
            observed_digest=observed_digest,
        )


def _validate_loaded_inventory(inventory: object) -> Mapping[str, str]:
    if not isinstance(inventory, Mapping):
        raise ProviderMalformedResponseError
    for name, digest in inventory.items():
        if not isinstance(name, str) or not name.strip():
            raise ProviderMalformedResponseError
        if not isinstance(digest, str) or not digest.strip():
            raise ProviderMalformedResponseError
    return inventory


def _inventory_digest(inventory: Mapping[str, str], model: str) -> str | None:
    observed = inventory.get(model)
    if observed is not None:
        return observed
    if model.endswith(":latest"):
        return inventory.get(model.removesuffix(":latest"))
    if ":" not in model:
        return inventory.get(f"{model}:latest")
    return None
