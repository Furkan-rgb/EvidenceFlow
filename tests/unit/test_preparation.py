from __future__ import annotations

from collections.abc import Mapping
from email.message import Message
from io import BytesIO
from typing import NoReturn
from urllib.error import HTTPError, URLError

import pytest

import app.preparation.ollama as ollama_preparation
from app.ai.config import ChatModelConfig, EmbeddingModelConfig, ModelsConfig
from app.preparation import (
    CheckSeverity,
    ModelTask,
    PreparationComponent,
    PreparationMode,
    PreparationReport,
    PreparationStatus,
    ProviderCheckerRegistry,
    ProviderHTTPError,
    ProviderMalformedResponseError,
    ProviderUnreachableError,
    failed_result,
    load_ollama_inventory,
    model_requirements,
    parse_ollama_inventory,
    passed_result,
    prepare_model_providers,
)

CHAT_DIGEST = "a" * 64
EMBEDDING_DIGEST = "b" * 64


class _HTTPResponse(BytesIO):
    status = 200


def _models(
    *,
    classification_provider: str = "ollama",
    classification_digest: str = CHAT_DIGEST,
    extraction_digest: str = CHAT_DIGEST,
    endpoint: str = "http://ollama.test:11434",
) -> ModelsConfig:
    return ModelsConfig.model_validate(
        {
            "classification": {
                "provider": classification_provider,
                "model": "chat-model",
                "model_digest": classification_digest,
                "base_url": endpoint,
            },
            "extraction": {
                "provider": "ollama",
                "model": "chat-model",
                "model_digest": extraction_digest,
                "base_url": endpoint,
            },
            "reporting": {
                "provider": "ollama",
                "model": "chat-model",
                "model_digest": CHAT_DIGEST,
                "base_url": endpoint,
            },
            "embeddings": {
                "provider": "ollama",
                "model": "embedding-model",
                "model_digest": EMBEDDING_DIGEST,
                "dimensions": 768,
                "base_url": endpoint,
            },
        }
    )


def test_model_requirements_select_tasks_by_operation() -> None:
    models = _models()

    runtime = model_requirements(models, mode=PreparationMode.RUNTIME)
    evaluation = model_requirements(models, mode=PreparationMode.EVALUATION)
    rebuild = model_requirements(models, mode=PreparationMode.POLICY_INDEX_REBUILD)

    expected_all = (
        ModelTask.CLASSIFICATION,
        ModelTask.EXTRACTION,
        ModelTask.REPORTING,
        ModelTask.EMBEDDINGS,
    )
    assert tuple(requirement.task for requirement in runtime) == expected_all
    assert tuple(requirement.task for requirement in evaluation) == expected_all
    assert tuple(requirement.task for requirement in rebuild) == (ModelTask.EMBEDDINGS,)


@pytest.mark.asyncio
async def test_runtime_preserves_each_task_when_models_are_shared() -> None:
    calls: list[str] = []

    async def inventory(endpoint: str, _timeout: float) -> Mapping[str, str]:
        calls.append(endpoint)
        return {"chat-model": CHAT_DIGEST, "embedding-model": EMBEDDING_DIGEST}

    report = await prepare_model_providers(
        _models(),
        mode=PreparationMode.RUNTIME,
        ollama_inventory_loader=inventory,
    )

    assert report.status is PreparationStatus.READY
    assert report.ready
    assert [result.task for result in report.results] == [
        ModelTask.CLASSIFICATION,
        ModelTask.EXTRACTION,
        ModelTask.REPORTING,
        ModelTask.EMBEDDINGS,
    ]
    assert {result.code for result in report.results} == {"model_ready"}
    assert calls == ["http://ollama.test:11434"]


@pytest.mark.asyncio
async def test_task_specific_digest_checks_do_not_overwrite_shared_model_identity() -> None:
    async def inventory(_endpoint: str, _timeout: float) -> Mapping[str, str]:
        return {"chat-model": CHAT_DIGEST, "embedding-model": EMBEDDING_DIGEST}

    report = await prepare_model_providers(
        _models(extraction_digest="c" * 64),
        mode=PreparationMode.RUNTIME,
        ollama_inventory_loader=inventory,
    )

    assert report.for_task(ModelTask.CLASSIFICATION)[0].code == "model_ready"
    extraction = report.for_task(ModelTask.EXTRACTION)[0]
    assert extraction.code == "model_digest_mismatch"
    assert extraction.expected_digest == "c" * 64
    assert extraction.observed_digest == CHAT_DIGEST
    assert report.status is PreparationStatus.BLOCKED


@pytest.mark.asyncio
async def test_policy_rebuild_checks_only_the_embedding_provider() -> None:
    models = _models(classification_provider="openai")

    async def inventory(_endpoint: str, _timeout: float) -> Mapping[str, str]:
        return {"embedding-model": EMBEDDING_DIGEST}

    report = await prepare_model_providers(
        models,
        mode=PreparationMode.POLICY_INDEX_REBUILD,
        ollama_inventory_loader=inventory,
    )

    assert report.ready
    assert len(report.results) == 1
    assert report.results[0].task is ModelTask.EMBEDDINGS
    assert report.results[0].code == "model_ready"


@pytest.mark.asyncio
async def test_typed_provider_without_adapter_fails_explicitly_per_task() -> None:
    async def inventory(_endpoint: str, _timeout: float) -> Mapping[str, str]:
        return {"chat-model": CHAT_DIGEST, "embedding-model": EMBEDDING_DIGEST}

    report = await prepare_model_providers(
        _models(classification_provider="openai"),
        mode=PreparationMode.RUNTIME,
        ollama_inventory_loader=inventory,
    )

    unsupported = report.for_task(ModelTask.CLASSIFICATION)[0]
    assert unsupported.code == "provider_not_implemented"
    assert unsupported.provider == "openai"
    assert unsupported.blocking
    assert "not implemented" in unsupported.message


@pytest.mark.parametrize(
    ("error", "expected_code"),
    [
        (ProviderUnreachableError(), "provider_unreachable"),
        (ProviderHTTPError(503), "provider_http_error"),
        (ProviderMalformedResponseError(), "provider_malformed_response"),
    ],
)
@pytest.mark.asyncio
async def test_ollama_probe_failures_have_distinct_codes(
    error: Exception, expected_code: str
) -> None:
    async def failing_inventory(_endpoint: str, _timeout: float) -> Mapping[str, str]:
        raise error

    report = await prepare_model_providers(
        _models(),
        mode=PreparationMode.RUNTIME,
        ollama_inventory_loader=failing_inventory,
    )

    assert {result.code for result in report.results} == {expected_code}
    assert all(result.blocking for result in report.results)
    if expected_code == "provider_http_error":
        assert {result.http_status for result in report.results} == {503}


@pytest.mark.asyncio
async def test_missing_model_and_digest_mismatch_are_distinct() -> None:
    async def inventory(_endpoint: str, _timeout: float) -> Mapping[str, str]:
        return {"chat-model": "c" * 64}

    report = await prepare_model_providers(
        _models(),
        mode=PreparationMode.RUNTIME,
        ollama_inventory_loader=inventory,
    )

    assert report.for_task(ModelTask.CLASSIFICATION)[0].code == "model_digest_mismatch"
    assert report.for_task(ModelTask.EMBEDDINGS)[0].code == "model_missing"


@pytest.mark.asyncio
async def test_diagnostics_redact_endpoint_credentials_and_query_values() -> None:
    secret_endpoint = "https://alice:super-secret@ollama.test:11434/private?token=hidden"

    async def inventory(_endpoint: str, _timeout: float) -> Mapping[str, str]:
        raise ProviderHTTPError(401)

    report = await prepare_model_providers(
        _models(endpoint=secret_endpoint),
        mode=PreparationMode.RUNTIME,
        ollama_inventory_loader=inventory,
    )

    rendered = " ".join(
        f"{result.message} {result.remediation} {result.endpoint}" for result in report.results
    )
    assert "super-secret" not in rendered
    assert "token" not in rendered
    assert "hidden" not in rendered
    assert {result.endpoint for result in report.results} == {"https://ollama.test:11434"}


def test_parser_validates_inventory_and_adds_latest_alias() -> None:
    assert parse_ollama_inventory(
        {"models": [{"name": "embedding-model:latest", "digest": EMBEDDING_DIGEST}]}
    ) == {
        "embedding-model:latest": EMBEDDING_DIGEST,
        "embedding-model": EMBEDDING_DIGEST,
    }

    with pytest.raises(ProviderMalformedResponseError):
        parse_ollama_inventory({"models": [{"name": "chat-model"}]})
    with pytest.raises(ProviderMalformedResponseError):
        parse_ollama_inventory({"unexpected": []})


@pytest.mark.asyncio
async def test_default_ollama_transport_maps_http_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(_url: str, *, timeout: float) -> NoReturn:
        del timeout
        raise HTTPError(_url, 502, "contains-provider-details", Message(), None)

    monkeypatch.setattr(ollama_preparation, "urlopen", fail)

    with pytest.raises(ProviderHTTPError) as error:
        await load_ollama_inventory("http://ollama.test:11434", 0.1)

    assert error.value.status_code == 502


@pytest.mark.asyncio
async def test_default_ollama_transport_maps_network_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(_url: str, *, timeout: float) -> NoReturn:
        del _url, timeout
        raise URLError("contains-provider-details")

    monkeypatch.setattr(ollama_preparation, "urlopen", fail)

    with pytest.raises(ProviderUnreachableError):
        await load_ollama_inventory("http://ollama.test:11434", 0.1)


@pytest.mark.asyncio
async def test_default_ollama_transport_maps_invalid_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def respond(_url: str, *, timeout: float) -> _HTTPResponse:
        del _url, timeout
        return _HTTPResponse(b"not-json")

    monkeypatch.setattr(ollama_preparation, "urlopen", respond)

    with pytest.raises(ProviderMalformedResponseError):
        await load_ollama_inventory("http://ollama.test:11434", 0.1)


def test_runtime_telemetry_warns_but_evaluation_telemetry_blocks() -> None:
    runtime_failure = failed_result(
        mode=PreparationMode.RUNTIME,
        code="telemetry_unreachable",
        component=PreparationComponent.TELEMETRY,
        message="Telemetry is unavailable.",
    )
    evaluation_failure = failed_result(
        mode=PreparationMode.EVALUATION,
        code="telemetry_unreachable",
        component=PreparationComponent.TELEMETRY,
        message="Telemetry is unavailable.",
    )

    runtime_report = PreparationReport.from_results(
        PreparationMode.RUNTIME, [runtime_failure]
    )
    evaluation_report = PreparationReport.from_results(
        PreparationMode.EVALUATION, [evaluation_failure]
    )

    assert runtime_failure.severity is CheckSeverity.WARNING
    assert runtime_report.ready
    assert runtime_report.status is PreparationStatus.DEGRADED
    assert evaluation_failure.severity is CheckSeverity.CRITICAL
    assert not evaluation_report.ready
    assert evaluation_report.status is PreparationStatus.BLOCKED


@pytest.mark.asyncio
async def test_unexpected_probe_errors_are_redacted() -> None:
    async def inventory(_endpoint: str, _timeout: float) -> Mapping[str, str]:
        raise RuntimeError("api_key=do-not-leak")

    report = await prepare_model_providers(
        _models(),
        mode=PreparationMode.RUNTIME,
        ollama_inventory_loader=inventory,
    )

    assert {result.code for result in report.results} == {"provider_probe_failed"}
    assert "do-not-leak" not in " ".join(result.message for result in report.results)


@pytest.mark.asyncio
async def test_incomplete_provider_adapter_results_block_startup() -> None:
    class IncompleteChecker:
        provider = "ollama"

        async def check(
            self,
            requirements: object,
            *,
            mode: PreparationMode,
            severity_resolver: object,
        ) -> tuple[object, ...]:
            del requirements, mode, severity_resolver
            return ()

    report = await prepare_model_providers(
        _models(),
        mode=PreparationMode.RUNTIME,
        registry=ProviderCheckerRegistry((IncompleteChecker(),)),
    )

    assert not report.ready
    assert [result.code for result in report.results] == [
        "provider_probe_incomplete",
        "provider_probe_incomplete",
        "provider_probe_incomplete",
        "provider_probe_incomplete",
    ]


@pytest.mark.asyncio
async def test_foreign_provider_adapter_result_blocks_startup() -> None:
    class ForeignChecker:
        provider = "ollama"

        async def check(
            self,
            requirements: object,
            *,
            mode: PreparationMode,
            severity_resolver: object,
        ) -> tuple[object, ...]:
            del requirements, mode, severity_resolver
            return (
                passed_result(
                    code="model_ready",
                    component=PreparationComponent.MODEL_PROVIDER,
                    message="Foreign task result.",
                    provider="another-provider",
                    model="chat-model",
                    task=ModelTask.CLASSIFICATION,
                ),
            )

    report = await prepare_model_providers(
        _models(),
        mode=PreparationMode.RUNTIME,
        registry=ProviderCheckerRegistry((ForeignChecker(),)),
    )

    assert not report.ready
    assert any(result.code == "provider_probe_invalid" for result in report.results)


def test_model_config_types_remain_compatible_with_requirement_builder() -> None:
    """Guard the public builder against config type changes."""

    models = ModelsConfig(
        classification=ChatModelConfig(
            provider="ollama", model="one", model_digest=CHAT_DIGEST
        ),
        extraction=ChatModelConfig(
            provider="ollama", model="two", model_digest=CHAT_DIGEST
        ),
        reporting=ChatModelConfig(
            provider="ollama", model="three", model_digest=CHAT_DIGEST
        ),
        embeddings=EmbeddingModelConfig(
            provider="ollama",
            model="four",
            model_digest=EMBEDDING_DIGEST,
            dimensions=768,
        ),
    )

    requirements = model_requirements(
        models,
        mode=PreparationMode.RUNTIME,
        default_ollama_base_url="http://fallback.test:11434",
    )

    assert {requirement.endpoint for requirement in requirements} == {
        "http://fallback.test:11434"
    }


@pytest.mark.asyncio
async def test_malformed_injected_inventory_is_not_treated_as_models_missing() -> None:
    async def inventory(_endpoint: str, _timeout: float) -> Mapping[str, str]:
        return {"chat-model": 123}  # type: ignore[dict-item]

    report = await prepare_model_providers(
        _models(),
        mode=PreparationMode.RUNTIME,
        ollama_inventory_loader=inventory,
    )

    assert {result.code for result in report.results} == {"provider_malformed_response"}


def test_report_combine_keeps_mode_and_recomputes_status() -> None:
    base = PreparationReport.from_results(PreparationMode.RUNTIME, ())
    warning = failed_result(
        mode=PreparationMode.RUNTIME,
        code="telemetry_unreachable",
        component=PreparationComponent.TELEMETRY,
        message="Telemetry is unavailable.",
    )

    combined = base.combine(warning)

    assert combined.mode is PreparationMode.RUNTIME
    assert combined.status is PreparationStatus.DEGRADED
    assert combined.results == (warning,)


def test_custom_severity_resolver_is_a_supported_hook() -> None:
    def warnings_only(
        _mode: PreparationMode, _component: PreparationComponent
    ) -> CheckSeverity:
        return CheckSeverity.WARNING

    result = failed_result(
        mode=PreparationMode.EVALUATION,
        code="custom_dependency_failed",
        component=PreparationComponent.STORAGE,
        message="A custom dependency failed.",
        severity_resolver=warnings_only,
    )

    assert result.severity is CheckSeverity.WARNING
    assert not result.blocking
