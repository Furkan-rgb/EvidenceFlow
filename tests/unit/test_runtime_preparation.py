from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pytest

from app.ai.config import ModelsConfig
from app.ai.fakes import DeterministicEmbeddingProvider
from app.config import Settings
from app.preparation import (
    CheckOutcome,
    CheckSeverity,
    ModelTask,
    PreparationComponent,
    PreparationMode,
    PreparationReport,
    PreparationStatus,
    failed_result,
    runtime,
)
from app.retrieval import PolicyIndexBuilder

PROJECT_ROOT = Path(__file__).parents[2]
POLICIES_DIR = PROJECT_ROOT / "policies"
RULES_PATH = PROJECT_ROOT / "config" / "review_rules.yaml"
CHAT_DIGEST = "a" * 64
EMBEDDING_DIGEST = "b" * 64


class _IndexEmbeddingProvider(DeterministicEmbeddingProvider):
    provider = "ollama"
    model = "embedding-model"
    model_digest = EMBEDDING_DIGEST


def _settings(tmp_path: Path, *, mlflow_enabled: bool = True) -> Settings:
    return Settings(
        _env_file=None,
        EVIDENCEFLOW_DATA_DIR=tmp_path / "data",
        EVIDENCEFLOW_RULES_CONFIG=RULES_PATH,
        EVIDENCEFLOW_POLICIES_DIR=POLICIES_DIR,
        EVIDENCEFLOW_MLFLOW_ENABLED=mlflow_enabled,
        MLFLOW_TRACKING_URI="http://mlflow.test:5001",
    )


def _models(
    *,
    classification_provider: str = "ollama",
    embedding_provider: str = "ollama",
) -> ModelsConfig:
    return ModelsConfig.model_validate(
        {
            "classification": {
                "provider": classification_provider,
                "model": "chat-model",
                "model_digest": CHAT_DIGEST,
                "base_url": "http://ollama.test:11434",
            },
            "extraction": {
                "provider": "ollama",
                "model": "chat-model",
                "model_digest": CHAT_DIGEST,
                "base_url": "http://ollama.test:11434",
            },
            "reporting": {
                "provider": "ollama",
                "model": "chat-model",
                "model_digest": CHAT_DIGEST,
                "base_url": "http://ollama.test:11434",
            },
            "embeddings": {
                "provider": embedding_provider,
                "model": "embedding-model",
                "model_digest": EMBEDDING_DIGEST,
                "dimensions": 8,
                "base_url": "http://ollama.test:11434",
            },
        }
    )


async def _build_compatible_policy_index(settings: Settings) -> None:
    provider = _IndexEmbeddingProvider(dimensions=8)
    await PolicyIndexBuilder(
        provider,
        dimensions=8,
        model_digest=EMBEDDING_DIGEST,
    ).rebuild(
        policies_dir=settings.policies_dir,
        index_path=settings.policy_index_path,
        manifest_path=settings.policy_manifest_path,
    )


@pytest.mark.asyncio
async def test_healthy_runtime_preparation_aggregates_every_dependency(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    models = _models()
    await _build_compatible_policy_index(settings)
    inventory_calls: list[tuple[str, float]] = []
    mlflow_calls: list[str] = []

    async def inventory(endpoint: str, timeout_seconds: float) -> Mapping[str, str]:
        inventory_calls.append((endpoint, timeout_seconds))
        return {"chat-model": CHAT_DIGEST, "embedding-model": EMBEDDING_DIGEST}

    async def mlflow_probe(tracking_uri: str) -> bool:
        mlflow_calls.append(tracking_uri)
        return True

    report = await runtime.prepare_application(
        settings,
        models,
        ollama_inventory_loader=inventory,
        mlflow_health_probe=mlflow_probe,
    )

    assert report.status is PreparationStatus.READY
    assert report.ready
    assert [result.code for result in report.results] == [
        "configuration_valid",
        "storage_ready",
        "database_ready",
        "checkpoint_store_ready",
        "model_ready",
        "model_ready",
        "model_ready",
        "model_ready",
        "policy_index_ready",
        "telemetry_ready",
    ]
    assert inventory_calls == [("http://ollama.test:11434", 2.0)]
    assert mlflow_calls == ["http://mlflow.test:5001"]


@pytest.mark.asyncio
async def test_mlflow_unavailable_warns_at_runtime_but_blocks_evaluation(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)

    async def unavailable(_tracking_uri: str) -> bool:
        return False

    runtime_result = await runtime._check_mlflow(
        settings,
        mode=PreparationMode.RUNTIME,
        health_probe=unavailable,
    )
    evaluation_result = await runtime._check_mlflow(
        settings,
        mode=PreparationMode.EVALUATION,
        health_probe=unavailable,
    )

    assert runtime_result.code == "telemetry_unavailable"
    assert runtime_result.severity is CheckSeverity.WARNING
    assert PreparationReport.from_results(
        PreparationMode.RUNTIME, [runtime_result]
    ).status is PreparationStatus.DEGRADED
    assert evaluation_result.code == "telemetry_unavailable"
    assert evaluation_result.severity is CheckSeverity.CRITICAL
    assert PreparationReport.from_results(
        PreparationMode.EVALUATION, [evaluation_result]
    ).status is PreparationStatus.BLOCKED


@pytest.mark.asyncio
async def test_disabled_mlflow_is_skipped_at_runtime_and_blocks_evaluation(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path, mlflow_enabled=False)

    async def unexpected_probe(_tracking_uri: str) -> bool:
        pytest.fail("disabled telemetry must not be probed")

    runtime_result = await runtime._check_mlflow(
        settings,
        mode=PreparationMode.RUNTIME,
        health_probe=unexpected_probe,
    )
    evaluation_result = await runtime._check_mlflow(
        settings,
        mode=PreparationMode.EVALUATION,
        health_probe=unexpected_probe,
    )

    assert runtime_result.code == "telemetry_disabled"
    assert runtime_result.outcome is CheckOutcome.SKIPPED
    assert runtime_result.severity is CheckSeverity.INFO
    assert evaluation_result.code == "telemetry_disabled"
    assert evaluation_result.outcome is CheckOutcome.FAILED
    assert evaluation_result.severity is CheckSeverity.CRITICAL


def test_storage_failure_is_critical_and_redacts_internal_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(tmp_path)

    def fail_atomic_write(_directory: Path) -> None:
        raise OSError("secret mount detail")

    monkeypatch.setattr(runtime, "_verify_atomic_write", fail_atomic_write)

    result = runtime._check_writable_storage(settings, PreparationMode.RUNTIME)

    assert result.code == "storage_unavailable"
    assert result.blocking
    assert "secret mount detail" not in result.message
    assert "secret mount detail" not in (result.remediation or "")


@pytest.mark.asyncio
async def test_business_database_failure_is_a_critical_storage_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(tmp_path)

    class FailingRepository:
        def __init__(self, _database_path: Path) -> None:
            pass

        async def migrate(self) -> None:
            raise RuntimeError("database password must not leak")

        async def health(self) -> bool:
            pytest.fail("health must not run after migration fails")

    monkeypatch.setattr(runtime, "SQLiteReviewRepository", FailingRepository)

    result = await runtime._check_business_database(
        settings,
        mode=PreparationMode.RUNTIME,
    )

    assert result.code == "database_unavailable"
    assert result.blocking
    assert "password" not in result.message
    assert "password" not in (result.remediation or "")


@pytest.mark.asyncio
async def test_checkpoint_database_failure_is_a_critical_storage_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(tmp_path)

    def fail_connect(_path: Path) -> None:
        raise OSError("private checkpoint path detail")

    monkeypatch.setattr(runtime.aiosqlite, "connect", fail_connect)

    result = await runtime._check_checkpoint_database(
        settings,
        mode=PreparationMode.RUNTIME,
    )

    assert result.code == "checkpoint_store_unavailable"
    assert result.blocking
    assert "private checkpoint path detail" not in result.message
    assert "private checkpoint path detail" not in (result.remediation or "")


@pytest.mark.asyncio
async def test_policy_index_is_skipped_when_embedding_provider_is_not_ready(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(tmp_path, mlflow_enabled=False)
    models = _models(embedding_provider="openai")

    async def inventory(_endpoint: str, _timeout: float) -> Mapping[str, str]:
        return {"chat-model": CHAT_DIGEST}

    def unexpected_index_check(*_args: object, **_kwargs: object) -> None:
        pytest.fail("an index cannot be validated before its embedder is ready")

    monkeypatch.setattr(runtime, "_check_policy_index", unexpected_index_check)

    report = await runtime.prepare_application(
        settings,
        models,
        ollama_inventory_loader=inventory,
    )

    embedding_failure = report.for_task(ModelTask.EMBEDDINGS)[0]
    assert embedding_failure.code == "provider_not_implemented"
    assert embedding_failure.blocking
    skipped = next(
        result for result in report.results if result.code == "policy_index_check_skipped"
    )
    assert skipped.outcome is CheckOutcome.SKIPPED
    assert skipped.component is PreparationComponent.POLICY_INDEX
    assert not settings.policy_index_path.exists()


@pytest.mark.asyncio
async def test_policy_rebuild_checks_only_embedder_corpus_and_local_storage(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    settings = settings.model_copy(
        update={"rules_config": tmp_path / "rules-that-are-not-needed.yaml"}
    )
    models = _models(classification_provider="openai")
    inventory_calls: list[str] = []

    async def inventory(endpoint: str, _timeout: float) -> Mapping[str, str]:
        inventory_calls.append(endpoint)
        return {"embedding-model": EMBEDDING_DIGEST}

    async def unexpected_mlflow_probe(_tracking_uri: str) -> bool:
        pytest.fail("policy-index rebuild preparation must not require MLflow")

    report = await runtime.prepare_application(
        settings,
        models,
        mode=PreparationMode.POLICY_INDEX_REBUILD,
        ollama_inventory_loader=inventory,
        mlflow_health_probe=unexpected_mlflow_probe,
    )

    assert report.ready
    assert [result.code for result in report.results] == [
        "configuration_valid",
        "storage_ready",
        "model_ready",
        "policy_corpus_ready",
    ]
    assert report.for_task(ModelTask.CLASSIFICATION) == ()
    assert report.for_task(ModelTask.EMBEDDINGS)[0].code == "model_ready"
    assert inventory_calls == ["http://ollama.test:11434"]
    assert not settings.database_path.exists()
    assert not settings.checkpoints_path.exists()
    assert not settings.policy_index_path.exists()
    assert not settings.policy_manifest_path.exists()


def test_require_prepared_raises_only_redacted_failure_codes() -> None:
    failure = failed_result(
        mode=PreparationMode.RUNTIME,
        code="provider_authentication_failed",
        component=PreparationComponent.MODEL_PROVIDER,
        message="api_key=must-not-escape",
        remediation="Bearer must-not-escape",
        provider="future-cloud-provider",
        model="private-deployment-name",
    )
    report = PreparationReport.from_results(PreparationMode.RUNTIME, [failure])

    with pytest.raises(runtime.PreparationBlockedError) as caught:
        runtime.require_prepared(report)

    rendered = str(caught.value)
    assert rendered == "EvidenceFlow preparation failed (provider_authentication_failed)."
    assert "must-not-escape" not in rendered
    assert "private-deployment-name" not in rendered
    assert caught.value.report is report


def test_require_prepared_accepts_a_ready_report() -> None:
    report = PreparationReport.from_results(PreparationMode.RUNTIME, ())

    runtime.require_prepared(report)
