from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

import app.bootstrap as bootstrap
from app.preparation import (
    PreparationBlockedError,
    PreparationComponent,
    PreparationMode,
    PreparationReport,
    failed_result,
)


@pytest.mark.asyncio
async def test_application_container_stops_before_composition_when_preparation_blocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = SimpleNamespace(ollama_base_url=None)
    models = SimpleNamespace()
    report = PreparationReport.from_results(
        PreparationMode.RUNTIME,
        [
            failed_result(
                mode=PreparationMode.RUNTIME,
                code="model_missing",
                component=PreparationComponent.MODEL_PROVIDER,
                message="Configured model is unavailable.",
                remediation="Install the configured model.",
            )
        ],
    )
    preparation_calls: list[tuple[object, object, PreparationMode]] = []
    construction_calls: list[str] = []

    monkeypatch.setattr(
        bootstrap,
        "load_models_config",
        lambda **_kwargs: models,
    )

    async def prepare_application(
        received_settings: object,
        received_models: object,
        *,
        mode: PreparationMode,
    ) -> PreparationReport:
        preparation_calls.append((received_settings, received_models, mode))
        return report

    monkeypatch.setattr(bootstrap, "prepare_application", prepare_application)

    def unexpected_construction(*_args: Any, **_kwargs: Any) -> None:
        construction_calls.append("constructed")

    for name in (
        "load_review_rules",
        "SQLiteReviewRepository",
        "LocalArtifactStore",
        "_build_tracer",
        "create_chat_model",
        "create_embedding_provider",
        "SqliteVecPolicyRetriever",
        "build_review_graph",
        "WorkflowRunner",
    ):
        monkeypatch.setattr(bootstrap, name, unexpected_construction)
    monkeypatch.setattr(bootstrap.aiosqlite, "connect", unexpected_construction)

    with pytest.raises(PreparationBlockedError) as caught:
        async with bootstrap.application_container(settings):
            pytest.fail("a blocked application container must never yield")

    assert caught.value.report is report
    assert preparation_calls == [(settings, models, PreparationMode.RUNTIME)]
    assert construction_calls == []


@pytest.mark.asyncio
async def test_application_container_redacts_invalid_model_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = SimpleNamespace(ollama_base_url=None)

    def invalid_models(**_kwargs: object) -> None:
        raise ValueError("api_key=must-not-escape")

    monkeypatch.setattr(bootstrap, "load_models_config", invalid_models)

    with pytest.raises(PreparationBlockedError) as caught:
        async with bootstrap.application_container(settings):
            pytest.fail("invalid model configuration must never start the app")

    assert str(caught.value) == "EvidenceFlow preparation failed (configuration_invalid)."
    assert "must-not-escape" not in str(caught.value)
    assert caught.value.__cause__ is None
