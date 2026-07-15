from __future__ import annotations

from types import SimpleNamespace

import pytest
import typer

from app import cli
from app.preparation import (
    ModelTask,
    PreparationComponent,
    PreparationMode,
    PreparationReport,
    failed_result,
    passed_result,
    skipped_result,
)


def _patch_preparation(
    monkeypatch: pytest.MonkeyPatch,
    report: PreparationReport,
) -> None:
    settings = SimpleNamespace()
    models = SimpleNamespace()
    monkeypatch.setattr(cli, "_settings_and_models", lambda: (settings, models))

    async def prepare_application(
        received_settings: object,
        received_models: object,
        *,
        mode: PreparationMode,
    ) -> PreparationReport:
        assert received_settings is settings
        assert received_models is models
        assert mode is PreparationMode.RUNTIME
        return report

    monkeypatch.setattr(cli, "prepare_application", prepare_application)


def test_prepare_allows_runtime_when_only_mlflow_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    report = PreparationReport.from_results(
        PreparationMode.RUNTIME,
        [
            passed_result(
                code="model_ready",
                component=PreparationComponent.MODEL_PROVIDER,
                message="Configured model is available.",
                task=ModelTask.CLASSIFICATION,
            ),
            failed_result(
                mode=PreparationMode.RUNTIME,
                code="telemetry_unavailable",
                component=PreparationComponent.TELEMETRY,
                message="MLflow is unavailable; runtime tracing will fail open.",
                remediation="Start MLflow to restore tracing.",
            ),
        ],
    )
    _patch_preparation(monkeypatch, report)

    cli.prepare()

    captured = capsys.readouterr()
    assert "EvidenceFlow preparation (runtime)" in captured.out
    assert "[ok] classification: Configured model is available." in captured.out
    assert "Preparation ready with warnings." in captured.out
    assert "[warning] telemetry: MLflow is unavailable" in captured.out
    assert "Fix: Start MLflow" in captured.out
    assert captured.err == ""


def test_prepare_exits_when_critical_checks_fail(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    report = PreparationReport.from_results(
        PreparationMode.RUNTIME,
        [
            failed_result(
                mode=PreparationMode.RUNTIME,
                code="model_missing",
                component=PreparationComponent.MODEL_PROVIDER,
                message="Configured model gemma4:12b-mlx is not installed.",
                remediation="Run ollama pull gemma4:12b-mlx.",
                task=ModelTask.CLASSIFICATION,
                provider="ollama",
                model="gemma4:12b-mlx",
            ),
            failed_result(
                mode=PreparationMode.RUNTIME,
                code="policy_index_missing",
                component=PreparationComponent.POLICY_INDEX,
                message="The policy index has not been built.",
                remediation="Run make rebuild.",
            ),
        ],
    )
    _patch_preparation(monkeypatch, report)

    with pytest.raises(typer.Exit) as error:
        cli.prepare()

    assert error.value.exit_code == 1
    captured = capsys.readouterr()
    assert "gemma4:12b-mlx" in captured.err
    assert "Fix: Run make rebuild." in captured.err
    assert "Preparation blocked by 2 critical check(s)." in captured.err
    assert "Preparation ready" not in captured.out


def test_doctor_is_a_compatibility_alias_for_prepare(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[None] = []

    def prepare() -> None:
        calls.append(None)

    monkeypatch.setattr(cli, "prepare", prepare)

    cli.doctor()

    assert calls == [None]


def test_prepare_reports_intentionally_disabled_mlflow(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    report = PreparationReport.from_results(
        PreparationMode.RUNTIME,
        [
            skipped_result(
                code="telemetry_disabled",
                component=PreparationComponent.TELEMETRY,
                message="MLflow tracing is intentionally disabled.",
            )
        ],
    )
    _patch_preparation(monkeypatch, report)

    cli.prepare()

    captured = capsys.readouterr()
    assert "[disabled] telemetry: MLflow tracing is intentionally disabled." in captured.out
    assert "Preparation ready." in captured.out
    assert "MLflow is unavailable" not in captured.err
