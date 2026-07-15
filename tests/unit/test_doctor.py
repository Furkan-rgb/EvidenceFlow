from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import typer

from app import cli


def test_doctor_allows_runtime_when_only_mlflow_is_unavailable(
    monkeypatch: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        cli,
        "_settings_and_models",
        lambda: (SimpleNamespace(), SimpleNamespace()),
    )

    async def checks(*_args: object) -> dict[str, object]:
        return {
            "ollama_ok": True,
            "missing_models": [],
            "digest_mismatches": {},
            "policy_index_ok": True,
            "policy_index_error": None,
            "mlflow_ok": False,
        }

    monkeypatch.setattr(cli, "_doctor_checks", checks)

    cli.doctor()

    captured = capsys.readouterr()
    assert "[ok] Ollama" in captured.out
    assert "[warning] MLflow is unavailable" in captured.err


def test_doctor_fails_when_models_or_policy_index_are_missing(
    monkeypatch: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        cli,
        "_settings_and_models",
        lambda: (SimpleNamespace(), SimpleNamespace()),
    )

    async def checks(*_args: object) -> dict[str, object]:
        return {
            "ollama_ok": False,
            "missing_models": ["gemma4:12b-mlx"],
            "digest_mismatches": {},
            "policy_index_ok": False,
            "policy_index_error": "index missing",
            "mlflow_ok": True,
        }

    monkeypatch.setattr(cli, "_doctor_checks", checks)

    with pytest.raises(typer.Exit) as error:
        cli.doctor()

    assert error.value.exit_code == 1
    captured = capsys.readouterr()
    assert "gemma4:12b-mlx" in captured.err
    assert "make rebuild" in captured.err


def test_doctor_reports_intentionally_disabled_mlflow(
    monkeypatch: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        cli,
        "_settings_and_models",
        lambda: (SimpleNamespace(), SimpleNamespace()),
    )

    async def checks(*_args: object) -> dict[str, object]:
        return {
            "ollama_ok": True,
            "missing_models": [],
            "digest_mismatches": {},
            "policy_index_ok": True,
            "policy_index_error": None,
            "mlflow_enabled": False,
            "mlflow_ok": True,
        }

    monkeypatch.setattr(cli, "_doctor_checks", checks)

    cli.doctor()

    captured = capsys.readouterr()
    assert "[disabled] MLflow tracing" in captured.out
    assert "MLflow is unavailable" not in captured.err
