from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import mlflow

from app import cli


class _HealthyTracer:
    healthy = True
    ever_failed = False

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        pass


def test_evaluate_removes_progress_only_after_mlflow_logging(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    bundles_dir = tmp_path / "bundles"
    bundles_dir.mkdir()
    output_dir = tmp_path / "results"
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    progress_path = data_dir / "evaluation-progress.json"
    settings = SimpleNamespace(
        data_dir=data_dir,
        mlflow_tracking_uri="file:///unused",
    )

    monkeypatch.setattr(cli, "_settings_and_models", lambda: (settings, object()))
    monkeypatch.setattr(cli, "_require_ollama", lambda *_args: None)
    monkeypatch.setattr(cli, "MlflowTracer", _HealthyTracer)

    async def fake_run_real_evaluation(
        **kwargs: Any,
    ) -> tuple[dict[str, object], tuple[Path, Path]]:
        assert kwargs["progress_path"] == progress_path
        progress_path.write_text("checkpoint", encoding="utf-8")
        output_dir.mkdir()
        json_path = output_dir / "evaluation-results.json"
        markdown_path = output_dir / "evaluation-results.md"
        json_path.write_text("{}\n", encoding="utf-8")
        markdown_path.write_text("# Results\n", encoding="utf-8")
        return (
            {
                "aggregate_metrics": {},
                "duration_seconds": 1.0,
                "bundle_count": 1,
                "bundle_results": [],
                "retrieval_metrics": None,
                "metadata": {},
            },
            (json_path, markdown_path),
        )

    monkeypatch.setattr(cli, "_run_real_evaluation", fake_run_real_evaluation)
    monkeypatch.setattr(mlflow, "set_tracking_uri", lambda _uri: None)
    monkeypatch.setattr(mlflow, "set_experiment", lambda _name: None)
    monkeypatch.setattr(mlflow, "log_metrics", lambda _metrics: None)
    monkeypatch.setattr(mlflow, "log_params", lambda _params: None)

    def log_artifacts(_path: str, *, artifact_path: str) -> None:
        assert artifact_path == "evaluation"
        assert progress_path.exists()

    monkeypatch.setattr(mlflow, "log_artifacts", log_artifacts)

    class RunContext:
        def __enter__(self) -> None:
            return None

        def __exit__(self, *_args: Any) -> None:
            assert progress_path.exists()

    monkeypatch.setattr(mlflow, "start_run", lambda **_kwargs: RunContext())

    cli.evaluate(bundles_dir=bundles_dir, output_dir=output_dir)

    assert not progress_path.exists()
