from __future__ import annotations

from typing import Any

import mlflow

from app.observability import MlflowTracer


class FailingSpan:
    def set_attribute(self, key: str, value: Any) -> None:
        del key, value
        raise RuntimeError("telemetry attribute failed")

    def set_outputs(self, outputs: Any) -> None:
        del outputs
        raise RuntimeError("telemetry output failed")


class FailingManager:
    def __enter__(self) -> FailingSpan:
        return FailingSpan()

    def __exit__(self, *args: object) -> None:
        del args
        raise RuntimeError("telemetry close failed")


def test_runtime_tracing_failures_do_not_change_workflow_outcome(
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(mlflow, "set_experiment", lambda _name: None)
    monkeypatch.setattr(mlflow, "start_span", lambda **_kwargs: FailingManager())
    tracer = MlflowTracer("file:///tmp/evidenceflow-test-mlflow", "test")

    with tracer.span("test") as span:
        assert span is not None
        span.set_attribute("safe", "value")
        span.set_outputs({"safe": True})

    assert tracer.healthy is False
    assert tracer.ever_failed is True


def test_application_error_is_preserved_when_trace_close_also_fails(
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(mlflow, "set_experiment", lambda _name: None)
    monkeypatch.setattr(mlflow, "start_span", lambda **_kwargs: FailingManager())
    tracer = MlflowTracer("file:///tmp/evidenceflow-test-mlflow", "test")

    try:
        with tracer.span("test"):
            raise ValueError("application failure")
    except ValueError as error:
        assert str(error) == "application failure"
    else:  # pragma: no cover - assertion guard
        raise AssertionError("application failure was swallowed")

    assert tracer.ever_failed is True
