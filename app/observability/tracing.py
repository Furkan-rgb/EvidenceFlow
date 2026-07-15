"""Small tracing interface that keeps MLflow out of domain and graph logic."""

from __future__ import annotations

import logging
import sys
from collections.abc import Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class Span(Protocol):
    def set_attribute(self, key: str, value: Any) -> None: ...

    def set_outputs(self, outputs: Any) -> None: ...


class Tracer(Protocol):
    @property
    def enabled(self) -> bool: ...

    @property
    def healthy(self) -> bool: ...

    @property
    def ever_failed(self) -> bool: ...

    def span(
        self,
        name: str,
        *,
        span_type: str = "CHAIN",
        attributes: Mapping[str, Any] | None = None,
    ) -> AbstractContextManager[Span | None]: ...


class NoOpTracer:
    @property
    def enabled(self) -> bool:
        return False

    @property
    def healthy(self) -> bool:
        return True

    @property
    def ever_failed(self) -> bool:
        return False

    @contextmanager
    def span(
        self,
        name: str,
        *,
        span_type: str = "CHAIN",
        attributes: Mapping[str, Any] | None = None,
    ) -> Iterator[None]:
        del name, span_type, attributes
        yield None


class MlflowTracer:
    """Best-effort runtime tracing; explicit evaluation checks remain fail-closed."""

    def __init__(self, tracking_uri: str, experiment_name: str) -> None:
        import mlflow

        self._mlflow = mlflow
        self._healthy = True
        self._ever_failed = False
        self._warned = False
        mlflow.set_tracking_uri(tracking_uri)
        try:
            mlflow.set_experiment(experiment_name)
        except Exception as error:  # pragma: no cover - depends on external server
            self._mark_failed(error)

    @property
    def enabled(self) -> bool:
        return True

    @property
    def healthy(self) -> bool:
        return self._healthy

    @property
    def ever_failed(self) -> bool:
        """Latch any loss of telemetry so evaluation can fail closed."""

        return self._ever_failed

    @contextmanager
    def span(
        self,
        name: str,
        *,
        span_type: str = "CHAIN",
        attributes: Mapping[str, Any] | None = None,
    ) -> Iterator[Span | None]:
        try:
            manager = self._mlflow.start_span(
                name=name,
                span_type=span_type,
                attributes=dict(attributes or {}),
            )
            span = manager.__enter__()
        except Exception as error:  # pragma: no cover - depends on external server
            self._mark_failed(error)
            yield None
            return
        self._healthy = True
        safe_span = _SafeSpan(span, self)
        try:
            yield safe_span
        except BaseException:
            try:
                manager.__exit__(*sys.exc_info())
            except Exception as error:  # pragma: no cover - external telemetry
                self._mark_failed(error)
            raise
        else:
            try:
                manager.__exit__(None, None, None)
            except Exception as error:  # pragma: no cover - external telemetry
                self._mark_failed(error)

    def _mark_failed(self, error: Exception) -> None:
        self._healthy = False
        self._ever_failed = True
        self._warn(error)

    def _warn(self, error: Exception) -> None:
        if self._warned:
            return
        logger.warning("MLflow tracing unavailable; review execution continues: %s", error)
        self._warned = True


class _SafeSpan:
    """Prevent telemetry mutations from changing runtime workflow outcomes."""

    def __init__(self, span: Any, owner: MlflowTracer) -> None:
        self._span = span
        self._owner = owner

    def set_attribute(self, key: str, value: Any) -> None:
        try:
            self._span.set_attribute(key, value)
        except Exception as error:  # pragma: no cover - external telemetry
            self._owner._mark_failed(error)

    def set_outputs(self, outputs: Any) -> None:
        try:
            self._span.set_outputs(outputs)
        except Exception as error:  # pragma: no cover - external telemetry
            self._owner._mark_failed(error)
