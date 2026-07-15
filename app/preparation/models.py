"""Typed preparation results shared by startup, CLI, and evaluation gates."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import StrEnum


class PreparationMode(StrEnum):
    """The operation whose dependencies are being prepared."""

    RUNTIME = "runtime"
    POLICY_INDEX_REBUILD = "policy_index_rebuild"
    EVALUATION = "evaluation"


class ModelTask(StrEnum):
    """Stable identities for independently configured model capabilities."""

    CLASSIFICATION = "classification"
    EXTRACTION = "extraction"
    REPORTING = "reporting"
    EMBEDDINGS = "embeddings"


class PreparationComponent(StrEnum):
    """Dependency categories used by the shared severity policy."""

    CONFIGURATION = "configuration"
    MODEL_PROVIDER = "model_provider"
    POLICY_INDEX = "policy_index"
    STORAGE = "storage"
    TELEMETRY = "telemetry"


class CheckOutcome(StrEnum):
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


class CheckSeverity(StrEnum):
    """Impact of a failed check on the requested preparation mode."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class PreparationStatus(StrEnum):
    READY = "ready"
    DEGRADED = "degraded"
    BLOCKED = "blocked"


SeverityResolver = Callable[[PreparationMode, PreparationComponent], CheckSeverity]


def default_failure_severity(
    mode: PreparationMode, component: PreparationComponent
) -> CheckSeverity:
    """Resolve whether a dependency failure blocks the requested operation.

    Runtime telemetry deliberately fails open. Evaluation telemetry fails closed
    so a metrics run can never silently lose required traces. All other dependency
    failures are critical in every mode.
    """

    if component is PreparationComponent.TELEMETRY and mode is not PreparationMode.EVALUATION:
        return CheckSeverity.WARNING
    return CheckSeverity.CRITICAL


@dataclass(frozen=True, slots=True)
class PreparationResult:
    """One redacted, machine-readable dependency check result."""

    code: str
    outcome: CheckOutcome
    severity: CheckSeverity
    component: PreparationComponent
    message: str
    remediation: str | None = None
    task: ModelTask | None = None
    provider: str | None = None
    model: str | None = None
    endpoint: str | None = None
    expected_digest: str | None = None
    observed_digest: str | None = None
    http_status: int | None = None

    @property
    def blocking(self) -> bool:
        return self.outcome is CheckOutcome.FAILED and self.severity is CheckSeverity.CRITICAL

    @property
    def warning(self) -> bool:
        return self.outcome is CheckOutcome.FAILED and self.severity is CheckSeverity.WARNING


@dataclass(frozen=True, slots=True)
class PreparationReport:
    """Aggregate readiness for one operation."""

    mode: PreparationMode
    results: tuple[PreparationResult, ...]

    @classmethod
    def from_results(
        cls,
        mode: PreparationMode,
        results: Iterable[PreparationResult],
    ) -> PreparationReport:
        return cls(mode=mode, results=tuple(results))

    @property
    def blocking_failures(self) -> tuple[PreparationResult, ...]:
        return tuple(result for result in self.results if result.blocking)

    @property
    def warnings(self) -> tuple[PreparationResult, ...]:
        return tuple(result for result in self.results if result.warning)

    @property
    def ready(self) -> bool:
        return not self.blocking_failures

    @property
    def status(self) -> PreparationStatus:
        if self.blocking_failures:
            return PreparationStatus.BLOCKED
        if self.warnings:
            return PreparationStatus.DEGRADED
        return PreparationStatus.READY

    def for_task(self, task: ModelTask) -> tuple[PreparationResult, ...]:
        return tuple(result for result in self.results if result.task is task)

    def combine(self, *results: PreparationResult) -> PreparationReport:
        """Return a report with additional checks while preserving its mode."""

        return PreparationReport(self.mode, (*self.results, *results))

    def merge(self, *reports: PreparationReport) -> PreparationReport:
        """Merge reports for the same operation into one ordered report."""

        if any(report.mode is not self.mode for report in reports):
            raise ValueError("Cannot merge preparation reports for different modes.")
        return PreparationReport(
            self.mode,
            (*self.results, *(result for report in reports for result in report.results)),
        )


def passed_result(
    *,
    code: str,
    component: PreparationComponent,
    message: str,
    remediation: str | None = None,
    task: ModelTask | None = None,
    provider: str | None = None,
    model: str | None = None,
    endpoint: str | None = None,
    expected_digest: str | None = None,
    observed_digest: str | None = None,
    http_status: int | None = None,
) -> PreparationResult:
    """Build a successful result with consistent informational severity."""

    return PreparationResult(
        code=code,
        outcome=CheckOutcome.PASSED,
        severity=CheckSeverity.INFO,
        component=component,
        message=message,
        remediation=remediation,
        task=task,
        provider=provider,
        model=model,
        endpoint=endpoint,
        expected_digest=expected_digest,
        observed_digest=observed_digest,
        http_status=http_status,
    )


def skipped_result(
    *,
    code: str,
    component: PreparationComponent,
    message: str,
    remediation: str | None = None,
    task: ModelTask | None = None,
    provider: str | None = None,
    model: str | None = None,
) -> PreparationResult:
    """Build an explicit non-blocking result for a check that could not run."""

    return PreparationResult(
        code=code,
        outcome=CheckOutcome.SKIPPED,
        severity=CheckSeverity.INFO,
        component=component,
        message=message,
        remediation=remediation,
        task=task,
        provider=provider,
        model=model,
    )


def failed_result(
    *,
    mode: PreparationMode,
    code: str,
    component: PreparationComponent,
    message: str,
    remediation: str | None = None,
    task: ModelTask | None = None,
    provider: str | None = None,
    model: str | None = None,
    endpoint: str | None = None,
    expected_digest: str | None = None,
    observed_digest: str | None = None,
    http_status: int | None = None,
    severity_resolver: SeverityResolver = default_failure_severity,
) -> PreparationResult:
    """Build a failed result using the operation's severity policy."""

    return PreparationResult(
        code=code,
        outcome=CheckOutcome.FAILED,
        severity=severity_resolver(mode, component),
        component=component,
        message=message,
        remediation=remediation,
        task=task,
        provider=provider,
        model=model,
        endpoint=endpoint,
        expected_digest=expected_digest,
        observed_digest=observed_digest,
        http_status=http_status,
    )
