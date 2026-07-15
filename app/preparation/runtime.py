"""Shared application preparation checks used by CLI and lifespan startup."""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Awaitable, Callable
from contextlib import suppress
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import urlopen
from uuid import uuid4

import aiosqlite
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from app.ai.config import ModelsConfig
from app.ai.models import create_embedding_provider
from app.config import Settings
from app.errors import EmbeddingIndexMismatchError, PolicyIndexMissingError
from app.persistence import SQLiteReviewRepository
from app.preparation.models import (
    CheckOutcome,
    ModelTask,
    PreparationComponent,
    PreparationMode,
    PreparationReport,
    PreparationResult,
    failed_result,
    passed_result,
    skipped_result,
)
from app.preparation.ollama import OllamaInventoryLoader
from app.preparation.providers import ProviderCheckerRegistry
from app.preparation.service import prepare_model_providers
from app.retrieval import SqliteVecPolicyRetriever
from app.retrieval.chunking import load_policy_corpus
from app.review import load_review_rules

MlflowHealthProbe = Callable[[str], Awaitable[bool]]


class PreparationBlockedError(RuntimeError):
    """Safe startup error carrying the complete redacted preparation report."""

    def __init__(self, report: PreparationReport) -> None:
        self.report = report
        codes = ", ".join(result.code for result in report.blocking_failures)
        super().__init__(f"EvidenceFlow preparation failed ({codes}).")


async def prepare_application(
    settings: Settings,
    models: ModelsConfig,
    *,
    mode: PreparationMode = PreparationMode.RUNTIME,
    provider_registry: ProviderCheckerRegistry | None = None,
    ollama_inventory_loader: OllamaInventoryLoader | None = None,
    mlflow_health_probe: MlflowHealthProbe | None = None,
) -> PreparationReport:
    """Check every dependency needed for one operation.

    The same function powers the explicit CLI preparation command and lifespan
    startup. Model providers remain injectable so future cloud adapters can add
    credential and authentication checks without changing the orchestration.
    """

    results: list[PreparationResult] = []
    results.append(_check_typed_configuration(settings, mode=mode))
    results.append(await asyncio.to_thread(_check_writable_storage, settings, mode))

    if mode is not PreparationMode.POLICY_INDEX_REBUILD:
        results.append(await _check_business_database(settings, mode=mode))
        results.append(await _check_checkpoint_database(settings, mode=mode))

    provider_report = await prepare_model_providers(
        models,
        mode=mode,
        default_ollama_base_url=settings.ollama_base_url,
        registry=provider_registry,
        ollama_inventory_loader=ollama_inventory_loader,
    )
    results.extend(provider_report.results)

    if mode is PreparationMode.POLICY_INDEX_REBUILD:
        results.append(await asyncio.to_thread(_check_policy_corpus, settings, mode))
    elif _task_ready(provider_report, ModelTask.EMBEDDINGS):
        results.append(await _check_policy_index(settings, models, mode))
    else:
        results.append(
            skipped_result(
                code="policy_index_check_skipped",
                component=PreparationComponent.POLICY_INDEX,
                message=(
                    "Policy-index compatibility was not checked because the "
                    "embedding provider is not ready."
                ),
                remediation=(
                    "Resolve the embedding-provider failure, then retry preparation."
                ),
                task=ModelTask.EMBEDDINGS,
            )
        )

    if mode is not PreparationMode.POLICY_INDEX_REBUILD:
        results.append(
            await _check_mlflow(
                settings,
                mode=mode,
                health_probe=mlflow_health_probe or mlflow_health,
            )
        )

    return PreparationReport.from_results(mode, results)


def require_prepared(report: PreparationReport) -> None:
    """Raise a redacted error when a report cannot start its requested operation."""

    if not report.ready:
        raise PreparationBlockedError(report)


def _check_typed_configuration(
    settings: Settings, *, mode: PreparationMode
) -> PreparationResult:
    try:
        if mode is not PreparationMode.POLICY_INDEX_REBUILD:
            load_review_rules(settings.rules_config)
    except Exception:
        return failed_result(
            mode=mode,
            code="configuration_invalid",
            component=PreparationComponent.CONFIGURATION,
            message="The deterministic review-rules configuration is invalid.",
            remediation="Correct config/review_rules.yaml and retry preparation.",
        )
    return passed_result(
        code="configuration_valid",
        component=PreparationComponent.CONFIGURATION,
        message=(
            "The model registry and review rules are valid."
            if mode is not PreparationMode.POLICY_INDEX_REBUILD
            else "The model registry is valid."
        ),
    )


def _check_writable_storage(
    settings: Settings, mode: PreparationMode
) -> PreparationResult:
    try:
        settings.ensure_directories()
        for directory in (settings.data_dir, settings.uploads_dir, settings.exports_dir):
            _verify_atomic_write(directory)
    except Exception:
        return failed_result(
            mode=mode,
            code="storage_unavailable",
            component=PreparationComponent.STORAGE,
            message="EvidenceFlow runtime storage is not writable.",
            remediation="Check the configured data directory and filesystem permissions.",
        )
    return passed_result(
        code="storage_ready",
        component=PreparationComponent.STORAGE,
        message="Runtime storage supports writable, atomic file replacement.",
    )


def _verify_atomic_write(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    token = uuid4().hex
    temporary = directory / f".evidenceflow-prepare-{token}.tmp"
    published = directory / f".evidenceflow-prepare-{token}.ready"
    try:
        with temporary.open("wb") as handle:
            handle.write(b"ready")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, published)
        if published.read_bytes() != b"ready":
            raise OSError("storage verification read did not match its write")
    finally:
        temporary.unlink(missing_ok=True)
        published.unlink(missing_ok=True)


async def _check_business_database(
    settings: Settings, *, mode: PreparationMode
) -> PreparationResult:
    try:
        repository = SQLiteReviewRepository(settings.database_path)
        await repository.migrate()
        healthy = await repository.health()
    except Exception:
        healthy = False
    if not healthy:
        return failed_result(
            mode=mode,
            code="database_unavailable",
            component=PreparationComponent.STORAGE,
            message="The review database could not be migrated or queried.",
            remediation="Check the data directory, SQLite files, and filesystem permissions.",
        )
    return passed_result(
        code="database_ready",
        component=PreparationComponent.STORAGE,
        message="The review database and numbered migrations are ready.",
    )


async def _check_checkpoint_database(
    settings: Settings, *, mode: PreparationMode
) -> PreparationResult:
    connection: aiosqlite.Connection | None = None
    try:
        connection = await aiosqlite.connect(settings.checkpoints_path)
        checkpointer = AsyncSqliteSaver(
            connection,
            serde=JsonPlusSerializer(pickle_fallback=False),
        )
        await checkpointer.setup()
        async with connection.execute("PRAGMA quick_check") as cursor:
            row = await cursor.fetchone()
        healthy = row == ("ok",)
    except Exception:
        healthy = False
    finally:
        if connection is not None:
            try:
                await connection.close()
            except Exception:
                healthy = False
    if not healthy:
        return failed_result(
            mode=mode,
            code="checkpoint_store_unavailable",
            component=PreparationComponent.STORAGE,
            message="The durable LangGraph checkpoint database is unavailable.",
            remediation="Check the data directory, checkpoint database, and permissions.",
        )
    return passed_result(
        code="checkpoint_store_ready",
        component=PreparationComponent.STORAGE,
        message="The durable LangGraph checkpoint database is ready.",
    )


def _task_ready(report: PreparationReport, task: ModelTask) -> bool:
    task_results = report.for_task(task)
    return bool(task_results) and all(
        result.outcome is CheckOutcome.PASSED for result in task_results
    )


def _check_policy_corpus(
    settings: Settings, mode: PreparationMode
) -> PreparationResult:
    try:
        corpus = load_policy_corpus(settings.policies_dir)
    except Exception:
        return failed_result(
            mode=mode,
            code="policy_corpus_invalid",
            component=PreparationComponent.POLICY_INDEX,
            message="The Markdown policy corpus is missing or invalid.",
            remediation="Correct the policies directory before rebuilding the index.",
        )
    return passed_result(
        code="policy_corpus_ready",
        component=PreparationComponent.POLICY_INDEX,
        message=(
            f"The policy corpus contains {corpus.document_count} documents and "
            f"{len(corpus.chunks)} indexable chunks."
        ),
    )


async def _check_policy_index(
    settings: Settings,
    models: ModelsConfig,
    mode: PreparationMode,
) -> PreparationResult:
    retriever: SqliteVecPolicyRetriever | None = None
    embedder = None
    try:
        embedder = create_embedding_provider(models.embeddings)
        retriever = await asyncio.to_thread(
            SqliteVecPolicyRetriever,
            embedder,
            dimensions=models.embeddings.dimensions,
            index_path=settings.policy_index_path,
            manifest_path=settings.policy_manifest_path,
            model_digest=models.embeddings.model_digest,
            policies_dir=settings.policies_dir,
        )
    except PolicyIndexMissingError:
        return failed_result(
            mode=mode,
            code="policy_index_missing",
            component=PreparationComponent.POLICY_INDEX,
            message="The local policy index is missing or invalid.",
            remediation="Run `make rebuild`, then retry preparation.",
        )
    except EmbeddingIndexMismatchError:
        return failed_result(
            mode=mode,
            code="policy_index_incompatible",
            component=PreparationComponent.POLICY_INDEX,
            message="The policy index does not match its corpus or configured embedder.",
            remediation="Run `make rebuild` with the intended embedding configuration.",
        )
    except Exception:
        return failed_result(
            mode=mode,
            code="policy_index_unavailable",
            component=PreparationComponent.POLICY_INDEX,
            message="The policy index could not be validated.",
            remediation="Check the local index files or run `make rebuild`.",
        )
    finally:
        if retriever is not None:
            with suppress(Exception):
                await asyncio.to_thread(retriever.close)
        if embedder is not None:
            await embedder.aclose()
    return passed_result(
        code="policy_index_ready",
        component=PreparationComponent.POLICY_INDEX,
        message="The policy index matches its corpus and configured embedder.",
        task=ModelTask.EMBEDDINGS,
        provider=models.embeddings.provider,
        model=models.embeddings.model,
    )


async def _check_mlflow(
    settings: Settings,
    *,
    mode: PreparationMode,
    health_probe: MlflowHealthProbe,
) -> PreparationResult:
    if not settings.mlflow_enabled:
        if mode is PreparationMode.EVALUATION:
            return failed_result(
                mode=mode,
                code="telemetry_disabled",
                component=PreparationComponent.TELEMETRY,
                message="MLflow tracing is disabled, but evaluation requires it.",
                remediation="Enable MLflow tracing and start the tracking server.",
            )
        return skipped_result(
            code="telemetry_disabled",
            component=PreparationComponent.TELEMETRY,
            message="MLflow tracing is intentionally disabled.",
        )

    try:
        healthy = await health_probe(settings.mlflow_tracking_uri)
    except asyncio.CancelledError:
        raise
    except Exception:
        healthy = False
    if not healthy:
        return failed_result(
            mode=mode,
            code="telemetry_unavailable",
            component=PreparationComponent.TELEMETRY,
            message="MLflow could not be reached.",
            remediation=(
                "Start `make mlflow`, correct its tracking URI, or disable "
                "runtime tracing."
            ),
        )
    return passed_result(
        code="telemetry_ready",
        component=PreparationComponent.TELEMETRY,
        message="MLflow tracing is reachable.",
    )


async def mlflow_health(tracking_uri: str) -> bool:
    """Probe HTTP tracking servers without blocking the event loop."""

    return await asyncio.to_thread(_mlflow_health_sync, tracking_uri)


def _mlflow_health_sync(tracking_uri: str) -> bool:
    parsed = urlparse(tracking_uri)
    if parsed.scheme not in {"http", "https"}:
        return True
    try:
        with urlopen(f"{tracking_uri.rstrip('/')}/health", timeout=2.0) as response:
            status = int(response.status)
            if not 200 <= status < 300:
                return False
            content_type = response.headers.get("content-type", "")
            if "application/json" in content_type:
                json.load(response)
            return True
    except (OSError, URLError, ValueError, json.JSONDecodeError):
        return False
