"""Lifespan-managed application composition root."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from typing import Any

import aiosqlite
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from app.ai import (
    LLMDocumentClassifier,
    LLMFieldExtractor,
    LLMReportComposer,
)
from app.ai.config import ModelsConfig, load_models_config
from app.ai.models import (
    LangChainEmbeddingProvider,
    create_chat_model,
    create_embedding_provider,
)
from app.config import Settings
from app.documents import PyMuPDFDocumentProcessor
from app.graph import WorkflowDependencies, build_review_graph
from app.observability import MlflowTracer, NoOpTracer, Tracer
from app.persistence import LocalArtifactStore, SQLiteReviewRepository
from app.ports import ArtifactStore, ReviewRepository
from app.preparation import (
    PreparationBlockedError,
    PreparationComponent,
    PreparationMode,
    PreparationReport,
    failed_result,
    prepare_application,
    require_prepared,
)
from app.retrieval import SqliteVecPolicyRetriever
from app.review import load_review_rules
from app.runner import WorkflowRunner


@dataclass(slots=True)
class ApplicationContainer:
    settings: Settings
    models: ModelsConfig
    repository: ReviewRepository
    artifact_store: ArtifactStore
    graph: Any
    runner: WorkflowRunner
    tracer: Tracer
    policy_index_healthy: bool
    model_runtime_healthy: bool
    preparation_report: PreparationReport


def model_task_metadata(
    models: ModelsConfig,
) -> dict[str, dict[str, str | int | float | bool]]:
    """Safe model identity attached to traces without prompts or document text."""

    result: dict[str, dict[str, str | int | float | bool]] = {}
    for task in ("classification", "extraction", "reporting", "embeddings"):
        config = getattr(models, task)
        metadata: dict[str, str | int | float | bool] = {
            "task": task,
            "provider": config.provider,
            "model": config.model,
        }
        if config.model_digest:
            metadata["model_digest"] = config.model_digest
        if task == "embeddings":
            metadata["dimensions"] = config.dimensions
        result[task] = metadata
    return result


def _build_tracer(settings: Settings) -> Tracer:
    if not settings.mlflow_enabled:
        return NoOpTracer()
    return MlflowTracer(settings.mlflow_tracking_uri, settings.mlflow_experiment_name)


@asynccontextmanager
async def application_container(
    settings: Settings | None = None,
) -> AsyncIterator[ApplicationContainer]:
    """Create and cleanly stop all long-lived local resources."""

    try:
        settings = settings or Settings()
        models = load_models_config(
            ollama_base_url=settings.ollama_base_url,
        )
    except Exception:
        configuration_report = PreparationReport.from_results(
            PreparationMode.RUNTIME,
            (
                failed_result(
                    mode=PreparationMode.RUNTIME,
                    code="configuration_invalid",
                    component=PreparationComponent.CONFIGURATION,
                    message="Runtime settings or the model registry are invalid.",
                    remediation="Correct the runtime settings or config/models.yaml.",
                ),
            ),
        )
        raise PreparationBlockedError(configuration_report) from None
    preparation_report = await prepare_application(
        settings,
        models,
        mode=PreparationMode.RUNTIME,
    )
    require_prepared(preparation_report)

    rules = load_review_rules(settings.rules_config)
    repository = SQLiteReviewRepository(settings.database_path)
    await repository.migrate()
    artifact_store = LocalArtifactStore(settings.uploads_dir, settings.exports_dir)
    tracer = _build_tracer(settings)

    retriever: SqliteVecPolicyRetriever | None = None
    embedder: LangChainEmbeddingProvider | None = None
    checkpoint_connection: aiosqlite.Connection | None = None
    runner: WorkflowRunner | None = None
    try:
        classifier = LLMDocumentClassifier(create_chat_model(models.classification))
        extractor = LLMFieldExtractor(create_chat_model(models.extraction))
        composer = LLMReportComposer(create_chat_model(models.reporting))
        embedder = create_embedding_provider(models.embeddings)
        retriever = SqliteVecPolicyRetriever(
            embedder,
            dimensions=models.embeddings.dimensions,
            index_path=settings.policy_index_path,
            manifest_path=settings.policy_manifest_path,
            model_digest=models.embeddings.model_digest,
            policies_dir=settings.policies_dir,
        )
        checkpoint_connection = await aiosqlite.connect(settings.checkpoints_path)
        checkpointer = AsyncSqliteSaver(
            checkpoint_connection,
            serde=JsonPlusSerializer(pickle_fallback=False),
        )
        await checkpointer.setup()
        graph = build_review_graph(
            WorkflowDependencies(
                processor=PyMuPDFDocumentProcessor(
                    artifact_store, max_pages=settings.max_pages
                ),
                classifier=classifier,
                extractor=extractor,
                retriever=retriever,
                report_composer=composer,
                rules=rules,
                tracer=tracer,
                task_metadata=model_task_metadata(models),
            ),
            checkpointer=checkpointer,
        )
        runner = WorkflowRunner(
            repository,
            graph,
            tracer,
            log_sensitive_content=settings.log_sensitive_content,
        )
        container = ApplicationContainer(
            settings=settings,
            models=models,
            repository=repository,
            artifact_store=artifact_store,
            graph=graph,
            runner=runner,
            tracer=tracer,
            policy_index_healthy=True,
            model_runtime_healthy=True,
            preparation_report=preparation_report,
        )
        await runner.start()
        yield container
    finally:
        if runner is not None:
            await runner.stop()
        if retriever is not None:
            with suppress(Exception):
                retriever.close()
        if embedder is not None:
            await embedder.aclose()
        if checkpoint_connection is not None:
            await checkpoint_connection.close()
