"""Lifespan-managed application composition root."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

import aiosqlite
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from app.ai import (
    LLMDocumentClassifier,
    LLMFieldExtractor,
    LLMReportComposer,
)
from app.ai.config import ModelsConfig, load_models_config
from app.ai.models import create_chat_model, create_embedding_provider
from app.config import Settings
from app.documents import PyMuPDFDocumentProcessor
from app.errors import EmbeddingIndexMismatchError, PolicyIndexMissingError
from app.graph import WorkflowDependencies, build_review_graph
from app.observability import MlflowTracer, NoOpTracer, Tracer
from app.persistence import LocalArtifactStore, SQLiteReviewRepository
from app.ports import ArtifactStore, ReviewRepository
from app.retrieval import SqliteVecPolicyRetriever
from app.review import load_review_rules
from app.runner import WorkflowRunner

logger = logging.getLogger(__name__)


class UnavailablePolicyRetriever:
    """Deferred startup failure that keeps health and indexing endpoints usable."""

    def __init__(self, error: Exception) -> None:
        self.error = error

    async def search(self, query: str, *, limit: int = 5) -> list[Any]:
        del query, limit
        if isinstance(self.error, PolicyIndexMissingError):
            raise self.error
        raise PolicyIndexMissingError(
            "The policy index is unavailable; rebuild it before reviewing documents."
        ) from self.error


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


def _model_names(models: ModelsConfig) -> set[str]:
    return {
        models.classification.model,
        models.extraction.model,
        models.reporting.model,
        models.embeddings.model,
    }


def model_digests(models: ModelsConfig) -> dict[str, str]:
    return {
        config.model: config.model_digest
        for config in (
            models.classification,
            models.extraction,
            models.reporting,
            models.embeddings,
        )
        if config.model_digest is not None
    }


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


def _ollama_inventory_sync(base_url: str) -> dict[str, str]:
    try:
        with urlopen(f"{base_url.rstrip('/')}/api/tags", timeout=2.0) as response:
            payload = json.load(response)
    except (OSError, URLError, ValueError, json.JSONDecodeError):
        return {}
    inventory: dict[str, str] = {}
    for item in payload.get("models", []):
        if not isinstance(item, dict) or not item.get("name") or not item.get("digest"):
            continue
        name = str(item["name"])
        digest = str(item["digest"])
        inventory[name] = digest
        if name.endswith(":latest"):
            inventory[name.removesuffix(":latest")] = digest
    return inventory


async def ollama_inventory(base_url: str) -> dict[str, str]:
    return await asyncio.to_thread(_ollama_inventory_sync, base_url)


async def probe_ollama(
    base_url: str,
    required: set[str],
    expected_digests: Mapping[str, str] | None = None,
) -> bool:
    inventory = await ollama_inventory(base_url)
    if not required <= set(inventory):
        return False
    return all(
        inventory.get(model) == digest
        for model, digest in dict(expected_digests or {}).items()
    )


def _build_tracer(settings: Settings) -> Tracer:
    if not settings.mlflow_enabled:
        return NoOpTracer()
    return MlflowTracer(settings.mlflow_tracking_uri, settings.mlflow_experiment_name)


@asynccontextmanager
async def application_container(
    settings: Settings | None = None,
) -> AsyncIterator[ApplicationContainer]:
    """Create and cleanly stop all long-lived local resources."""

    settings = settings or Settings()
    settings.ensure_directories()
    models = load_models_config(
        settings.models_config,
        classification_model=settings.classification_model,
        extraction_model=settings.extraction_model,
        reporting_model=settings.reporting_model,
        embedding_model=settings.embedding_model,
        ollama_base_url=settings.ollama_base_url,
    )
    rules = load_review_rules(settings.rules_config)
    repository = SQLiteReviewRepository(settings.database_path)
    await repository.migrate()
    artifact_store = LocalArtifactStore(settings.uploads_dir, settings.exports_dir)
    tracer = _build_tracer(settings)

    inventory = await ollama_inventory(settings.ollama_base_url)
    expected_digests = model_digests(models)
    model_runtime_healthy = _model_names(models) <= set(inventory) and all(
        inventory.get(model) == digest
        for model, digest in expected_digests.items()
    )

    classifier = LLMDocumentClassifier(create_chat_model(models.classification))
    extractor = LLMFieldExtractor(create_chat_model(models.extraction))
    composer = LLMReportComposer(create_chat_model(models.reporting))
    embedder = create_embedding_provider(models.embeddings)
    policy_index_healthy = True
    try:
        observed_embedding_digest = inventory.get(models.embeddings.model)
        if (
            observed_embedding_digest is not None
            and models.embeddings.model_digest is not None
            and observed_embedding_digest != models.embeddings.model_digest
        ):
            raise EmbeddingIndexMismatchError(
                "The running embedding model digest differs from the configured "
                "policy index identity; rebuild with the intended model.",
                details={
                    "model": models.embeddings.model,
                    "expected_digest": models.embeddings.model_digest,
                    "observed_digest": observed_embedding_digest,
                },
            )
        retriever: Any = SqliteVecPolicyRetriever(
            embedder,
            dimensions=models.embeddings.dimensions,
            index_path=settings.policy_index_path,
            manifest_path=settings.policy_manifest_path,
            model_digest=models.embeddings.model_digest,
            policies_dir=settings.policies_dir,
        )
    except Exception as error:
        logger.warning("Policy retrieval is unavailable at startup: %s", error)
        retriever = UnavailablePolicyRetriever(error)
        policy_index_healthy = False

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
        policy_index_healthy=policy_index_healthy,
        model_runtime_healthy=model_runtime_healthy,
    )
    await runner.start()
    try:
        yield container
    finally:
        await runner.stop()
        await checkpoint_connection.close()
