"""Local EvidenceFlow developer and evaluation commands."""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from typing import Annotated
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import urlopen

import typer

from app.ai import LLMDocumentClassifier, LLMFieldExtractor, LLMReportComposer
from app.ai.config import ModelsConfig, load_models_config
from app.ai.models import create_chat_model, create_embedding_provider
from app.bootstrap import (
    model_digests,
    model_task_metadata,
    ollama_inventory,
    probe_ollama,
)
from app.config import Settings
from app.domain import (
    DocumentType,
    PageContent,
    ProcessedDocument,
    ProcessorMetadata,
)
from app.evaluation import generate_evaluation_data, run_evaluation
from app.evaluation.workflow_adapter import (
    PolicyRetrieverEvaluationAdapter,
    WorkflowEvaluationAdapter,
)
from app.observability import MlflowTracer
from app.retrieval import PolicyIndexBuilder, SqliteVecPolicyRetriever
from app.review import build_verified_review, load_review_rules

app = typer.Typer(
    no_args_is_help=True,
    help="EvidenceFlow V1 local runtime, indexing, evaluation, and smoke commands.",
)


def _settings_and_models() -> tuple[Settings, ModelsConfig]:
    settings = Settings()
    settings.ensure_directories()
    models = load_models_config(
        settings.models_config,
        classification_model=settings.classification_model,
        extraction_model=settings.extraction_model,
        reporting_model=settings.reporting_model,
        embedding_model=settings.embedding_model,
        ollama_base_url=settings.ollama_base_url,
    )
    return settings, models


def _sha256_files(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _implementation_sha256() -> str:
    """Hash the versioned runtime, configuration, and lock inputs."""

    repository_root = Path(__file__).resolve().parents[1]
    paths = [
        *repository_root.joinpath("app").rglob("*.py"),
        *(
            path
            for path in repository_root.joinpath("config").rglob("*")
            if path.is_file()
        ),
        repository_root / "pyproject.toml",
        repository_root / "uv.lock",
    ]
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.relative_to(repository_root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _required_models(models: ModelsConfig) -> set[str]:
    return {
        models.classification.model,
        models.extraction.model,
        models.reporting.model,
        models.embeddings.model,
    }


def _require_ollama(settings: Settings, models: ModelsConfig) -> None:
    if not asyncio.run(
        probe_ollama(
            settings.ollama_base_url,
            _required_models(models),
            model_digests(models),
        )
    ):
        typer.echo(
            "Ollama is unavailable, a configured model is missing, or a model "
            "digest differs from config/models.yaml.",
            err=True,
        )
        raise typer.Exit(1)


def _mlflow_health_sync(tracking_uri: str) -> bool:
    parsed = urlparse(tracking_uri)
    if parsed.scheme not in {"http", "https"}:
        return True
    try:
        with urlopen(f"{tracking_uri.rstrip('/')}/health", timeout=2.0) as response:
            status = int(response.status)
            return 200 <= status < 300
    except (OSError, URLError, ValueError):
        return False


async def _doctor_checks(
    settings: Settings, models: ModelsConfig
) -> dict[str, object]:
    inventory = await ollama_inventory(settings.ollama_base_url)
    required = _required_models(models)
    expected_digests = model_digests(models)
    missing_models = sorted(required - set(inventory))
    digest_mismatches = {
        model: {"expected": digest, "actual": inventory.get(model)}
        for model, digest in expected_digests.items()
        if inventory.get(model) != digest
    }
    ollama_ok = not missing_models and not digest_mismatches

    policy_error: str | None = None
    try:
        SqliteVecPolicyRetriever(
            create_embedding_provider(models.embeddings),
            dimensions=models.embeddings.dimensions,
            index_path=settings.policy_index_path,
            manifest_path=settings.policy_manifest_path,
            model_digest=models.embeddings.model_digest,
            policies_dir=settings.policies_dir,
        )
    except Exception as error:
        policy_error = str(error)

    mlflow_enabled = settings.mlflow_enabled
    mlflow_ok = not mlflow_enabled or await asyncio.to_thread(
        _mlflow_health_sync, settings.mlflow_tracking_uri
    )
    return {
        "ollama_ok": ollama_ok,
        "missing_models": missing_models,
        "digest_mismatches": digest_mismatches,
        "policy_index_ok": policy_error is None,
        "policy_index_error": policy_error,
        "mlflow_enabled": mlflow_enabled,
        "mlflow_ok": mlflow_ok,
    }


@app.command()
def doctor() -> None:
    """Fail early on missing critical local runtime dependencies."""

    try:
        settings, models = _settings_and_models()
        checks = asyncio.run(_doctor_checks(settings, models))
    except Exception as error:
        typer.echo(f"[failed] configuration: {error}", err=True)
        raise typer.Exit(1) from error

    typer.echo("EvidenceFlow preflight")
    if checks["ollama_ok"]:
        typer.echo("[ok] Ollama and configured model digests")
    else:
        missing = checks["missing_models"]
        mismatches = checks["digest_mismatches"]
        typer.echo(
            f"[failed] Ollama models (missing={missing}, digest_mismatches={mismatches})",
            err=True,
        )
    if checks["policy_index_ok"]:
        typer.echo("[ok] policy index, manifest, corpus, chunks, and vectors")
    else:
        typer.echo(
            f"[failed] policy index: {checks['policy_index_error']}", err=True
        )
    if not checks.get("mlflow_enabled", True):
        typer.echo("[disabled] MLflow tracing")
    elif checks["mlflow_ok"]:
        typer.echo("[ok] MLflow")
    else:
        typer.echo(
            "[warning] MLflow is unavailable; runtime remains fail-open and /health "
            "will report degraded telemetry.",
            err=True,
        )

    if not checks["ollama_ok"] or not checks["policy_index_ok"]:
        typer.echo(
            "Preflight failed. Start Ollama/pull the configured models or run "
            "`make rebuild`, then retry.",
            err=True,
        )
        raise typer.Exit(1)


@app.command()
def run(
    host: Annotated[str, typer.Option(help="Interface on which to listen.")] = "127.0.0.1",
    port: Annotated[int, typer.Option(min=1, max=65535)] = 8000,
    reload: Annotated[bool, typer.Option(help="Reload when source files change.")] = False,
) -> None:
    """Start the FastAPI application and vanilla-JavaScript UI."""

    import uvicorn

    uvicorn.run("app.main:app", host=host, port=port, reload=reload)


@app.command("rebuild-policy-index")
def rebuild_policy_index_command(
    policies_dir: Annotated[
        Path, typer.Option(file_okay=False, dir_okay=True)
    ] = Path("policies"),
) -> None:
    """Atomically rebuild the sqlite-vec policy index with embeddinggemma."""

    settings, models = _settings_and_models()
    _require_ollama(settings, models)
    embedder = create_embedding_provider(models.embeddings)
    manifest = asyncio.run(
        PolicyIndexBuilder(
            embedder,
            dimensions=models.embeddings.dimensions,
            model_digest=models.embeddings.model_digest,
        ).rebuild(
            policies_dir=policies_dir,
            index_path=settings.policy_index_path,
            manifest_path=settings.policy_manifest_path,
        )
    )
    typer.echo(
        f"Indexed {manifest.document_count} policies / {manifest.chunk_count} chunks "
        f"with {manifest.model} ({manifest.dimensions} dimensions)."
    )


@app.command("generate-eval-data")
def generate_eval_data_command(
    output_dir: Annotated[
        Path, typer.Option(file_okay=False, dir_okay=True)
    ] = Path("eval/bundles"),
    overwrite: Annotated[
        bool, typer.Option(help="Replace an existing generated corpus.")
    ] = False,
) -> None:
    """Generate the deterministic 20-bundle synthetic PDF corpus."""

    bundles = generate_evaluation_data(output_dir, overwrite=overwrite)
    typer.echo(f"Generated {len(bundles)} bundles under {output_dir}.")


async def _run_real_evaluation(
    *,
    bundles_dir: Path,
    output_dir: Path,
    settings: Settings,
    models: ModelsConfig,
    tracer: MlflowTracer,
    progress_path: Path,
) -> tuple[dict[str, object], tuple[Path, Path]]:
    rules = load_review_rules(settings.rules_config)
    embedder = create_embedding_provider(models.embeddings)
    retriever = SqliteVecPolicyRetriever(
        embedder,
        dimensions=models.embeddings.dimensions,
        index_path=settings.policy_index_path,
        manifest_path=settings.policy_manifest_path,
        model_digest=models.embeddings.model_digest,
        policies_dir=settings.policies_dir,
    )
    adapter = WorkflowEvaluationAdapter(
        models=models,
        rules=rules,
        retriever=retriever,
        tracer=tracer,
        max_pages=settings.max_pages,
    )
    configuration_hash = _sha256_files(
        [settings.models_config, settings.rules_config, settings.policy_manifest_path]
    )
    result = await run_evaluation(
        bundles_dir,
        adapter,
        retrieval_adapter=PolicyRetrieverEvaluationAdapter(
            retriever,
            tracer=tracer,
            task_metadata=model_task_metadata(models)["embeddings"],
        ),
        metadata={
            "classification_model": models.classification.model,
            "classification_model_digest": models.classification.model_digest,
            "extraction_model": models.extraction.model,
            "extraction_model_digest": models.extraction.model_digest,
            "reporting_model": models.reporting.model,
            "reporting_model_digest": models.reporting.model_digest,
            "embedding_model": models.embeddings.model,
            "embedding_model_digest": models.embeddings.model_digest,
            "configuration_sha256": configuration_hash,
            "implementation_sha256": _implementation_sha256(),
        },
        progress_path=progress_path,
    )
    if not tracer.healthy or tracer.ever_failed:
        raise RuntimeError("MLflow tracing became unavailable during evaluation")
    paths = result.write(output_dir)
    return result.as_dict(), paths


@app.command()
def evaluate(
    bundles_dir: Annotated[
        Path, typer.Option(exists=True, file_okay=False, dir_okay=True)
    ] = Path("eval/bundles"),
    output_dir: Annotated[
        Path, typer.Option(file_okay=False, dir_okay=True)
    ] = Path("eval/results"),
) -> None:
    """Run real Gemma/Ollama evaluation; MLflow availability is mandatory."""

    import mlflow

    settings, models = _settings_and_models()
    _require_ollama(settings, models)
    tracer = MlflowTracer(settings.mlflow_tracking_uri, "evidenceflow-evaluation")
    if not tracer.healthy or tracer.ever_failed:
        typer.echo("MLflow is unavailable; evaluation tracing fails closed.", err=True)
        raise typer.Exit(1)
    progress_path = settings.data_dir / "evaluation-progress.json"
    try:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment("evidenceflow-evaluation")
        with mlflow.start_run(run_name="gemma-v1-20-bundle-evaluation"):
            result, paths = asyncio.run(
                _run_real_evaluation(
                    bundles_dir=bundles_dir,
                    output_dir=output_dir,
                    settings=settings,
                    models=models,
                    tracer=tracer,
                    progress_path=progress_path,
                )
            )
            metrics = result["aggregate_metrics"]
            assert isinstance(metrics, dict)
            numeric_metrics = {
                key: float(value)
                for key, value in metrics.items()
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            }
            counts = metrics.get("counts")
            if isinstance(counts, dict):
                numeric_metrics.update(
                    {
                        f"count_{key}": float(value)
                        for key, value in counts.items()
                        if isinstance(value, (int, float))
                        and not isinstance(value, bool)
                    }
                )
            duration_seconds = result.get("duration_seconds")
            bundle_count = result.get("bundle_count")
            if isinstance(duration_seconds, (int, float)):
                numeric_metrics["evaluation_duration_seconds"] = float(
                    duration_seconds
                )
            if isinstance(bundle_count, (int, float)):
                numeric_metrics["bundle_count"] = float(bundle_count)
            bundle_results = result.get("bundle_results")
            if isinstance(bundle_results, list):
                durations = [
                    float(item["duration_seconds"])
                    for item in bundle_results
                    if isinstance(item, dict)
                    and isinstance(item.get("duration_seconds"), (int, float))
                ]
                if durations:
                    numeric_metrics.update(
                        {
                            "bundle_latency_min_seconds": min(durations),
                            "bundle_latency_mean_seconds": sum(durations)
                            / len(durations),
                            "bundle_latency_max_seconds": max(durations),
                        }
                    )
            retrieval = result.get("retrieval_metrics")
            if isinstance(retrieval, dict):
                numeric_metrics.update(
                    {
                        f"retrieval_{key}": float(value)
                        for key, value in retrieval.items()
                        if isinstance(value, (int, float))
                        and not isinstance(value, bool)
                    }
                )
            mlflow.log_metrics(numeric_metrics)
            metadata = result.get("metadata")
            if isinstance(metadata, dict):
                mlflow.log_params(
                    {
                        str(key): str(value)
                        for key, value in metadata.items()
                        if value is not None
                    }
                )
            mlflow.log_artifacts(str(output_dir), artifact_path="evaluation")
        progress_path.unlink(missing_ok=True)
    except typer.Exit:
        raise
    except Exception as error:
        typer.echo(f"Evaluation failed: {error}", err=True)
        raise typer.Exit(1) from error
    typer.echo(f"Wrote genuine evaluation results to {paths[0]} and {paths[1]}.")


async def _ollama_smoke(settings: Settings, models: ModelsConfig) -> dict[str, object]:
    if not await probe_ollama(
        settings.ollama_base_url,
        _required_models(models),
        model_digests(models),
    ):
        raise RuntimeError("Ollama is unavailable or one of the configured models is missing")
    document = ProcessedDocument(
        document_id="ollama-smoke-application",
        filename="application_form.pdf",
        pages=[
            PageContent(
                page_number=1,
                text=(
                    "BUSINESS ONBOARDING APPLICATION\n"
                    "Legal company name: Smoke Test B.V.\n"
                    "Registration number: 12345678\n"
                    "Estimated annual revenue: EUR 1000000\n"
                    "Number of employees: 10\nEND"
                ),
            )
        ],
        processor_metadata=ProcessorMetadata(processor="ollama-smoke", version="1"),
    )
    classification = await LLMDocumentClassifier(
        create_chat_model(models.classification)
    ).classify(document)
    extraction = await LLMFieldExtractor(
        create_chat_model(models.extraction)
    ).extract(document, DocumentType.APPLICATION_FORM)
    embedder = create_embedding_provider(models.embeddings)
    embedding = await embedder.embed_query("registration number conflict")
    verified = build_verified_review(
        review_id="ollama-smoke-review",
        classifications=[],
        extractions=[],
        effective_fields=[],
        findings=[],
    )
    report = await LLMReportComposer(create_chat_model(models.reporting)).compose(
        verified, []
    )
    return {
        "classification": classification.document_type.value,
        "extracted_field_count": len(extraction.fields),
        "embedding_dimensions": len(embedding),
        "report_status": report.status.value,
    }


@app.command("ollama-smoke")
def ollama_smoke_command() -> None:
    """Exercise all configured real Ollama task adapters."""

    settings, models = _settings_and_models()
    try:
        result = asyncio.run(_ollama_smoke(settings, models))
    except Exception as error:
        typer.echo(f"Ollama smoke failed: {error}", err=True)
        raise typer.Exit(1) from error
    typer.echo(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    app()
