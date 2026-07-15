"""Local EvidenceFlow developer and evaluation commands."""

from __future__ import annotations

import asyncio
import hashlib
import json
from contextlib import suppress
from pathlib import Path
from typing import Annotated

import typer

from app.ai import LLMDocumentClassifier, LLMFieldExtractor, LLMReportComposer
from app.ai.config import MODELS_CONFIG_PATH, ModelsConfig, load_models_config
from app.ai.models import (
    LangChainEmbeddingProvider,
    create_chat_model,
    create_embedding_provider,
)
from app.bootstrap import model_task_metadata
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
from app.preparation import (
    CheckOutcome,
    PreparationMode,
    PreparationReport,
    prepare_model_providers,
)
from app.preparation.runtime import prepare_application, require_prepared
from app.retrieval import (
    PolicyIndexBuilder,
    PolicyIndexManifest,
    SqliteVecPolicyRetriever,
)
from app.review import build_verified_review, load_review_rules

app = typer.Typer(
    no_args_is_help=True,
    help="EvidenceFlow V1 local runtime, indexing, evaluation, and smoke commands.",
)


def _settings_and_models() -> tuple[Settings, ModelsConfig]:
    settings = Settings()
    models = load_models_config(
        ollama_base_url=settings.ollama_base_url,
    )
    return settings, models


def _sha256_files(paths: list[Path]) -> str:
    repository_root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in paths:
        resolved = path.resolve()
        try:
            identity = resolved.relative_to(repository_root).as_posix()
        except ValueError:
            identity = path.as_posix()
        digest.update(identity.encode("utf-8"))
        digest.update(b"\0")
        digest.update(resolved.read_bytes())
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


def _render_preparation(report: PreparationReport) -> None:
    error_stream = not report.ready
    typer.echo(
        f"EvidenceFlow preparation ({report.mode.value.replace('_', ' ')})",
        err=error_stream,
    )
    for result in report.results:
        if result.outcome is CheckOutcome.PASSED:
            label = "ok"
        elif result.outcome is CheckOutcome.SKIPPED:
            label = "disabled" if result.code == "telemetry_disabled" else "skipped"
        elif result.warning:
            label = "warning"
        else:
            label = "failed"
        scope = (
            result.task.value if result.task is not None else result.component.value
        ).replace("_", " ")
        typer.echo(f"[{label}] {scope}: {result.message}", err=error_stream)
        if result.outcome is CheckOutcome.FAILED and result.remediation:
            typer.echo(f"         Fix: {result.remediation}", err=error_stream)

    if report.ready:
        suffix = " with warnings" if report.warnings else ""
        typer.echo(f"Preparation ready{suffix}.")
    else:
        typer.echo(
            f"Preparation blocked by {len(report.blocking_failures)} critical check(s).",
            err=error_stream,
        )


def _prepare_or_exit(
    mode: PreparationMode,
    *,
    policies_dir: Path | None = None,
) -> tuple[Settings, ModelsConfig, PreparationReport]:
    try:
        settings, models = _settings_and_models()
        if policies_dir is not None:
            settings = settings.model_copy(update={"policies_dir": policies_dir})
        report = asyncio.run(prepare_application(settings, models, mode=mode))
    except Exception as error:
        typer.echo(
            "[failed] configuration: runtime settings or config/models.yaml "
            "could not be loaded.",
            err=True,
        )
        raise typer.Exit(1) from error
    _render_preparation(report)
    if not report.ready:
        raise typer.Exit(1)
    return settings, models, report


@app.command()
def prepare() -> None:
    """Validate every dependency required before the app accepts work."""

    _prepare_or_exit(PreparationMode.RUNTIME)


@app.command()
def doctor() -> None:
    """Compatibility alias for the preparation command."""

    prepare()


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
    """Atomically rebuild the sqlite-vec index with the configured embedder."""

    settings, models, _report = _prepare_or_exit(
        PreparationMode.POLICY_INDEX_REBUILD,
        policies_dir=policies_dir,
    )
    embedder = create_embedding_provider(models.embeddings)
    manifest = asyncio.run(
        _rebuild_policy_index(settings=settings, models=models, embedder=embedder)
    )
    typer.echo(
        f"Indexed {manifest.document_count} policies / {manifest.chunk_count} chunks "
        f"with {manifest.model} ({manifest.dimensions} dimensions)."
    )


async def _rebuild_policy_index(
    *,
    settings: Settings,
    models: ModelsConfig,
    embedder: LangChainEmbeddingProvider,
) -> PolicyIndexManifest:
    try:
        return await PolicyIndexBuilder(
            embedder,
            dimensions=models.embeddings.dimensions,
            model_digest=models.embeddings.model_digest,
        ).rebuild(
            policies_dir=settings.policies_dir,
            index_path=settings.policy_index_path,
            manifest_path=settings.policy_manifest_path,
        )
    finally:
        await embedder.aclose()


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
    retriever: SqliteVecPolicyRetriever | None = None
    try:
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
            [MODELS_CONFIG_PATH, settings.rules_config, settings.policy_manifest_path]
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
    finally:
        if retriever is not None:
            with suppress(Exception):
                retriever.close()
        await embedder.aclose()


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

    settings, models, _report = _prepare_or_exit(PreparationMode.EVALUATION)
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


async def _ollama_smoke(
    settings: Settings,
    models: ModelsConfig,
    *,
    prepared: bool = False,
) -> dict[str, object]:
    if not prepared:
        provider_report = await prepare_model_providers(
            models,
            mode=PreparationMode.RUNTIME,
            default_ollama_base_url=settings.ollama_base_url,
        )
        require_prepared(provider_report)
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
    try:
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
    finally:
        await embedder.aclose()
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
        provider_report = asyncio.run(
            prepare_model_providers(
                models,
                mode=PreparationMode.RUNTIME,
                default_ollama_base_url=settings.ollama_base_url,
            )
        )
        _render_preparation(provider_report)
        require_prepared(provider_report)
        result = asyncio.run(_ollama_smoke(settings, models, prepared=True))
    except Exception as error:
        typer.echo(f"Ollama smoke failed: {error}", err=True)
        raise typer.Exit(1) from error
    typer.echo(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    app()
