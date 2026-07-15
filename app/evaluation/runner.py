"""Adapter-oriented evaluation runner; no provider or workflow is hard-wired here."""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
import platform
import tempfile
import time
from collections.abc import Awaitable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from app.evaluation.metrics import (
    AccuracyScore,
    CitationValidityScore,
    PrecisionRecallScore,
    RetrievalScore,
    citation_validity,
    classification_accuracy,
    conflict_precision_recall,
    conflicts_from_findings,
    field_extraction_accuracy,
    missing_document_detection_accuracy,
    policy_retrieval_metrics,
    review_required_accuracy,
    review_routing_accuracy,
)

EVALUATION_PROGRESS_SCHEMA_VERSION = 1


@runtime_checkable
class EvaluationAdapter(Protocol):
    """Boundary implemented by a fake or real EvidenceFlow workflow."""

    def evaluate_bundle(
        self, bundle_path: Path, ground_truth: Mapping[str, Any]
    ) -> Mapping[str, Any] | Awaitable[Mapping[str, Any]]:
        """Return predictions for one generated evaluation bundle."""


@runtime_checkable
class PolicyQueryAdapter(Protocol):
    """Optional boundary for evaluating a policy retriever independently."""

    def retrieve_policy_ids(
        self, query: str, *, k: int
    ) -> Sequence[str] | Awaitable[Sequence[str]]:
        """Return ranked evidence IDs for one policy query."""


@dataclass(slots=True)
class EvaluationRunResult:
    """Serializable result of a complete corpus run."""

    generated_at: str
    duration_seconds: float
    bundle_count: int
    aggregate_metrics: dict[str, Any]
    bundle_results: list[dict[str, Any]]
    retrieval_metrics: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "1.0",
            "generated_at": self.generated_at,
            "duration_seconds": self.duration_seconds,
            "bundle_count": self.bundle_count,
            "aggregate_metrics": self.aggregate_metrics,
            "retrieval_metrics": self.retrieval_metrics,
            "bundle_results": self.bundle_results,
            "metadata": self.metadata,
        }

    def to_markdown(self) -> str:
        metrics = self.aggregate_metrics
        rows = [
            ("Document classification accuracy", metrics["classification_accuracy"]),
            ("Field extraction accuracy", metrics["field_extraction_accuracy"]),
            (
                "Missing-document detection accuracy",
                metrics["missing_document_detection_accuracy"],
            ),
            ("Conflict precision", metrics["conflict_precision"]),
            ("Conflict recall", metrics["conflict_recall"]),
            ("Human-review routing accuracy", metrics["review_routing_accuracy"]),
            ("Human-review required accuracy", metrics["review_required_accuracy"]),
            ("Report citation validity", metrics["report_citation_validity"]),
            ("Report unknown-ID rate", metrics["report_unknown_id_rate"]),
        ]
        if self.retrieval_metrics is not None:
            rows.extend(
                (
                    ("Policy HitRate@5", self.retrieval_metrics["hit_rate_at_k"]),
                    ("Policy Recall@5", self.retrieval_metrics["recall_at_k"]),
                    ("Policy MRR@5", self.retrieval_metrics["mrr_at_k"]),
                    ("Policy nDCG@5", self.retrieval_metrics["ndcg_at_k"]),
                )
            )
        body = "\n".join(f"| {name} | {value:.4f} |" for name, value in rows)
        identity_labels = (
            ("classification_model", "Classification model"),
            ("classification_model_digest", "Classification model digest"),
            ("extraction_model", "Extraction model"),
            ("extraction_model_digest", "Extraction model digest"),
            ("reporting_model", "Reporting model"),
            ("reporting_model_digest", "Reporting model digest"),
            ("embedding_model", "Embedding model"),
            ("embedding_model_digest", "Embedding model digest"),
            ("dataset_sha256", "Dataset SHA-256"),
            ("configuration_sha256", "Configuration SHA-256"),
            ("policy_queries_sha256", "Policy-query labels SHA-256"),
            ("python", "Python"),
        )
        identity_rows = [
            f"| {label} | `{self.metadata[key]}` |"
            for key, label in identity_labels
            if self.metadata.get(key) is not None
        ]
        known_keys = {key for key, _ in identity_labels}
        identity_rows.extend(
            f"| {key} | `{value}` |"
            for key, value in sorted(self.metadata.items())
            if key not in known_keys
            and isinstance(value, (str, int, float, bool))
        )
        counts = metrics.get("counts", {})
        count_rows = "\n".join(
            f"| {str(key).replace('_', ' ').title()} | {value} |"
            for key, value in sorted(counts.items())
        )
        bundle_latencies = [
            float(item["duration_seconds"])
            for item in self.bundle_results
            if isinstance(item.get("duration_seconds"), (int, float))
        ]
        if bundle_latencies:
            latency_rows = (
                f"| Minimum bundle latency | {min(bundle_latencies):.2f}s |\n"
                f"| Mean bundle latency | "
                f"{sum(bundle_latencies) / len(bundle_latencies):.2f}s |\n"
                f"| Maximum bundle latency | {max(bundle_latencies):.2f}s |\n"
                f"| Total evaluation duration | {self.duration_seconds:.2f}s |"
            )
        else:
            latency_rows = f"| Total evaluation duration | {self.duration_seconds:.2f}s |"
        return (
            "# EvidenceFlow Evaluation Results\n\n"
            f"Generated: `{self.generated_at}`  \n"
            f"Bundles: **{self.bundle_count}**  \n"
            f"Duration: **{self.duration_seconds:.2f}s**\n\n"
            "## Reproducibility identity\n\n"
            "| Identity | Value |\n"
            "| --- | --- |\n"
            f"{'\n'.join(identity_rows)}\n\n"
            "## Quality metrics\n\n"
            "| Metric | Result |\n"
            "| --- | ---: |\n"
            f"{body}\n\n"
            "## Counts\n\n"
            "| Count | Value |\n"
            "| --- | ---: |\n"
            f"{count_rows}\n\n"
            "## Latency\n\n"
            "| Measure | Value |\n"
            "| --- | ---: |\n"
            f"{latency_rows}\n"
        )

    def write(self, output_directory: str | Path) -> tuple[Path, Path]:
        destination = Path(output_directory)
        destination.mkdir(parents=True, exist_ok=True)
        json_path = destination / "evaluation-results.json"
        markdown_path = destination / "evaluation-results.md"
        json_path.write_text(
            json.dumps(self.as_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        markdown_path.write_text(self.to_markdown(), encoding="utf-8")
        return json_path, markdown_path


async def _resolve(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


def _combine_accuracy(scores: Sequence[AccuracyScore]) -> AccuracyScore:
    return AccuracyScore(
        correct=sum(score.correct for score in scores),
        total=sum(score.total for score in scores),
    )


def _combine_conflicts(scores: Sequence[PrecisionRecallScore]) -> PrecisionRecallScore:
    return PrecisionRecallScore(
        true_positive=sum(score.true_positive for score in scores),
        false_positive=sum(score.false_positive for score in scores),
        false_negative=sum(score.false_negative for score in scores),
    )


def _combine_citations(scores: Sequence[CitationValidityScore]) -> CitationValidityScore:
    return CitationValidityScore(
        valid=sum(score.valid for score in scores),
        unknown=sum(score.unknown for score in scores),
    )


def _load_bundles(dataset_directory: Path) -> list[tuple[Path, dict[str, Any]]]:
    bundles: list[tuple[Path, dict[str, Any]]] = []
    for ground_truth_path in sorted(dataset_directory.glob("bundle_*/ground_truth.json")):
        ground_truth = json.loads(ground_truth_path.read_text(encoding="utf-8"))
        bundles.append((ground_truth_path.parent, ground_truth))
    if not bundles:
        raise FileNotFoundError(
            f"No bundle_*/ground_truth.json files found under {dataset_directory}"
        )
    return bundles


def _actual_conflicts(prediction: Mapping[str, Any]) -> set[str]:
    explicit = prediction.get("conflicts")
    if isinstance(explicit, Sequence) and not isinstance(explicit, (str, bytes)):
        return {str(item.get("field") if isinstance(item, Mapping) else item) for item in explicit}
    findings = prediction.get("findings", ())
    return conflicts_from_findings(findings if isinstance(findings, Sequence) else ())


def _actual_review_required(prediction: Mapping[str, Any]) -> bool:
    if "review_required" in prediction:
        return bool(prediction["review_required"])
    pending = prediction.get("pending_reviews") or prediction.get("review_items") or ()
    return bool(pending)


def _actual_review_reasons(prediction: Mapping[str, Any]) -> set[str]:
    explicit = prediction.get("review_routing_reasons")
    if isinstance(explicit, Sequence) and not isinstance(explicit, (str, bytes)):
        return {str(reason) for reason in explicit}

    reasons: set[str] = set()
    pending = prediction.get("review_items") or prediction.get("pending_reviews") or ()
    if not isinstance(pending, Sequence) or isinstance(pending, (str, bytes)):
        return reasons
    for item in pending:
        if not isinstance(item, Mapping):
            continue
        item_type = str(item.get("type") or item.get("item_type") or "")
        field_name = item.get("field_name") or item.get("field")
        if item_type == "low_confidence_classification":
            reasons.add("low_confidence:classification")
        elif item_type == "low_confidence_field" and field_name is not None:
            reasons.add(f"low_confidence:{field_name}")
        elif item_type == "field_conflict" and field_name is not None:
            reasons.add(f"conflict:{field_name}")
    return reasons


def _report_citations(prediction: Mapping[str, Any]) -> CitationValidityScore:
    report = prediction.get("report")
    if not isinstance(report, Mapping):
        return CitationValidityScore(valid=0, unknown=0)
    finding_ids = [str(item) for item in report.get("finding_ids", ())]
    policy_ids = [str(item) for item in report.get("policy_evidence_ids", ())]
    for section in report.get("sections", ()):
        if isinstance(section, Mapping):
            finding_ids.extend(str(item) for item in section.get("finding_ids", ()))
            policy_ids.extend(str(item) for item in section.get("policy_evidence_ids", ()))
    supported_finding_ids = {
        str(item.get("finding_id"))
        for item in prediction.get("findings", ())
        if isinstance(item, Mapping) and item.get("finding_id") is not None
    }
    supported_policy_ids = {
        str(item) for item in prediction.get("retrieved_policy_evidence_ids", ())
    }
    if not supported_policy_ids:
        supported_policy_ids = {
            str(item.get("evidence_id"))
            for item in prediction.get("policy_evidence", ())
            if isinstance(item, Mapping) and item.get("evidence_id") is not None
        }
    finding_score = citation_validity(finding_ids, supported_finding_ids)
    policy_score = citation_validity(policy_ids, supported_policy_ids)
    return _combine_citations((finding_score, policy_score))


async def _evaluate_retrieval(
    retriever: PolicyQueryAdapter,
    queries_path: Path,
    *,
    k: int,
) -> RetrievalScore:
    payload = _load_query_labels(queries_path)
    labelled_queries = payload.get("queries", payload)
    ranked: dict[str, Sequence[str]] = {}
    for item in labelled_queries:
        ranked[str(item["query_id"])] = await _resolve(
            retriever.retrieve_policy_ids(str(item["query"]), k=k)
        )
    return policy_retrieval_metrics(labelled_queries, ranked, k=k)


def _load_query_labels(queries_path: Path) -> Any:
    return json.loads(queries_path.read_text(encoding="utf-8"))


def _directory_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_mapping(value: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    """Return an isolated JSON representation suitable for a durable checkpoint."""

    try:
        encoded = json.dumps(dict(value), allow_nan=False, sort_keys=True)
        decoded = json.loads(encoded)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{label} must be JSON serializable") from error
    if not isinstance(decoded, dict):
        raise TypeError(f"{label} must serialize to a JSON object")
    return decoded


def _progress_identity(
    *,
    dataset_sha256: str,
    policy_queries_sha256: str | None,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "cache_schema_version": EVALUATION_PROGRESS_SCHEMA_VERSION,
        "dataset_sha256": dataset_sha256,
        "policy_queries_sha256": policy_queries_sha256,
        "python": platform.python_version(),
        "metadata": _json_mapping(metadata, label="Evaluation metadata"),
    }


def _valid_progress_entries(payload: Any, identity: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        return {}
    if payload.get("schema_version") != EVALUATION_PROGRESS_SCHEMA_VERSION:
        return {}
    if payload.get("identity") != identity:
        return {}
    entries = payload.get("entries")
    if not isinstance(entries, Mapping):
        return {}

    valid: dict[str, Any] = {}
    for key, entry in entries.items():
        if not isinstance(key, str) or not isinstance(entry, Mapping):
            continue
        prediction = entry.get("prediction")
        duration = entry.get("duration_seconds")
        if not isinstance(prediction, Mapping):
            continue
        if (
            not isinstance(duration, (int, float))
            or isinstance(duration, bool)
            or not math.isfinite(duration)
            or duration < 0
        ):
            continue
        try:
            json_prediction = _json_mapping(prediction, label="Cached prediction")
        except TypeError:
            continue
        valid[key] = {
            "prediction": json_prediction,
            "duration_seconds": float(duration),
        }
    return valid


def _load_progress(path: Path, identity: Mapping[str, Any]) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeError):
        return {}
    return _valid_progress_entries(payload, identity)


def _write_progress(
    path: Path,
    *,
    identity: Mapping[str, Any],
    entries: Mapping[str, Any],
) -> None:
    """Durably replace the progress file without exposing a partial JSON document."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": EVALUATION_PROGRESS_SCHEMA_VERSION,
        "identity": identity,
        "entries": entries,
    }
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(payload, handle, allow_nan=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


async def run_evaluation(
    dataset_directory: str | Path,
    adapter: EvaluationAdapter,
    *,
    retrieval_adapter: PolicyQueryAdapter | None = None,
    queries_path: str | Path = "eval/queries/policy_relevance.json",
    retrieval_k: int = 5,
    metadata: Mapping[str, Any] | None = None,
    progress_path: str | Path | None = None,
) -> EvaluationRunResult:
    """Evaluate an adapter over every generated bundle in deterministic order."""

    dataset_path = Path(dataset_directory)
    query_labels_path = Path(queries_path)
    bundles = _load_bundles(dataset_path)
    dataset_sha256 = _directory_hash(dataset_path)
    policy_queries_sha256 = (
        _file_hash(query_labels_path)
        if retrieval_adapter is not None
        else None
    )
    supplied_metadata = _json_mapping(metadata or {}, label="Evaluation metadata")
    cache_identity = _progress_identity(
        dataset_sha256=dataset_sha256,
        policy_queries_sha256=policy_queries_sha256,
        metadata=supplied_metadata,
    )
    cache_path = Path(progress_path) if progress_path is not None else None
    progress_entries: dict[str, Any] = {}
    if cache_path is not None:
        progress_entries = _load_progress(cache_path, cache_identity)
        # Replace malformed or incompatible progress immediately, even if the next
        # provider call fails before it can produce a new checkpoint.
        _write_progress(
            cache_path,
            identity=cache_identity,
            entries=progress_entries,
        )
    classification_scores: list[AccuracyScore] = []
    extraction_scores: list[AccuracyScore] = []
    missing_scores: list[AccuracyScore] = []
    conflict_scores: list[PrecisionRecallScore] = []
    routing_scores: list[AccuracyScore] = []
    review_required_scores: list[AccuracyScore] = []
    citation_scores: list[CitationValidityScore] = []
    bundle_results: list[dict[str, Any]] = []
    cache_hit_count = 0

    for bundle_path, ground_truth in bundles:
        bundle_key = bundle_path.relative_to(dataset_path).as_posix()
        cached = progress_entries.get(bundle_key)
        if isinstance(cached, Mapping):
            cache_hit = True
            prediction = cached["prediction"]
            bundle_duration = float(cached["duration_seconds"])
            cache_hit_count += 1
        else:
            cache_hit = False
            bundle_started = time.perf_counter()
            resolved_prediction = await _resolve(
                adapter.evaluate_bundle(bundle_path, ground_truth)
            )
            if not isinstance(resolved_prediction, Mapping):
                raise TypeError("Evaluation adapter predictions must be mappings")
            bundle_duration = time.perf_counter() - bundle_started
            prediction = _json_mapping(
                resolved_prediction,
                label=f"Prediction for {bundle_key}",
            )
            if cache_path is not None:
                progress_entries[bundle_key] = {
                    "prediction": prediction,
                    "duration_seconds": bundle_duration,
                }
                _write_progress(
                    cache_path,
                    identity=cache_identity,
                    entries=progress_entries,
                )
        expected_documents = ground_truth.get("documents", ())
        actual_documents = prediction.get("documents", ())
        findings = prediction.get("findings", ())
        classification = classification_accuracy(expected_documents, actual_documents)
        extraction = field_extraction_accuracy(expected_documents, actual_documents)
        missing = missing_document_detection_accuracy(
            ground_truth.get("expected_findings", ()), findings
        )
        conflicts = conflict_precision_recall(
            ground_truth.get("expected_conflicts", ()), _actual_conflicts(prediction)
        )
        expected_routing = ground_truth.get("expected_review_routing", {})
        routing = review_routing_accuracy(
            expected_routing.get("reasons", ()),
            _actual_review_reasons(prediction),
        )
        required = review_required_accuracy(
            bool(expected_routing.get("required")),
            _actual_review_required(prediction),
        )
        citations = _report_citations(prediction)
        classification_scores.append(classification)
        extraction_scores.append(extraction)
        missing_scores.append(missing)
        conflict_scores.append(conflicts)
        routing_scores.append(routing)
        review_required_scores.append(required)
        citation_scores.append(citations)
        bundle_results.append(
            {
                "bundle_id": ground_truth.get("bundle_id", bundle_path.name),
                "category": ground_truth.get("category"),
                "duration_seconds": bundle_duration,
                "cache_hit": cache_hit,
                "classification": classification.as_dict(),
                "extraction": extraction.as_dict(),
                "missing_document_detection": missing.as_dict(),
                "conflicts": conflicts.as_dict(),
                "review_routing": routing.as_dict(),
                "review_required": required.as_dict(),
                "report_citations": citations.as_dict(),
            }
        )

    classification_total = _combine_accuracy(classification_scores)
    extraction_total = _combine_accuracy(extraction_scores)
    missing_total = _combine_accuracy(missing_scores)
    conflict_total = _combine_conflicts(conflict_scores)
    routing_total = _combine_accuracy(routing_scores)
    review_required_total = _combine_accuracy(review_required_scores)
    citation_total = _combine_citations(citation_scores)
    aggregate = {
        "classification_accuracy": classification_total.value,
        "field_extraction_accuracy": extraction_total.value,
        "missing_document_detection_accuracy": missing_total.value,
        "conflict_precision": conflict_total.precision,
        "conflict_recall": conflict_total.recall,
        "conflict_f1": conflict_total.f1,
        "review_routing_accuracy": routing_total.value,
        "review_required_accuracy": review_required_total.value,
        "report_citation_validity": citation_total.validity_rate,
        "report_unknown_id_rate": citation_total.unknown_id_rate,
        "counts": {
            "documents": classification_total.total,
            "labelled_fields": extraction_total.total,
            "conflict_true_positive": conflict_total.true_positive,
            "conflict_false_positive": conflict_total.false_positive,
            "conflict_false_negative": conflict_total.false_negative,
            "report_citations": citation_total.total,
            "review_routes": routing_total.total,
        },
    }

    retrieval: RetrievalScore | None = None
    retrieval_duration = 0.0
    if retrieval_adapter is not None:
        retrieval_started = time.perf_counter()
        retrieval = await _evaluate_retrieval(
            retrieval_adapter,
            query_labels_path,
            k=retrieval_k,
        )
        retrieval_duration = time.perf_counter() - retrieval_started

    run_metadata = {
        "python": platform.python_version(),
        "dataset_sha256": dataset_sha256,
        **supplied_metadata,
        "cache_enabled": cache_path is not None,
        "cache_hit_count": cache_hit_count,
        "cache_schema_version": EVALUATION_PROGRESS_SCHEMA_VERSION,
    }
    if retrieval_adapter is not None:
        run_metadata["policy_queries_sha256"] = policy_queries_sha256
    return EvaluationRunResult(
        generated_at=datetime.now(UTC).isoformat(),
        duration_seconds=(
            sum(float(item["duration_seconds"]) for item in bundle_results)
            + retrieval_duration
        ),
        bundle_count=len(bundles),
        aggregate_metrics=aggregate,
        retrieval_metrics=retrieval.as_dict() if retrieval is not None else None,
        bundle_results=bundle_results,
        metadata=run_metadata,
    )
