from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from app.evaluation.runner import run_evaluation


class PerfectAdapter:
    async def evaluate_bundle(
        self, bundle_path: Path, ground_truth: dict[str, Any]
    ) -> dict[str, Any]:
        documents = []
        for expected in ground_truth["documents"]:
            documents.append(
                {
                    "file_name": expected["file_name"],
                    "document_type": expected["document_type"],
                    "fields": expected["expected_fields"],
                }
            )
        findings = []
        for index, expected in enumerate(ground_truth["expected_findings"]):
            findings.append({"finding_id": f"finding-{index}", **expected})
        return {
            "documents": documents,
            "findings": findings,
            "review_required": ground_truth["expected_review_routing"]["required"],
            "review_routing_reasons": ground_truth["expected_review_routing"][
                "reasons"
            ],
            "report": {"finding_ids": [finding["finding_id"] for finding in findings]},
        }


class PerfectRetriever:
    def __init__(self, answers: dict[str, list[str]]) -> None:
        self.answers = answers

    def retrieve_policy_ids(self, query: str, *, k: int) -> list[str]:
        return self.answers[query][:k]


def _write_bundle(dataset: Path, bundle_number: int) -> None:
    bundle_id = f"bundle_{bundle_number:03d}"
    bundle = dataset / bundle_id
    bundle.mkdir(parents=True)
    ground_truth = {
        "bundle_id": bundle_id,
        "category": "checkpoint-test",
        "documents": [
            {
                "file_name": "application.pdf",
                "document_type": "application_form",
                "expected_fields": {"employee_count": {"value": bundle_number}},
            }
        ],
        "expected_findings": [],
        "expected_conflicts": [],
        "expected_review_routing": {"required": False, "reasons": []},
    }
    (bundle / "ground_truth.json").write_text(json.dumps(ground_truth))


class RecordingAdapter(PerfectAdapter):
    def __init__(self, *, fail_once_on: str | None = None) -> None:
        self.calls: list[str] = []
        self.fail_once_on = fail_once_on
        self.failed = False

    async def evaluate_bundle(
        self, bundle_path: Path, ground_truth: dict[str, Any]
    ) -> dict[str, Any]:
        self.calls.append(bundle_path.name)
        if bundle_path.name == self.fail_once_on and not self.failed:
            self.failed = True
            raise RuntimeError("simulated late provider failure")
        return await super().evaluate_bundle(bundle_path, ground_truth)


@pytest.mark.asyncio
async def test_runner_aggregates_real_counts_and_writes_results(tmp_path: Path) -> None:
    dataset = tmp_path / "bundles"
    bundle = dataset / "bundle_001"
    bundle.mkdir(parents=True)
    ground_truth = {
        "bundle_id": "bundle_001",
        "category": "test",
        "documents": [
            {
                "file_name": "application.pdf",
                "document_type": "application_form",
                "expected_fields": {"employee_count": {"value": 42}},
            }
        ],
        "expected_findings": [{"type": "field_conflict", "field": "employee_count"}],
        "expected_conflicts": ["employee_count"],
        "expected_review_routing": {
            "required": True,
            "reasons": ["conflict:employee_count"],
        },
    }
    (bundle / "ground_truth.json").write_text(json.dumps(ground_truth))

    queries = tmp_path / "queries.json"
    query_payload = {
        "queries": [
            {
                "query_id": "q1",
                "query": "employee count conflict",
                "relevant_evidence_ids": ["POLICY:1:chunk-0"],
            }
        ]
    }
    queries.write_text(json.dumps(query_payload))
    result = await run_evaluation(
        dataset,
        PerfectAdapter(),
        retrieval_adapter=PerfectRetriever({"employee count conflict": ["POLICY:1:chunk-0"]}),
        queries_path=queries,
        metadata={
            "classification_model": "deterministic-fake",
            "classification_model_digest": "a" * 64,
            "extraction_model": "deterministic-fake",
            "extraction_model_digest": "b" * 64,
            "reporting_model": "deterministic-fake",
            "reporting_model_digest": "c" * 64,
            "embedding_model": "deterministic-embedding",
            "embedding_model_digest": "d" * 64,
            "configuration_sha256": "e" * 64,
        },
    )

    assert result.bundle_count == 1
    assert result.aggregate_metrics["classification_accuracy"] == 1
    assert result.aggregate_metrics["field_extraction_accuracy"] == 1
    assert result.aggregate_metrics["conflict_precision"] == 1
    assert result.aggregate_metrics["conflict_recall"] == 1
    assert result.aggregate_metrics["review_routing_accuracy"] == 1
    assert result.aggregate_metrics["review_required_accuracy"] == 1
    assert result.aggregate_metrics["report_unknown_id_rate"] == 0
    assert result.retrieval_metrics is not None
    assert result.retrieval_metrics["hit_rate_at_k"] == 1
    assert len(result.metadata["dataset_sha256"]) == 64
    assert len(result.metadata["policy_queries_sha256"]) == 64

    json_path, markdown_path = result.write(tmp_path / "results")
    assert (
        json.loads(json_path.read_text())["metadata"]["classification_model"]
        == "deterministic-fake"
    )
    markdown = markdown_path.read_text()
    assert "Document classification accuracy" in markdown
    assert "Classification model digest" in markdown
    assert "Dataset SHA-256" in markdown
    assert "Configuration SHA-256" in markdown
    assert "Policy-query labels SHA-256" in markdown
    assert "Labelled Fields" in markdown
    assert "Mean bundle latency" in markdown


@pytest.mark.asyncio
async def test_progress_resumes_exact_identity_after_late_failure(tmp_path: Path) -> None:
    dataset = tmp_path / "bundles"
    _write_bundle(dataset, 1)
    _write_bundle(dataset, 2)
    progress_path = tmp_path / "evaluation-progress.json"
    adapter = RecordingAdapter(fail_once_on="bundle_002")

    with pytest.raises(RuntimeError, match="simulated late provider failure"):
        await run_evaluation(
            dataset,
            adapter,
            metadata={"implementation_sha256": "a" * 64},
            progress_path=progress_path,
        )

    checkpoint = json.loads(progress_path.read_text())
    assert list(checkpoint["entries"]) == ["bundle_001"]
    assert checkpoint["identity"]["policy_queries_sha256"] is None
    cached_duration = checkpoint["entries"]["bundle_001"]["duration_seconds"]

    result = await run_evaluation(
        dataset,
        adapter,
        metadata={"implementation_sha256": "a" * 64},
        progress_path=progress_path,
    )

    assert adapter.calls == ["bundle_001", "bundle_002", "bundle_002"]
    assert result.metadata["cache_hit_count"] == 1
    assert [item["cache_hit"] for item in result.bundle_results] == [True, False]
    assert result.bundle_results[0]["duration_seconds"] == cached_duration
    assert result.duration_seconds == sum(
        item["duration_seconds"] for item in result.bundle_results
    )


@pytest.mark.asyncio
async def test_progress_identity_mismatch_recomputes_all_bundles(tmp_path: Path) -> None:
    dataset = tmp_path / "bundles"
    _write_bundle(dataset, 1)
    _write_bundle(dataset, 2)
    progress_path = tmp_path / "evaluation-progress.json"
    first_adapter = RecordingAdapter()
    second_adapter = RecordingAdapter()

    await run_evaluation(
        dataset,
        first_adapter,
        metadata={"implementation_sha256": "a" * 64},
        progress_path=progress_path,
    )
    result = await run_evaluation(
        dataset,
        second_adapter,
        metadata={"implementation_sha256": "b" * 64},
        progress_path=progress_path,
    )

    assert first_adapter.calls == ["bundle_001", "bundle_002"]
    assert second_adapter.calls == ["bundle_001", "bundle_002"]
    assert result.metadata["cache_hit_count"] == 0
    assert all(item["cache_hit"] is False for item in result.bundle_results)


@pytest.mark.asyncio
async def test_malformed_progress_is_replaced_safely(tmp_path: Path) -> None:
    dataset = tmp_path / "bundles"
    _write_bundle(dataset, 1)
    progress_path = tmp_path / "evaluation-progress.json"
    progress_path.write_text("{not valid JSON", encoding="utf-8")
    adapter = RecordingAdapter()

    result = await run_evaluation(dataset, adapter, progress_path=progress_path)

    checkpoint = json.loads(progress_path.read_text())
    assert checkpoint["schema_version"] == 1
    assert list(checkpoint["entries"]) == ["bundle_001"]
    assert adapter.calls == ["bundle_001"]
    assert result.metadata["cache_hit_count"] == 0
