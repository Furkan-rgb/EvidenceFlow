"""Opt-in smoke test for the complete real-model LangGraph workflow."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from app.ai.config import load_models_config
from app.ai.models import create_embedding_provider
from app.config import Settings
from app.evaluation.workflow_adapter import WorkflowEvaluationAdapter
from app.observability import NoOpTracer
from app.retrieval import SqliteVecPolicyRetriever
from app.review import load_review_rules

pytestmark = [
    pytest.mark.ollama,
    pytest.mark.skipif(
        os.getenv("EVIDENCEFLOW_RUN_OLLAMA_TESTS") != "1",
        reason="set EVIDENCEFLOW_RUN_OLLAMA_TESTS=1 to exercise local Ollama",
    ),
]


@pytest.mark.asyncio
async def test_real_models_complete_a_consistent_langgraph_review() -> None:
    root = Path(__file__).parents[2]
    settings = Settings()
    models = load_models_config(ollama_base_url=settings.ollama_base_url)
    retriever = SqliteVecPolicyRetriever(
        create_embedding_provider(models.embeddings),
        dimensions=models.embeddings.dimensions,
        index_path=settings.policy_index_path,
        manifest_path=settings.policy_manifest_path,
        model_digest=models.embeddings.model_digest,
        policies_dir=settings.policies_dir,
    )
    adapter = WorkflowEvaluationAdapter(
        models=models,
        rules=load_review_rules(settings.rules_config),
        retriever=retriever,
        tracer=NoOpTracer(),
        max_pages=settings.max_pages,
    )
    bundle = root / "eval/bundles/bundle_001"
    ground_truth: dict[str, Any] = json.loads(
        (bundle / "ground_truth.json").read_text(encoding="utf-8")
    )

    prediction = await adapter.evaluate_bundle(bundle, ground_truth)

    expected_types = {
        str(document["file_name"]): str(document["document_type"])
        for document in ground_truth["documents"]
    }
    actual_types = {
        str(document["file_name"]): str(document["document_type"])
        for document in prediction["documents"]
    }
    assert actual_types == expected_types
    report = prediction["report"]
    assert isinstance(report, dict)
    assert report["status"] == "complete"
    assert report["company_name"]
    assert set(report.get("finding_ids", ())) <= {
        str(finding["finding_id"]) for finding in prediction["findings"]
    }
    assert set(report.get("policy_evidence_ids", ())) <= set(
        prediction["retrieved_policy_evidence_ids"]
    )
