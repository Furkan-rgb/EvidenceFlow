from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from app.domain import ReviewReport
from app.graph import build_review_graph
from app.main import create_app
from app.persistence import LocalArtifactStore, SQLiteReviewRepository
from tests.integration.test_review_graph import dependencies, initial_state

pytestmark = pytest.mark.e2e


@pytest.mark.asyncio
async def test_completed_workflow_exports_only_verified_references(
    tmp_path: Path,
) -> None:
    review_id = "review-export-e2e"
    thread_id = "thread-export-e2e"
    graph = build_review_graph(dependencies("complete"))
    completed = await graph.ainvoke(initial_state(review_id, thread_id))

    repository = SQLiteReviewRepository(tmp_path / "reviews.db")
    await repository.migrate()
    artifact_store = LocalArtifactStore(
        tmp_path / "uploads", tmp_path / "exports"
    )
    await repository.create(
        review_id,
        thread_id,
        [
            {
                "document_id": item["document_id"],
                "filename": item["filename"],
                "artifact_id": item["artifact_id"],
                "sha256": f"sha-{item['document_id']}",
                "size_bytes": item["size_bytes"],
                "content_type": "application/pdf",
            }
            for item in completed["uploaded_documents"]
        ],
    )
    await repository.save_snapshot(review_id, completed)
    await repository.save_report(
        review_id,
        ReviewReport.model_validate(completed["report"]),
        completed["report_markdown"],
    )

    app = create_app()
    app.state.container = SimpleNamespace(
        repository=repository,
        artifact_store=artifact_store,
        runner=SimpleNamespace(wake=lambda: None),
        settings=SimpleNamespace(),
    )
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://test"
    ) as client:
        response = await client.get(f"/api/v1/reviews/{review_id}/export.json")
        markdown = await client.get(f"/api/v1/reviews/{review_id}/export.md")

    assert response.status_code == 200
    payload = response.json()
    known_findings = {
        item["finding_id"] for item in payload["review"]["findings"]
    }
    known_policy = {
        item["evidence_id"] for item in payload["review"]["policy_evidence"]
    }
    for section in payload["report"]["sections"]:
        assert set(section["finding_ids"]) <= known_findings
        assert set(section["policy_evidence_ids"]) <= known_policy
    assert markdown.status_code == 200
    assert markdown.text == completed["report_markdown"]
