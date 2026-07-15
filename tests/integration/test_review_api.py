from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import httpx
import pytest
import pytest_asyncio

from app.domain import ReportStatus, ReviewReport
from app.main import create_app
from app.persistence import LocalArtifactStore, SQLiteReviewRepository


class NoOpRunner:
    def __init__(self) -> None:
        self.wake_count = 0

    def wake(self) -> None:
        self.wake_count += 1


class ProgressGraph:
    async def aget_state(self, _configuration: object) -> SimpleNamespace:
        return SimpleNamespace(
            next=("normalize_and_check_completeness",),
            values={
                "classifications": [
                    {"document_id": "one"},
                    {"document_id": "two"},
                ],
                "extraction_results": [
                    {"fields": [{"field_id": "one:name"}, {"field_id": "one:id"}]}
                ],
                "findings": [{"finding_id": "finding-one"}],
            }
        )


@dataclass
class APIHarness:
    client: httpx.AsyncClient
    repository: SQLiteReviewRepository
    artifact_store: LocalArtifactStore
    uploads_dir: Path
    container: SimpleNamespace
    runner: NoOpRunner


@pytest_asyncio.fixture
async def api(tmp_path: Path) -> AsyncIterator[APIHarness]:
    repository = SQLiteReviewRepository(tmp_path / "evidenceflow.db")
    await repository.migrate()
    uploads_dir = tmp_path / "uploads"
    artifact_store = LocalArtifactStore(uploads_dir, tmp_path / "exports")
    runner = NoOpRunner()
    container = SimpleNamespace(
        repository=repository,
        artifact_store=artifact_store,
        runner=runner,
        settings=SimpleNamespace(max_file_bytes=128, max_bundle_bytes=180),
        tracer=SimpleNamespace(healthy=True),
        policy_index_healthy=False,
        model_runtime_healthy=False,
    )
    app = create_app()
    # ASGITransport does not enter application lifespan. Supplying the test
    # container directly keeps every API call model-free and local.
    app.state.container = container
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield APIHarness(
            client, repository, artifact_store, uploads_dir, container, runner
        )


def pdf(label: str, *, size: int | None = None) -> bytes:
    content = b"%PDF-1.7\n" + label.encode()
    if size is not None:
        content += b"x" * max(0, size - len(content))
    return content


async def create_review(
    client: httpx.AsyncClient,
    *documents: tuple[str, bytes, str],
) -> httpx.Response:
    return await client.post(
        "/api/v1/reviews",
        files=[
            ("files", (filename, content, content_type))
            for filename, content, content_type in documents
        ],
    )


async def add_pending_item(
    repository: SQLiteReviewRepository,
    review_id: str,
    *,
    item_id: str = "review-field-financial:employee_count",
) -> None:
    await repository.save_review_items(
        review_id,
        [
            {
                "review_item_id": item_id,
                "type": "low_confidence_field",
                "document_id": "financial",
                "field_id": "financial:employee_count",
                "field_name": "employee_count",
                "extracted_value": 42,
                "confidence": 0.6,
                "expected_value_type": "integer",
                "evidence": [],
            }
        ],
    )


def assert_safe_error(response: httpx.Response, code: str) -> dict[str, Any]:
    assert response.headers["x-request-id"]
    payload = cast(dict[str, Any], response.json())
    assert set(payload) == {"error"}
    assert payload["error"]["code"] == code
    assert payload["error"]["request_id"] == response.headers["x-request-id"]
    assert "traceback" not in response.text.lower()
    return payload


@pytest.mark.asyncio
async def test_create_poll_and_duplicate_document_instances_are_preserved(
    api: APIHarness,
) -> None:
    response = await create_review(
        api.client,
        ("application-form.pdf", pdf("first form"), "application/pdf"),
        ("application-form.pdf", pdf("second form"), "application/pdf"),
        ("extract.pdf", pdf("extract"), "application/pdf"),
    )
    assert response.status_code == 202
    created = response.json()
    assert created["status"] == "processing"
    assert api.runner.wake_count == 1

    poll = await api.client.get(f"/api/v1/reviews/{created['review_id']}")
    assert poll.status_code == 200
    body = poll.json()
    assert body["status"] == "processing"
    assert len(body["documents"]) == 3
    assert body["summary"] == {
        "document_count": 3,
        "classified_count": 0,
        "extracted_field_count": 0,
        "finding_count": 0,
        "pending_review_count": 0,
    }
    assert body["progress"]["current_step_id"] == "document_processing"
    assert [step["status"] for step in body["progress"]["steps"]] == [
        "current",
        "upcoming",
        "upcoming",
        "upcoming",
        "upcoming",
        "upcoming",
        "upcoming",
    ]
    assert len({item["document_id"] for item in body["documents"]}) == 3
    assert [item["filename"] for item in body["documents"]].count(
        "application-form.pdf"
    ) == 2


@pytest.mark.asyncio
async def test_processing_poll_projects_live_checkpoint_progress(api: APIHarness) -> None:
    created = await create_review(
        api.client,
        ("one.pdf", pdf("one"), "application/pdf"),
        ("two.pdf", pdf("two"), "application/pdf"),
    )
    api.container.graph = ProgressGraph()

    response = await api.client.get(
        f"/api/v1/reviews/{created.json()['review_id']}"
    )

    assert response.json()["summary"] == {
        "document_count": 2,
        "classified_count": 2,
        "extracted_field_count": 2,
        "finding_count": 1,
        "pending_review_count": 0,
    }
    progress = response.json()["progress"]
    assert progress["current_step_id"] == "completeness"
    assert "normalize_and_check_completeness" not in response.text
    assert [step["status"] for step in progress["steps"]] == [
        "completed",
        "completed",
        "completed",
        "current",
        "upcoming",
        "upcoming",
        "upcoming",
    ]


@pytest.mark.asyncio
async def test_review_interrupt_projects_the_waiting_human_stage(
    api: APIHarness,
) -> None:
    created = await create_review(
        api.client,
        ("financial.pdf", pdf("field review"), "application/pdf"),
    )
    review_id = created.json()["review_id"]
    await api.repository.save_snapshot(
        review_id,
        {
            "review_stage": "field",
            "processed_documents": [{}],
            "classifications": [{}],
            "extraction_results": [{"fields": []}],
            "effective_fields": [],
        },
    )
    await add_pending_item(api.repository, review_id)

    response = await api.client.get(f"/api/v1/reviews/{review_id}")

    assert response.status_code == 200
    progress = response.json()["progress"]
    assert progress["current_step_id"] == "completeness"
    assert [step["status"] for step in progress["steps"]] == [
        "completed",
        "completed",
        "completed",
        "current",
        "upcoming",
        "upcoming",
        "upcoming",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("filename", "content", "content_type", "expected_status", "code"),
    [
        ("document.txt", pdf("text"), "application/pdf", 422, "invalid_upload"),
        ("document.pdf", pdf("mime"), "text/plain", 422, "invalid_upload"),
        ("document.pdf", b"not a PDF", "application/pdf", 422, "invalid_upload"),
        ("document.pdf", pdf("large", size=129), "application/pdf", 413, "upload_too_large"),
    ],
)
async def test_file_validation_returns_safe_structured_errors(
    api: APIHarness,
    filename: str,
    content: bytes,
    content_type: str,
    expected_status: int,
    code: str,
) -> None:
    response = await create_review(api.client, (filename, content, content_type))
    assert response.status_code == expected_status
    assert_safe_error(response, code)


@pytest.mark.asyncio
async def test_count_bundle_limits_and_exact_duplicates_are_preserved(
    api: APIHarness,
) -> None:
    too_many = await create_review(
        api.client,
        *((f"{index}.pdf", pdf(str(index)), "application/pdf") for index in range(6)),
    )
    assert too_many.status_code == 422
    assert_safe_error(too_many, "invalid_upload")

    too_large = await create_review(
        api.client,
        ("first.pdf", pdf("first", size=100), "application/pdf"),
        ("second.pdf", pdf("second", size=100), "application/pdf"),
    )
    assert too_large.status_code == 413
    assert_safe_error(too_large, "upload_too_large")
    assert not list(api.uploads_dir.rglob("*.pdf"))

    content = pdf("identical")
    duplicate = await create_review(
        api.client,
        ("first.pdf", content, "application/pdf"),
        ("second.pdf", content, "application/pdf"),
    )
    assert duplicate.status_code == 202
    duplicate_review = await api.repository.get(duplicate.json()["review_id"])
    assert duplicate_review is not None
    assert len(duplicate_review["documents"]) == 2
    assert len({item["document_id"] for item in duplicate_review["documents"]}) == 2
    assert len({item["sha256"] for item in duplicate_review["documents"]}) == 1


@pytest.mark.asyncio
async def test_late_invalid_file_does_not_leave_upload_artifacts(
    api: APIHarness,
) -> None:
    response = await create_review(
        api.client,
        ("valid.pdf", pdf("valid"), "application/pdf"),
        ("invalid.pdf", b"not a PDF", "application/pdf"),
    )

    assert response.status_code == 422
    assert_safe_error(response, "invalid_upload")
    assert not list(api.uploads_dir.rglob("*.pdf"))


@pytest.mark.asyncio
async def test_upload_filename_is_reduced_to_safe_basename(api: APIHarness) -> None:
    response = await create_review(
        api.client,
        ("../../application.pdf", pdf("safe path"), "application/pdf"),
    )
    assert response.status_code == 202
    review = await api.repository.get(response.json()["review_id"])
    assert review is not None
    assert review["documents"][0]["filename"] == "application.pdf"
    artifact_id = str(review["documents"][0]["artifact_id"])
    assert ".." not in artifact_id
    assert await api.artifact_store.read_upload(artifact_id) == pdf("safe path")


@pytest.mark.asyncio
async def test_report_poll_exports_and_not_ready_errors(api: APIHarness) -> None:
    response = await create_review(
        api.client,
        ("application.pdf", pdf("report"), "application/pdf"),
    )
    review_id = response.json()["review_id"]

    for suffix in ("report", "export.json", "export.md"):
        not_ready = await api.client.get(f"/api/v1/reviews/{review_id}/{suffix}")
        assert not_ready.status_code == 409
        assert_safe_error(not_ready, "report_not_ready")

    report = ReviewReport(
        company_name="Acme BV",
        status=ReportStatus.COMPLETE,
        executive_summary="The synthetic package is complete.",
        sections=[],
    )
    markdown = "# EvidenceFlow review report\n"
    await api.repository.save_report(review_id, report, markdown)

    poll = await api.client.get(f"/api/v1/reviews/{review_id}")
    assert poll.json()["status"] == "completed"
    assert poll.json()["report_available"] is True
    assert poll.json()["progress"]["current_step_id"] is None
    assert {step["status"] for step in poll.json()["progress"]["steps"]} == {
        "completed"
    }
    report_response = await api.client.get(f"/api/v1/reviews/{review_id}/report")
    assert report_response.json() == report.model_dump(mode="json")
    json_export = await api.client.get(f"/api/v1/reviews/{review_id}/export.json")
    assert json_export.status_code == 200
    assert json_export.json()["report"] == report.model_dump(mode="json")
    assert "attachment" in json_export.headers["content-disposition"]
    markdown_export = await api.client.get(f"/api/v1/reviews/{review_id}/export.md")
    assert markdown_export.text == markdown
    assert markdown_export.headers["content-type"].startswith("text/markdown")


@pytest.mark.asyncio
async def test_evidence_is_review_scoped_and_unknown_resources_are_safe(
    api: APIHarness,
) -> None:
    first = await create_review(
        api.client,
        ("first.pdf", pdf("first evidence"), "application/pdf"),
    )
    second = await create_review(
        api.client,
        ("second.pdf", pdf("second evidence"), "application/pdf"),
    )
    first_id = first.json()["review_id"]
    second_id = second.json()["review_id"]
    first_review = await api.repository.get(first_id)
    assert first_review is not None
    document_id = first_review["documents"][0]["document_id"]

    evidence = await api.client.get(
        f"/api/v1/reviews/{first_id}/documents/{document_id}"
    )
    assert evidence.status_code == 200
    assert evidence.content == pdf("first evidence")
    assert evidence.headers["cache-control"] == "private, no-store"

    cross_review = await api.client.get(
        f"/api/v1/reviews/{second_id}/documents/{document_id}"
    )
    assert cross_review.status_code == 404
    assert_safe_error(cross_review, "document_not_found")
    missing_review = await api.client.get("/api/v1/reviews/review_missing")
    assert missing_review.status_code == 404
    assert_safe_error(missing_review, "review_not_found")


@pytest.mark.asyncio
async def test_resume_is_atomic_for_repeated_and_concurrent_requests(
    api: APIHarness,
) -> None:
    created = await create_review(
        api.client,
        ("application.pdf", pdf("resume"), "application/pdf"),
    )
    review_id = created.json()["review_id"]
    item_id = "review-field-financial:employee_count"
    await add_pending_item(api.repository, review_id, item_id=item_id)
    payload = {
        "decisions": [
            {"review_item_id": item_id, "action": "correct", "value": 40}
        ]
    }

    responses = await asyncio.gather(
        api.client.post(f"/api/v1/reviews/{review_id}/resume", json=payload),
        api.client.post(f"/api/v1/reviews/{review_id}/resume", json=payload),
    )
    assert sorted(response.status_code for response in responses) == [202, 409]
    conflict = next(response for response in responses if response.status_code == 409)
    assert_safe_error(conflict, "review_not_resumable")

    repeated = await api.client.post(
        f"/api/v1/reviews/{review_id}/resume", json=payload
    )
    assert repeated.status_code == 409
    assert_safe_error(repeated, "review_not_resumable")


@pytest.mark.asyncio
async def test_resume_requires_exact_pending_item_set_and_valid_action(
    api: APIHarness,
) -> None:
    created = await create_review(
        api.client,
        ("application.pdf", pdf("invalid resume"), "application/pdf"),
    )
    review_id = created.json()["review_id"]
    item_id = "review-field-financial:employee_count"
    await add_pending_item(api.repository, review_id, item_id=item_id)

    unknown = await api.client.post(
        f"/api/v1/reviews/{review_id}/resume",
        json={
            "decisions": [
                {"review_item_id": "unknown-item", "action": "approve"}
            ]
        },
    )
    assert unknown.status_code == 422
    assert_safe_error(unknown, "invalid_review_decision")

    invalid_action = await api.client.post(
        f"/api/v1/reviews/{review_id}/resume",
        json={
            "decisions": [
                {"review_item_id": item_id, "action": "mark_unresolved"}
            ]
        },
    )
    assert invalid_action.status_code == 422
    assert_safe_error(invalid_action, "invalid_review_decision")


@pytest.mark.asyncio
async def test_unexpected_failures_do_not_expose_exception_details(
    api: APIHarness,
) -> None:
    created = await create_review(
        api.client,
        ("application.pdf", pdf("secret error"), "application/pdf"),
    )
    review_id = created.json()["review_id"]
    review = await api.repository.get(review_id)
    assert review is not None
    document_id = review["documents"][0]["document_id"]

    class BrokenStore:
        async def read_upload(self, artifact_id: str) -> bytes:
            del artifact_id
            raise RuntimeError("private filesystem detail: /secret/path")

    api.container.artifact_store = BrokenStore()
    response = await api.client.get(
        f"/api/v1/reviews/{review_id}/documents/{document_id}"
    )
    assert response.status_code == 500
    assert_safe_error(response, "internal_error")
    assert "/secret/path" not in response.text
