"""Review creation, polling, durable resume, reports, exports, and evidence."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any, cast
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Response, UploadFile, status

from app.api.dependencies import get_container
from app.api.schemas import (
    CreateReviewResponse,
    ResumeReviewRequest,
    ResumeReviewResponse,
)
from app.domain import ReviewDecision, ReviewItem
from app.errors import (
    DocumentNotFoundError,
    InvalidReviewDecisionError,
    InvalidUploadError,
    ReportNotReadyError,
    ReviewNotFoundError,
    ReviewNotResumableError,
    UploadTooLargeError,
)
from app.review import ReviewDecisionValidationError, validate_review_decisions

router = APIRouter(prefix="/api/v1/reviews", tags=["reviews"])
_CONTROL_CHARACTERS = re.compile(r"[\x00-\x1f\x7f]")
_ALLOWED_CONTENT_TYPES = {"application/pdf", "application/octet-stream"}


def _safe_filename(filename: str | None) -> str:
    # Browsers may submit either POSIX or Windows-style path components.
    name = Path((filename or "document.pdf").replace("\\", "/")).name
    name = _CONTROL_CHARACTERS.sub("", name).strip()[:120]
    if not name or not name.lower().endswith(".pdf"):
        raise InvalidUploadError("Every uploaded file must use a .pdf extension")
    return name


async def _read_limited(upload: UploadFile, limit: int) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while chunk := await upload.read(1024 * 1024):
        total += len(chunk)
        if total > limit:
            raise UploadTooLargeError(f"{upload.filename or 'PDF'} exceeds the file-size limit")
        chunks.append(chunk)
    return b"".join(chunks)


async def _require_review(container: Any, review_id: str) -> dict[str, Any]:
    review = await container.repository.get(review_id)
    if review is None:
        raise ReviewNotFoundError("Review was not found")
    return cast(dict[str, Any], review)


def _review_summary(
    review: dict[str, Any], snapshot: dict[str, Any]
) -> dict[str, int]:
    extractions = list(snapshot.get("extraction_results") or [])
    existing = {
        str(key): int(value)
        for key, value in dict(snapshot.get("summary") or {}).items()
        if isinstance(value, int) and not isinstance(value, bool)
    }
    existing.update(
        {
            "document_count": len(review["documents"]),
            "classified_count": len(snapshot.get("classifications") or []),
            "extracted_field_count": sum(
                len(item.get("fields") or []) for item in extractions
            ),
            "finding_count": len(snapshot.get("findings") or []),
            "pending_review_count": len(review["pending_reviews"]),
        }
    )
    return existing


async def _latest_checkpoint_snapshot(
    container: Any, review: dict[str, Any], snapshot: dict[str, Any]
) -> dict[str, Any]:
    graph = getattr(container, "graph", None)
    if review["status"] != "processing" or graph is None:
        return snapshot
    try:
        checkpoint = await graph.aget_state(
            {"configurable": {"thread_id": str(review["thread_id"])}}
        )
    except Exception:
        # Progress projection is optional; durable business state remains canonical.
        return snapshot
    values = getattr(checkpoint, "values", None)
    if isinstance(values, Mapping):
        return {str(key): value for key, value in values.items()}
    return snapshot


@router.post("", response_model=CreateReviewResponse, status_code=status.HTTP_202_ACCEPTED)
async def create_review(
    files: Annotated[list[UploadFile], File(...)],
    container: Annotated[Any, Depends(get_container)],
) -> CreateReviewResponse:
    if not 1 <= len(files) <= 5:
        raise InvalidUploadError("Upload between one and five PDF documents")

    review_id = f"review_{uuid4().hex}"
    thread_id = f"thread_{uuid4().hex}"
    buffered: list[tuple[dict[str, object], bytes]] = []
    bundle_bytes = 0
    for upload in files:
        filename = _safe_filename(upload.filename)
        if upload.content_type not in _ALLOWED_CONTENT_TYPES:
            raise InvalidUploadError(f"{filename} is not identified as a PDF")
        content = await _read_limited(upload, container.settings.max_file_bytes)
        bundle_bytes += len(content)
        if bundle_bytes > container.settings.max_bundle_bytes:
            raise UploadTooLargeError("The combined PDF bundle exceeds the size limit")
        if not content.startswith(b"%PDF-"):
            raise InvalidUploadError(f"{filename} does not have a valid PDF signature")
        digest = hashlib.sha256(content).hexdigest()
        document_id = f"document_{uuid4().hex}"
        buffered.append(
            (
                {
                    "document_id": document_id,
                    "filename": filename,
                    "sha256": digest,
                    "size_bytes": len(content),
                    "content_type": "application/pdf",
                },
                content,
            )
        )

    # Validate and buffer the complete bundle before creating any filesystem
    # artifacts. A bad later file must not orphan uploads accepted earlier in
    # this request.
    rows: list[dict[str, object]] = []
    for row, content in buffered:
        artifact_id = await container.artifact_store.save_upload_bytes(
            review_id, str(row["document_id"]), content
        )
        row["artifact_id"] = artifact_id
        rows.append(row)
    await container.repository.create(review_id, thread_id, rows)
    container.runner.wake()
    return CreateReviewResponse(review_id=review_id, thread_id=thread_id)


@router.get("/{review_id}")
async def get_review(
    review_id: str, container: Annotated[Any, Depends(get_container)]
) -> dict[str, Any]:
    review = await _require_review(container, review_id)
    snapshot = dict(review.get("snapshot") or {})
    snapshot = await _latest_checkpoint_snapshot(container, review, snapshot)
    snapshot.update(
        {
            "review_id": review_id,
            "thread_id": review["thread_id"],
            "status": review["status"],
            "report_status": review.get("report_status"),
            "documents": snapshot.get("documents", review["documents"]),
            "pending_reviews": review["pending_reviews"],
            "summary": _review_summary(review, snapshot),
            "report_available": review["report"] is not None,
            "error": review.get("error"),
            "revision": review["revision"],
        }
    )
    return snapshot


@router.post(
    "/{review_id}/resume",
    response_model=ResumeReviewResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def resume_review(
    review_id: str,
    request: ResumeReviewRequest,
    container: Annotated[Any, Depends(get_container)],
) -> ResumeReviewResponse:
    review = await _require_review(container, review_id)
    if review["status"] != "needs_review":
        raise ReviewNotResumableError("Review is not waiting for human decisions")
    pending = [ReviewItem.model_validate(item) for item in review["pending_reviews"]]
    decisions = [
        ReviewDecision.model_validate(
            decision.model_dump(mode="json", exclude_none=True)
        )
        for decision in request.decisions
    ]
    try:
        validated = validate_review_decisions(pending, decisions)
    except ReviewDecisionValidationError as error:
        raise InvalidReviewDecisionError(
            str(error), details=error.details
        ) from error
    await container.repository.begin_resume(
        review_id,
        [decision.model_dump(mode="json", exclude_none=True) for decision in validated],
    )
    container.runner.wake()
    return ResumeReviewResponse(review_id=review_id)


@router.get("/{review_id}/report")
async def get_report(
    review_id: str, container: Annotated[Any, Depends(get_container)]
) -> dict[str, Any]:
    review = await _require_review(container, review_id)
    if review["report"] is None:
        raise ReportNotReadyError("The review report is not ready")
    return dict(review["report"])


@router.get("/{review_id}/export.json")
async def export_json(
    review_id: str, container: Annotated[Any, Depends(get_container)]
) -> Response:
    review = await _require_review(container, review_id)
    if review["report"] is None:
        raise ReportNotReadyError("The review report is not ready")
    payload = {
        "review_id": review_id,
        "review": review["snapshot"],
        "report": review["report"],
    }
    return Response(
        json.dumps(payload, indent=2, ensure_ascii=False),
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{review_id}.json"'},
    )


@router.get("/{review_id}/export.md")
async def export_markdown(
    review_id: str, container: Annotated[Any, Depends(get_container)]
) -> Response:
    review = await _require_review(container, review_id)
    if review["report_markdown"] is None:
        raise ReportNotReadyError("The review report is not ready")
    return Response(
        review["report_markdown"],
        media_type="text/markdown",
        headers={"Content-Disposition": f'attachment; filename="{review_id}.md"'},
    )


@router.get("/{review_id}/documents/{document_id}")
async def get_document(
    review_id: str,
    document_id: str,
    container: Annotated[Any, Depends(get_container)],
) -> Response:
    await _require_review(container, review_id)
    document = await container.repository.get_document(review_id, document_id)
    if document is None:
        raise DocumentNotFoundError("The requested document was not found")
    content = await container.artifact_store.read_upload(str(document["artifact_id"]))
    filename = str(document["filename"]).replace('"', "")
    return Response(
        content,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'inline; filename="{filename}"',
            "X-Content-Type-Options": "nosniff",
            "Cache-Control": "private, no-store",
        },
    )
