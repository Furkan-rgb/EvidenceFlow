"""Application-facing capability protocols."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

from app.domain import (
    DocumentClassification,
    DocumentType,
    ExtractionResult,
    PolicyEvidence,
    ProcessedDocument,
    ReviewReport,
    UploadedDocument,
    VerifiedReview,
)


class DocumentProcessor(Protocol):
    async def process(self, document: UploadedDocument) -> ProcessedDocument: ...


class DocumentClassifier(Protocol):
    async def classify(self, document: ProcessedDocument) -> DocumentClassification: ...


class FieldExtractor(Protocol):
    async def extract(
        self, document: ProcessedDocument, document_type: DocumentType
    ) -> ExtractionResult: ...


class ReportComposer(Protocol):
    async def compose(
        self, review: VerifiedReview, policy_evidence: list[PolicyEvidence]
    ) -> ReviewReport: ...


class EmbeddingProvider(Protocol):
    provider: str
    model: str

    async def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    async def embed_query(self, text: str) -> list[float]: ...


class PolicyRetriever(Protocol):
    async def search(self, query: str, *, limit: int = 5) -> list[PolicyEvidence]: ...


class ArtifactStore(Protocol):
    async def save_upload_bytes(
        self, review_id: str, document_id: str, content: bytes
    ) -> str: ...

    async def read_upload(self, artifact_id: str) -> bytes: ...

    async def save_export(self, review_id: str, name: str, content: bytes) -> str: ...


class ReviewRepository(Protocol):
    async def health(self) -> bool: ...

    async def create(
        self,
        review_id: str,
        thread_id: str,
        documents: Sequence[dict[str, object]],
    ) -> None: ...

    async def get(self, review_id: str) -> dict[str, Any] | None: ...

    async def get_document(
        self, review_id: str, document_id: str
    ) -> dict[str, Any] | None: ...

    async def update_status(self, review_id: str, status: str, **values: object) -> None: ...

    async def save_snapshot(self, review_id: str, snapshot: object) -> None: ...

    async def save_review_items(
        self, review_id: str, items: Sequence[object]
    ) -> None: ...

    async def save_review_decisions(
        self, review_id: str, decisions: list[object]
    ) -> None: ...

    async def begin_resume(
        self, review_id: str, decisions: Sequence[object]
    ) -> str: ...

    async def save_report(self, review_id: str, report: object, markdown: str) -> None: ...

    async def claim_next_job(self) -> dict[str, Any] | None: ...

    async def finish_job(self, job_id: str, *, failed: bool = False) -> None: ...

    async def recover_jobs(self) -> int: ...
