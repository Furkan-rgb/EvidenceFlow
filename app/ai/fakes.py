"""Deterministic capability fakes for model-free graph and API tests."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping

from app.domain import (
    DocumentClassification,
    DocumentType,
    ExtractionResult,
    PolicyEvidence,
    ProcessedDocument,
    ReviewReport,
    VerifiedReview,
)


class FakeDocumentClassifier:
    def __init__(self, results: Mapping[str, DocumentClassification]) -> None:
        self._results = dict(results)
        self.calls: list[str] = []

    async def classify(self, document: ProcessedDocument) -> DocumentClassification:
        self.calls.append(document.document_id)
        try:
            return self._results[document.document_id].model_copy(deep=True)
        except KeyError as exc:
            raise AssertionError(
                f"No fake classification configured for {document.document_id}"
            ) from exc


class FakeFieldExtractor:
    def __init__(
        self, results: Mapping[tuple[str, DocumentType], ExtractionResult]
    ) -> None:
        self._results = dict(results)
        self.calls: list[tuple[str, DocumentType]] = []

    async def extract(
        self, document: ProcessedDocument, document_type: DocumentType
    ) -> ExtractionResult:
        key = (document.document_id, document_type)
        self.calls.append(key)
        try:
            return self._results[key].model_copy(deep=True)
        except KeyError as exc:
            raise AssertionError(f"No fake extraction configured for {key}") from exc


class FakeReportComposer:
    def __init__(self, report: ReviewReport) -> None:
        self._report = report
        self.calls: list[tuple[VerifiedReview, list[PolicyEvidence]]] = []

    async def compose(
        self, review: VerifiedReview, policy_evidence: list[PolicyEvidence]
    ) -> ReviewReport:
        self.calls.append((review.model_copy(deep=True), list(policy_evidence)))
        return self._report.model_copy(deep=True)


class DeterministicEmbeddingProvider:
    """Small hashing embedder for tests; it does not make model-quality claims."""

    provider = "fake"
    model = "deterministic-hash-v1"

    def __init__(self, dimensions: int = 16) -> None:
        if dimensions < 2:
            raise ValueError("dimensions must be at least 2")
        self.dimensions = dimensions
        self.document_calls: list[list[str]] = []
        self.query_calls: list[str] = []

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.document_calls.append(list(texts))
        return [self._embed(text) for text in texts]

    async def embed_query(self, text: str) -> list[float]:
        self.query_calls.append(text)
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        values = [0.0] * self.dimensions
        for token in text.casefold().split():
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dimensions
            values[bucket] += 1.0
        magnitude = math.sqrt(sum(value * value for value in values))
        if magnitude == 0:
            values[0] = 1.0
            return values
        return [value / magnitude for value in values]


class FakePolicyRetriever:
    def __init__(self, evidence: list[PolicyEvidence]) -> None:
        self._evidence = list(evidence)
        self.calls: list[tuple[str, int]] = []

    async def search(self, query: str, *, limit: int = 5) -> list[PolicyEvidence]:
        self.calls.append((query, limit))
        return [item.model_copy(deep=True) for item in self._evidence[:limit]]
