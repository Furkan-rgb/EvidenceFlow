"""Typed application errors exposed safely at process boundaries."""

from __future__ import annotations

from typing import Any


class EvidenceFlowError(Exception):
    """Base error carrying a stable machine-readable code."""

    code = "evidenceflow_error"
    status_code = 500
    retryable = False

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class UnsupportedProviderError(EvidenceFlowError):
    code = "unsupported_provider"
    status_code = 422


class ModelUnavailableError(EvidenceFlowError):
    code = "model_unavailable"
    status_code = 503
    retryable = True


class InvalidStructuredOutputError(EvidenceFlowError):
    code = "invalid_structured_output"


class EmbeddingIndexMismatchError(EvidenceFlowError):
    code = "embedding_index_mismatch"
    status_code = 503


class PolicyIndexMissingError(EvidenceFlowError):
    code = "policy_index_missing"
    status_code = 503


class UnsupportedDocumentError(EvidenceFlowError):
    code = "unsupported_document"
    status_code = 422


class InvalidReviewDecisionError(EvidenceFlowError):
    code = "invalid_review_decision"
    status_code = 422


class ReviewNotFoundError(EvidenceFlowError):
    code = "review_not_found"
    status_code = 404


class ReviewNotResumableError(EvidenceFlowError):
    code = "review_not_resumable"
    status_code = 409


class DocumentNotFoundError(EvidenceFlowError):
    code = "document_not_found"
    status_code = 404


class ReportNotReadyError(EvidenceFlowError):
    code = "report_not_ready"
    status_code = 409


class InvalidUploadError(EvidenceFlowError):
    code = "invalid_upload"
    status_code = 422


class UploadTooLargeError(EvidenceFlowError):
    code = "upload_too_large"
    status_code = 413
