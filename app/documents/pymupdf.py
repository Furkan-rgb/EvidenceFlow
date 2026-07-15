"""Text-only PDF processing implemented behind the document processor port."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Protocol

import pymupdf

from app.domain import PageContent, ProcessedDocument, ProcessorMetadata, UploadedDocument
from app.errors import UnsupportedDocumentError


class ArtifactReader(Protocol):
    """Minimum artifact capability needed by a document processor."""

    async def read_upload(self, artifact_id: str) -> bytes: ...


class PyMuPDFDocumentProcessor:
    """Extract ordered, one-based page text without leaking PyMuPDF objects.

    V1 intentionally rejects encrypted, corrupt, over-sized, and effectively
    textless PDFs. OCR is a future document-processor implementation rather than
    a hidden fallback in this adapter.
    """

    def __init__(
        self,
        artifact_reader: ArtifactReader,
        *,
        max_pages: int = 50,
        min_useful_characters: int = 20,
    ) -> None:
        if max_pages < 1:
            raise ValueError("max_pages must be at least 1")
        if min_useful_characters < 1:
            raise ValueError("min_useful_characters must be at least 1")
        self._max_pages = max_pages
        self._min_useful_characters = min_useful_characters
        self._artifact_reader = artifact_reader

    async def process(self, document: UploadedDocument) -> ProcessedDocument:
        """Process an uploaded PDF off the event loop."""

        content = await self._artifact_reader.read_upload(document.artifact_id)
        return await asyncio.to_thread(self._process_sync, document, content)

    def _process_sync(self, document: UploadedDocument, content: bytes) -> ProcessedDocument:
        if not content.startswith(b"%PDF-"):
            raise UnsupportedDocumentError(
                "The uploaded file is not a structurally valid PDF.",
                details={"document_id": document.document_id, "reason": "invalid_pdf_signature"},
            )

        try:
            pdf: pymupdf.Document = pymupdf.open(  # type: ignore[no-untyped-call]
                stream=content, filetype="pdf"
            )
        except (pymupdf.FileDataError, pymupdf.EmptyFileError, RuntimeError, ValueError) as exc:
            raise UnsupportedDocumentError(
                "The uploaded PDF is corrupt or cannot be opened.",
                details={"document_id": document.document_id, "reason": "invalid_pdf"},
            ) from exc

        try:
            # ``needs_pass`` is false when a PDF has an empty user password,
            # even though its contents are still protected by an encryption
            # dictionary. V1 rejects encrypted input consistently rather than
            # treating that special case as an ordinary text PDF.
            encryption_kind, _ = pdf.xref_get_key(  # type: ignore[no-untyped-call]
                -1, "Encrypt"
            )
            if pdf.needs_pass or encryption_kind != "null":
                raise UnsupportedDocumentError(
                    "Encrypted PDFs are not supported in EvidenceFlow V1.",
                    details={"document_id": document.document_id, "reason": "encrypted_pdf"},
                )
            if pdf.is_repaired:
                raise UnsupportedDocumentError(
                    "The uploaded PDF is corrupt or requires structural repair.",
                    details={"document_id": document.document_id, "reason": "invalid_pdf"},
                )
            if pdf.page_count < 1:
                raise UnsupportedDocumentError(
                    "The PDF contains no pages.",
                    details={"document_id": document.document_id, "reason": "empty_pdf"},
                )
            if pdf.page_count > self._max_pages:
                raise UnsupportedDocumentError(
                    f"The PDF exceeds the {self._max_pages}-page V1 limit.",
                    details={
                        "document_id": document.document_id,
                        "reason": "page_limit_exceeded",
                        "page_count": pdf.page_count,
                        "max_pages": self._max_pages,
                    },
                )

            pages: list[PageContent] = []
            for index in range(pdf.page_count):
                page: pymupdf.Page = pdf.load_page(index)  # type: ignore[no-untyped-call]
                pages.append(
                    PageContent(
                        page_number=index + 1,
                        text=page.get_text(  # type: ignore[no-untyped-call]
                            "text", sort=True
                        ).strip(),
                    )
                )
        except UnsupportedDocumentError:
            raise
        except (pymupdf.FileDataError, RuntimeError, ValueError) as exc:
            raise UnsupportedDocumentError(
                "Text could not be extracted from the uploaded PDF.",
                details={"document_id": document.document_id, "reason": "extraction_failed"},
            ) from exc
        finally:
            pdf.close()  # type: ignore[no-untyped-call]

        useful_character_count = self._useful_character_count(pages)
        if useful_character_count < self._min_useful_characters:
            raise UnsupportedDocumentError(
                "The PDF has no useful extractable text as a whole; "
                "scanned documents and OCR are outside EvidenceFlow V1.",
                details={
                    "document_id": document.document_id,
                    "reason": "ocr_required",
                    "page_numbers": [page.page_number for page in pages],
                    "useful_character_count": useful_character_count,
                    "minimum_useful_characters": self._min_useful_characters,
                },
            )

        return ProcessedDocument(
            document_id=document.document_id,
            filename=document.filename,
            pages=pages,
            processor_metadata=ProcessorMetadata(
                processor="pymupdf",
                version=pymupdf.VersionBind,
            ),
        )

    @staticmethod
    def _useful_character_count(pages: Sequence[PageContent]) -> int:
        return sum(
            character.isalnum()
            for page in pages
            for character in page.text
        )
