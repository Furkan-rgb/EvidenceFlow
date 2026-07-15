from __future__ import annotations

import pymupdf
import pytest

from app.documents import PyMuPDFDocumentProcessor
from app.domain import UploadedDocument
from app.errors import UnsupportedDocumentError


class MemoryArtifactReader:
    def __init__(self, content: bytes) -> None:
        self.content = content
        self.artifact_ids: list[str] = []

    async def read_upload(self, artifact_id: str) -> bytes:
        self.artifact_ids.append(artifact_id)
        return self.content


def make_pdf(*page_texts: str) -> bytes:
    document = pymupdf.open()
    for text in page_texts:
        page = document.new_page()
        if text:
            page.insert_text((72, 72), text)
    content = document.tobytes()
    document.close()
    return content


def make_encrypted_pdf(*, user_password: str) -> bytes:
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), "Useful encrypted company onboarding evidence")
    content = document.tobytes(
        encryption=pymupdf.PDF_ENCRYPT_AES_256,
        owner_pw="owner-secret",
        user_pw=user_password,
    )
    document.close()
    return content


def upload(content: bytes) -> UploadedDocument:
    return UploadedDocument(
        document_id="doc-1",
        filename="application.pdf",
        artifact_id="opaque-artifact-1",
        size_bytes=len(content),
    )


@pytest.mark.asyncio
async def test_extracts_text_with_one_based_page_numbers() -> None:
    content = make_pdf(
        "Application form for Acme Corporation registration NL12345678",
        "Annual revenue EUR 1200000 and employee count 42",
    )
    reader = MemoryArtifactReader(content)

    result = await PyMuPDFDocumentProcessor(reader).process(upload(content))

    assert reader.artifact_ids == ["opaque-artifact-1"]
    assert [page.page_number for page in result.pages] == [1, 2]
    assert "Acme Corporation" in result.pages[0].text
    assert "employee count 42" in result.pages[1].text
    assert result.processor_metadata.processor == "pymupdf"
    assert result.processor_metadata.version


@pytest.mark.asyncio
async def test_retains_blank_pages_when_document_has_useful_text() -> None:
    content = make_pdf(
        "Application form for Acme Corporation registration NL12345678",
        "",
        "Annual revenue EUR 1200000 and employee count 42",
    )

    result = await PyMuPDFDocumentProcessor(
        MemoryArtifactReader(content)
    ).process(upload(content))

    assert [page.page_number for page in result.pages] == [1, 2, 3]
    assert result.pages[1].text == ""
    assert "employee count 42" in result.pages[2].text


@pytest.mark.asyncio
async def test_rejects_an_entirely_textless_pdf_with_ocr_limitation() -> None:
    content = make_pdf("", "")

    with pytest.raises(UnsupportedDocumentError) as error:
        await PyMuPDFDocumentProcessor(MemoryArtifactReader(content)).process(upload(content))

    assert error.value.details["reason"] == "ocr_required"
    assert error.value.details["page_numbers"] == [1, 2]
    assert error.value.details["useful_character_count"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("user_password", ["user-secret", ""])
async def test_rejects_encrypted_pdfs_including_empty_user_password(
    user_password: str,
) -> None:
    content = make_encrypted_pdf(user_password=user_password)

    with pytest.raises(UnsupportedDocumentError) as error:
        await PyMuPDFDocumentProcessor(MemoryArtifactReader(content)).process(upload(content))

    assert error.value.details["reason"] == "encrypted_pdf"


@pytest.mark.asyncio
async def test_rejects_invalid_pdf_signature() -> None:
    content = b"not a pdf"

    with pytest.raises(UnsupportedDocumentError) as error:
        await PyMuPDFDocumentProcessor(MemoryArtifactReader(content)).process(upload(content))

    assert error.value.details["reason"] == "invalid_pdf_signature"


@pytest.mark.asyncio
async def test_rejects_corrupt_pdf_with_valid_signature() -> None:
    content = b"%PDF-1.7\nthis is not a valid PDF structure"

    with pytest.raises(UnsupportedDocumentError) as error:
        await PyMuPDFDocumentProcessor(MemoryArtifactReader(content)).process(upload(content))

    assert error.value.details["reason"] == "invalid_pdf"


@pytest.mark.asyncio
async def test_rejects_pdf_that_pymupdf_would_silently_repair() -> None:
    content = make_pdf("Useful company onboarding evidence in a damaged PDF")
    prefix, trailer = content.rsplit(b"startxref\n", 1)
    _, suffix = trailer.split(b"\n", 1)
    damaged = prefix + b"startxref\n0\n" + suffix

    with pytest.raises(UnsupportedDocumentError) as error:
        await PyMuPDFDocumentProcessor(MemoryArtifactReader(damaged)).process(
            upload(damaged)
        )

    assert error.value.details["reason"] == "invalid_pdf"


@pytest.mark.asyncio
async def test_rejects_page_limit() -> None:
    content = make_pdf(
        "First useful page with enough characters for extraction",
        "Second useful page with enough characters for extraction",
    )

    with pytest.raises(UnsupportedDocumentError) as error:
        await PyMuPDFDocumentProcessor(
            MemoryArtifactReader(content), max_pages=1
        ).process(upload(content))

    assert error.value.details["reason"] == "page_limit_exceeded"
