"""Opt-in adapter smoke tests against the locally configured Ollama models."""

from __future__ import annotations

import os

import pytest

from app.ai.classification.service import LLMDocumentClassifier
from app.ai.config import load_models_config
from app.ai.extraction.service import LLMFieldExtractor
from app.ai.models.factory import create_chat_model, create_embedding_provider
from app.ai.reporting.service import LLMReportComposer
from app.config import Settings
from app.domain import (
    DocumentType,
    PageContent,
    PolicyEvidence,
    ProcessedDocument,
    ProcessorMetadata,
    ReportStatus,
    VerifiedReview,
)

pytestmark = [
    pytest.mark.ollama,
    pytest.mark.skipif(
        os.getenv("EVIDENCEFLOW_RUN_OLLAMA_TESTS") != "1",
        reason="set EVIDENCEFLOW_RUN_OLLAMA_TESTS=1 to exercise local Ollama",
    ),
]


def configured_models():
    settings = Settings()
    return load_models_config(ollama_base_url=settings.ollama_base_url)


@pytest.mark.asyncio
async def test_configured_chat_model_supports_domain_structured_output() -> None:
    config = configured_models()
    classifier = LLMDocumentClassifier(create_chat_model(config.classification))
    document = ProcessedDocument(
        document_id="ollama-smoke-application",
        filename="application.pdf",
        pages=[
            PageContent(
                page_number=1,
                text=(
                    "BUSINESS ONBOARDING APPLICATION FORM\n"
                    "Company name: Acme B.V.\nRegistration number: NL12345678\n"
                    "Annual revenue EUR: 1200000\nEmployee count: 42"
                ),
            )
        ],
        processor_metadata=ProcessorMetadata(processor="ollama-smoke"),
    )

    result = await classifier.classify(document)

    assert result.document_id == document.document_id
    assert result.document_type is DocumentType.APPLICATION_FORM


@pytest.mark.asyncio
async def test_configured_embedding_model_has_manifest_dimensions() -> None:
    config = configured_models()
    provider = create_embedding_provider(config.embeddings)

    vector = await provider.embed_query("company registration evidence")

    assert len(vector) == config.embeddings.dimensions


@pytest.mark.asyncio
async def test_configured_extraction_and_reporting_schemas() -> None:
    config = configured_models()
    document = ProcessedDocument(
        document_id="ollama-smoke-extraction",
        filename="application.pdf",
        pages=[
            PageContent(
                page_number=1,
                text=(
                    "BUSINESS ONBOARDING APPLICATION FORM\n"
                    "Company name: Acme B.V.\nRegistration number: NL12345678\n"
                    "Annual revenue EUR: 1200000\nEmployee count: 42"
                ),
            )
        ],
        processor_metadata=ProcessorMetadata(processor="ollama-smoke"),
    )
    extraction = await LLMFieldExtractor(
        create_chat_model(config.extraction)
    ).extract(document, DocumentType.APPLICATION_FORM)
    evidence = PolicyEvidence(
        evidence_id="EFP-ONBOARDING:1.4:chunk-0",
        policy_id="EFP-ONBOARDING",
        title="Company Onboarding Requirements",
        section_id="1.4",
        text="A complete package has required evidence and no actionable conflicts.",
        score=0.9,
        source_path="company-onboarding-requirements.md",
    )
    report = await LLMReportComposer(create_chat_model(config.reporting)).compose(
        VerifiedReview(
            review_id="ollama-smoke-review",
            company_name="Acme B.V.",
            status=ReportStatus.COMPLETE,
            extractions=[extraction],
        ),
        [evidence],
    )

    assert {field.field_name for field in extraction.fields} == {
        "company_name",
        "registration_number",
        "annual_revenue_eur",
        "employee_count",
    }
    assert report.company_name == "Acme B.V."
    assert report.status is ReportStatus.COMPLETE
