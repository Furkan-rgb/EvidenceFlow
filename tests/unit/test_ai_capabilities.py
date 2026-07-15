from __future__ import annotations

from pathlib import Path

import pytest

from app.ai.classification.service import LLMDocumentClassifier
from app.ai.config import ChatModelConfig, load_models_config
from app.ai.extraction.service import LLMFieldExtractor
from app.ai.models.factory import create_chat_model
from app.ai.reporting.service import LLMReportComposer
from app.domain import (
    DocumentType,
    Finding,
    FindingSeverity,
    FindingType,
    PageContent,
    PolicyEvidence,
    ProcessedDocument,
    ProcessorMetadata,
    ReportStatus,
    VerifiedReview,
)
from app.errors import UnsupportedProviderError


class StubRunnable:
    def __init__(self, responses: list[object]) -> None:
        self.responses = responses
        self.calls = 0

    async def ainvoke(self, _messages: object) -> object:
        response = self.responses[self.calls]
        self.calls += 1
        if isinstance(response, Exception):
            raise response
        return response


class StubChatModel:
    def __init__(self, responses: list[object]) -> None:
        self.runnable = StubRunnable(responses)
        self.schemas: list[type[object]] = []
        self.methods: list[str] = []

    def with_structured_output(
        self, schema: type[object], *, method: str, include_raw: bool = False
    ) -> StubRunnable:
        del include_raw
        self.schemas.append(schema)
        self.methods.append(method)
        return self.runnable


def processed_document() -> ProcessedDocument:
    return ProcessedDocument(
        document_id="doc-1",
        filename="application.pdf",
        pages=[
            PageContent(
                page_number=1,
                text=(
                    "Application Form\nCompany name: Acme B.V.\n"
                    "Registration: NL123\nRevenue EUR: 1200000\nEmployees: 42"
                ),
            )
        ],
        processor_metadata=ProcessorMetadata(processor="test"),
    )


@pytest.mark.asyncio
async def test_classifier_uses_json_schema_and_injects_known_identity() -> None:
    model = StubChatModel(
        [
            {
                "document_type": "application_form",
                "confidence": 0.9,
                "reasoning_summary": "Application fields are present.",
            },
        ]
    )

    result = await LLMDocumentClassifier(model).classify(processed_document())  # type: ignore[arg-type]

    assert result.document_type is DocumentType.APPLICATION_FORM
    assert model.methods == ["json_schema"]
    assert model.runnable.calls == 1
    assert result.document_id == "doc-1"


@pytest.mark.asyncio
async def test_classifier_rejects_model_authored_review_metadata() -> None:
    reviewed_payload = {
        "document_type": "unknown",
        "confidence": 0.5,
        "reasoning_summary": "The document type is ambiguous.",
        "effective_document_type": "application_form",
        "was_human_reviewed": True,
        "review_decision_id": "model-invented-decision",
    }
    clean_payload = {
        "document_type": "application_form",
        "confidence": 0.9,
        "reasoning_summary": "Application fields are present.",
    }
    model = StubChatModel([reviewed_payload, clean_payload])

    result = await LLMDocumentClassifier(model).classify(processed_document())  # type: ignore[arg-type]

    assert model.runnable.calls == 2
    assert result.was_human_reviewed is False
    assert result.effective_document_type is None


@pytest.mark.asyncio
async def test_extractor_returns_typed_provenance() -> None:
    model = StubChatModel(
        [
            {
                "company_name": {
                    "field_id": "doc-1:company_name",
                    "value": "Acme B.V.",
                    "confidence": 0.98,
                    "evidence": [
                        {
                            "page_number": 1,
                            "source_text": "Company name: Acme B.V.",
                        }
                    ],
                },
                "registration_number": {
                    "field_id": "doc-1:registration_number",
                    "value": "NL123",
                    "confidence": 0.97,
                    "evidence": [
                        {"page_number": 1, "source_text": "Registration: NL123"}
                    ],
                },
                "annual_revenue_eur": {
                    "field_id": "doc-1:annual_revenue_eur",
                    "value": 1200000,
                    "confidence": 0.96,
                    "evidence": [
                        {"page_number": 1, "source_text": "Revenue EUR: 1200000"}
                    ],
                },
                "employee_count": {
                    "field_id": "doc-1:employee_count",
                    "value": 42,
                    "confidence": 0.99,
                    "evidence": [
                        {"page_number": 1, "source_text": "Employees: 42"}
                    ],
                },
            }
        ]
    )

    result = await LLMFieldExtractor(model).extract(  # type: ignore[arg-type]
        processed_document(), DocumentType.APPLICATION_FORM
    )

    assert [field.field_name for field in result.fields] == [
        "company_name",
        "registration_number",
        "annual_revenue_eur",
        "employee_count",
    ]
    assert result.fields[0].evidence[0].page_number == 1


@pytest.mark.asyncio
async def test_extractor_canonicalizes_model_authored_field_identifiers() -> None:
    payload = {
        "company_name": {
            "field_id": "stale-document:company_name",
            "value": "Acme B.V.",
            "confidence": 0.98,
            "evidence": [
                {"page_number": 1, "source_text": "Company name: Acme B.V."}
            ],
        },
        "registration_number": {
            "field_id": "registration_number",
            "value": "NL123",
            "confidence": 0.97,
            "evidence": [
                {"page_number": 1, "source_text": "Registration: NL123"}
            ],
        },
        "annual_revenue_eur": {
            "field_id": "wrong",
            "value": 1200000,
            "confidence": 0.96,
            "evidence": [
                {"page_number": 1, "source_text": "Revenue EUR: 1200000"}
            ],
        },
        "employee_count": {
            "field_id": "wrong",
            "value": 42,
            "confidence": 0.99,
            "evidence": [{"page_number": 1, "source_text": "Employees: 42"}],
        },
    }
    model = StubChatModel([payload])

    result = await LLMFieldExtractor(model).extract(  # type: ignore[arg-type]
        processed_document(), DocumentType.APPLICATION_FORM
    )

    assert [field.field_id for field in result.fields] == [
        "doc-1:company_name",
        "doc-1:registration_number",
        "doc-1:annual_revenue_eur",
        "doc-1:employee_count",
    ]
    assert model.runnable.calls == 1


@pytest.mark.asyncio
async def test_extractor_derives_redundant_clarification_text() -> None:
    document = ProcessedDocument(
        document_id="doc-correspondence",
        filename="correspondence.pdf",
        pages=[
            PageContent(
                page_number=1,
                text=(
                    "SUPPORTING CORRESPONDENCE\nCompany name: Acme B.V.\n"
                    "The registered and operating addresses are the same."
                ),
            )
        ],
        processor_metadata=ProcessorMetadata(processor="test"),
    )
    model = StubChatModel(
        [
            {
                "company_name": {
                    "field_id": "wrong:company_name",
                    "value": "Acme B.V.",
                    "confidence": 0.99,
                    "evidence": [
                        {"page_number": 1, "source_text": "Company name: Acme B.V."}
                    ],
                },
                "clarification_statements": [
                    {
                        "statement_id": "wrong",
                        "topic": "operating address",
                        "value": "The registered and operating addresses are the same.",
                        "confidence": 1.0,
                        "evidence": [
                            {
                                "page_number": 1,
                                "source_text": (
                                    "The registered and operating addresses are the same."
                                ),
                            }
                        ],
                    }
                ],
            }
        ]
    )

    result = await LLMFieldExtractor(model).extract(  # type: ignore[arg-type]
        document, DocumentType.SUPPORTING_CORRESPONDENCE
    )

    assert result.clarification_statements[0].statement_id == (
        "doc-correspondence:clarification:0"
    )
    assert result.clarification_statements[0].text == (
        "The registered and operating addresses are the same."
    )
    assert model.runnable.calls == 1


@pytest.mark.asyncio
async def test_extractor_wraps_explicitly_null_missing_field() -> None:
    model = StubChatModel(
        [
            {
                "company_name": {
                    "field_id": "doc-1:company_name",
                    "value": "Acme B.V.",
                    "confidence": 0.98,
                    "evidence": [
                        {"page_number": 1, "source_text": "Company name: Acme B.V."}
                    ],
                },
                "registration_number": {
                    "field_id": "doc-1:registration_number",
                    "value": "NL123",
                    "confidence": 0.97,
                    "evidence": [
                        {"page_number": 1, "source_text": "Registration: NL123"}
                    ],
                },
                "annual_revenue_eur": None,
                "employee_count": {
                    "field_id": "doc-1:employee_count",
                    "value": 42,
                    "confidence": 0.99,
                    "evidence": [
                        {"page_number": 1, "source_text": "Employees: 42"}
                    ],
                },
            }
        ]
    )

    result = await LLMFieldExtractor(model).extract(  # type: ignore[arg-type]
        processed_document(), DocumentType.APPLICATION_FORM
    )

    missing = next(
        field for field in result.fields if field.field_name == "annual_revenue_eur"
    )
    assert missing.field_id == "doc-1:annual_revenue_eur"
    assert missing.value is None
    assert missing.evidence == []
    assert model.runnable.calls == 1


@pytest.mark.asyncio
async def test_extractor_replaces_paraphrase_with_unique_exact_page_line() -> None:
    payload = {
        "company_name": {
            "field_id": "doc-1:company_name",
            "value": "Acme B.V.",
            "confidence": 0.98,
            "evidence": [
                {"page_number": 1, "source_text": "The company is Acme B.V."}
            ],
        },
        "registration_number": {
            "field_id": "doc-1:registration_number",
            "value": "NL123",
            "confidence": 0.97,
            "evidence": [
                {"page_number": 1, "source_text": "Registration: NL123"}
            ],
        },
        "annual_revenue_eur": {
            "field_id": "doc-1:annual_revenue_eur",
            "value": 1200000,
            "confidence": 0.96,
            "evidence": [
                {"page_number": 1, "source_text": "Revenue EUR: 1200000"}
            ],
        },
        "employee_count": {
            "field_id": "doc-1:employee_count",
            "value": 42,
            "confidence": 0.99,
            "evidence": [{"page_number": 1, "source_text": "Employees: 42"}],
        },
    }
    model = StubChatModel([payload])

    result = await LLMFieldExtractor(model).extract(  # type: ignore[arg-type]
        processed_document(), DocumentType.APPLICATION_FORM
    )

    company = next(field for field in result.fields if field.field_name == "company_name")
    assert company.evidence[0].source_text == "Company name: Acme B.V."
    assert model.runnable.calls == 1


@pytest.mark.asyncio
async def test_report_retries_invented_reference_and_imposes_canonical_values() -> None:
    invalid = {
        "executive_summary": "One issue needs follow-up.",
        "sections": [
            {
                "title": "Issues",
                "summary": "A required document is missing.",
                "finding_ids": ["invented-finding"],
                "policy_evidence_ids": [],
            }
        ],
    }
    valid = {
        **invalid,
        "sections": [
            {
                "title": "Issues",
                "summary": "A required document is missing.",
                "finding_ids": ["finding-missing-financial"],
                "policy_evidence_ids": ["EFP-ONBOARDING:1.2:chunk-0"],
            }
        ],
    }
    model = StubChatModel([invalid, valid])
    finding = Finding(
        finding_id="finding-missing-financial",
        type=FindingType.MISSING_DOCUMENT,
        severity=FindingSeverity.HIGH,
        message="Financial statement is missing.",
        document_type=DocumentType.FINANCIAL_STATEMENT,
    )
    review = VerifiedReview(
        review_id="review-1",
        company_name="Acme B.V.",
        status=ReportStatus.INCOMPLETE,
        findings=[finding],
    )
    evidence = PolicyEvidence(
        evidence_id="EFP-ONBOARDING:1.2:chunk-0",
        policy_id="EFP-ONBOARDING",
        title="Company Onboarding Requirements",
        section_id="1.2",
        text="A missing required document makes the package incomplete.",
        score=0.9,
        source_path="company-onboarding-requirements.md",
    )

    report = await LLMReportComposer(model).compose(review, [evidence])  # type: ignore[arg-type]

    assert model.runnable.calls == 2
    assert report.company_name == "Acme B.V."
    assert report.status is ReportStatus.INCOMPLETE
    assert report.sections[0].finding_ids == ["finding-missing-financial"]


def test_model_registry_defaults_and_overrides() -> None:
    root = Path(__file__).parents[2]

    config = load_models_config(
        root / "config/models.yaml",
        extraction_model="local-extraction-override",
        ollama_base_url="http://ollama.test:11434",
    )

    assert config.classification.model == "gemma4:12b-mlx"
    assert config.extraction.model == "local-extraction-override"
    assert config.reporting.model == "gemma4:12b-mlx"
    assert config.embeddings.model == "embeddinggemma"
    assert config.embeddings.model_digest is not None
    assert config.embeddings.dimensions == 768
    assert config.classification.base_url == "http://ollama.test:11434"
    assert config.classification.timeout_seconds == 300
    assert config.extraction.timeout_seconds == 600
    assert config.reporting.timeout_seconds == 600


def test_same_model_override_preserves_digest_and_changed_model_clears_it() -> None:
    root = Path(__file__).parents[2]

    same = load_models_config(
        root / "config/models.yaml",
        classification_model="gemma4:12b-mlx",
        embedding_model="embeddinggemma",
    )
    changed = load_models_config(
        root / "config/models.yaml",
        classification_model="another-chat-model",
        embedding_model="another-embedding-model",
    )

    assert same.classification.model_digest is not None
    assert same.embeddings.model_digest is not None
    assert changed.classification.model_digest is None
    assert changed.embeddings.model_digest is None


def test_unsupported_chat_provider_fails_without_fallback() -> None:
    config = ChatModelConfig(provider="openai", model="example")

    with pytest.raises(UnsupportedProviderError):
        create_chat_model(config)
