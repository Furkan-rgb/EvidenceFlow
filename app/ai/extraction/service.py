"""LangChain-backed, schema-validated field extraction."""

from __future__ import annotations

import re
from datetime import date

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from app.ai.extraction.prompts import EXTRACTION_SYSTEM_PROMPT, FIELD_SPECIFICATIONS
from app.ai.extraction.schemas import (
    ApplicationFormModelOutput,
    CompanyExtractModelOutput,
    FinancialStatementModelOutput,
    IntegerModelField,
    ModelClarificationStatement,
    ModelExtractionOutput,
    ModelField,
    NumberModelField,
    StringModelField,
    SupportingCorrespondenceModelOutput,
)
from app.ai.structured import invoke_structured
from app.domain import (
    ApplicationFormExtraction,
    ClarificationStatement,
    CompanyExtractExtraction,
    DocumentType,
    EvidenceReference,
    ExtractedField,
    ExtractionResult,
    FinancialStatementExtraction,
    ProcessedDocument,
    SupportingCorrespondenceExtraction,
    UnknownDocumentExtraction,
)

_EXPECTED_FIELDS: dict[DocumentType, tuple[str, ...]] = {
    DocumentType.APPLICATION_FORM: (
        "company_name",
        "registration_number",
        "annual_revenue_eur",
        "employee_count",
    ),
    DocumentType.COMPANY_EXTRACT: (
        "company_name",
        "registration_number",
        "incorporation_date",
    ),
    DocumentType.FINANCIAL_STATEMENT: (
        "company_name",
        "annual_revenue_eur",
        "reporting_year",
        "employee_count",
    ),
    DocumentType.SUPPORTING_CORRESPONDENCE: ("company_name",),
    DocumentType.UNKNOWN: (),
}

_MODEL_RESULT_SCHEMAS: dict[DocumentType, type[ModelExtractionOutput]] = {
    DocumentType.APPLICATION_FORM: ApplicationFormModelOutput,
    DocumentType.COMPANY_EXTRACT: CompanyExtractModelOutput,
    DocumentType.FINANCIAL_STATEMENT: FinancialStatementModelOutput,
    DocumentType.SUPPORTING_CORRESPONDENCE: SupportingCorrespondenceModelOutput,
}

_DOMAIN_RESULT_SCHEMAS: dict[DocumentType, type[ExtractionResult]] = {
    DocumentType.APPLICATION_FORM: ApplicationFormExtraction,
    DocumentType.COMPANY_EXTRACT: CompanyExtractExtraction,
    DocumentType.FINANCIAL_STATEMENT: FinancialStatementExtraction,
    DocumentType.SUPPORTING_CORRESPONDENCE: SupportingCorrespondenceExtraction,
}

_MODEL_FIELD_SCHEMAS: dict[str, type[ModelField]] = {
    "company_name": StringModelField,
    "registration_number": StringModelField,
    "annual_revenue_eur": NumberModelField,
    "employee_count": IntegerModelField,
    "incorporation_date": StringModelField,
    "reporting_year": IntegerModelField,
}


class LLMFieldExtractor:
    def __init__(self, model: BaseChatModel) -> None:
        self._model = model

    async def extract(
        self, document: ProcessedDocument, document_type: DocumentType
    ) -> ExtractionResult:
        if document_type == DocumentType.UNKNOWN:
            return UnknownDocumentExtraction(
                document_id=document.document_id,
                document_type=document_type,
                fields=[],
            )

        page_text = "\n\n".join(
            f"<page number=\"{page.page_number}\">\n{page.text}\n</page>"
            for page in document.pages
        )
        messages = [
            SystemMessage(content=EXTRACTION_SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"document_id: {document.document_id}\n"
                    f"document_type: {document_type.value}\n"
                    f"Required field specification:\n{FIELD_SPECIFICATIONS[document_type]}\n"
                    f"Document pages:\n{page_text}"
                )
            ),
        ]

        model_result = await invoke_structured(
            self._model,
            _MODEL_RESULT_SCHEMAS[document_type],
            messages,
            capability=f"field extraction for {document_type.value}",
            validator=lambda result: self._validate_model_result(
                result, document, document_type
            ),
        )
        return self._to_domain_result(model_result, document, document_type)

    def _validate_model_result(
        self,
        result: ModelExtractionOutput,
        document: ProcessedDocument,
        document_type: DocumentType,
    ) -> ModelExtractionOutput:
        # Field and statement identifiers are workflow-owned identities, not
        # evidence extracted from the document. Models occasionally reproduce
        # the requested suffix with a stale or shortened document ID even when
        # every extracted value and citation is valid. Canonicalize these
        # mechanical IDs in code, while continuing to validate all model-owned
        # values, confidences, and provenance strictly.
        result = result.model_copy(deep=True)
        for field_name in _EXPECTED_FIELDS[document_type]:
            if getattr(result, field_name) is None:
                setattr(
                    result,
                    field_name,
                    _MODEL_FIELD_SCHEMAS[field_name](
                        field_id=f"{document.document_id}:{field_name}",
                        value=None,
                        confidence=1.0,
                        evidence=[],
                    ),
                )
        pages = {page.page_number: page.text for page in document.pages}
        for field_name, field in self._model_fields(result, document_type):
            field.field_id = f"{document.document_id}:{field_name}"
            self._validate_value(field_name, field.value)
            if field.value is not None and not field.evidence:
                raise ValueError(f"non-null field {field_name} has no evidence")
            if field.value is None and field.evidence:
                raise ValueError(f"null field {field_name} must not contain evidence")
            for evidence in field.evidence:
                page_text = pages.get(evidence.page_number)
                if page_text is None:
                    raise ValueError("evidence references a page outside the document")
                evidence.source_text = self._exact_source_text(
                    evidence.source_text,
                    page_text,
                    value=field.value,
                )
        statements = self._model_statements(result)
        for statement_index, statement in enumerate(statements):
            statement.statement_id = (
                f"{document.document_id}:clarification:{statement_index}"
            )
            if statement.text is None:
                if statement.value is not None:
                    statement.text = str(statement.value)
                elif statement.evidence:
                    statement.text = statement.evidence[0].source_text
                else:  # The schema requires evidence; retain an explicit guard.
                    raise ValueError("clarification statement has no usable text")
            for evidence in statement.evidence:
                page_text = pages.get(evidence.page_number)
                if page_text is None:
                    raise ValueError("clarification evidence references an unknown page")
                evidence.source_text = self._exact_source_text(
                    evidence.source_text,
                    page_text,
                    value=statement.value if statement.value is not None else statement.text,
                )
        return result

    @staticmethod
    def _exact_source_text(
        source_text: str,
        page_text: str,
        *,
        value: object,
    ) -> str:
        """Return an exact page line for a uniquely supported model citation.

        PDF extraction can change whitespace and models occasionally paraphrase
        the surrounding label. We never retain that paraphrase. A correction is
        allowed only when the model's value occurs on exactly one non-empty line
        of the cited page; otherwise provenance validation still fails.
        """

        if source_text in page_text:
            return source_text
        lines = [line.strip() for line in page_text.splitlines() if line.strip()]
        normalized_source = " ".join(source_text.split()).casefold()
        whitespace_matches = [
            line
            for line in lines
            if " ".join(line.split()).casefold() == normalized_source
        ]
        if len(whitespace_matches) == 1:
            return whitespace_matches[0]

        value_matches = [
            line for line in lines if LLMFieldExtractor._line_contains_value(line, value)
        ]
        if len(value_matches) == 1:
            return value_matches[0]
        raise ValueError("evidence source_text is not an exact, uniquely supported page line")

    @staticmethod
    def _line_contains_value(line: str, value: object) -> bool:
        if value is None or isinstance(value, bool):
            return False
        if isinstance(value, (int, float)):
            compact_line = re.sub(r"[\s,_]", "", line)
            candidates = {str(value)}
            if isinstance(value, float):
                candidates.add(format(value, "g"))
            return any(
                re.search(rf"(?<!\d){re.escape(candidate)}(?!\d)", compact_line)
                is not None
                for candidate in candidates
            )
        normalized_value = " ".join(str(value).split()).casefold()
        normalized_line = " ".join(line.split()).casefold()
        return bool(normalized_value) and normalized_value in normalized_line

    def _to_domain_result(
        self,
        result: ModelExtractionOutput,
        document: ProcessedDocument,
        document_type: DocumentType,
    ) -> ExtractionResult:
        fields = [
            ExtractedField(
                field_id=field.field_id,
                document_id=document.document_id,
                field_name=field_name,
                value=field.value,
                confidence=field.confidence,
                evidence=[
                    EvidenceReference(
                        document_id=document.document_id,
                        page_number=item.page_number,
                        source_text=item.source_text,
                    )
                    for item in field.evidence
                ],
            )
            for field_name, field in self._model_fields(result, document_type)
        ]
        statements = [
            self._to_domain_statement(item, document.document_id)
            for item in self._model_statements(result)
        ]
        return _DOMAIN_RESULT_SCHEMAS[document_type](
            document_id=document.document_id,
            document_type=document_type,
            fields=fields,
            clarification_statements=statements,
        )

    @staticmethod
    def _model_fields(
        result: ModelExtractionOutput, document_type: DocumentType
    ) -> list[tuple[str, ModelField]]:
        expected = _EXPECTED_FIELDS[document_type]
        fields: list[tuple[str, ModelField]] = []
        for field_name in expected:
            field = getattr(result, field_name)
            if not isinstance(field, ModelField):
                raise ValueError(f"field {field_name} was not canonicalized")
            fields.append((field_name, field))
        return fields

    @staticmethod
    def _model_statements(
        result: ModelExtractionOutput,
    ) -> list[ModelClarificationStatement]:
        if isinstance(result, SupportingCorrespondenceModelOutput):
            return result.clarification_statements
        return []

    @staticmethod
    def _to_domain_statement(
        statement: ModelClarificationStatement, document_id: str
    ) -> ClarificationStatement:
        assert statement.text is not None
        return ClarificationStatement(
            statement_id=statement.statement_id,
            topic=statement.topic,
            text=statement.text,
            value=statement.value,
            confidence=statement.confidence,
            evidence=[
                EvidenceReference(
                    document_id=document_id,
                    page_number=item.page_number,
                    source_text=item.source_text,
                )
                for item in statement.evidence
            ],
        )

    @staticmethod
    def _validate_value(field_name: str, value: object) -> None:
        if value is None:
            return
        if field_name in {"company_name", "registration_number"}:
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
        elif field_name == "annual_revenue_eur":
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
                raise ValueError("annual_revenue_eur must be a non-negative number")
        elif field_name == "employee_count":
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError("employee_count must be a non-negative integer")
        elif field_name == "reporting_year":
            if isinstance(value, bool) or not isinstance(value, int) or not 1900 <= value <= 2200:
                raise ValueError("reporting_year must be a four-digit integer")
        elif field_name == "incorporation_date":
            if not isinstance(value, str):
                raise ValueError("incorporation_date must be an ISO date string")
            try:
                date.fromisoformat(value)
            except ValueError as exc:
                raise ValueError("incorporation_date must be a valid ISO date") from exc
