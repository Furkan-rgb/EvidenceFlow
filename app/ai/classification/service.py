"""LangChain-backed implementation of the document-classifier capability."""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from app.ai.classification.prompts import CLASSIFICATION_SYSTEM_PROMPT
from app.ai.classification.schemas import ModelDocumentClassification
from app.ai.structured import invoke_structured
from app.domain import DocumentClassification, ProcessedDocument


class LLMDocumentClassifier:
    def __init__(self, model: BaseChatModel) -> None:
        self._model = model

    async def classify(self, document: ProcessedDocument) -> DocumentClassification:
        page_text = "\n\n".join(
            f"<page number=\"{page.page_number}\">\n{page.text}\n</page>"
            for page in document.pages
        )
        messages = [
            SystemMessage(content=CLASSIFICATION_SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"document_id: {document.document_id}\n"
                    f"filename: {document.filename}\n\n{page_text}"
                )
            ),
        ]

        model_result = await invoke_structured(
            self._model,
            ModelDocumentClassification,
            messages,
            capability="document classification",
        )
        return DocumentClassification(
            document_id=document.document_id,
            document_type=model_result.document_type,
            confidence=model_result.confidence,
            reasoning_summary=model_result.reasoning_summary,
        )
