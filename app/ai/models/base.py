"""Narrow factory protocols used only inside the AI infrastructure layer."""

from __future__ import annotations

from typing import Protocol

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel

from app.ai.config import ChatModelConfig, EmbeddingModelConfig
from app.ports import EmbeddingProvider

__all__ = ["ChatModelFactory", "EmbeddingModelFactory", "EmbeddingProvider"]


class ChatModelFactory(Protocol):
    def create(self, config: ChatModelConfig) -> BaseChatModel: ...


class EmbeddingModelFactory(Protocol):
    def create(self, config: EmbeddingModelConfig) -> Embeddings: ...
