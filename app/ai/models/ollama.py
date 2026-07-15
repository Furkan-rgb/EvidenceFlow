"""Ollama-specific LangChain construction. No business code imports this module."""

from __future__ import annotations

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_ollama import ChatOllama, OllamaEmbeddings

from app.ai.config import ChatModelConfig, EmbeddingModelConfig
from app.errors import UnsupportedProviderError


class OllamaChatModelFactory:
    def create(self, config: ChatModelConfig) -> BaseChatModel:
        if config.provider != "ollama":
            raise UnsupportedProviderError(
                f"Chat provider '{config.provider}' is not implemented in V1.",
                details={"provider": config.provider},
            )
        return ChatOllama(
            model=config.model,
            temperature=config.temperature,
            base_url=config.base_url,
            reasoning=False,
            num_predict=config.max_output_tokens,
            client_kwargs={"timeout": config.timeout_seconds},
            async_client_kwargs={"timeout": config.timeout_seconds},
            validate_model_on_init=False,
        )


class OllamaEmbeddingModelFactory:
    def create(self, config: EmbeddingModelConfig) -> Embeddings:
        if config.provider != "ollama":
            raise UnsupportedProviderError(
                f"Embedding provider '{config.provider}' is not implemented in V1.",
                details={"provider": config.provider},
            )
        return OllamaEmbeddings(model=config.model, base_url=config.base_url)
