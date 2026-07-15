"""Provider-selecting entry points for application bootstrap."""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel

from app.ai.config import ChatModelConfig, EmbeddingModelConfig
from app.ai.models.embedding import LangChainEmbeddingProvider
from app.ai.models.ollama import OllamaChatModelFactory, OllamaEmbeddingModelFactory
from app.errors import UnsupportedProviderError


def create_chat_model(config: ChatModelConfig) -> BaseChatModel:
    if config.provider == "ollama":
        return OllamaChatModelFactory().create(config)
    raise UnsupportedProviderError(
        f"Chat provider '{config.provider}' is not implemented in V1.",
        details={"provider": config.provider},
    )


def create_embedding_provider(config: EmbeddingModelConfig) -> LangChainEmbeddingProvider:
    if config.provider != "ollama":
        raise UnsupportedProviderError(
            f"Embedding provider '{config.provider}' is not implemented in V1.",
            details={"provider": config.provider},
        )
    embeddings = OllamaEmbeddingModelFactory().create(config)
    return LangChainEmbeddingProvider(
        embeddings,
        provider=config.provider,
        model=config.model,
        dimensions=config.dimensions,
        model_digest=config.model_digest,
    )
