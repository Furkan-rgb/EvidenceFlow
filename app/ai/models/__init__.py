"""LangChain model factories and the application embedding adapter."""

from app.ai.models.embedding import LangChainEmbeddingProvider
from app.ai.models.factory import create_chat_model, create_embedding_provider

__all__ = ["LangChainEmbeddingProvider", "create_chat_model", "create_embedding_provider"]
