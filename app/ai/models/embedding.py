"""Convert a LangChain embedding integration into the EvidenceFlow port."""

from __future__ import annotations

import asyncio
import math

from langchain_core.embeddings import Embeddings

from app.errors import ModelUnavailableError


class LangChainEmbeddingProvider:
    """Validated embedding provider with no LangChain types at its public edge."""

    def __init__(
        self,
        embeddings: Embeddings,
        *,
        provider: str,
        model: str,
        dimensions: int,
        model_digest: str | None = None,
    ) -> None:
        if dimensions < 1:
            raise ValueError("dimensions must be positive")
        self._embeddings = embeddings
        self.provider = provider
        self.model = model
        self.dimensions = dimensions
        self.model_digest = model_digest

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            vectors = await self._embeddings.aembed_documents(texts)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise ModelUnavailableError(
                f"Embedding model '{self.model}' is unavailable.",
                details={"provider": self.provider, "model": self.model},
            ) from exc
        return self._validate(vectors, expected_count=len(texts))

    async def embed_query(self, text: str) -> list[float]:
        try:
            vector = await self._embeddings.aembed_query(text)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise ModelUnavailableError(
                f"Embedding model '{self.model}' is unavailable.",
                details={"provider": self.provider, "model": self.model},
            ) from exc
        return self._validate([vector], expected_count=1)[0]

    def _validate(
        self, vectors: list[list[float]], *, expected_count: int
    ) -> list[list[float]]:
        if len(vectors) != expected_count:
            raise ModelUnavailableError(
                "Embedding provider returned an unexpected vector count.",
                details={"expected": expected_count, "actual": len(vectors)},
            )
        validated: list[list[float]] = []
        for vector in vectors:
            if len(vector) != self.dimensions or not all(math.isfinite(value) for value in vector):
                raise ModelUnavailableError(
                    "Embedding provider returned an incompatible vector.",
                    details={"expected_dimensions": self.dimensions, "actual": len(vector)},
                )
            validated.append([float(value) for value in vector])
        return validated
