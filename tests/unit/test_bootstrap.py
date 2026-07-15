from __future__ import annotations

from typing import Any

import pytest

import app.bootstrap as bootstrap


@pytest.mark.asyncio
async def test_ollama_probe_requires_configured_digest(monkeypatch: Any) -> None:
    async def inventory(_base_url: str) -> dict[str, str]:
        return {"gemma4:12b-mlx": "correct", "embeddinggemma": "embed"}

    monkeypatch.setattr(bootstrap, "ollama_inventory", inventory)

    assert await bootstrap.probe_ollama(
        "http://ollama.test",
        {"gemma4:12b-mlx", "embeddinggemma"},
        {"gemma4:12b-mlx": "correct", "embeddinggemma": "embed"},
    )
    assert not await bootstrap.probe_ollama(
        "http://ollama.test",
        {"gemma4:12b-mlx", "embeddinggemma"},
        {"gemma4:12b-mlx": "stale", "embeddinggemma": "embed"},
    )
