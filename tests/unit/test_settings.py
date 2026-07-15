from __future__ import annotations

import pytest

from app.config import Settings


def test_local_mlflow_default_avoids_the_macos_airplay_port(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    settings = Settings(_env_file=None)

    assert settings.mlflow_tracking_uri == "http://127.0.0.1:5001"
