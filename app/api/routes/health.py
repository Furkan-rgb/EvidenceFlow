"""Liveness and safe dependency state."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends

from app.api.dependencies import get_container
from app.preparation import PreparationComponent

router = APIRouter(tags=["health"])


@router.get("/health")
async def health(container: Annotated[Any, Depends(get_container)]) -> dict[str, object]:
    database = await container.repository.health()
    policy_index = bool(getattr(container, "policy_index_healthy", False))
    model_runtime = bool(getattr(container, "model_runtime_healthy", False))
    preparation = getattr(container, "preparation_report", None)
    storage = _component_ready(preparation, PreparationComponent.STORAGE)
    tracing_enabled = bool(getattr(container.tracer, "enabled", True))
    tracing = bool(container.tracer.healthy)
    dependencies = {
        "database": "ok" if database else "unavailable",
        "storage": "ok" if storage else "unavailable",
        "policy_index": "ok" if policy_index else "unavailable",
        "model_providers": "ok" if model_runtime else "unavailable",
        "mlflow": "disabled" if not tracing_enabled else ("ok" if tracing else "degraded"),
    }
    critical = database and storage
    ready = critical and policy_index and model_runtime
    degraded = ready and tracing_enabled and not tracing
    return {
        "status": (
            "degraded"
            if degraded
            else ("ok" if ready else ("degraded" if critical else "unavailable"))
        ),
        "ready": ready,
        "dependencies": dependencies,
    }


def _component_ready(report: Any, component: PreparationComponent) -> bool:
    if report is None:
        return True
    results = [result for result in report.results if result.component is component]
    return bool(results) and not any(result.blocking for result in results)
