"""Deterministic synthetic-data generation and evaluation helpers."""

from app.evaluation.generator import generate_evaluation_data
from app.evaluation.runner import EvaluationAdapter, EvaluationRunResult, run_evaluation
from app.evaluation.scenarios import SCENARIOS, Scenario, get_scenarios

__all__ = [
    "SCENARIOS",
    "EvaluationAdapter",
    "EvaluationRunResult",
    "Scenario",
    "generate_evaluation_data",
    "get_scenarios",
    "run_evaluation",
]
