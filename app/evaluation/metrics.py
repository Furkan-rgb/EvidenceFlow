"""Pure, model-independent evaluation metrics for EvidenceFlow."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
from typing import Any


@dataclass(frozen=True, slots=True)
class AccuracyScore:
    correct: int
    total: int

    @property
    def value(self) -> float:
        return self.correct / self.total if self.total else 1.0

    def as_dict(self) -> dict[str, int | float]:
        return {"value": self.value, **asdict(self)}


@dataclass(frozen=True, slots=True)
class PrecisionRecallScore:
    true_positive: int
    false_positive: int
    false_negative: int

    @property
    def precision(self) -> float:
        denominator = self.true_positive + self.false_positive
        return self.true_positive / denominator if denominator else 1.0

    @property
    def recall(self) -> float:
        denominator = self.true_positive + self.false_negative
        return self.true_positive / denominator if denominator else 1.0

    @property
    def f1(self) -> float:
        denominator = self.precision + self.recall
        return 2 * self.precision * self.recall / denominator if denominator else 0.0

    def as_dict(self) -> dict[str, int | float]:
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            **asdict(self),
        }


@dataclass(frozen=True, slots=True)
class RetrievalScore:
    query_count: int
    hit_count: int
    recall_sum: float
    reciprocal_rank_sum: float
    ndcg_sum: float
    k: int

    @property
    def hit_rate(self) -> float:
        return self.hit_count / self.query_count if self.query_count else 1.0

    @property
    def recall(self) -> float:
        return self.recall_sum / self.query_count if self.query_count else 1.0

    @property
    def mrr(self) -> float:
        return self.reciprocal_rank_sum / self.query_count if self.query_count else 1.0

    @property
    def ndcg(self) -> float:
        return self.ndcg_sum / self.query_count if self.query_count else 1.0

    def as_dict(self) -> dict[str, int | float]:
        return {
            "hit_rate_at_k": self.hit_rate,
            "recall_at_k": self.recall,
            "mrr_at_k": self.mrr,
            "ndcg_at_k": self.ndcg,
            **asdict(self),
        }


@dataclass(frozen=True, slots=True)
class CitationValidityScore:
    valid: int
    unknown: int

    @property
    def total(self) -> int:
        return self.valid + self.unknown

    @property
    def validity_rate(self) -> float:
        return self.valid / self.total if self.total else 1.0

    @property
    def unknown_id_rate(self) -> float:
        return self.unknown / self.total if self.total else 0.0

    def as_dict(self) -> dict[str, int | float]:
        return {
            "validity_rate": self.validity_rate,
            "unknown_id_rate": self.unknown_id_rate,
            "valid": self.valid,
            "unknown": self.unknown,
            "total": self.total,
        }


def _as_document_index(
    documents: Mapping[str, Any] | Sequence[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    if isinstance(documents, Mapping):
        return {
            str(file_name): value
            for file_name, value in documents.items()
            if isinstance(value, Mapping)
        }
    result: dict[str, Mapping[str, Any]] = {}
    for document in documents:
        file_name = document.get("file_name") or document.get("filename")
        if file_name is not None:
            result[str(file_name)] = document
    return result


def classification_accuracy(
    expected_documents: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    actual_documents: Mapping[str, Any] | Sequence[Mapping[str, Any]],
) -> AccuracyScore:
    """Score expected document types by stable source filename."""

    expected = _as_document_index(expected_documents)
    actual = _as_document_index(actual_documents)
    correct = sum(
        1
        for file_name, document in expected.items()
        if file_name in actual
        and actual[file_name].get("document_type") == document.get("document_type")
    )
    return AccuracyScore(correct=correct, total=len(expected))


def _field_mapping(document: Mapping[str, Any]) -> Mapping[str, Any]:
    for key in ("expected_fields", "fields", "extracted_fields"):
        fields = document.get(key)
        if isinstance(fields, Mapping):
            return fields
        if isinstance(fields, Sequence) and not isinstance(fields, (str, bytes)):
            mapped: dict[str, Any] = {}
            for field in fields:
                if isinstance(field, Mapping):
                    name = field.get("name") or field.get("field_name")
                    if name is not None:
                        mapped[str(name)] = field
            return mapped
    return {}


def _field_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        if "effective_value" in value:
            return value["effective_value"]
        if "value" in value:
            return value["value"]
        if "extracted_value" in value:
            return value["extracted_value"]
    return value


def values_equal(expected: Any, actual: Any) -> bool:
    """Compare typed extraction values without accepting semantic drift.

    Numeric string formatting is ignored (``2350000`` equals ``2350000.00``),
    while arbitrary strings remain case-sensitive after surrounding whitespace is
    removed. Booleans are deliberately not treated as integers.
    """

    expected = _field_value(expected)
    actual = _field_value(actual)
    if expected is None or actual is None:
        return expected is actual
    if isinstance(expected, bool) or isinstance(actual, bool):
        return expected is actual
    if isinstance(expected, (int, float, Decimal)) or isinstance(actual, (int, float, Decimal)):
        try:
            return Decimal(str(expected)) == Decimal(str(actual))
        except InvalidOperation:
            return False
    return str(expected).strip() == str(actual).strip()


def field_extraction_accuracy(
    expected_documents: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    actual_documents: Mapping[str, Any] | Sequence[Mapping[str, Any]],
) -> AccuracyScore:
    """Score every labelled field by filename and canonical field name."""

    expected = _as_document_index(expected_documents)
    actual = _as_document_index(actual_documents)
    correct = 0
    total = 0
    for file_name, expected_document in expected.items():
        expected_fields = _field_mapping(expected_document)
        actual_fields = _field_mapping(actual.get(file_name, {}))
        for field_name, expected_value in expected_fields.items():
            total += 1
            if field_name in actual_fields and values_equal(
                expected_value, actual_fields[field_name]
            ):
                correct += 1
    return AccuracyScore(correct=correct, total=total)


def finding_key(finding: Mapping[str, Any]) -> tuple[str, str, str]:
    """Return a stable semantic key independent of generated finding IDs."""

    finding_type = str(finding.get("type") or finding.get("finding_type") or "")
    return (
        finding_type,
        str(finding.get("document_type") or ""),
        str(finding.get("field") or finding.get("field_name") or ""),
    )


def missing_document_detection_accuracy(
    expected_findings: Sequence[Mapping[str, Any]],
    actual_findings: Sequence[Mapping[str, Any]],
) -> AccuracyScore:
    """Score exact missing-document detection for one or more bundles."""

    expected = {
        finding_key(finding)
        for finding in expected_findings
        if finding_key(finding)[0] == "missing_document"
    }
    actual = {
        finding_key(finding)
        for finding in actual_findings
        if finding_key(finding)[0] == "missing_document"
    }
    return AccuracyScore(correct=int(expected == actual), total=1)


def conflict_precision_recall(
    expected_conflicts: Iterable[str], actual_conflicts: Iterable[str]
) -> PrecisionRecallScore:
    expected = {str(value) for value in expected_conflicts}
    actual = {str(value) for value in actual_conflicts}
    return PrecisionRecallScore(
        true_positive=len(expected & actual),
        false_positive=len(actual - expected),
        false_negative=len(expected - actual),
    )


def conflicts_from_findings(findings: Sequence[Mapping[str, Any]]) -> set[str]:
    return {
        str(finding.get("field") or finding.get("field_name"))
        for finding in findings
        if finding_key(finding)[0] == "field_conflict"
        and (finding.get("field") or finding.get("field_name")) is not None
    }


def review_required_accuracy(expected_required: bool, actual_required: bool) -> AccuracyScore:
    """Score the binary decision to involve a reviewer."""

    return AccuracyScore(correct=int(expected_required == actual_required), total=1)


def review_routing_accuracy(
    expected_reasons: Iterable[str], actual_reasons: Iterable[str]
) -> AccuracyScore:
    """Score the exact reason/type set used to route a review.

    A generic review-required flag is insufficient: routing a low-confidence
    field because of an unrelated conflict must not count as a correct route.
    """

    expected = {str(reason) for reason in expected_reasons}
    actual = {str(reason) for reason in actual_reasons}
    return AccuracyScore(correct=int(expected == actual), total=1)


def _dcg(relevances: Sequence[int]) -> float:
    return sum(relevance / math.log2(rank + 2) for rank, relevance in enumerate(relevances))


def policy_retrieval_metrics(
    labelled_queries: Sequence[Mapping[str, Any]],
    ranked_results: Mapping[str, Sequence[str]],
    *,
    k: int = 5,
) -> RetrievalScore:
    """Compute macro HitRate, Recall, MRR and nDCG for labelled policy queries."""

    if k < 1:
        raise ValueError("k must be at least one")
    hit_count = 0
    recall_sum = 0.0
    reciprocal_rank_sum = 0.0
    ndcg_sum = 0.0
    for query in labelled_queries:
        query_id = str(query["query_id"])
        relevant = {str(item) for item in query.get("relevant_evidence_ids", ())}
        ranked = [str(item) for item in ranked_results.get(query_id, ())[:k]]
        relevance = [int(item in relevant) for item in ranked]
        relevant_retrieved = sum(relevance)
        if relevant_retrieved:
            hit_count += 1
            first_rank = relevance.index(1) + 1
            reciprocal_rank_sum += 1 / first_rank
        recall_sum += relevant_retrieved / len(relevant) if relevant else 1.0
        ideal = [1] * min(len(relevant), k)
        ideal_dcg = _dcg(ideal)
        ndcg_sum += _dcg(relevance) / ideal_dcg if ideal_dcg else 1.0
    return RetrievalScore(
        query_count=len(labelled_queries),
        hit_count=hit_count,
        recall_sum=recall_sum,
        reciprocal_rank_sum=reciprocal_rank_sum,
        ndcg_sum=ndcg_sum,
        k=k,
    )


def citation_validity(
    cited_ids: Iterable[str], supported_ids: Iterable[str]
) -> CitationValidityScore:
    supported = {str(item) for item in supported_ids}
    cited = [str(item) for item in cited_ids]
    valid = sum(item in supported for item in cited)
    return CitationValidityScore(valid=valid, unknown=len(cited) - valid)
