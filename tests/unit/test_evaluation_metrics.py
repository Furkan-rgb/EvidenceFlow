from __future__ import annotations

from app.evaluation.metrics import (
    citation_validity,
    classification_accuracy,
    conflict_precision_recall,
    field_extraction_accuracy,
    missing_document_detection_accuracy,
    policy_retrieval_metrics,
    review_required_accuracy,
    review_routing_accuracy,
    values_equal,
)


def test_classification_and_field_accuracy_use_filename_and_canonical_field() -> None:
    expected = [
        {
            "file_name": "application.pdf",
            "document_type": "application_form",
            "expected_fields": {
                "registration_number": {"value": "00123456"},
                "annual_revenue_eur": {"value": "2350000.00"},
            },
        },
        {
            "file_name": "extract.pdf",
            "document_type": "company_extract",
            "expected_fields": {"registration_number": {"value": "00123456"}},
        },
    ]
    actual = [
        {
            "file_name": "application.pdf",
            "document_type": "application_form",
            "fields": {
                "registration_number": {"value": "00123456"},
                "annual_revenue_eur": {"value": 2_350_000},
            },
        },
        {
            "file_name": "extract.pdf",
            "document_type": "unknown",
            "fields": {"registration_number": {"value": "123456"}},
        },
    ]

    assert classification_accuracy(expected, actual).as_dict() == {
        "value": 0.5,
        "correct": 1,
        "total": 2,
    }
    assert field_extraction_accuracy(expected, actual).value == 2 / 3
    assert values_equal("00123456", "123456") is False
    assert values_equal("2350000.00", 2_350_000) is True


def test_missing_conflict_and_routing_metrics() -> None:
    expected_findings = [{"type": "missing_document", "document_type": "financial_statement"}]
    actual_findings = [{"finding_type": "missing_document", "document_type": "financial_statement"}]
    assert missing_document_detection_accuracy(expected_findings, actual_findings).value == 1
    assert missing_document_detection_accuracy(expected_findings, []).value == 0

    conflicts = conflict_precision_recall(
        {"registration_number", "annual_revenue_eur"},
        {"registration_number", "employee_count"},
    )
    assert conflicts.true_positive == 1
    assert conflicts.false_positive == 1
    assert conflicts.false_negative == 1
    assert conflicts.precision == 0.5
    assert conflicts.recall == 0.5
    assert review_required_accuracy(True, True).value == 1
    assert review_required_accuracy(True, False).value == 0
    assert review_routing_accuracy(
        {"low_confidence:employee_count"},
        {"low_confidence:employee_count"},
    ).value == 1
    assert review_routing_accuracy(
        {"low_confidence:employee_count"}, {"conflict:employee_count"}
    ).value == 0


def test_policy_metrics_compute_rank_sensitive_results() -> None:
    labelled = [
        {"query_id": "one", "relevant_evidence_ids": ["A", "B"]},
        {"query_id": "two", "relevant_evidence_ids": ["C"]},
    ]
    ranked = {"one": ["X", "A", "B"], "two": ["Z", "Y", "C"]}
    result = policy_retrieval_metrics(labelled, ranked, k=3)

    assert result.hit_rate == 1.0
    assert result.recall == 1.0
    assert result.mrr == (1 / 2 + 1 / 3) / 2
    assert 0 < result.ndcg < 1


def test_citation_validity_rejects_every_unknown_id() -> None:
    result = citation_validity(
        ["finding-known", "policy-known", "invented"],
        ["finding-known", "policy-known"],
    )

    assert result.valid == 2
    assert result.unknown == 1
    assert result.validity_rate == 2 / 3
    assert result.unknown_id_rate == 1 / 3
