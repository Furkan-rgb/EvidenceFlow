from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import fitz

from app.evaluation.generator import generate_evaluation_data
from app.evaluation.scenarios import get_scenarios


def _tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def test_scenario_catalogue_has_exact_required_distribution() -> None:
    scenarios = get_scenarios()
    categories = Counter(scenario.category for scenario in scenarios)

    assert len(scenarios) == 20
    assert categories == {
        "complete_consistent": 4,
        "missing_required_document": 3,
        "registration_number_conflict": 3,
        "revenue_conflict": 3,
        "company_name_formatting_variant": 2,
        "low_confidence_extraction": 1,
        "low_confidence_classification": 1,
        "unknown_document": 1,
        "incomplete_financial_document": 1,
        "human_correction": 1,
    }
    assert all(3 <= len(scenario.documents) <= 5 for scenario in scenarios)


def test_ground_truth_is_derived_from_rendered_scenario() -> None:
    scenario = get_scenarios()[0]
    ground_truth = scenario.as_ground_truth()

    assert ground_truth["bundle_id"] == "bundle_001"
    assert ground_truth["documents"][0]["file_name"] == "application_form.pdf"
    assert (
        ground_truth["documents"][0]["expected_fields"]["registration_number"]["value"]
        == "12345678"
    )
    assert ground_truth["expected_review_routing"] == {"required": False, "reasons": []}

    incomplete = get_scenarios()[18].as_ground_truth()
    financial = next(
        document
        for document in incomplete["documents"]
        if document["document_type"] == "financial_statement"
    )
    assert financial["expected_fields"]["annual_revenue_eur"]["value"] is None


def test_generator_creates_readable_deterministic_pdf_bundles(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    generated = generate_evaluation_data(first)
    generate_evaluation_data(second)

    assert len(generated) == 20
    assert _tree_hash(first) == _tree_hash(second)
    ground_truth = json.loads((first / "bundle_016" / "ground_truth.json").read_text())
    assert ground_truth["category"] == "low_confidence_extraction"
    assert ground_truth["expected_review_routing"]["required"] is True

    pdf_path = first / "bundle_001" / "documents" / "application_form.pdf"
    with fitz.open(pdf_path) as document:
        assert document.page_count == 1
        text = document[0].get_text()
    assert "Northwind Logistics B.V." in text
    assert "Registration number: 12345678" in text
    assert "SYNTHETIC DATA" in text


def test_generator_refuses_to_replace_nonempty_directory_by_default(tmp_path: Path) -> None:
    destination = tmp_path / "bundles"
    destination.mkdir()
    (destination / "keep.txt").write_text("owned by user")

    try:
        generate_evaluation_data(destination)
    except FileExistsError:
        pass
    else:
        raise AssertionError("expected a FileExistsError")

    assert (destination / "keep.txt").read_text() == "owned by user"
