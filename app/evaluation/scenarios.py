"""Single source of truth for EvidenceFlow's 20 synthetic evaluation bundles.

The same scenario objects drive PDF contents and ``ground_truth.json``. Keeping
those two artifacts coupled prevents a common evaluation failure mode where a
fixture is updated without updating its labels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Literal

DocumentType = Literal[
    "application_form",
    "company_extract",
    "financial_statement",
    "supporting_correspondence",
    "unknown",
]


@dataclass(frozen=True, slots=True)
class ExpectedField:
    """A field rendered in a document and its extraction label."""

    name: str
    value: str | int | Decimal | None
    page_number: int = 1
    source_text: str | None = None
    ambiguous: bool = False

    def as_ground_truth(self) -> dict[str, Any]:
        value: str | int | None = (
            format(self.value, "f") if isinstance(self.value, Decimal) else self.value
        )
        return {
            "value": value,
            "page_number": self.page_number,
            "source_text": self.source_text,
            "ambiguous": self.ambiguous,
        }


@dataclass(frozen=True, slots=True)
class SyntheticDocument:
    """A deterministic text PDF specification."""

    file_name: str
    document_type: DocumentType
    title: str
    lines: tuple[str, ...]
    expected_fields: tuple[ExpectedField, ...] = ()
    classification_review_required: bool = False
    expected_clarification_statements: tuple[dict[str, Any], ...] = ()

    def as_ground_truth(self) -> dict[str, Any]:
        return {
            "file_name": self.file_name,
            "document_type": self.document_type,
            "classification_review_required": self.classification_review_required,
            "expected_fields": {item.name: item.as_ground_truth() for item in self.expected_fields},
            "expected_clarification_statements": list(self.expected_clarification_statements),
        }


@dataclass(frozen=True, slots=True)
class Scenario:
    """A complete, immutable evaluation-bundle definition."""

    bundle_id: str
    category: str
    description: str
    documents: tuple[SyntheticDocument, ...]
    expected_findings: tuple[dict[str, Any], ...] = ()
    expected_conflicts: tuple[str, ...] = ()
    expected_review_reasons: tuple[str, ...] = ()
    expected_status_before_review: Literal["complete", "incomplete", "needs_follow_up"] = "complete"
    reviewer_decisions: tuple[dict[str, Any], ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_ground_truth(self) -> dict[str, Any]:
        return {
            "schema_version": "1.0",
            "bundle_id": self.bundle_id,
            "category": self.category,
            "description": self.description,
            "documents": [document.as_ground_truth() for document in self.documents],
            "expected_findings": list(self.expected_findings),
            "expected_conflicts": list(self.expected_conflicts),
            "expected_review_routing": {
                "required": bool(self.expected_review_reasons),
                "reasons": list(self.expected_review_reasons),
            },
            "expected_status_before_review": self.expected_status_before_review,
            "reviewer_decisions": list(self.reviewer_decisions),
            "metadata": self.metadata,
        }


def _application(
    company: str,
    registration: str,
    revenue: str | int | Decimal,
    employees: int | None,
    *,
    employee_line: str | None = None,
    employee_expected: int | None = None,
    employee_ambiguous: bool = False,
) -> SyntheticDocument:
    lines = [
        "BUSINESS ONBOARDING APPLICATION",
        f"Legal company name: {company}",
        f"Registration number: {registration}",
        f"Estimated annual revenue: EUR {revenue}",
    ]
    fields: list[ExpectedField] = [
        ExpectedField("company_name", company, source_text=lines[1]),
        ExpectedField("registration_number", registration, source_text=lines[2]),
        ExpectedField("annual_revenue_eur", Decimal(str(revenue)), source_text=lines[3]),
    ]
    if employee_line is not None:
        lines.append(employee_line)
        fields.append(
            ExpectedField(
                "employee_count",
                employee_expected,
                source_text=employee_line,
                ambiguous=employee_ambiguous,
            )
        )
    elif employees is not None:
        line = f"Number of employees: {employees}"
        lines.append(line)
        fields.append(ExpectedField("employee_count", employees, source_text=line))
    lines.extend(("Applicant declaration: The submitted information is complete.", "END"))
    return SyntheticDocument(
        "application_form.pdf",
        "application_form",
        "Business Onboarding Application",
        tuple(lines),
        tuple(fields),
    )


def _extract(
    company: str,
    registration: str,
    incorporation_date: str = "2017-04-12",
) -> SyntheticDocument:
    lines = (
        "OFFICIAL COMPANY REGISTER EXTRACT",
        f"Legal company name: {company}",
        f"Registration number: {registration}",
        f"Date of incorporation: {incorporation_date}",
        "Register status: Active",
        "END OF EXTRACT",
    )
    return SyntheticDocument(
        "company_extract.pdf",
        "company_extract",
        "Company Register Extract",
        lines,
        (
            ExpectedField("company_name", company, source_text=lines[1]),
            ExpectedField("registration_number", registration, source_text=lines[2]),
            ExpectedField("incorporation_date", incorporation_date, source_text=lines[3]),
        ),
    )


def _financial(
    company: str,
    revenue: str | int | Decimal | None,
    employees: int | None,
    reporting_year: int | None = 2025,
) -> SyntheticDocument:
    lines = ["ANNUAL FINANCIAL STATEMENT", f"Reporting entity: {company}"]
    fields: list[ExpectedField] = [ExpectedField("company_name", company, source_text=lines[1])]
    if reporting_year is not None:
        line = f"Reporting year: {reporting_year}"
        lines.append(line)
        fields.append(ExpectedField("reporting_year", reporting_year, source_text=line))
    else:
        fields.append(ExpectedField("reporting_year", None))
    if revenue is not None:
        line = f"Revenue for the reporting year: EUR {revenue}"
        lines.append(line)
        fields.append(ExpectedField("annual_revenue_eur", Decimal(str(revenue)), source_text=line))
    else:
        fields.append(ExpectedField("annual_revenue_eur", None))
    if employees is not None:
        line = f"Average employee count at year end: {employees}"
        lines.append(line)
        fields.append(ExpectedField("employee_count", employees, source_text=line))
    lines.extend(("Prepared for synthetic evaluation purposes.", "END OF STATEMENT"))
    return SyntheticDocument(
        "financial_statement.pdf",
        "financial_statement",
        "Annual Financial Statement",
        tuple(lines),
        tuple(fields),
    )


def _correspondence(
    company: str,
    *,
    file_name: str = "supporting_correspondence.pdf",
    ambiguous_classification: bool = False,
) -> SyntheticDocument:
    lines = (
        "SUPPORTING CORRESPONDENCE",
        f"Re: onboarding clarification for {company}",
        "Topic: operating address",
        "Clarification: The registered and operating addresses are the same.",
        "This note does not replace any mandatory company document.",
    )
    return SyntheticDocument(
        file_name,
        "supporting_correspondence",
        "Supporting Correspondence",
        lines,
        (ExpectedField("company_name", company, source_text=lines[1]),),
        classification_review_required=ambiguous_classification,
        expected_clarification_statements=(
            {
                "topic": "operating_address",
                "text": "The registered and operating addresses are the same.",
                "value": None,
                "page_number": 1,
                "source_text": lines[3],
            },
        ),
    )


def _unknown() -> SyntheticDocument:
    lines = (
        "OFFICE CAFETERIA MENU",
        "Monday: vegetable soup and bread",
        "Tuesday: pasta salad",
        "This document contains no company onboarding evidence.",
    )
    return SyntheticDocument(
        "irrelevant_attachment.pdf",
        "unknown",
        "Irrelevant Attachment",
        lines,
    )


def _complete(
    number: int,
    company: str,
    registration: str,
    revenue: int,
    employees: int,
    *,
    include_correspondence: bool = False,
) -> Scenario:
    documents: tuple[SyntheticDocument, ...] = (
        _application(company, registration, revenue, employees),
        _extract(company, registration),
        _financial(company, revenue, employees),
    )
    if include_correspondence:
        documents += (_correspondence(company),)
    return Scenario(
        f"bundle_{number:03d}",
        "complete_consistent",
        "A complete package whose required values agree across documents.",
        documents,
    )


def _missing(
    number: int,
    company: str,
    registration: str,
    revenue: int,
    employees: int,
    missing: DocumentType,
) -> Scenario:
    all_documents = {
        "application_form": _application(company, registration, revenue, employees),
        "company_extract": _extract(company, registration),
        "financial_statement": _financial(company, revenue, employees),
    }
    documents = (
        *(document for kind, document in all_documents.items() if kind != missing),
        _correspondence(company),
    )
    return Scenario(
        f"bundle_{number:03d}",
        "missing_required_document",
        f"A package missing its required {missing}.",
        documents,
        expected_findings=(
            {
                "type": "missing_document",
                "severity": "high",
                "document_type": missing,
            },
        ),
        expected_status_before_review="incomplete",
    )


def _registration_conflict(
    number: int,
    company: str,
    application_registration: str,
    extract_registration: str,
) -> Scenario:
    return Scenario(
        f"bundle_{number:03d}",
        "registration_number_conflict",
        "The application and register extract contain different registration numbers.",
        (
            _application(company, application_registration, 2_350_000, 42),
            _extract(company, extract_registration),
            _financial(company, 2_350_000, 42),
        ),
        expected_findings=(
            {
                "type": "field_conflict",
                "severity": "high",
                "field": "registration_number",
            },
        ),
        expected_conflicts=("registration_number",),
        expected_review_reasons=("conflict:registration_number",),
        expected_status_before_review="needs_follow_up",
    )


def _revenue_conflict(
    number: int,
    company: str,
    registration: str,
    application_revenue: int,
    financial_revenue: int,
) -> Scenario:
    return Scenario(
        f"bundle_{number:03d}",
        "revenue_conflict",
        "Revenue values differ by more than the allowed two percent tolerance.",
        (
            _application(company, registration, application_revenue, 42),
            _extract(company, registration),
            _financial(company, financial_revenue, 42),
        ),
        expected_findings=(
            {
                "type": "field_conflict",
                "severity": "high",
                "field": "annual_revenue_eur",
            },
        ),
        expected_conflicts=("annual_revenue_eur",),
        expected_review_reasons=("conflict:annual_revenue_eur",),
        expected_status_before_review="needs_follow_up",
    )


SCENARIOS: tuple[Scenario, ...] = (
    _complete(1, "Northwind Logistics B.V.", "12345678", 2_350_000, 42),
    _complete(2, "Alder Marine N.V.", "20441876", 4_800_000, 67, include_correspondence=True),
    _complete(3, "Brightwave Consulting B.V.", "31800542", 920_000, 18),
    _complete(
        4,
        "Cedarworks Europe B.V.",
        "44712009",
        12_600_000,
        154,
        include_correspondence=True,
    ),
    _missing(5, "Dune Analytics B.V.", "50112784", 1_200_000, 23, "financial_statement"),
    _missing(6, "Elm City Trading B.V.", "61449022", 3_100_000, 38, "company_extract"),
    _missing(7, "Fjord Systems B.V.", "72990114", 7_450_000, 81, "application_form"),
    _registration_conflict(8, "Gable Foods B.V.", "12345678", "12345687"),
    _registration_conflict(9, "Harborline B.V.", "80441230", "80441203"),
    _registration_conflict(10, "Ion Works N.V.", "91882741", "91887241"),
    _revenue_conflict(11, "Juniper Freight B.V.", "11774420", 2_350_000, 2_900_000),
    _revenue_conflict(12, "Keystone Labs B.V.", "22559018", 8_100_000, 7_400_000),
    _revenue_conflict(13, "Lumen Retail B.V.", "33009127", 950_000, 1_100_000),
    Scenario(
        "bundle_014",
        "company_name_formatting_variant",
        "Case, punctuation, ampersand and legal-form presentation should normalise to a match.",
        (
            _application("Northstar & Partners B.V.", "44117730", 1_750_000, 29),
            _extract("NORTHSTAR AND PARTNERS BV", "44117730"),
            _financial("Northstar and Partners B.V.", 1_750_000, 29),
        ),
        metadata={"normalised_company_name": "northstar and partners bv"},
    ),
    Scenario(
        "bundle_015",
        "company_name_formatting_variant",
        "Hyphenation, spacing, case and legal-form punctuation should normalise to a match.",
        (
            _application("Delta-Tech Holdings N.V.", "55220981", 5_300_000, 51),
            _extract("DELTA TECH HOLDINGS NV", "55220981"),
            _financial("Delta Tech Holdings N.V.", 5_300_000, 51),
        ),
        metadata={"normalised_company_name": "delta tech holdings nv"},
    ),
    Scenario(
        "bundle_016",
        "low_confidence_extraction",
        "An employee count is expressed approximately in words and should be reviewed.",
        (
            _application(
                "Mosaic Services B.V.",
                "66400291",
                2_050_000,
                None,
                employee_line=(
                    "At year end, the organisation had an average workforce of "
                    "approximately forty-two FTE."
                ),
                employee_expected=42,
                employee_ambiguous=True,
            ),
            _extract("Mosaic Services B.V.", "66400291"),
            _financial("Mosaic Services B.V.", 2_050_000, 42),
        ),
        expected_review_reasons=("low_confidence:employee_count",),
    ),
    Scenario(
        "bundle_017",
        "low_confidence_classification",
        "A short financial clarification letter has intentionally ambiguous document cues.",
        (
            _application("Nimbus Health B.V.", "77081443", 6_200_000, 74),
            _extract("Nimbus Health B.V.", "77081443"),
            _financial("Nimbus Health B.V.", 6_200_000, 74),
            SyntheticDocument(
                "ambiguous_note.pdf",
                "supporting_correspondence",
                "Ambiguous Financial Note",
                (
                    "FINANCIAL INFORMATION NOTE",
                    "For Nimbus Health B.V.",
                    "The enclosed statement remains the authoritative financial record.",
                    "Please contact the applicant if clarification is required.",
                ),
                (
                    ExpectedField(
                        "company_name",
                        "Nimbus Health B.V.",
                        source_text="For Nimbus Health B.V.",
                    ),
                ),
                classification_review_required=True,
            ),
        ),
        expected_review_reasons=("low_confidence:classification",),
    ),
    Scenario(
        "bundle_018",
        "unknown_document",
        "A complete package includes one irrelevant attachment classified as unknown.",
        (
            _application("Orchard Mobility B.V.", "88114405", 3_750_000, 46),
            _extract("Orchard Mobility B.V.", "88114405"),
            _financial("Orchard Mobility B.V.", 3_750_000, 46),
            _unknown(),
        ),
    ),
    Scenario(
        "bundle_019",
        "incomplete_financial_document",
        "The financial statement is recognised but has no revenue value.",
        (
            _application("Pioneer Textiles B.V.", "99220461", 1_480_000, 33),
            _extract("Pioneer Textiles B.V.", "99220461"),
            _financial("Pioneer Textiles B.V.", None, 33),
        ),
        expected_findings=(
            {
                "type": "missing_required_field",
                "severity": "high",
                "document_type": "financial_statement",
                "field": "annual_revenue_eur",
            },
        ),
        expected_status_before_review="incomplete",
    ),
    Scenario(
        "bundle_020",
        "human_correction",
        "An unclear employee-count value must be corrected before validation can pass.",
        (
            _application(
                "Quarry Digital B.V.",
                "10993372",
                2_800_000,
                None,
                employee_line="Number of employees: 4? (final digit unclear)",
                employee_expected=42,
                employee_ambiguous=True,
            ),
            _extract("Quarry Digital B.V.", "10993372"),
            _financial("Quarry Digital B.V.", 2_800_000, 42),
        ),
        expected_review_reasons=("low_confidence:employee_count",),
        reviewer_decisions=({"field": "employee_count", "action": "correct", "value": 42},),
        metadata={"expected_status_after_review": "complete"},
    ),
)


def get_scenarios() -> tuple[Scenario, ...]:
    """Return the fixed V1 scenario catalogue after checking its invariants."""

    if len(SCENARIOS) != 20:
        raise AssertionError("The V1 evaluation corpus must contain exactly 20 bundles")
    ids = [scenario.bundle_id for scenario in SCENARIOS]
    if len(ids) != len(set(ids)):
        raise AssertionError("Evaluation bundle IDs must be unique")
    for scenario in SCENARIOS:
        if not 3 <= len(scenario.documents) <= 5:
            raise AssertionError(
                f"{scenario.bundle_id} must contain between three and five documents"
            )
        file_names = [document.file_name for document in scenario.documents]
        if len(file_names) != len(set(file_names)):
            raise AssertionError(f"{scenario.bundle_id} contains duplicate file names")
    return SCENARIOS
