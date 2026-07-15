from __future__ import annotations

from decimal import Decimal

import pytest

from app.review import (
    ValueNormalizationError,
    normalize_company_name,
    normalize_employee_count,
    normalize_registration_number,
    normalize_revenue,
    symmetric_percentage_difference,
    within_symmetric_percent_tolerance,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (" ACME, B.V. ", "acme bv"),
        ("Acme BV", "acme bv"),
        ("Research & Development Limited", "research and development ltd"),
        ("Müller GmbH", "müller gmbh"),
    ],
)
def test_company_name_normalization_preserves_legal_form(
    raw: str, expected: str
) -> None:
    assert normalize_company_name(raw) == expected


def test_registration_number_preserves_leading_zeroes() -> None:
    assert normalize_registration_number(" nl-00.123 456 ") == "NL00123456"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("EUR 1.234,50", Decimal("1234.5")),
        ("€1,234.50", Decimal("1234.5")),
        (1_000_000.25, Decimal("1000000.25")),
    ],
)
def test_revenue_normalization_uses_decimal(raw: str | float, expected: Decimal) -> None:
    assert normalize_revenue(raw) == expected


def test_revenue_uses_symmetric_inclusive_percentage_tolerance() -> None:
    assert symmetric_percentage_difference(Decimal(99), Decimal(101)) == Decimal(2)
    assert within_symmetric_percent_tolerance(Decimal(99), Decimal(101), 2)
    assert not within_symmetric_percent_tolerance(
        Decimal("98.99"), Decimal("101.01"), 2
    )


def test_employee_count_requires_non_negative_integral_value() -> None:
    assert normalize_employee_count("42") == 42
    with pytest.raises(ValueNormalizationError, match="integer"):
        normalize_employee_count("42.5")
    with pytest.raises(ValueNormalizationError, match="non-negative"):
        normalize_employee_count(-1)
