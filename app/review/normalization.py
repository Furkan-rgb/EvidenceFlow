"""Deterministic field normalization and numeric comparison semantics."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from datetime import date, datetime
from decimal import Decimal, InvalidOperation

from app.domain.extractions import FieldValue, NormalizedValue


class ValueNormalizationError(ValueError):
    """Raised when a non-null field value cannot be normalized safely."""


_SPACE_RE = re.compile(r"\s+")
_LEGAL_SUFFIX_TOKEN_GROUPS: tuple[tuple[str, ...], ...] = (
    ("b", "v"),
    ("n", "v"),
    ("l", "l", "c"),
    ("l", "t", "d"),
    ("p", "l", "c"),
    ("g", "m", "b", "h"),
    ("i", "n", "c"),
)
_LEGAL_FORM_ALIASES = {
    "limited": "ltd",
    "incorporated": "inc",
    "corporation": "corp",
}


def _text_tokens(value: str) -> list[str]:
    value = unicodedata.normalize("NFKC", value).casefold().replace("&", " and ")
    characters = [
        character
        if character.isspace()
        or unicodedata.category(character).startswith(("L", "N"))
        else " "
        for character in value
    ]
    return _SPACE_RE.sub(" ", "".join(characters)).strip().split()


def normalize_company_name(value: str) -> str:
    """Canonicalize presentation while retaining the company's legal form."""

    if not isinstance(value, str):
        raise ValueNormalizationError("company_name must be a string")
    tokens = _text_tokens(value)
    if not tokens:
        raise ValueNormalizationError("company_name cannot be blank")

    # Punctuation in forms such as B.V. produces single-letter tokens. Join only
    # recognized trailing legal forms so ordinary word punctuation stays a space.
    for legal_form in sorted(_LEGAL_SUFFIX_TOKEN_GROUPS, key=len, reverse=True):
        if tuple(tokens[-len(legal_form) :]) == legal_form:
            tokens[-len(legal_form) :] = ["".join(legal_form)]
            break
    tokens = [_LEGAL_FORM_ALIASES.get(token, token) for token in tokens]
    return " ".join(tokens)


def normalise_company_name(value: str) -> str:
    """British-spelling alias used in architecture documentation."""

    return normalize_company_name(value)


def normalize_registration_number(value: str | int) -> str:
    """Remove presentation separators without converting away leading zeroes."""

    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueNormalizationError("registration_number must be text or an integer")
    raw = unicodedata.normalize("NFKC", str(value)).upper()
    normalized = "".join(character for character in raw if character.isalnum())
    if not normalized:
        raise ValueNormalizationError("registration_number cannot be blank")
    return normalized


def _decimal_from_text(value: str) -> Decimal:
    raw = unicodedata.normalize("NFKC", value).strip()
    negative_parentheses = raw.startswith("(") and raw.endswith(")")
    raw = raw.removeprefix("(").removesuffix(")")
    raw = re.sub(r"(?i)\bEUR\b", "", raw).replace("€", "")
    raw = raw.replace(" ", "").replace("'", "")
    if not raw:
        raise ValueNormalizationError("numeric value cannot be blank")

    comma = raw.rfind(",")
    dot = raw.rfind(".")
    if comma >= 0 and dot >= 0:
        decimal_separator = "," if comma > dot else "."
        thousands_separator = "." if decimal_separator == "," else ","
        raw = raw.replace(thousands_separator, "").replace(decimal_separator, ".")
    elif comma >= 0:
        fraction_digits = len(raw) - comma - 1
        raw = raw.replace(",", "." if 0 < fraction_digits <= 2 else "")
    elif dot >= 0:
        fraction_digits = len(raw) - dot - 1
        if fraction_digits == 3 and raw.count(".") >= 1:
            raw = raw.replace(".", "")

    if negative_parentheses:
        raw = f"-{raw}"
    try:
        return Decimal(raw)
    except InvalidOperation as exc:
        raise ValueNormalizationError(f"invalid numeric value: {value!r}") from exc


def normalize_revenue(value: str | int | float | Decimal) -> Decimal:
    if isinstance(value, bool):
        raise ValueNormalizationError("annual_revenue_eur must be numeric")
    if isinstance(value, Decimal):
        result = value
    elif isinstance(value, int):
        result = Decimal(value)
    elif isinstance(value, float):
        result = Decimal(str(value))
    elif isinstance(value, str):
        result = _decimal_from_text(value)
    else:
        raise ValueNormalizationError("annual_revenue_eur must be numeric")
    if not result.is_finite() or result < 0:
        raise ValueNormalizationError("annual_revenue_eur must be finite and non-negative")
    return result.normalize() if result else Decimal(0)


def normalize_employee_count(value: str | int | float | Decimal) -> int:
    if isinstance(value, bool):
        raise ValueNormalizationError("employee_count must be an integer")
    if isinstance(value, Decimal):
        number = value
    elif isinstance(value, int):
        number = Decimal(value)
    elif isinstance(value, float):
        number = Decimal(str(value))
    elif isinstance(value, str):
        number = _decimal_from_text(value)
    else:
        raise ValueNormalizationError("employee_count must be an integer")
    if not number.is_finite():
        raise ValueNormalizationError("employee_count must be finite")
    if number < 0:
        raise ValueNormalizationError("employee_count must be non-negative")
    if number != number.to_integral_value():
        raise ValueNormalizationError("employee_count must be an integer")
    return int(number)


def normalize_reporting_year(value: str | int) -> int:
    if isinstance(value, bool):
        raise ValueNormalizationError("reporting_year must be an integer year")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueNormalizationError("reporting_year must be an integer year") from exc
    if not 1000 <= result <= 9999 or str(value).strip() not in {
        str(result),
        f"{result}.0",
    }:
        raise ValueNormalizationError("reporting_year must be a four-digit year")
    return result


def normalize_date(value: str | date | datetime) -> str:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if not isinstance(value, str):
        raise ValueNormalizationError("date value must be text or a date")
    candidate = unicodedata.normalize("NFKC", value).strip()
    try:
        return date.fromisoformat(candidate).isoformat()
    except ValueError:
        pass
    for pattern in ("%d-%m-%Y", "%d/%m/%Y", "%Y/%m/%d", "%d %B %Y"):
        try:
            return datetime.strptime(candidate, pattern).date().isoformat()
        except ValueError:
            continue
    raise ValueNormalizationError(f"invalid date value: {value!r}")


def normalize_text(value: str) -> str:
    if not isinstance(value, str):
        raise ValueNormalizationError("text field must be a string")
    normalized = _SPACE_RE.sub(" ", unicodedata.normalize("NFKC", value)).strip()
    if not normalized:
        raise ValueNormalizationError("text field cannot be blank")
    return normalized


def normalize_field_value(field_name: str, value: FieldValue) -> NormalizedValue:
    """Normalize a known semantic field with deterministic, field-specific rules."""

    if field_name == "company_name":
        return normalize_company_name(value)  # type: ignore[arg-type]
    if field_name == "registration_number":
        return normalize_registration_number(value)  # type: ignore[arg-type]
    if field_name == "annual_revenue_eur":
        return normalize_revenue(value)
    if field_name == "employee_count":
        return normalize_employee_count(value)
    if field_name == "reporting_year":
        return normalize_reporting_year(value)  # type: ignore[arg-type]
    if field_name == "incorporation_date":
        return normalize_date(value)  # type: ignore[arg-type]
    if isinstance(value, str):
        return normalize_text(value)
    if isinstance(value, float):
        return Decimal(str(value))
    return value


def normalise_field_value(field_name: str, value: FieldValue) -> NormalizedValue:
    return normalize_field_value(field_name, value)


def symmetric_percentage_difference(left: Decimal, right: Decimal) -> Decimal:
    """Return symmetric percentage difference, avoiding source-order bias."""

    left = abs(left)
    right = abs(right)
    if left == right:
        return Decimal(0)
    mean = (left + right) / Decimal(2)
    if mean == 0:
        return Decimal(0)
    return abs(left - right) / mean * Decimal(100)


def within_symmetric_percent_tolerance(
    left: Decimal,
    right: Decimal,
    tolerance_percent: float | Decimal,
) -> bool:
    tolerance = Decimal(str(tolerance_percent))
    return symmetric_percentage_difference(left, right) <= tolerance


def all_pairs_match(values: Iterable[NormalizedValue], predicate: object) -> bool:
    """Return true only when every pair matches (important for 3+ duplicates)."""

    sequence = list(values)
    compare = predicate
    return all(
        compare(sequence[left], sequence[right])  # type: ignore[operator]
        for left in range(len(sequence))
        for right in range(left + 1, len(sequence))
    )
