---
policy_id: EFP-DATA-QUALITY
title: Data Quality Guidelines
---

# Data Quality Guidelines

## 4.1 Confidence thresholds

Document classifications below the configured classification threshold require
review before extraction. Non-null extracted fields below the configured field
threshold require approval or correction before cross-document validation.

## 4.2 Typed values

Revenue must be a non-negative EUR decimal, employee count and reporting year must
be integers, and dates must be valid ISO dates. A model response that cannot be
validated against the expected type is not verified data.

## 4.3 Null and unsupported values

The absence of a required value must be recorded explicitly. A null value does
not require low-confidence approval, but it does create an incomplete-document
finding when the field is required.

## 4.4 Provenance quality

Every non-null important field must cite its document, a one-based page number,
and a concise exact source-text span. Confidence alone is not a substitute for
provenance.
