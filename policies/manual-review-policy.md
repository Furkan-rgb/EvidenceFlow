---
policy_id: EFP-MANUAL-REVIEW
title: Manual Review Policy
---

# Manual Review Policy

## 5.1 Classification review

A reviewer may approve a proposed low-confidence document type or correct it to
one of the supported document types. Extraction must use the effective reviewed
classification.

## 5.2 Field review

For each low-confidence non-null field, a reviewer must either approve the
extracted value or provide a value valid for the field type. A review batch is
accepted only when every pending item has one valid decision.

## 5.3 Conflict review

A reviewer may select one of the cited submitted values, provide a typed corrected
value, or mark the conflict unresolved. The system must not assign implicit
authority based on upload order or document type.

## 5.4 Audit trail

Original model output and citations are immutable. Each review decision records
the review item, action, effective value when applicable, and UTC decision time.
Repeated or concurrent decisions for an already consumed review batch are rejected.
