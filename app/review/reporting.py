"""Deterministic report status, canonical identity, and reference validation."""

from __future__ import annotations

from collections.abc import Sequence

from app.domain.extractions import (
    DocumentClassification,
    EffectiveFieldValue,
    ExtractionResult,
)
from app.domain.findings import Finding, FindingType
from app.domain.policy import PolicyEvidence
from app.domain.reports import ReportNarrative, ReportStatus, ReviewReport
from app.domain.reviews import ReviewDecision, ReviewItem, VerifiedReview


class ReportReferenceError(ValueError):
    """Raised when composed prose references facts it was not given."""

    def __init__(
        self,
        message: str,
        *,
        unknown_finding_ids: set[str] | None = None,
        unknown_policy_evidence_ids: set[str] | None = None,
    ) -> None:
        super().__init__(message)
        self.unknown_finding_ids = unknown_finding_ids or set()
        self.unknown_policy_evidence_ids = unknown_policy_evidence_ids or set()


def determine_report_status(
    findings: Sequence[Finding], pending_review_items: Sequence[ReviewItem] = ()
) -> ReportStatus:
    unresolved = [finding for finding in findings if not finding.resolved]
    if any(
        finding.type
        in {FindingType.MISSING_DOCUMENT, FindingType.MISSING_REQUIRED_FIELD}
        for finding in unresolved
    ):
        return ReportStatus.INCOMPLETE
    if unresolved or pending_review_items:
        return ReportStatus.NEEDS_FOLLOW_UP
    return ReportStatus.COMPLETE


def derive_canonical_company_name(
    effective_fields: Sequence[EffectiveFieldValue],
) -> str | None:
    """Return a presentation value only when deterministic values agree."""

    eligible_types = {
        "application_form": 0,
        "company_extract": 1,
        "financial_statement": 2,
    }
    candidates = [
        field
        for field in effective_fields
        if field.field_name == "company_name"
        and field.document_type.value in eligible_types
        and field.effective_value is not None
        and field.normalized_value is not None
    ]
    normalized = {str(field.normalized_value) for field in candidates}
    if len(normalized) != 1:
        return None
    chosen = min(
        candidates,
        key=lambda field: (
            eligible_types[field.document_type.value],
            field.document_id,
            field.field_id,
        ),
    )
    return str(chosen.effective_value)


def build_verified_review(
    *,
    review_id: str,
    classifications: Sequence[DocumentClassification],
    extractions: Sequence[ExtractionResult],
    effective_fields: Sequence[EffectiveFieldValue],
    findings: Sequence[Finding],
    review_decisions: Sequence[ReviewDecision] = (),
    pending_review_items: Sequence[ReviewItem] = (),
) -> VerifiedReview:
    """Seal a deterministic snapshot for the report-composer boundary."""

    return VerifiedReview(
        review_id=review_id,
        company_name=derive_canonical_company_name(effective_fields),
        status=determine_report_status(findings, pending_review_items),
        classifications=list(classifications),
        extractions=list(extractions),
        effective_fields=list(effective_fields),
        findings=list(findings),
        review_decisions=list(review_decisions),
        pending_review_items=list(pending_review_items),
    )


def validate_report_references(
    report: ReportNarrative,
    review: VerifiedReview,
    policy_evidence: Sequence[PolicyEvidence],
) -> ReportNarrative:
    finding_ids = {finding.finding_id for finding in review.findings}
    evidence_ids = {evidence.evidence_id for evidence in policy_evidence}
    claimed_findings = {
        finding_id for section in report.sections for finding_id in section.finding_ids
    }
    claimed_evidence = {
        evidence_id
        for section in report.sections
        for evidence_id in section.policy_evidence_ids
    }
    unknown_findings = claimed_findings - finding_ids
    unknown_evidence = claimed_evidence - evidence_ids
    if unknown_findings or unknown_evidence:
        raise ReportReferenceError(
            "report contains unsupported finding or policy-evidence references",
            unknown_finding_ids=unknown_findings,
            unknown_policy_evidence_ids=unknown_evidence,
        )
    return report


def finalize_report(
    report: ReportNarrative,
    review: VerifiedReview,
    policy_evidence: Sequence[PolicyEvidence],
) -> ReviewReport:
    """Reject invented references and impose code-owned status and company name."""

    validate_report_references(report, review, policy_evidence)
    return ReviewReport(
        company_name=review.company_name,
        status=review.status,
        executive_summary=report.executive_summary,
        sections=report.sections,
    )


def render_report_markdown(report: ReviewReport) -> str:
    """Render only a previously validated structured report."""

    lines = [
        "# EvidenceFlow review report",
        "",
        f"**Company:** {report.company_name or 'Not determinable'}",
        f"**Status:** {report.status.value}",
        "",
        "## Executive summary",
        "",
        report.executive_summary,
    ]
    for section in report.sections:
        lines.extend(["", f"## {section.title}", "", section.summary])
        if section.finding_ids:
            lines.extend(
                ["", "Findings: " + ", ".join(f"`{item}`" for item in section.finding_ids)]
            )
        if section.policy_evidence_ids:
            lines.extend(
                [
                    "",
                    "Policy evidence: "
                    + ", ".join(f"`{item}`" for item in section.policy_evidence_ids),
                ]
            )
    return "\n".join(lines) + "\n"
