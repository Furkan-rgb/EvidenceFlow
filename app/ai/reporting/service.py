"""LLM narrative generation constrained by deterministic review truth."""

from __future__ import annotations

import json

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from app.ai.reporting.prompts import REPORTING_SYSTEM_PROMPT
from app.ai.reporting.schemas import ModelReportNarrative
from app.ai.structured import invoke_structured
from app.domain import PolicyEvidence, ReportNarrative, ReviewReport, VerifiedReview
from app.review import finalize_report


class LLMReportComposer:
    def __init__(self, model: BaseChatModel) -> None:
        self._model = model

    async def compose(
        self, review: VerifiedReview, policy_evidence: list[PolicyEvidence]
    ) -> ReviewReport:
        review_json = json.dumps(review.model_dump(mode="json"), sort_keys=True)
        evidence_json = json.dumps(
            [item.model_dump(mode="json") for item in policy_evidence], sort_keys=True
        )
        messages = [
            SystemMessage(content=REPORTING_SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"Verified review JSON:\n{review_json}\n\n"
                    f"Retrieved policy evidence JSON:\n{evidence_json}"
                )
            ),
        ]

        model_narrative = await invoke_structured(
            self._model,
            ModelReportNarrative,
            messages,
            capability="verified review reporting",
            validator=lambda result: self._validate_references(
                result, review, policy_evidence
            ),
        )
        narrative = ReportNarrative.model_validate(model_narrative.model_dump())
        return finalize_report(narrative, review, policy_evidence)

    @staticmethod
    def _validate_references(
        narrative: ModelReportNarrative,
        review: VerifiedReview,
        policy_evidence: list[PolicyEvidence],
    ) -> ModelReportNarrative:
        known_findings = {finding.finding_id for finding in review.findings}
        known_evidence = {item.evidence_id for item in policy_evidence}
        claimed_findings = {
            finding_id
            for section in narrative.sections
            for finding_id in section.finding_ids
        }
        claimed_evidence = {
            evidence_id
            for section in narrative.sections
            for evidence_id in section.policy_evidence_ids
        }
        unknown_findings = sorted(claimed_findings - known_findings)
        unknown_evidence = sorted(claimed_evidence - known_evidence)
        if unknown_findings or unknown_evidence:
            raise ValueError(
                "unsupported report references: "
                f"findings={unknown_findings}, policy_evidence={unknown_evidence}"
            )
        return narrative
