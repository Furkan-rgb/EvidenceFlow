import assert from "node:assert/strict";
import test from "node:test";

import { deriveManualDecisions } from "../../frontend/js/decision-data.js";

test("joins conflict selection with its field and effective audited value", () => {
  const decision = {
    decision_id: "decision-conflict",
    review_item_id: "review-conflict-registration",
    action: "select_value",
    selected_field_id: "extract:registration_number",
  };
  const decisions = deriveManualDecisions({
    review_decisions: [decision],
    review_decision_audits: [{
      decision,
      review_item_type: "field_conflict",
      original_values: ["12345678", "12345687"],
      effective_value: "12345687",
    }],
    resolved_review_items: [{
      review_item_id: "review-conflict-registration",
      type: "field_conflict",
      field_name: "registration_number",
      resolved_by_decision_id: "decision-conflict",
    }],
  });

  assert.deepEqual(decisions, [{
    label: "Registration number",
    action_label: "Selected submitted value",
    outcome: "Effective value: 12345687",
    review_item_id: "review-conflict-registration",
    decision_id: "decision-conflict",
  }]);
});

test("makes corrected classifications and unresolved conflicts explicit", () => {
  const classification = {
    decision_id: "decision-classification",
    review_item_id: "review-classification-note",
    action: "correct",
    value: "supporting_correspondence",
  };
  const unresolved = {
    decision_id: "decision-unresolved",
    review_item_id: "review-conflict-revenue",
    action: "mark_unresolved",
  };
  const decisions = deriveManualDecisions({
    review_decisions: [classification, unresolved],
    review_decision_audits: [
      {
        decision: classification,
        review_item_type: "low_confidence_classification",
        original_values: ["unknown"],
        effective_value: "supporting_correspondence",
      },
      {
        decision: unresolved,
        review_item_type: "field_conflict",
        original_values: [100, 120],
        effective_value: null,
      },
    ],
    resolved_review_items: [
      {
        review_item_id: "review-classification-note",
        type: "low_confidence_classification",
        resolved_by_decision_id: "decision-classification",
      },
      {
        review_item_id: "review-conflict-revenue",
        type: "field_conflict",
        field_name: "annual_revenue_eur",
        resolved_by_decision_id: "decision-unresolved",
      },
    ],
  });

  assert.deepEqual(
    decisions.map(({ label, action_label: action, outcome }) => ({ label, action, outcome })),
    [
      {
        label: "Document classification",
        action: "Corrected",
        outcome: "Effective type: supporting_correspondence",
      },
      {
        label: "Annual revenue EUR",
        action: "Marked unresolved",
        outcome: "No effective value was selected; the finding remains unresolved.",
      },
    ],
  );
});

test("preserves useful fallbacks for older decision-only snapshots", () => {
  assert.deepEqual(
    deriveManualDecisions({
      decisions: [{
        review_item_id: "legacy-employee-count",
        action: "correct",
        field_name: "employee_count",
        value: 42,
      }],
    }),
    [{
      label: "Employee count",
      action_label: "Corrected",
      outcome: "Effective value: 42",
      review_item_id: "legacy-employee-count",
      decision_id: null,
    }],
  );

  assert.deepEqual(
    deriveManualDecisions({}, { manual_review_decisions: ["Reviewer approved evidence"] }),
    [{
      label: "Reviewer approved evidence",
      action_label: "Recorded decision",
      outcome: "This decision came from a legacy report snapshot.",
    }],
  );
});
