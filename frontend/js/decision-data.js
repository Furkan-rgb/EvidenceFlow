function firstNonEmptyArray(...values) {
  return values.find((value) => Array.isArray(value) && value.length) || [];
}

function humanise(value) {
  return String(value || "review item")
    .replaceAll("_", " ")
    .replace(/^./, (character) => character.toUpperCase())
    .replace(/\beur\b/gi, "EUR")
    .replace(/\bid\b/gi, "ID");
}

function owns(object, key) {
  return object !== null
    && typeof object === "object"
    && Object.prototype.hasOwnProperty.call(object, key);
}

function itemLabel(item, decision) {
  const explicit = decision.field_name
    || decision.field
    || item?.field_name
    || item?.field;
  if (explicit) return humanise(explicit);
  const itemType = item?.type || item?.item_type;
  if (itemType === "low_confidence_classification") return "Document classification";
  return humanise(decision.label || decision.review_item_id || "review item");
}

function actionLabel(action) {
  return {
    approve: "Approved",
    correct: "Corrected",
    select_value: "Selected submitted value",
    mark_unresolved: "Marked unresolved",
  }[action] || humanise(action || "recorded");
}

function effectiveOutcome(decision, audit, item) {
  if (decision.action === "mark_unresolved") {
    return "No effective value was selected; the finding remains unresolved.";
  }

  if (owns(audit, "effective_value") && audit.effective_value !== null) {
    const noun = item?.type === "low_confidence_classification" ? "type" : "value";
    return `Effective ${noun}: ${audit.effective_value}`;
  }
  if (owns(decision, "effective_value") && decision.effective_value !== null) {
    return `Effective value: ${decision.effective_value}`;
  }
  if (owns(decision, "value") && decision.value !== null) {
    return `Effective value: ${decision.value}`;
  }
  if (decision.action === "select_value" && decision.selected_field_id) {
    return `Selected field: ${decision.selected_field_id}`;
  }
  if (decision.action === "approve") {
    const approved = item?.extracted_value ?? item?.proposed_document_type;
    if (approved !== null && approved !== undefined) return `Effective value: ${approved}`;
  }
  return "The decision was recorded in the audit trail.";
}

/**
 * Join immutable decisions with their audit outcome and resolved review item.
 *
 * Current graph snapshots expose all three collections. Older exports may only
 * contain raw or pre-enriched decisions, so every join has a safe fallback.
 */
export function deriveManualDecisions(review = {}, report = {}) {
  const audits = firstNonEmptyArray(
    review.review_decision_audits,
    report.review_decision_audits,
  ).filter((audit) => audit && typeof audit === "object");
  const resolvedItems = firstNonEmptyArray(
    review.resolved_review_items,
    report.resolved_review_items,
  ).filter((item) => item && typeof item === "object");
  const explicitDecisions = firstNonEmptyArray(
    report.manual_review_decisions,
    review.review_decisions,
    review.decisions,
  );

  const auditByDecisionId = new Map();
  const auditByItemId = new Map();
  audits.forEach((audit) => {
    const decision = audit.decision;
    if (!decision || typeof decision !== "object") return;
    if (decision.decision_id) auditByDecisionId.set(decision.decision_id, audit);
    if (decision.review_item_id) auditByItemId.set(decision.review_item_id, audit);
  });
  const itemByDecisionId = new Map();
  const itemByItemId = new Map();
  resolvedItems.forEach((item) => {
    if (item.resolved_by_decision_id) {
      itemByDecisionId.set(item.resolved_by_decision_id, item);
    }
    if (item.review_item_id) itemByItemId.set(item.review_item_id, item);
  });

  const decisions = [];
  const seen = new Set();
  function add(rawDecision) {
    if (typeof rawDecision === "string") {
      const key = `text:${rawDecision}`;
      if (!seen.has(key)) {
        seen.add(key);
        decisions.push({
          label: rawDecision,
          action_label: "Recorded decision",
          outcome: "This decision came from a legacy report snapshot.",
        });
      }
      return;
    }
    if (!rawDecision || typeof rawDecision !== "object") return;
    const identifiers = [
      rawDecision.decision_id && `decision:${rawDecision.decision_id}`,
      rawDecision.review_item_id && `item:${rawDecision.review_item_id}`,
    ].filter(Boolean);
    if (identifiers.some((identifier) => seen.has(identifier))) return;
    identifiers.forEach((identifier) => seen.add(identifier));
    if (!identifiers.length) seen.add(`anonymous:${decisions.length}`);
    if (rawDecision.label && rawDecision.action_label && rawDecision.outcome) {
      decisions.push({
        label: rawDecision.label,
        action_label: rawDecision.action_label,
        outcome: rawDecision.outcome,
        review_item_id: rawDecision.review_item_id || null,
        decision_id: rawDecision.decision_id || null,
      });
      return;
    }
    const audit = auditByDecisionId.get(rawDecision.decision_id)
      || auditByItemId.get(rawDecision.review_item_id)
      || null;
    const item = itemByDecisionId.get(rawDecision.decision_id)
      || itemByItemId.get(rawDecision.review_item_id)
      || null;
    const itemContext = item || (audit ? { type: audit.review_item_type } : null);
    decisions.push({
      label: itemLabel(itemContext, rawDecision),
      action_label: actionLabel(rawDecision.action),
      outcome: effectiveOutcome(rawDecision, audit, itemContext),
      review_item_id: rawDecision.review_item_id || null,
      decision_id: rawDecision.decision_id || null,
    });
  }

  explicitDecisions.forEach(add);
  audits.forEach((audit) => add(audit.decision));
  return decisions;
}
