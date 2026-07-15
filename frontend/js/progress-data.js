const VALID_STATUSES = new Set(["completed", "current", "upcoming"]);

export const WORKFLOW_STEPS = Object.freeze([
  Object.freeze({
    id: "document_processing",
    label: "Read PDF documents",
    activeLabel: "Reading PDF documents…",
    description: "Validate the files and extract text with 1-based page provenance.",
  }),
  Object.freeze({
    id: "classification",
    label: "Classify documents",
    activeLabel: "Classifying documents…",
    description: "Identify each supported document type and pause if confidence is low.",
  }),
  Object.freeze({
    id: "extraction",
    label: "Extract required fields",
    activeLabel: "Extracting required fields…",
    description: "Read type-specific values and retain their source citations.",
  }),
  Object.freeze({
    id: "completeness",
    label: "Normalize and check completeness",
    activeLabel: "Normalizing and checking completeness…",
    description: "Standardize values and check the required documents and fields.",
  }),
  Object.freeze({
    id: "cross_check",
    label: "Cross-check evidence",
    activeLabel: "Cross-checking evidence…",
    description: "Compare duplicate and overlapping values and surface any conflicts.",
  }),
  Object.freeze({
    id: "policy_retrieval",
    label: "Retrieve policy evidence",
    activeLabel: "Retrieving policy evidence…",
    description: "Match the findings to relevant sections in the local policy index.",
  }),
  Object.freeze({
    id: "report_composition",
    label: "Compose the validated report",
    activeLabel: "Composing the validated report…",
    description: "Draft the narrative, then validate its status and references in code.",
  }),
]);

const STEP_BY_ID = new Map(WORKFLOW_STEPS.map((step) => [step.id, step]));

/**
 * Join API-owned progress states to frontend-owned reviewer copy.
 *
 * Unknown identifiers and statuses are deliberately ignored. This keeps API
 * content out of the DOM and makes a partially malformed response fail safely.
 */
export function workflowSteps(progress) {
  const apiSteps = Array.isArray(progress?.steps) ? progress.steps : [];
  const statusById = new Map();
  let firstCurrentId = null;

  apiSteps.forEach((step) => {
    if (!step || !STEP_BY_ID.has(step.id) || !VALID_STATUSES.has(step.status)) return;
    if (!statusById.has(step.id)) statusById.set(step.id, step.status);
    if (firstCurrentId === null && step.status === "current") firstCurrentId = step.id;
  });

  const declaredCurrentId = STEP_BY_ID.has(progress?.current_step_id)
    ? progress.current_step_id
    : firstCurrentId;

  return WORKFLOW_STEPS.map((step, index) => {
    let status = statusById.get(step.id) || "upcoming";
    if (step.id === declaredCurrentId) status = "current";
    else if (status === "current") status = "upcoming";
    return { ...step, status, position: index + 1 };
  });
}

export function currentProgressStep(progress) {
  return workflowSteps(progress).find((step) => step.status === "current") || null;
}

export function nextProgressStep(progress) {
  const steps = workflowSteps(progress);
  const currentIndex = steps.findIndex((step) => step.status === "current");
  if (currentIndex < 0) return null;
  return steps.slice(currentIndex + 1).find((step) => step.status === "upcoming") || null;
}

export function progressTransitionAnnouncement(previousStepId, progress) {
  const current = currentProgressStep(progress);
  if (!current || current.id === previousStepId) return null;
  const previous = STEP_BY_ID.get(previousStepId);
  const completed = previous && WORKFLOW_STEPS.indexOf(previous) < current.position - 1
    ? `${previous.label} completed. `
    : "";
  return `${completed}Now on step ${current.position} of ${WORKFLOW_STEPS.length}: ${current.label}.`;
}

function sortedSummaryEntries(summary) {
  if (!summary || typeof summary !== "object" || Array.isArray(summary)) return [];
  return Object.entries(summary).sort(([left], [right]) => left.localeCompare(right));
}

function extractedFieldCount(review) {
  return (review.extractions || review.extraction_results || []).reduce(
    (total, extraction) => total + (extraction.fields?.length ?? 0),
    0,
  );
}

/**
 * Describe only data rendered by the processing screen. Live graph checkpoints
 * can advance without changing the durable review revision, so revision is not
 * part of this signature.
 */
export function processingSignature(review = {}) {
  return JSON.stringify({
    has_progress: Boolean(review.progress),
    progress: workflowSteps(review.progress).map(({ id, status }) => [id, status]),
    summary: sortedSummaryEntries(review.summary),
    fallbacks: {
      document_count: review.documents?.length ?? 0,
      classification_count: review.classifications?.length ?? 0,
      extracted_field_count: extractedFieldCount(review),
      finding_count: review.findings?.length ?? 0,
      pending_review_count: review.pending_reviews?.length ?? 0,
    },
  });
}
