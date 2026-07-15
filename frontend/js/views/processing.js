import { currentProgressStep, workflowSteps, WORKFLOW_STEPS } from "../progress-data.js";

const STATUS_LABELS = {
  completed: "Done",
  current: "Working",
  upcoming: "Coming up",
};

function count(summary, keys, fallback) {
  for (const key of keys) {
    if (summary?.[key] === undefined || summary[key] === null) continue;
    const value = Number(summary[key]);
    if (Number.isFinite(value)) return Math.max(0, Math.trunc(value));
  }
  return fallback;
}

function trackerMarkup(progress) {
  return workflowSteps(progress).map((step) => {
    const marker = step.status === "completed" ? "✓" : step.position;
    const currentAttribute = step.status === "current" ? ' aria-current="step"' : "";
    return `
      <li class="workflow-step" data-status="${step.status}"${currentAttribute}>
        <span class="workflow-step-marker" aria-hidden="true">${marker}</span>
        <div class="workflow-step-copy">
          <strong>${step.label}</strong>
          <p>${step.description}</p>
        </div>
        <span class="workflow-step-status">${STATUS_LABELS[step.status]}</span>
      </li>
    `;
  }).join("");
}

export function renderProcessing(review = {}) {
  const summary = review.summary || {};
  const documentCount = count(
    summary,
    ["document_count"],
    review.documents?.length ?? 0,
  );
  const classified = count(
    summary,
    ["classified_count", "classification_count"],
    review.classifications?.length ?? 0,
  );
  const fields = count(
    summary,
    ["extracted_field_count", "extraction_count"],
    (review.extractions || review.extraction_results || []).reduce(
      (total, extraction) => total + (extraction.fields?.length ?? 0),
      0,
    ),
  );
  const findings = count(
    summary,
    ["finding_count", "conflict_count"],
    review.findings?.length ?? 0,
  );
  const pending = count(
    summary,
    ["pending_review_count", "review_item_count"],
    review.pending_reviews?.length ?? 0,
  );
  const current = currentProgressStep(review.progress);
  const eyebrow = current
    ? `Review in progress · Step ${current.position} of ${WORKFLOW_STEPS.length}`
    : "Loading review";
  const heading = current?.activeLabel || "Loading review…";
  const description = current?.description || "Fetching the latest durable workflow checkpoint.";
  const packageLabel = documentCount
    ? `${documentCount} document${documentCount === 1 ? "" : "s"} in this review`
    : "Preparing document totals";

  const section = document.createElement("section");
  section.className = "processing-shell";
  section.setAttribute("aria-labelledby", "processing-heading");
  section.innerHTML = `
    <div class="processing-panel panel">
      <div class="processing-header">
        <div class="spinner" aria-hidden="true"></div>
        <div>
          <p class="eyebrow">${eyebrow}</p>
          <h1 id="processing-heading">${heading}</h1>
          <p>${description}</p>
          <p class="processing-package">${packageLabel}</p>
        </div>
      </div>

      <section class="workflow-progress" aria-labelledby="workflow-progress-heading">
        <div class="workflow-progress-heading">
          <h2 id="workflow-progress-heading">Workflow progress</h2>
          <p>${current ? `${current.position} of ${WORKFLOW_STEPS.length} stages` : "Waiting for progress data"}</p>
        </div>
        <ol class="workflow-steps" role="list">
          ${trackerMarkup(review.progress)}
        </ol>
      </section>

      <div class="summary-grid" aria-label="Results collected so far">
        <div class="summary-card"><strong>${classified}</strong><span>documents classified</span></div>
        <div class="summary-card"><strong>${fields}</strong><span>fields extracted</span></div>
        <div class="summary-card"><strong>${findings}</strong><span>findings identified</span></div>
        <div class="summary-card"><strong>${pending}</strong><span>items need review</span></div>
      </div>
    </div>
  `;
  return section;
}
