import {
  currentProgressStep,
  nextProgressStep,
  workflowSteps,
  WORKFLOW_STEPS,
} from "../progress-data.js";

const STATUS_LABELS = {
  completed: "Completed",
  current: "In progress",
  upcoming: "Upcoming",
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
        </div>
        <span class="workflow-step-status">${STATUS_LABELS[step.status]}</span>
      </li>
    `;
  }).join("");
}

function stageValue(progress, stepId, value, { current = "Working…", upcoming = "Not started" } = {}) {
  const step = workflowSteps(progress).find(({ id }) => id === stepId);
  if (step?.status === "completed") return String(value);
  if (step?.status === "current") return current;
  return upcoming;
}

function progressSummary(progress) {
  const steps = workflowSteps(progress);
  const completed = steps.filter(({ status }) => status === "completed").length;
  const upcoming = steps.filter(({ status }) => status === "upcoming").length;
  return `${completed} completed · 1 in progress · ${upcoming} upcoming`;
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
  const current = currentProgressStep(review.progress);
  const next = nextProgressStep(review.progress);
  const packageLabel = documentCount
    ? `${documentCount} document${documentCount === 1 ? "" : "s"} in this review`
    : "Preparing document totals";

  const section = document.createElement("section");
  section.className = "processing-shell";
  section.setAttribute("aria-labelledby", "processing-heading");
  if (!current) {
    section.innerHTML = `
      <div class="processing-panel panel">
        <div class="processing-header processing-header-loading">
          <div class="spinner" aria-hidden="true"></div>
          <div>
            <p class="eyebrow">Review in progress</p>
            <h1 id="processing-heading">Preparing the review…</h1>
            <p>Fetching the latest saved workflow state.</p>
            <p class="processing-package">${packageLabel}</p>
          </div>
        </div>
        <p class="processing-reassurance">
          The workflow will appear here as soon as the first saved stage is available.
        </p>
      </div>
    `;
    return section;
  }

  const classifiedValue = stageValue(review.progress, "classification", classified);
  const fieldsValue = stageValue(review.progress, "extraction", fields);
  const findingsValue = stageValue(
    review.progress,
    "cross_check",
    findings,
    { current: "Checking…", upcoming: "Not checked" },
  );

  section.innerHTML = `
    <div class="processing-panel panel">
      <p class="eyebrow">Review in progress</p>
      <div class="active-stage">
        <div class="spinner" aria-hidden="true"></div>
        <div class="active-stage-copy">
          <div class="active-stage-meta">
            <span>Step ${current.position} of ${WORKFLOW_STEPS.length}</span>
            <span class="active-stage-status">In progress</span>
          </div>
          <h1 id="processing-heading">${current.label}</h1>
          <p>${current.description}</p>
          ${next ? `<p class="processing-next"><strong>Next:</strong> ${next.label}</p>` : ""}
        </div>
      </div>
      <p class="processing-reassurance">
        Updates automatically. Model-backed stages can take a few minutes; processing continues locally if you leave this page.
      </p>

      <section class="workflow-progress" aria-labelledby="workflow-progress-heading">
        <div class="workflow-progress-heading">
          <h2 id="workflow-progress-heading">Review workflow</h2>
          <p>${progressSummary(review.progress)}</p>
        </div>
        <ol class="workflow-steps" role="list" aria-label="Review workflow stages">
          ${trackerMarkup(review.progress)}
        </ol>
      </section>

      <section class="processing-results" aria-labelledby="processing-results-heading">
        <h2 id="processing-results-heading">Results collected so far</h2>
        <dl class="processing-stats">
          <div><dt>Documents in package</dt><dd>${documentCount}</dd></div>
          <div><dt>Documents classified</dt><dd>${classifiedValue}</dd></div>
          <div><dt>Fields extracted</dt><dd>${fieldsValue}</dd></div>
          <div><dt>Findings so far</dt><dd>${findingsValue}</dd></div>
        </dl>
      </section>
    </div>
  `;
  return section;
}
