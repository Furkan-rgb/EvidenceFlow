import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  currentProgressStep,
  nextProgressStep,
  processingSignature,
  progressTransitionAnnouncement,
  workflowSteps,
  WORKFLOW_STEPS,
} from "../../frontend/js/progress-data.js";
import { resumedReviewState } from "../../frontend/js/state.js";
import { renderProcessing } from "../../frontend/js/views/processing.js";

const EXTRACTION_PROGRESS = {
  current_step_id: "extraction",
  steps: [
    { id: "document_processing", status: "completed" },
    { id: "classification", status: "completed" },
    { id: "extraction", status: "current" },
    { id: "completeness", status: "upcoming" },
    { id: "cross_check", status: "upcoming" },
    { id: "policy_retrieval", status: "upcoming" },
    { id: "report_composition", status: "upcoming" },
  ],
};

const PROCESSING_STYLES = readFileSync(
  new URL("../../frontend/css/app.css", import.meta.url),
  "utf8",
);

function processingMarkup(review) {
  const previousDocument = globalThis.document;
  globalThis.document = {
    createElement: () => ({
      className: "",
      innerHTML: "",
      setAttribute(name, value) {
        this[name] = value;
      },
    }),
  };
  try {
    return renderProcessing(review).innerHTML;
  } finally {
    if (previousDocument === undefined) delete globalThis.document;
    else globalThis.document = previousDocument;
  }
}

test("uses the seven stable workflow steps in reviewer order", () => {
  assert.deepEqual(
    WORKFLOW_STEPS.map(({ id, label }) => [id, label]),
    [
      ["document_processing", "Read PDF documents"],
      ["classification", "Classify documents"],
      ["extraction", "Extract required fields"],
      ["completeness", "Normalize and check completeness"],
      ["cross_check", "Cross-check evidence"],
      ["policy_retrieval", "Retrieve policy evidence"],
      ["report_composition", "Compose the validated report"],
    ],
  );
  assert.match(WORKFLOW_STEPS[0].description, /1-based page provenance/);
  assert.match(WORKFLOW_STEPS[5].description, /local policy index/);
  assert.match(WORKFLOW_STEPS[6].description, /validate its status and references in code/);
});

test("preserves API states while allowing exactly one declared current step", () => {
  const progress = {
    current_step_id: "extraction",
    steps: [
      { id: "document_processing", status: "completed" },
      { id: "classification", status: "current" },
      { id: "extraction", status: "upcoming" },
      { id: "untrusted_stage", status: "current", label: "<img src=x>" },
      { id: "cross_check", status: "invalid" },
    ],
  };

  const steps = workflowSteps(progress);
  assert.equal(steps.filter(({ status }) => status === "current").length, 1);
  assert.equal(steps.find(({ id }) => id === "document_processing").status, "completed");
  assert.equal(steps.find(({ id }) => id === "classification").status, "upcoming");
  assert.equal(steps.find(({ id }) => id === "extraction").status, "current");
  assert.equal(steps.find(({ id }) => id === "cross_check").status, "upcoming");
  assert.equal(currentProgressStep(progress).id, "extraction");
  assert.equal(nextProgressStep(progress).id, "completeness");
  assert.equal(steps.some(({ id }) => id === "untrusted_stage"), false);
});

test("renders an accessible current, completed, and upcoming tracker", () => {
  const markup = processingMarkup({
    status: "processing",
    progress: EXTRACTION_PROGRESS,
    summary: {
      document_count: 3,
      classified_count: 3,
      extracted_field_count: 0,
      finding_count: 0,
      pending_review_count: 0,
    },
  });

  assert.match(markup, /<ol class="workflow-steps" role="list" aria-label="Review workflow stages">/);
  assert.equal((markup.match(/aria-current="step"/g) || []).length, 1);
  assert.match(markup, /data-status="completed"/);
  assert.match(markup, /data-status="current"/);
  assert.match(markup, /data-status="upcoming"/);
  assert.match(markup, />Completed<\/span>/);
  assert.match(markup, />In progress<\/span>/);
  assert.match(markup, />Upcoming<\/span>/);
  assert.match(markup, /Step 3 of 7/);
  assert.match(markup, /<h1 id="processing-heading">Extract required fields<\/h1>/);
  assert.match(markup, /<strong>Next:<\/strong> Normalize and check completeness/);
  assert.match(markup, /2 completed · 1 in progress · 4 upcoming/);
  assert.match(markup, /<dl class="processing-stats">/);
  assert.match(markup, /<dt>Fields extracted<\/dt><dd>Working…<\/dd>/);
  assert.match(markup, /<dt>Findings so far<\/dt><dd>Not checked<\/dd>/);
  assert.doesNotMatch(markup, /untrusted_stage|<img src=x>/);
});

test("gives completed, current, and upcoming rows distinct full-row styles", () => {
  assert.match(PROCESSING_STYLES, /\.workflow-step\[data-status="completed"\]\s*\{/);
  assert.match(PROCESSING_STYLES, /\.workflow-step\[data-status="current"\]\s*\{/);
  assert.match(PROCESSING_STYLES, /\.workflow-step\[data-status="upcoming"\]\s*\{/);
});

test("shows an honest loading state until authoritative progress arrives", () => {
  const markup = processingMarkup({ review_id: "review-loading" });

  assert.match(markup, /<h1 id="processing-heading">Preparing the review…<\/h1>/);
  assert.match(markup, /Fetching the latest saved workflow state/);
  assert.doesNotMatch(markup, /aria-current="step"/);
  assert.doesNotMatch(markup, /workflow-steps|processing-stats|Not started|Not checked/);
});

test("explicit zero summary counts override stale fallback arrays", () => {
  const markup = processingMarkup({
    progress: EXTRACTION_PROGRESS,
    summary: {
      classified_count: 0,
      extracted_field_count: 0,
      finding_count: 0,
      pending_review_count: 0,
    },
    classifications: [{}, {}],
    extraction_results: [{ fields: [{}, {}] }],
    findings: [{}, {}],
    pending_reviews: [{}, {}],
  });

  assert.match(markup, /<dt>Documents classified<\/dt><dd>0<\/dd>/);
  assert.match(markup, /<dt>Fields extracted<\/dt><dd>Working…<\/dd>/);
  assert.match(markup, /<dt>Findings so far<\/dt><dd>Not checked<\/dd>/);
  assert.doesNotMatch(markup, /items need review/);
});

test("processing signatures react to progress and summary, but not revision", () => {
  const review = {
    revision: 1,
    progress: EXTRACTION_PROGRESS,
    summary: { document_count: 3, classified_count: 3 },
    documents: [{}, {}, {}],
  };

  assert.equal(processingSignature(review), processingSignature({ ...review, revision: 99 }));
  assert.notEqual(
    processingSignature(review),
    processingSignature({ ...review, summary: { ...review.summary, classified_count: 2 } }),
  );
  assert.notEqual(
    processingSignature(review),
    processingSignature({
      ...review,
      progress: {
        ...EXTRACTION_PROGRESS,
        current_step_id: "completeness",
        steps: EXTRACTION_PROGRESS.steps.map((step) => ({
          ...step,
          status: step.id === "extraction"
            ? "completed"
            : step.id === "completeness" ? "current" : step.status,
        })),
      },
    }),
  );
});

test("announces only real step transitions", () => {
  assert.equal(
    progressTransitionAnnouncement(null, EXTRACTION_PROGRESS),
    "Now on step 3 of 7: Extract required fields.",
  );
  assert.equal(
    progressTransitionAnnouncement("classification", EXTRACTION_PROGRESS),
    "Classify documents completed. Now on step 3 of 7: Extract required fields.",
  );
  assert.equal(progressTransitionAnnouncement("extraction", EXTRACTION_PROGRESS), null);
  assert.equal(progressTransitionAnnouncement(null, null), null);
});

test("resuming clears stale pending items while retaining other review data", () => {
  const resumed = resumedReviewState(
    {
      status: "needs_review",
      pending_reviews: [{ review_item_id: "one" }],
      pending_review_items: [{ review_item_id: "one" }],
      review_items: [{ review_item_id: "one" }],
      summary: { document_count: 3, pending_review_count: 1 },
    },
    { review_id: "review-one", status: "processing" },
  );

  assert.equal(resumed.status, "processing");
  assert.deepEqual(resumed.pending_reviews, []);
  assert.deepEqual(resumed.pending_review_items, []);
  assert.deepEqual(resumed.review_items, []);
  assert.deepEqual(resumed.summary, { document_count: 3, pending_review_count: 0 });
});
