import { createFindingCard } from "../components/finding-card.js";
import { createReviewCard } from "../components/review-card.js";

function summaryValue(review, key, fallback) {
  return review.summary?.[key] ?? fallback ?? 0;
}

export function renderReview(reviewId, review, { onSubmit }) {
  const fragment = document.createDocumentFragment();
  const pending = review.pending_reviews || review.pending_review_items || review.review_items || [];
  const findings = review.findings || review.verified_review?.findings || [];
  const companyName = review.company_name
    || review.verified_review?.company_name
    || review.summary?.company_name
    || "Onboarding package";
  const toolbar = document.createElement("header");
  toolbar.className = "review-toolbar";
  const title = document.createElement("div");
  title.innerHTML = `<p class="eyebrow">Human review required</p><h1 class="workspace-title"></h1><p class="review-id"></p>`;
  title.querySelector("h1").textContent = companyName;
  title.querySelector(".review-id").textContent = `Review ${reviewId}`;
  const status = document.createElement("span");
  status.className = "status-pill status-pill-needs-review";
  status.textContent = `${pending.length} pending`;
  toolbar.append(title, status);

  const summaries = document.createElement("section");
  summaries.className = "summary-grid";
  summaries.setAttribute("aria-label", "Review summary");
  const values = [
    [summaryValue(review, "document_count", review.documents?.length), "documents"],
    [summaryValue(
      review,
      "extracted_field_count",
      (review.extractions || review.extraction_results || []).reduce(
        (total, extraction) => total + (extraction.fields?.length ?? 0),
        0,
      ),
    ), "extracted fields"],
    [summaryValue(review, "finding_count", findings.length), "findings"],
    [summaryValue(review, "pending_review_count", pending.length), "pending reviews"],
  ];
  values.forEach(([value, label]) => {
    const card = document.createElement("div");
    card.className = "summary-card";
    const strong = document.createElement("strong");
    strong.textContent = value;
    const span = document.createElement("span");
    span.textContent = label;
    card.append(strong, span);
    summaries.append(card);
  });

  const workspace = document.createElement("div");
  workspace.className = "workspace-grid";
  const main = document.createElement("section");
  main.className = "workspace-main";
  const heading = document.createElement("div");
  heading.className = "section-heading";
  heading.innerHTML = `<h2>Decisions</h2><p>Every item must be decided together</p>`;
  main.append(heading);
  const form = document.createElement("form");
  form.className = "review-form";
  form.noValidate = true;
  const cards = pending.map((item, index) => createReviewCard(reviewId, item, index));
  cards.forEach((card) => form.append(card.element));
  const actions = document.createElement("div");
  actions.className = "form-actions";
  const instruction = document.createElement("p");
  instruction.textContent = "Original model values and evidence remain in the audit trail.";
  const submit = document.createElement("button");
  submit.className = "button";
  submit.type = "submit";
  submit.textContent = "Submit decisions and continue";
  actions.append(instruction, submit);
  form.append(actions);
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const decisions = cards.map((card) => card.readDecision());
    if (decisions.some((decision) => decision === null)) return;
    submit.disabled = true;
    submit.textContent = "Submitting…";
    try {
      await onSubmit(decisions);
    } catch (error) {
      const message = document.createElement("p");
      message.className = "error-message";
      message.setAttribute("role", "alert");
      message.textContent = error instanceof Error ? error.message : "The decisions could not be submitted.";
      actions.before(message);
      submit.disabled = false;
      submit.textContent = "Submit decisions and continue";
    }
  });
  main.append(form);

  const aside = document.createElement("aside");
  aside.className = "workspace-aside";
  const findingsHeading = document.createElement("div");
  findingsHeading.className = "section-heading";
  const findingsTitle = document.createElement("h2");
  findingsTitle.textContent = "Deterministic findings";
  findingsHeading.append(findingsTitle);
  const findingList = document.createElement("div");
  findingList.className = "finding-list";
  if (findings.length) findings.forEach((finding) => findingList.append(createFindingCard(reviewId, finding)));
  else {
    const empty = document.createElement("p");
    empty.className = "empty-state";
    empty.textContent = "No deterministic findings at this stage.";
    findingList.append(empty);
  }
  aside.append(findingsHeading, findingList);
  workspace.append(main, aside);
  fragment.append(toolbar, summaries, workspace);
  return fragment;
}
