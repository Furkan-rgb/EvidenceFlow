import { createReview, getReport, getReview, resumeReview } from "./api.js";
import { clearRoute, reviewIdFromHash, setReviewRoute } from "./router.js";
import { resetState, state, updateState } from "./state.js";
import { renderProcessing } from "./views/processing.js";
import { renderReport } from "./views/report.js";
import { renderReview } from "./views/review.js";
import { renderUpload } from "./views/upload.js";

const POLL_INTERVAL_MS = 1500;
const app = document.querySelector("#app");
const liveRegion = document.querySelector("#live-region");
let pollTimer = null;
let routeGeneration = 0;

function announce(message) {
  liveRegion.textContent = "";
  window.setTimeout(() => { liveRegion.textContent = message; }, 20);
}

function setDocumentTitle(label) {
  document.title = label ? `${label} · EvidenceFlow` : "EvidenceFlow";
}

function show(content, { focus = true } = {}) {
  app.replaceChildren(content);
  if (focus) {
    const heading = app.querySelector("h1");
    if (heading) {
      heading.tabIndex = -1;
      heading.focus({ preventScroll: true });
    } else app.focus({ preventScroll: true });
  }
}

function stopPolling() {
  if (pollTimer !== null) window.clearTimeout(pollTimer);
  pollTimer = null;
}

function showFatal(title, message, { allowRetry = true } = {}) {
  stopPolling();
  const panel = document.createElement("section");
  panel.className = "fatal-panel panel";
  const heading = document.createElement("h1");
  heading.textContent = title;
  const copy = document.createElement("p");
  copy.textContent = message;
  panel.append(heading, copy);
  if (allowRetry) {
    const button = document.createElement("button");
    button.className = "button button-secondary";
    button.type = "button";
    button.textContent = state.reviewId ? "Retry loading review" : "Start a new review";
    button.addEventListener("click", () => state.reviewId ? loadReview(state.reviewId) : showUpload());
    panel.append(button);
  }
  show(panel);
  setDocumentTitle("Unable to continue");
  announce(message);
}

function showUpload() {
  stopPolling();
  resetState();
  const upload = renderUpload({
    onStart: async (files) => {
      const created = await createReview(files);
      updateState({ reviewId: created.review_id, status: created.status, review: created });
      show(renderProcessing({ ...created, summary: { document_count: files.length } }));
      setDocumentTitle("Processing");
      announce("Upload accepted. Review processing has started.");
      setReviewRoute(created.review_id);
    },
  });
  show(upload);
  setDocumentTitle("");
}

async function renderCurrentReview(review, generation) {
  if (generation !== routeGeneration) return;
  const status = review.status;
  updateState({ reviewId: review.review_id || state.reviewId, status, review });
  if (status === "processing") {
    show(renderProcessing(review), { focus: false });
    setDocumentTitle("Processing");
    pollTimer = window.setTimeout(() => loadReview(state.reviewId, { renderLoading: false, generation }), POLL_INTERVAL_MS);
    return;
  }
  if (status === "needs_review") {
    stopPolling();
    show(renderReview(state.reviewId, review, {
      onSubmit: async (decisions) => {
        const resumed = await resumeReview(state.reviewId, decisions);
        announce("Decisions saved. Review processing has resumed.");
        updateState({ status: resumed.status || "processing", review: { ...review, ...resumed } });
        show(renderProcessing({ ...review, ...resumed }));
        await loadReview(state.reviewId, { renderLoading: false, generation });
      },
    }));
    setDocumentTitle("Review required");
    announce(`${(review.pending_reviews || review.review_items || []).length} reviewer decisions are required.`);
    return;
  }
  if (status === "completed") {
    stopPolling();
    try {
      const report = await getReport(state.reviewId);
      if (generation !== routeGeneration) return;
      updateState({ report });
      show(renderReport(state.reviewId, review, report));
      setDocumentTitle("Final report");
      announce("The review is complete and its final report is available.");
    } catch (error) {
      showFatal("Report unavailable", error instanceof Error ? error.message : "The final report could not be loaded.");
    }
    return;
  }
  if (status === "failed") {
    showFatal("Review failed", review.error?.message || review.error_message || "EvidenceFlow could not complete this review.");
    return;
  }
  showFatal("Unknown review state", `The API returned an unsupported status: ${status || "none"}.`);
}

async function loadReview(reviewId, { renderLoading = true, generation = routeGeneration } = {}) {
  stopPolling();
  updateState({ reviewId });
  if (renderLoading) {
    show(renderProcessing({ review_id: reviewId }), { focus: false });
    setDocumentTitle("Loading review");
  }
  try {
    const review = await getReview(reviewId);
    await renderCurrentReview(review, generation);
  } catch (error) {
    if (generation !== routeGeneration) return;
    showFatal("Review unavailable", error instanceof Error ? error.message : "The review could not be loaded.");
  }
}

function route() {
  routeGeneration += 1;
  const generation = routeGeneration;
  stopPolling();
  const reviewId = reviewIdFromHash();
  if (reviewId) loadReview(reviewId, { generation });
  else showUpload();
}

document.querySelectorAll("[data-new-review]").forEach((link) => link.addEventListener("click", (event) => {
  event.preventDefault();
  clearRoute();
}));
window.addEventListener("hashchange", route);
window.addEventListener("beforeunload", stopPolling);
route();
