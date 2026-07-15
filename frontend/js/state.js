export const state = {
  reviewId: null,
  status: null,
  review: null,
  report: null,
};

const listeners = new Set();

export function updateState(patch) {
  Object.assign(state, patch);
  listeners.forEach((listener) => listener(state));
}

export function resetState() {
  updateState({ reviewId: null, status: null, review: null, report: null });
}

export function resumedReviewState(review = {}, resumed = {}) {
  return {
    ...review,
    ...resumed,
    status: resumed.status || "processing",
    pending_reviews: [],
    pending_review_items: [],
    review_items: [],
    summary: {
      ...(review.summary || {}),
      ...(resumed.summary || {}),
      pending_review_count: 0,
    },
  };
}

export function subscribe(listener) {
  listeners.add(listener);
  return () => listeners.delete(listener);
}
