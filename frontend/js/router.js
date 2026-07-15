const REVIEW_PREFIX = "review=";

export function reviewIdFromHash(hash = window.location.hash) {
  const value = hash.startsWith("#") ? hash.slice(1) : hash;
  if (value.startsWith(REVIEW_PREFIX)) {
    try {
      return decodeURIComponent(value.slice(REVIEW_PREFIX.length)) || null;
    } catch {
      return null;
    }
  }
  const pathMatch = value.match(/^\/?reviews\/([^/?#]+)$/);
  if (pathMatch) {
    try {
      return decodeURIComponent(pathMatch[1]);
    } catch {
      return null;
    }
  }
  return null;
}

export function setReviewRoute(reviewId, { replace = false } = {}) {
  const hash = `#${REVIEW_PREFIX}${encodeURIComponent(reviewId)}`;
  if (replace) history.replaceState(null, "", hash);
  else window.location.hash = hash;
}

export function clearRoute() {
  history.pushState(null, "", `${window.location.pathname}${window.location.search}`);
  window.dispatchEvent(new HashChangeEvent("hashchange"));
}
