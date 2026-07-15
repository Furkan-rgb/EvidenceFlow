const API_ROOT = "/api/v1";

export class ApiError extends Error {
  constructor(message, { status = 0, code = "request_failed", details = null, requestId = null } = {}) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
    this.details = details;
    this.requestId = requestId;
  }
}

async function request(path, options = {}) {
  let response;
  try {
    response = await fetch(path, {
      credentials: "same-origin",
      headers: { Accept: "application/json", ...(options.headers || {}) },
      ...options,
    });
  } catch (error) {
    throw new ApiError("EvidenceFlow could not reach the local API. Check that the application is running.", {
      code: "network_error",
      details: error instanceof Error ? error.message : null,
    });
  }

  const contentType = response.headers.get("content-type") || "";
  const body = contentType.includes("application/json")
    ? await response.json().catch(() => null)
    : await response.text().catch(() => "");
  if (!response.ok) {
    const apiError = body && typeof body === "object" ? body.error : null;
    throw new ApiError(apiError?.message || `The request failed with status ${response.status}.`, {
      status: response.status,
      code: apiError?.code || "request_failed",
      details: apiError?.details || null,
      requestId: apiError?.request_id || response.headers.get("x-request-id"),
    });
  }
  return body;
}

export async function createReview(files) {
  const form = new FormData();
  files.forEach((file) => form.append("files", file, file.name));
  return request(`${API_ROOT}/reviews`, { method: "POST", body: form });
}

export function getReview(reviewId) {
  return request(`${API_ROOT}/reviews/${encodeURIComponent(reviewId)}`);
}

export function resumeReview(reviewId, decisions) {
  return request(`${API_ROOT}/reviews/${encodeURIComponent(reviewId)}/resume`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ decisions }),
  });
}

export function getReport(reviewId) {
  return request(`${API_ROOT}/reviews/${encodeURIComponent(reviewId)}/report`);
}

export function documentUrl(reviewId, documentId, pageNumber = null) {
  const base = `${API_ROOT}/reviews/${encodeURIComponent(reviewId)}/documents/${encodeURIComponent(documentId)}`;
  return pageNumber ? `${base}#page=${encodeURIComponent(pageNumber)}` : base;
}

export function exportUrl(reviewId, format) {
  if (format !== "json" && format !== "md") throw new TypeError("Unsupported export format");
  return `${API_ROOT}/reviews/${encodeURIComponent(reviewId)}/export.${format}`;
}
