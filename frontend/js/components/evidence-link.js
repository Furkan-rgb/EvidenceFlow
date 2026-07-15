import { documentUrl } from "../api.js";

export function evidenceDocumentId(evidence) {
  return evidence?.document_id || evidence?.documentId || evidence?.source?.document_id || null;
}

export function evidencePage(evidence) {
  const value = evidence?.page_number
    ?? evidence?.page
    ?? evidence?.source?.page_number
    ?? evidence?.evidence?.[0]?.page_number;
  return Number.isInteger(Number(value)) && Number(value) > 0 ? Number(value) : null;
}

export function createEvidenceLink(reviewId, evidence, label = "Open evidence") {
  const documentId = evidenceDocumentId(evidence);
  if (!documentId) {
    const unavailable = document.createElement("span");
    unavailable.className = "reference-meta";
    unavailable.textContent = "Source document unavailable";
    return unavailable;
  }
  const page = evidencePage(evidence);
  const link = document.createElement("a");
  link.className = "evidence-link";
  link.href = documentUrl(reviewId, documentId, page);
  link.target = "_blank";
  link.rel = "noopener";
  link.textContent = page ? `${label} · page ${page}` : label;
  link.setAttribute("aria-label", page ? `${label}, opens PDF at page ${page}` : `${label}, opens PDF`);
  return link;
}

export function createEvidenceExcerpt(evidence) {
  const excerpt = evidence?.excerpt
    || evidence?.source_text
    || evidence?.text
    || evidence?.quote
    || evidence?.evidence?.[0]?.source_text;
  if (!excerpt) return null;
  const wrapper = document.createElement("div");
  wrapper.className = "evidence-block";
  const quote = document.createElement("blockquote");
  quote.textContent = `“${excerpt}”`;
  wrapper.append(quote);
  return wrapper;
}
