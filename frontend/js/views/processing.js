function count(summary, ...keys) {
  for (const key of keys) if (summary?.[key] !== undefined) return Number(summary[key]) || 0;
  return 0;
}

export function renderProcessing(review = {}) {
  const summary = review.summary || {};
  const documentCount = count(summary, "document_count") || (review.documents?.length ?? 0);
  const classified = count(summary, "classified_count", "classification_count") || (review.classifications?.length ?? 0);
  const fields = count(summary, "extracted_field_count", "extraction_count")
    || (review.extractions || review.extraction_results || []).reduce((total, extraction) => total + (extraction.fields?.length ?? 0), 0);
  const findings = count(summary, "finding_count", "conflict_count") || (review.findings?.length ?? 0);
  const pending = count(summary, "pending_review_count", "review_item_count") || (review.pending_reviews?.length ?? 0);
  const section = document.createElement("section");
  section.className = "processing-shell";
  section.innerHTML = `
    <div class="processing-panel panel">
      <div class="processing-header">
        <div class="spinner" aria-hidden="true"></div>
        <div>
          <p class="eyebrow">Review in progress</p>
          <h1>Processing ${documentCount || "your"} document${documentCount === 1 ? "" : "s"}…</h1>
          <p>EvidenceFlow is reading the package and applying deterministic cross-document checks.</p>
        </div>
      </div>
      <div class="progress-track" aria-hidden="true"><span></span></div>
      <div class="summary-grid" aria-label="Current review progress">
        <div class="summary-card"><strong>${classified}</strong><span>documents classified</span></div>
        <div class="summary-card"><strong>${fields}</strong><span>fields extracted</span></div>
        <div class="summary-card"><strong>${findings}</strong><span>findings identified</span></div>
        <div class="summary-card"><strong>${pending}</strong><span>items need review</span></div>
      </div>
    </div>
  `;
  return section;
}
