function asArray(value) {
  if (Array.isArray(value)) return value;
  if (value === null || value === undefined) return [];
  return [value];
}

function humanise(value) {
  return String(value || "evidence")
    .replaceAll("_", " ")
    .replace(/^./, (character) => character.toUpperCase());
}

function documentIdOf(evidence, fallback = null) {
  return evidence?.document_id
    || evidence?.documentId
    || evidence?.source?.document_id
    || fallback;
}

function pageOf(evidence) {
  const candidate = evidence?.page_number ?? evidence?.page ?? evidence?.source?.page_number;
  const page = Number(candidate);
  return Number.isInteger(page) && page > 0 ? page : null;
}

function firstAvailableArray(...values) {
  return values.find((value) => Array.isArray(value) && value.length) || [];
}

/**
 * Derive final-report document links from persisted review provenance.
 *
 * Effective fields are canonical: report narratives are intentionally not
 * allowed to invent document evidence. Finding references are included as an
 * additional source. Legacy report references are used only when persisted
 * review provenance is unavailable. Every source is de-duplicated by its
 * visible field/document/page identity.
 */
export function deriveDocumentEvidence(review = {}, report = {}) {
  const documents = [
    ...asArray(review.documents),
    ...asArray(review.uploaded_documents),
  ];
  const filenames = new Map(
    documents
      .filter((document) => document?.document_id)
      .map((document) => [document.document_id, document.filename || document.document_id]),
  );
  const effectiveFields = firstAvailableArray(
    review.effective_fields,
    review.verified_review?.effective_fields,
  );
  const findings = firstAvailableArray(
    review.findings,
    review.verified_review?.findings,
    report.findings,
  );
  const fieldNamesById = new Map(
    effectiveFields
      .filter((field) => field?.field_id)
      .map((field) => [field.field_id, field.field_name]),
  );
  const result = [];
  const seen = new Set();

  function add(evidence, { fieldName = null, documentId = null, label = null } = {}) {
    if (!evidence || typeof evidence !== "object") return;
    const resolvedDocumentId = documentIdOf(evidence, documentId);
    if (!resolvedDocumentId) return;
    const page = pageOf(evidence);
    const resolvedFieldName = fieldName || fieldNamesById.get(evidence.field_id) || null;
    const documentLabel = evidence.filename || filenames.get(resolvedDocumentId) || resolvedDocumentId;
    const resolvedLabel = label
      || [resolvedFieldName && humanise(resolvedFieldName), documentLabel].filter(Boolean).join(" · ");
    const key = [resolvedFieldName || resolvedLabel, resolvedDocumentId, page || "document"].join("|");
    if (seen.has(key)) return;
    seen.add(key);
    result.push({
      ...evidence,
      document_id: resolvedDocumentId,
      ...(page ? { page_number: page } : {}),
      label: resolvedLabel || documentLabel,
    });
  }

  effectiveFields.forEach((field) => {
    asArray(field?.evidence).forEach((evidence) => add(evidence, {
      fieldName: field.field_name,
      documentId: field.document_id,
    }));
  });
  findings.forEach((finding) => {
    asArray(finding?.evidence || finding?.evidence_references).forEach((evidence) => add(evidence, {
      fieldName: finding.field_name || finding.field,
      documentId: finding.document_id,
    }));
  });
  if (!result.length) {
    asArray(report.document_evidence || report.evidence_references).forEach((evidence) => add(evidence, {
      fieldName: evidence.field_name || evidence.field,
      label: evidence.label,
    }));
  }

  return result;
}
