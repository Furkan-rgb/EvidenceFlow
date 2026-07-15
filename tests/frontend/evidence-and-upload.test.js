import assert from "node:assert/strict";
import test from "node:test";

import { createEvidenceLink } from "../../frontend/js/components/evidence-link.js";
import { deriveDocumentEvidence } from "../../frontend/js/evidence-data.js";
import {
  appendFileInstances,
  removeFileInstance,
} from "../../frontend/js/upload-selection.js";

test("derives and de-duplicates labelled evidence from effective fields", () => {
  const sharedEvidence = {
    document_id: "document-application",
    page_number: 2,
    source_text: "Company name: Acme B.V.",
  };
  const review = {
    documents: [
      { document_id: "document-application", filename: "application.pdf" },
      { document_id: "document-extract", filename: "extract.pdf" },
    ],
    effective_fields: [
      {
        field_id: "field-company-name",
        field_name: "company_name",
        document_id: "document-application",
        evidence: [sharedEvidence, { ...sharedEvidence }],
      },
      {
        field_id: "field-registration",
        field_name: "registration_number",
        document_id: "document-extract",
        evidence: [{
          document_id: "document-extract",
          page_number: 1,
          source_text: "Registration: NL12345678",
        }],
      },
    ],
    findings: [{
      field_name: "company_name",
      evidence: [{ field_id: "field-company-name", ...sharedEvidence }],
    }],
  };

  const evidence = deriveDocumentEvidence(review, {
    document_evidence: [{
      document_id: "narrative-only-document",
      page_number: 9,
      label: "Narrative-only reference",
    }],
  });

  assert.deepEqual(
    evidence.map((item) => [item.label, item.document_id, item.page_number]),
    [
      ["Company name · application.pdf", "document-application", 2],
      ["Registration number · extract.pdf", "document-extract", 1],
    ],
  );
});

test("evidence links label the field, document, and one-based PDF page", () => {
  globalThis.document = {
    createElement: () => ({
      setAttribute(name, value) {
        this[name] = value;
      },
    }),
  };
  try {
    const link = createEvidenceLink("review-1", {
      document_id: "document-application",
      page_number: 2,
    }, "Company name · application.pdf");

    assert.equal(link.textContent, "Company name · application.pdf · page 2");
    assert.equal(
      link.href,
      "/api/v1/reviews/review-1/documents/document-application#page=2",
    );
    assert.equal(
      link["aria-label"],
      "Company name · application.pdf, opens PDF at page 2",
    );
  } finally {
    delete globalThis.document;
  }
});

test("preserves metadata-identical files and removes only one selected instance", () => {
  const duplicate = {
    name: "application.pdf",
    size: 1234,
    lastModified: 42,
    type: "application/pdf",
  };
  let nextId = 0;
  const selected = appendFileInstances([], [duplicate, { ...duplicate }], () => {
    nextId += 1;
    return `upload-${nextId}`;
  });

  assert.equal(selected.length, 2);
  assert.deepEqual(selected.map((entry) => entry.instanceId), ["upload-1", "upload-2"]);

  const remaining = removeFileInstance(selected, "upload-1");
  assert.equal(remaining.length, 1);
  assert.equal(remaining[0].instanceId, "upload-2");
  assert.deepEqual(remaining[0].file, duplicate);
});
