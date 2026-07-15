export const REVIEWER_SAMPLES = Object.freeze([
  Object.freeze({
    bundle: "bundle_001",
    label: "Clean happy path",
    outcome: "A complete package with agreeing values.",
  }),
  Object.freeze({
    bundle: "bundle_008",
    label: "Registration conflict",
    outcome: "Two cited registration numbers require a reviewer decision.",
  }),
  Object.freeze({
    bundle: "bundle_016",
    label: "Low-confidence field",
    outcome: "An approximate employee count pauses for field review.",
  }),
  Object.freeze({
    bundle: "bundle_017",
    label: "Classification review",
    outcome: "An ambiguous note pauses before its fields are extracted.",
  }),
  Object.freeze({
    bundle: "bundle_020",
    label: "Typed correction",
    outcome: "An unclear employee count demonstrates an effective override.",
  }),
]);

export const REVIEWER_STEPS = Object.freeze([
  "Choose one bundle and upload every PDF in its documents folder.",
  "Watch the workflow move from processing to a review pause or a completed report.",
  "At a pause, open every evidence link, verify the cited PDF page, and decide each item.",
  "Continue the review, inspect findings and policy citations, then export JSON or Markdown.",
]);

export const DECISION_STAGES = Object.freeze([
  Object.freeze({
    title: "Read locally with provenance",
    detail: "Text-based PDFs are processed on this machine. Every extracted value keeps its source document, 1-based page number, and excerpt.",
  }),
  Object.freeze({
    title: "Create structured proposals",
    detail: "Gemma (gemma4:12b-mlx) classifies documents and extracts typed fields into constrained schemas. Its outputs are proposals, not final decisions.",
  }),
  Object.freeze({
    title: "Apply deterministic checks",
    detail: "YAML configuration and Python rules check required material, confidence thresholds, normalized identity values, revenue tolerance, exact employee counts, and duplicate conflicts.",
  }),
  Object.freeze({
    title: "Pause for accountable review",
    detail: "Classification, low-confidence field, and conflict decisions are persisted. Corrections become effective overrides while original model output and immutable audit decisions remain intact.",
  }),
  Object.freeze({
    title: "Retrieve supporting policy",
    detail: "EmbeddingGemma searches an index built from six Markdown policies and attaches the most relevant sections as policy evidence.",
  }),
  Object.freeze({
    title: "Validate the report in code",
    detail: "Gemma writes narrative sections only. Code owns the canonical company name and status, and rejects references to unknown findings or policy sections.",
  }),
  Object.freeze({
    title: "Trace without exposing documents",
    detail: "MLflow records safe workflow, model, retrieval, timing, and count metadata. Document contents, prompts, and source excerpts are not logged by default.",
  }),
]);

export const POLICY_EXPLANATION = "The six policies cover onboarding requirements, financial documents, company identification, data quality, manual review, and document evidence. Retrieved policy sections support findings and report citations; they do not replace the deterministic decision rules.";

export const V1_BOUNDARY = "EvidenceFlow V1 is a local demonstration for synthetic, digitally generated text PDFs. It does not perform OCR on scans and does not provide cloud processing or production authentication.";

export function sampleDocumentsPath(bundle) {
  return `eval/bundles/${bundle}/documents/`;
}
