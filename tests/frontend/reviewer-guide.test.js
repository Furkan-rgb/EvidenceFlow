import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  DECISION_STAGES,
  POLICY_EXPLANATION,
  REVIEWER_SAMPLES,
  REVIEWER_STEPS,
  V1_BOUNDARY,
  sampleDocumentsPath,
} from "../../frontend/js/reviewer-guide.js";

test("reviewer walkthrough points to the five intentional sample journeys", () => {
  assert.deepEqual(
    REVIEWER_SAMPLES.map(({ bundle, label }) => [bundle, label]),
    [
      ["bundle_001", "Clean happy path"],
      ["bundle_008", "Registration conflict"],
      ["bundle_016", "Low-confidence field"],
      ["bundle_017", "Classification review"],
      ["bundle_020", "Typed correction"],
    ],
  );
  assert.equal(
    sampleDocumentsPath("bundle_008"),
    "eval/bundles/bundle_008/documents/",
  );
  assert.ok(REVIEWER_STEPS.some((step) => step.includes("evidence link")));
  assert.ok(REVIEWER_STEPS.some((step) => step.includes("export JSON or Markdown")));
});

test("decision explanation distinguishes models, rules, policies, and V1 limits", () => {
  const explanation = DECISION_STAGES.map(({ title, detail }) => `${title} ${detail}`).join(" ");

  assert.match(explanation, /1-based page number/);
  assert.match(explanation, /gemma4:12b-mlx/);
  assert.match(explanation, /YAML configuration and Python rules/);
  assert.match(explanation, /persisted/);
  assert.match(explanation, /EmbeddingGemma/);
  assert.match(explanation, /Code owns the canonical company name and status/);
  assert.match(explanation, /MLflow/);
  assert.match(POLICY_EXPLANATION, /six policies/);
  assert.match(POLICY_EXPLANATION, /do not replace the deterministic decision rules/);
  assert.match(V1_BOUNDARY, /does not perform OCR/);
});

test("the shell has no footer and the decision explanation uses native details", () => {
  const index = readFileSync(new URL("../../frontend/index.html", import.meta.url), "utf8");
  const uploadView = readFileSync(new URL("../../frontend/js/views/upload.js", import.meta.url), "utf8");
  const styles = readFileSync(new URL("../../frontend/css/app.css", import.meta.url), "utf8");

  assert.doesNotMatch(index, /<footer\b/i);
  assert.doesNotMatch(index, /site-footer/);
  assert.match(index, /Source · AGPL-3\.0/);
  assert.match(index, /github\.com\/Furkan-rgb\/EvidenceFlow/);
  assert.match(uploadView, /document\.createElement\("details"\)/);
  assert.match(uploadView, /How EvidenceFlow reaches a decision/);
  assert.match(uploadView, /Reviewer walkthrough/);
  assert.match(uploadView, /http:\/\/127\.0\.0\.1:5001/);
  assert.doesNotMatch(uploadView, /What to include/);
  assert.match(styles, /\.button:focus-within\s*\{/);
});
