# Evaluation and reproducibility

EvidenceFlow includes a real-model benchmark over 20 deterministic synthetic review packages. The evaluator runs the same LangGraph workflow and deterministic validation used by the application, simulates labelled reviewer decisions when the workflow interrupts, and compares the resulting prediction with checked-in ground truth.

The committed benchmark outputs are:

- [machine-readable JSON](../eval/results/evaluation-results.json), including per-bundle results and run identity;
- [generated Markdown](../eval/results/evaluation-results.md), containing the compact generated scorecard;
- [policy-retrieval relevance labels](../eval/queries/policy_relevance.json).

The values below describe the recorded run identified by those artifacts. They are not estimates and should not be treated as measurements of a different model, dataset, configuration, or implementation hash.

## Corpus and ground truth

The deterministic ReportLab generator uses the same immutable scenario definitions for both PDF contents and each bundle's `ground_truth.json`. This keeps fixtures and labels coupled: changing a scenario changes the inputs and expected outputs from one source of truth instead of allowing them to drift independently.

Every `eval/bundles/bundle_NNN/` directory is one independent company-review case containing:

- a `documents/` directory with the PDFs supplied to the workflow;
- a `ground_truth.json` file used only by the evaluator;
- expected document types and typed field values;
- expected findings, conflicts, report status, and review-routing reasons;
- labelled reviewer decisions for cases intended to interrupt.

The ground-truth file is never supplied wholesale to a model. Expected classification,
extraction, finding, and routing labels are not included in those model calls. The
evaluator loads labels to score predictions and, at a genuine workflow interrupt, to
provide the decision that a simulated reviewer would have submitted. That decision
then becomes part of subsequent verified state, just as a real reviewer's decision
would, and may therefore be present in the reporting input. If the model fails to
trigger an expected interrupt, the evaluator records a routing miss instead of forcing
that pause into the workflow.

The 20 cases cover these scenario families:

| Scenario family | Bundles |
| --- | ---: |
| Complete, consistent packages | 4 |
| Missing required document | 3 |
| Registration-number conflict | 3 |
| Revenue conflict outside tolerance | 3 |
| Company-name formatting variant that should normalize | 2 |
| Low-confidence or ambiguous classification/extraction | 2 |
| Unknown or irrelevant document | 1 |
| Incomplete financial statement | 1 |
| Human correction before validation passes | 1 |
| **Total** | **20** |

The definitions live in [`app/evaluation/scenarios.py`](../app/evaluation/scenarios.py). Generated PDFs and labels are checked in so a benchmark does not silently generate a different corpus immediately before scoring it.

### Context isolation

Bundles do not share a chat transcript or growing prompt context. Each classification and extraction call receives one current document. Reporting receives only the verified state and retrieved policy evidence for the current bundle. Retrieval queries are evaluated independently.

Consequently, running `bundle_020` after `bundle_001` does not place the preceding 19 bundles in the model context window. Each provider request starts with the inputs for its current task. The evaluation progress cache described below is a durable restart aid; it is not conversational memory or an Ollama context shared between bundles.

## Running the evaluation

The checked-in corpus can be evaluated directly. Regenerate it only when deliberately changing the scenario source:

```bash
make generate-data
```

This replaces `eval/bundles`. The explicit equivalent is:

```bash
uv run python -m app.cli generate-eval-data \
  --output-dir eval/bundles \
  --overwrite
```

Before evaluating, ensure that the configured Ollama models are installed, the compatible policy index has been built, and the local MLflow server is running:

```bash
make rebuild
make mlflow
```

Keep MLflow running in its own terminal. Then run:

```bash
make evaluate
```

`make evaluate` runs the dependency doctor before invoking the real benchmark. The explicit evaluation command is:

```bash
uv run python -m app.cli evaluate \
  --bundles-dir eval/bundles \
  --output-dir eval/results
```

The benchmark is intentionally slow because it makes real local-model calls. The recorded run took approximately 66 minutes. Do not use `make generate-data` or `make evaluate` as routine application-startup steps.

### MLflow fails closed during evaluation

Ordinary EvidenceFlow runtime tracing is fail-open, but benchmark tracing is fail-closed. Evaluation requires a healthy MLflow connection before it starts, checks tracing throughout workflow execution, and checks again before accepting the completed run. If MLflow becomes unavailable, the command exits unsuccessfully rather than presenting a partially observed benchmark as complete.

Evaluation uses the `evidenceflow-evaluation` MLflow experiment. The completed run logs aggregate metrics, counts, latency, reproducibility parameters, generated result artifacts, and per-bundle workflow traces. Document text, prompts, and source excerpts are not logged by default.

### Interrupted-run progress cache

The CLI writes an ignored `data/evaluation-progress.json` while the benchmark is running. Each completed bundle prediction is replaced atomically in that file so an interruption does not require repeating compatible completed bundles.

Cache entries are accepted only when the cache schema, dataset hash, policy-query-label hash, Python version, and supplied model/configuration/implementation metadata match. Malformed or incompatible entries are discarded. A successful complete run removes the progress file; a failed run leaves compatible progress available for a retry.

The result artifact records whether this mechanism was enabled and how many predictions it reused. The recorded run had `cache_enabled: true` and `cache_hit_count: 0`, so all 20 bundle predictions came from fresh workflow executions during that run.

## Metric definitions

The metric implementations are model-independent functions in [`app/evaluation/metrics.py`](../app/evaluation/metrics.py).

| Metric | What is scored |
| --- | --- |
| Document-classification accuracy | Exact expected document type by stable source filename across every labelled PDF. |
| Typed field-extraction accuracy | Every labelled field by filename and canonical field name. Numeric formatting differences compare through decimal values; strings remain case-sensitive after trimming, and booleans are not treated as integers. |
| Missing-document detection accuracy | Exact equality of the expected and predicted `missing_document` finding keys for each bundle. |
| Conflict precision, recall, and F1 | Expected versus predicted conflicting canonical field names, aggregated as true positives, false positives, and false negatives. |
| Exact review-routing accuracy | Exact equality of the expected and observed review-reason set for a bundle. A pause for the wrong reason does not count as correct. |
| Binary review-required accuracy | Whether the workflow correctly decided that any reviewer involvement was required, regardless of the reason. |
| Report citation validity | Whether every finding and policy-evidence ID cited by the report exists in the verified findings or retrieved policy evidence supplied for that report. |
| Report unknown-ID rate | Fraction of report references that are absent from those supported IDs. |
| Policy HitRate@5 | Fraction of labelled queries with at least one relevant evidence ID among the first five results. |
| Policy Recall@5 | Macro-average fraction of each query's relevant evidence IDs retrieved in the first five results. |
| Policy MRR@5 | Macro-average reciprocal rank of the first relevant result within the first five. |
| Policy nDCG@5 | Macro-average discounted ranking quality relative to the ideal ordering within the first five. |
| Latency and counts | Per-bundle duration, total benchmark duration, PDF/field/citation/review-route totals, and conflict confusion counts. |

Policy retrieval is evaluated against eight separately labelled queries in [`eval/queries/policy_relevance.json`](../eval/queries/policy_relevance.json). Deterministic rule correctness is also exercised independently by the model-free test suite; the benchmark is not a substitute for boundary and invariant tests.

## Recorded real-model run

The genuine local run completed at `2026-07-14T23:11:16.055625+00:00` with `gemma4:12b-mlx` for classification, extraction, and reporting and `embeddinggemma` for retrieval.

It processed 20 bundles and 64 PDFs in `3971.48` seconds. No bundle prediction came from the evaluation progress cache.

### Quality results

| Metric | Measured result |
| --- | ---: |
| Document classification accuracy | 1.0000 (64/64) |
| Field extraction accuracy | 0.9953 (214/215) |
| Missing-document detection accuracy | 1.0000 |
| Conflict precision / recall / F1 | 1.0000 / 1.0000 / 1.0000 |
| Exact review-routing accuracy | 0.8500 (17/20) |
| Binary review-required accuracy | 0.8500 (17/20) |
| Report citation validity / unknown-ID rate | 1.0000 / 0.0000 |
| Policy HitRate / Recall / MRR / nDCG at 5 | 1.0000 / 0.7500 / 0.8750 / 0.7246 |

### Counts and latency

| Measure | Result |
| --- | ---: |
| Conflict true positives / false positives / false negatives | 6 / 0 / 0 |
| Labelled fields | 215 |
| Report references | 75 |
| Policy queries | 8 |
| Minimum bundle latency | 160.92s |
| Mean bundle latency | 198.52s |
| Maximum bundle latency | 292.26s |
| Total duration | 3971.48s |
| Progress cache enabled / hits | true / 0 |
| Progress cache schema | 1 |

### Known misses

The three routing misses were deliberately retained in the committed results:

| Bundle | Labelled scenario | Observed limitation |
| --- | --- | --- |
| `bundle_016` | Low-confidence employee-count extraction | The real model did not request the expected field-review pause. |
| `bundle_017` | Ambiguous attachment classification | The real model did not request the expected classification-review pause. |
| `bundle_020` | Human correction before validation | The real model did not request the expected correction pause. |

These are model-quality results, not evaluator exceptions. They explain both the exact-routing and binary-review-required scores of `0.85`. The artifacts were not edited to turn the missed pauses into successes.

The remaining headline outcomes were 64/64 document classifications, 214/215 labelled field values, all six labelled conflicts found with no false positive, and all 75 report references validated against supported IDs. HitRate@5 of `1.0` means every policy query retrieved at least one relevant chunk; Recall@5 of `0.75` also shows that the top five did not contain every labelled relevant chunk on average.

## Reproducibility identity

The recorded artifacts bind the measurements to concrete model, corpus, configuration, query-label, and implementation identities:

| Identity | SHA-256 / digest |
| --- | --- |
| Gemma model used for classification, extraction, and reporting | `197a75677efb4b634352a8bdf24fd4781f1ea2c0b0c11f2b391ea7e0fcdcf01c` |
| EmbeddingGemma model | `85462619ee721b466c5927d109d4cb765861907d5417b9109caebc4e614679f1` |
| Dataset | `4d2dbef7a5821d12e178c5387c196eb4f58da05de4e71fa4b6f4affc1c6e47d9` |
| Configuration | `556991efbcec38c779fb2a1f53ef04938fbcc8c11f14a3dd676e203c630ac728` |
| Implementation | `473de267f8a8cf8d6003cff3cb8e8d0525d4f1964592a47dbd8ca18ca8ca5d69` |
| Policy-query labels | `d397b93bb60c66f1f0f6d557310885d9d29aba611ca58c42c2fd3acee74cddb8` |

The run used Python `3.12.13`. Classification, extraction, and reporting each recorded the same Gemma model name and digest independently; the condensed table does not imply a shared prompt or context.

For the complete per-bundle measurements and uncondensed metadata, use the committed [JSON result](../eval/results/evaluation-results.json). The [generated Markdown result](../eval/results/evaluation-results.md) is produced from that result object and provides a compact human-readable mirror.

## When to rerun

Keep the committed results as a historical measurement of their recorded identity. Rerun `make evaluate` before claiming metrics for a changed identity, including changes to:

- classification, extraction, reporting, retrieval, routing, or review code;
- model names, model digests, prompts, structured-output schemas, or provider adapters;
- deterministic review rules or report validation that affect predictions;
- policies, embedding identity, policy-index preprocessing, or retrieval behavior;
- evaluation scenarios, generated PDFs, ground truth, policy-query labels, or metric definitions;
- dependencies or configuration that can materially change execution.

Documentation-only or frontend-only changes do not invalidate what the historical artifact measured, but the old implementation hash still does not describe the newer repository state. If presenting the numbers as results for the current checkout rather than as a historical run, rerun and commit the newly generated JSON and Markdown artifacts.

When scenario definitions change, first run `make generate-data`. When policies, embedding configuration, or index preprocessing change, rebuild the policy index with `make rebuild`. Start MLflow before the benchmark and keep it healthy until `make evaluate` completes successfully.
