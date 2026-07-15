# Local development and operations

This guide is the detailed local runbook for EvidenceFlow V1. The application is a
single-machine portfolio/demo system for synthetic documents. It has no authentication
boundary, so keep the API and supporting services bound to loopback and never use real
customer, personal, regulated, or confidential data.

## Prerequisites

| Requirement | Notes |
| --- | --- |
| macOS or Linux | The checked-in workflow is local and does not use containers. |
| [`uv`](https://docs.astral.sh/uv/) | Installs Python 3.12.13 and the locked Python dependencies. |
| [Ollama](https://ollama.com/) | Must be running locally and able to serve the configured chat and embedding models. |
| Sufficient memory | The default chat model is `gemma4:12b-mlx`. |
| Node.js 22 | Needed only for frontend tests; the browser UI has no build step or runtime package installation. |

Run all repository commands below from the repository root.

## Fresh setup order

Use this sequence for a new clone. Commands labelled as separate terminals remain
running.

1. Install the exact Python version and synchronize the committed lockfile:

   ```bash
   uv python install 3.12.13
   make setup
   ```

   The checked-in defaults do not require a `.env` file. You can delete it if you do
   not need local operational overrides.

2. Start Ollama in a separate terminal if the desktop application or service is not
   already running:

   ```bash
   ollama serve
   ```

3. Pull the checked-in model aliases and confirm that Ollama can see them:

   ```bash
   ollama pull gemma4:12b-mlx
   ollama pull embeddinggemma
   ollama list
   ```

4. Optional but recommended: start MLflow in another terminal before the application
   so the named experiment is initialized when EvidenceFlow starts:

   ```bash
   make mlflow
   ```

5. Build the ignored local policy index, then run the explicit preparation gate:

   ```bash
   make rebuild
   make prepare
   ```

6. Start the API, workflow worker, and frontend. `make start` repeats the preparation
   gate so readiness cannot be skipped accidentally:

   ```bash
   make start
   ```

7. Open `http://127.0.0.1:8000/`. For a first review, upload the three PDFs from
   `eval/bundles/bundle_001/documents/`. Upload the PDFs only, not
   `ground_truth.json`.

`make start` occupies its terminal until it is stopped. MLflow is optional for an
ordinary review: if tracing is enabled but its server is unavailable, preparation emits
a warning and the review still runs. Evaluation requires healthy MLflow throughout.

## Model configuration and aliases

[`config/models.yaml`](../config/models.yaml) independently configures classification,
extraction, reporting, and embeddings. The checked-in defaults use
`gemma4:12b-mlx` for all three chat tasks and `embeddinggemma` with 768 dimensions for
policy retrieval. Model digests are pinned so preparation can detect an unexpected local
model behind a familiar tag.

`models.yaml` is the single source of truth for provider and model selection. It owns
each task's provider, model alias, digest, default endpoint, timeout, and generation
settings. Model aliases are deliberately not overridden through `.env`, so inspecting
one file shows exactly what the application will use.

Ollama aliases and available builds can differ by platform. If
`gemma4:12b-mlx` is unavailable under that exact name, install a local Gemma model that
supports the required JSON-schema structured output, then edit the relevant entries in
`models.yaml`. For one shared replacement, update all three chat tasks:

```yaml
models:
  classification:
    model: your-gemma-alias
  extraction:
    model: your-gemma-alias
  reporting:
    model: your-gemma-alias
```

Also set the matching digest and retain the other required fields in the real file.
Run `make prepare` and `make smoke` before relying on an alternative model. Smoke
testing exercises classification, extraction, reporting, and embeddings against the
configured Ollama runtime. Changing the embedding model is more involved: its
configured dimensions and identity must match the vectors it returns, and the policy
index must then be rebuilt.

`.env` is optional and reserved for operational settings such as local data paths,
upload limits, telemetry, or an `OLLAMA_BASE_URL` endpoint override. The complete list
is in [`../.env.example`](../.env.example). You can also export those values in your
shell instead of creating a file.

V1 implements only Ollama adapters. If a cloud adapter is added later, its
provider/model/deployment identity will still belong in `models.yaml`; API keys and
tokens must stay in environment variables or a secret store. The provider's preparation
check should verify that credentials exist and can authenticate, and that the configured
deployment is reachable, without printing secret values.

## Preparation, startup, and health

`make prepare` is the single fail-early readiness flow. It reports each check before
the application accepts work:

| Check | Failure behavior |
| --- | --- |
| Typed YAML configuration and supported provider adapters | Critical: exit non-zero. |
| Writable runtime storage and local database readiness | Critical: exit non-zero. |
| Provider connectivity, configured task models, and pinned digests | Critical: exit non-zero. |
| Policy-index presence and compatibility with policies/embedder | Critical: exit non-zero. |
| MLflow connectivity when tracing is enabled | Warning for interactive runtime; tracing degrades safely. |

Critical failures include a safe explanation and remediation without a stack trace or
credential value. `make doctor` remains as a compatibility alias for `make prepare`.
If tracing is explicitly disabled, preparation reports it as disabled rather than
unavailable.

Preparation deliberately uses lightweight provider inventory/authentication checks; it
does not load the 12B chat model and run inference on every restart. Run `make smoke`
after installing or changing a model to verify real structured-output and embedding
calls.

`make start` always runs the same preparation flow first. On success it starts FastAPI,
the durable single-worker queue, LangGraph checkpointing, and the static frontend at
`http://127.0.0.1:8000/`. It also applies pending business-database migrations during
application lifespan startup.

Evaluation is stricter than an interactive review. `make evaluate` first prepares the
application, then requires MLflow to remain healthy for the full benchmark. Disabled,
unreachable, or failed evaluation tracing makes the evaluation fail closed rather than
publishing an incompletely observed result.

After startup, inspect the safe dependency summary with:

```bash
curl --fail http://127.0.0.1:8000/health
```

Keep the application on `127.0.0.1`. V1 does not implement production authentication,
authorization, tenancy, or an internet-facing security boundary.

## MLflow tracing

The `make mlflow` command used during setup starts the local server on
`http://127.0.0.1:5001/`, with its SQLite tracking database under `data/` and artifacts
under `data/mlartifacts/`. Port 5001 avoids the port 5000 listener commonly used by
macOS AirPlay Receiver.

To inspect an interactive review:

1. Open the `evidenceflow` experiment.
2. Select **Traces**.
3. Open the `workflow.execute` root span.
4. Inspect child spans such as `document.process`, `ai.classification`,
   `ai.extraction`, `policy.retrieve`, and `ai.report`.

The spans record model/provider identity, correlation identifiers, counts, latency,
and provider-reported token usage when available. Document text, prompts, and source
excerpts are not logged by default. The separate `evidenceflow-evaluation` experiment
contains benchmark runs and per-bundle traces.

For a custom port, change both the server and application setting. Export the
application setting in the shell that starts EvidenceFlow, or put it in an optional
`.env` file if you want it to persist:

```bash
MLFLOW_PORT=5100 make mlflow
```

```dotenv
MLFLOW_TRACKING_URI=http://127.0.0.1:5100
```

The frontend convenience link always targets the checked-in port 5001, so open a
custom MLflow URL manually. `MLFLOW_HOST` also controls the Make target's bind host;
leave it on `127.0.0.1` for this local application.

To run ordinary reviews without tracing, export the following value (or place it in an
optional `.env`) and restart EvidenceFlow:

```bash
export EVIDENCEFLOW_MLFLOW_ENABLED=false
```

The equivalent optional `.env` entry is:

```dotenv
EVIDENCEFLOW_MLFLOW_ENABLED=false
```

Evaluation is deliberately fail-closed for telemetry and cannot run with MLflow
disabled or unavailable.

## Policy index

The policy index is a local sqlite-vec search structure over section-aware chunks from
the six Markdown files in [`../policies/`](../policies/). It lets a baseline query and
finding-specific queries retrieve stable policy evidence for report composition. It
does not implement the deterministic business rules in
[`config/review_rules.yaml`](../config/review_rules.yaml).

Run `make rebuild`:

- after every fresh clone, because generated index files are ignored by Git;
- after changing any file in `policies/`;
- after changing the embedding provider, model, digest, or dimensions; or
- after changing the index preprocessing, chunking profile, or related compatibility
  settings.

Changes to deterministic review rules or only the reporting chat model do not
invalidate the index. Rebuilding writes and validates a complete temporary database
before atomically replacing the active generation. The embedded manifest records the
provider/model identity, digest, dimensions, preprocessing and chunk settings, corpus
hash, counts, build identifier, and timestamp. Startup and retrieval refuse an
incompatible or stale generation with rebuild guidance.

## Sample review journeys

Each checked-in bundle is an independent synthetic company-review case. In the
frontend, select every PDF in that bundle's `documents/` directory. Do not upload the
adjacent `ground_truth.json`; it is an evaluator label rather than review evidence.

The star-marked rows are the most useful first demonstrations. `bundle_001` is the
happy path, `bundle_008` reliably exercises deterministic conflict review, and
`bundle_005` demonstrates an incomplete result.

| Bundle | Scenario | Expected reviewer journey |
| --- | --- | --- |
| **`bundle_001` ★** | Complete, consistent package | No review pause; final status `complete`. |
| `bundle_002` | Complete, consistent package | No review pause; final status `complete`. |
| `bundle_003` | Complete, consistent package | No review pause; final status `complete`. |
| `bundle_004` | Complete, consistent package | No review pause; final status `complete`. |
| **`bundle_005` ★** | Missing financial statement | Missing required evidence; final status `incomplete`. |
| `bundle_006` | Missing company extract | Missing required evidence; final status `incomplete`. |
| `bundle_007` | Missing application form | Missing required evidence; final status `incomplete`. |
| **`bundle_008` ★** | Registration-number conflict | Deterministic conflict pause; select/correct a cited value or leave it unresolved. |
| `bundle_009` | Registration-number conflict | Same conflict family with different synthetic facts. |
| `bundle_010` | Registration-number conflict | Same conflict family with different synthetic facts. |
| `bundle_011` | Revenue conflict | Revenue differs by more than the symmetric 2% tolerance; conflict review is required. |
| `bundle_012` | Revenue conflict | Same conflict family with different synthetic facts. |
| `bundle_013` | Revenue conflict | Same conflict family with different synthetic facts. |
| `bundle_014` | Company-name formatting variant | Case, punctuation, ampersand, and legal-form variants normalize to agreement. |
| `bundle_015` | Company-name formatting variant | Hyphenation, spacing, case, and legal-form punctuation normalize to agreement. |
| `bundle_016` | Ambiguous employee count | Labelled to require low-confidence field review. |
| `bundle_017` | Ambiguous attachment | Labelled to require classification review. |
| `bundle_018` | Unknown attachment | Complete package plus an irrelevant document classified as `unknown`. |
| `bundle_019` | Incomplete financial statement | Recognized financial statement lacks revenue; final status `incomplete`. |
| `bundle_020` | Human correction | Labelled to require a typed employee-count correction before validation. |

Low-confidence routing depends on the model's returned confidence. In the committed
real-model evaluation, `bundle_016`, `bundle_017`, and `bundle_020` did not pause even
though their labels expect review. This measured limitation is why `bundle_008` is the
recommended reliable demonstration of human decision handling.

While a review runs, the frontend polls every 1.5 seconds and displays the current
checkpoint-derived stage. Long local model calls can keep one stage active for several
minutes. When a review pauses, inspect each evidence link, provide exactly one decision
for every pending item, and resume. The final screen exposes the structured report and
JSON/Markdown downloads. The review ID in the URL hash restores the same local review
after a refresh.

## Make command reference

Run `make` or `make help` to print this command list from the Makefile.

| Command | When to use it |
| --- | --- |
| `make help` | Print all documented Make targets. |
| `make setup` | First clone or whenever `uv.lock` changes. |
| `make prepare` | Run every startup-readiness check and fail early on critical errors. |
| `make doctor` | Compatibility alias for `make prepare`. |
| `make mlflow` | Start local MLflow; required throughout evaluation. |
| `make rebuild` | Build or atomically replace the policy index. |
| `make start` | Run preparation, then start the API, worker, and frontend. |
| `make generate-data` | Deliberately regenerate all 20 deterministic PDF bundles. |
| `make evaluate` | Run the genuine 20-bundle local-model benchmark and write result artifacts. |
| `make smoke` | Exercise every configured real Ollama task adapter. |
| `make test-ollama` | Run opt-in Ollama adapter tests and the real-model workflow smoke test. |
| `make test` | Run all model-free Python and frontend tests. |
| `make frontend-check` | Run only the build-free frontend Node test suite. |
| `make lint` | Run Ruff over the repository. |
| `make typecheck` | Run strict mypy checks over `app/`. |
| `make check` | Run lock verification, setup, lint, typing, and all model-free tests as CI does. |

The MLflow target accepts `MLFLOW_HOST` and `MLFLOW_PORT` Make variables. Runtime
settings may be exported in the shell or stored in an optional ignored `.env`; use
[`../.env.example`](../.env.example) as the key/value reference. Provider/model identity
and task behavior always belong in [`config/models.yaml`](../config/models.yaml).

## Troubleshooting

### Preparation reports missing models or digest mismatches

Confirm that Ollama is running and inspect `ollama list`. Pull the exact configured
aliases again, or edit the relevant task definitions in `config/models.yaml` as
described above. A same-name digest mismatch is intentional protection against
silently running a different model;
align the local model with [`config/models.yaml`](../config/models.yaml) rather than
ignoring the warning.

### Preparation reports a missing, stale, or incompatible policy index

Confirm that the embedding model configured in `models.yaml` is available, then run
`make rebuild`. If an embedding-model change is intentional, verify its dimensions and
digest before rebuilding. Do not copy an index from a machine whose model digest or
policy corpus differs.

### MLflow is unavailable or an old checkout still targets port 5000

Start `make mlflow` before EvidenceFlow. The default tracking URI is
`http://127.0.0.1:5001`; port 5000 is commonly occupied by macOS AirPlay Receiver. Use
the paired custom-port settings above if 5001 is also occupied, or disable tracing for
ordinary reviews. Restart EvidenceFlow after changing the environment.

### Port 8000 is already in use

Identify the local listener before starting another application instance:

```bash
lsof -nP -iTCP:8000 -sTCP:LISTEN
```

Stop the stale EvidenceFlow process or the other local service, then run `make start`
again. Do not expose V1 by changing the bind host to a public interface.

### A sample upload is rejected

Upload between one and five digitally generated text PDFs. The defaults allow at most
10 MiB per file, 25 MiB per bundle, and 50 pages per PDF. Scans, encrypted files,
corrupt files, and effectively textless PDFs are intentionally rejected because V1 has
no OCR fallback. Select files from one bundle's `documents/` directory and exclude its
JSON label.

### A workflow stage appears to take a long time

Classification, extraction, and reporting are local model calls and can take minutes
on a laptop. The progress tracker advances at durable LangGraph node boundaries rather
than estimating percentages or per-document completion inside a node. Check the app
terminal for a safe failure message, verify Ollama is responsive, and inspect MLflow
latency spans when tracing is enabled.

### Tests cannot find Node.js

Node.js is not needed to run the UI, but `make test`, `make frontend-check`, and
`make check` invoke the frontend's Node test suite. Install Node.js 22 and retry; there
are no npm dependencies to install.
