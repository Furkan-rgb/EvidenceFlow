.DEFAULT_GOAL := help

.PHONY: help setup doctor mlflow rebuild start generate-data evaluate smoke \
	test-ollama test frontend-check lint typecheck check

help: ## Show the available developer commands.
	@awk 'BEGIN {FS = ":.*## "; printf "EvidenceFlow developer commands:\n\n"} /^[a-zA-Z0-9_-]+:.*## / {printf "  %-16s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

setup: ## Install the locked Python environment and all development extras.
	uv sync --locked --all-extras --dev

doctor: ## Check Ollama models/digests, the policy index, and MLflow connectivity.
	uv run python -m app.cli doctor

mlflow: ## Start the local MLflow server at http://127.0.0.1:5000.
	uv run mlflow server \
		--host 127.0.0.1 \
		--port 5000 \
		--backend-store-uri sqlite:///data/mlflow.db \
		--default-artifact-root ./data/mlartifacts

rebuild: ## Atomically rebuild the policy index with the configured embedder.
	uv run python -m app.cli rebuild-policy-index

start: doctor ## Validate critical dependencies, then start the app and UI.
	uv run python -m app.cli run --host 127.0.0.1 --port 8000

generate-data: ## Regenerate the deterministic 20-bundle synthetic PDF corpus.
	uv run python -m app.cli generate-eval-data --output-dir eval/bundles --overwrite

evaluate: doctor ## Run the real 20-bundle evaluation (MLflow is mandatory).
	uv run python -m app.cli evaluate --bundles-dir eval/bundles --output-dir eval/results

smoke: ## Exercise every configured real Ollama task adapter.
	uv run python -m app.cli ollama-smoke

test-ollama: ## Run the opt-in Ollama integration test suite.
	EVIDENCEFLOW_RUN_OLLAMA_TESTS=1 uv run pytest -m ollama

test: ## Run model-free Python and frontend tests.
	uv run pytest -m "not ollama"
	npm test

frontend-check: ## Run the build-free frontend's Node test suite.
	npm test

lint: ## Run Ruff over the repository.
	uv run ruff check .

typecheck: ## Run strict mypy checks over the application.
	uv run mypy app

check: ## Run the complete model-free local/CI quality gate.
	uv lock --check
	$(MAKE) setup
	$(MAKE) lint
	$(MAKE) typecheck
	$(MAKE) test
