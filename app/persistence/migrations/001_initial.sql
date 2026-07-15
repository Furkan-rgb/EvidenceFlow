PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS reviews (
    review_id TEXT PRIMARY KEY,
    thread_id TEXT NOT NULL UNIQUE,
    status TEXT NOT NULL CHECK (status IN ('processing', 'needs_review', 'completed', 'failed')),
    report_status TEXT,
    revision INTEGER NOT NULL DEFAULT 0,
    snapshot_json TEXT NOT NULL DEFAULT '{}',
    error_json TEXT,
    resume_count INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS documents (
    document_id TEXT PRIMARY KEY,
    review_id TEXT NOT NULL REFERENCES reviews(review_id) ON DELETE CASCADE,
    filename TEXT NOT NULL,
    artifact_id TEXT NOT NULL UNIQUE,
    sha256 TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    content_type TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_documents_review ON documents(review_id);

CREATE TABLE IF NOT EXISTS review_items (
    review_item_id TEXT NOT NULL,
    review_id TEXT NOT NULL REFERENCES reviews(review_id) ON DELETE CASCADE,
    item_type TEXT NOT NULL,
    state TEXT NOT NULL CHECK (state IN ('pending', 'decided')),
    fingerprint TEXT,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (review_id, review_item_id)
);
CREATE INDEX IF NOT EXISTS ix_review_items_pending
    ON review_items(review_id, state);

CREATE TABLE IF NOT EXISTS review_decisions (
    decision_id TEXT PRIMARY KEY,
    review_id TEXT NOT NULL REFERENCES reviews(review_id) ON DELETE CASCADE,
    review_item_id TEXT NOT NULL,
    action TEXT NOT NULL,
    value_json TEXT,
    selected_field_id TEXT,
    actor TEXT NOT NULL,
    decided_at TEXT NOT NULL,
    UNIQUE (review_id, review_item_id),
    FOREIGN KEY (review_id, review_item_id)
        REFERENCES review_items(review_id, review_item_id)
);

CREATE TABLE IF NOT EXISTS reports (
    review_id TEXT PRIMARY KEY REFERENCES reviews(review_id) ON DELETE CASCADE,
    report_json TEXT NOT NULL,
    markdown TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS workflow_jobs (
    job_id TEXT PRIMARY KEY,
    review_id TEXT NOT NULL REFERENCES reviews(review_id) ON DELETE CASCADE,
    kind TEXT NOT NULL CHECK (kind IN ('start', 'resume', 'recover')),
    status TEXT NOT NULL CHECK (status IN ('queued', 'running', 'completed', 'failed')),
    payload_json TEXT NOT NULL DEFAULT '{}',
    attempts INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_workflow_jobs_queue
    ON workflow_jobs(status, created_at);

CREATE TABLE IF NOT EXISTS review_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    review_id TEXT NOT NULL REFERENCES reviews(review_id) ON DELETE CASCADE,
    event_type TEXT NOT NULL,
    payload_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_review_events_review
    ON review_events(review_id, event_id);
