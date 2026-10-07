-- Reference PostgreSQL queue schema for enterprise-runtime adapter.
CREATE TABLE IF NOT EXISTS orchestrator_tasks (
    task_id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    dedupe_key TEXT NOT NULL UNIQUE,
    status TEXT NOT NULL CHECK (status IN ('queued','processing','completed','failed','dead')),
    attempt INTEGER NOT NULL DEFAULT 0,
    max_attempts INTEGER NOT NULL DEFAULT 8 CHECK (max_attempts > 0),
    available_at BIGINT NOT NULL,
    priority INTEGER NOT NULL DEFAULT 0,
    partition TEXT NOT NULL DEFAULT 'default',
    worker_id TEXT,
    claim_id TEXT,
    lease_expires_at BIGINT,
    last_error TEXT NOT NULL DEFAULT '',
    created_at BIGINT NOT NULL,
    updated_at BIGINT NOT NULL
);

CREATE INDEX IF NOT EXISTS orchestrator_tasks_ready_idx
    ON orchestrator_tasks (status, available_at, priority DESC, created_at);
CREATE INDEX IF NOT EXISTS orchestrator_tasks_lease_idx
    ON orchestrator_tasks (status, lease_expires_at);
CREATE INDEX IF NOT EXISTS orchestrator_tasks_tenant_idx
    ON orchestrator_tasks (tenant_id, status);
