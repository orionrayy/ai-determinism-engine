import test from "node:test";
import assert from "node:assert/strict";
import {RecoveryWatchdog, runScheduledRecoveryWatchdog} from "./src/recovery_watchdog.mjs";

function makeStorage() {
  const values = new Map();
  return {
    async get(key) { return values.get(key); },
    async put(key, value) { values.set(key, structuredClone(value)); },
    async transaction(callback) {
      return callback({
        get: async (key) => values.get(key),
        put: async (key, value) => values.set(key, structuredClone(value)),
      });
    },
    values,
  };
}

function jsonResponse(value, status = 200) {
  return new Response(JSON.stringify(value), {
    status,
    headers: {"content-type": "application/json"},
  });
}

function makeEnv(storage, token = "test-token") {
  const watchdog = new RecoveryWatchdog({storage}, {});
  return {
    GITHUB_RECOVERY_TOKEN: token,
    RECOVERY_WATCHDOG: {
      idFromName(name) { return name; },
      get() {
        return {fetch: (request, init) => {
          const url = new URL(request);
          const body = JSON.parse(init.body);
          return watchdog.fetch(new Request(url, {
            method: init.method,
            headers: init.headers,
            body: JSON.stringify(body),
          }));
        }};
      },
    },
  };
}

function runRecord(nowSeconds, overrides = {}) {
  const createdAt = new Date((nowSeconds - 3600) * 1000).toISOString();
  return {
    id: 12345,
    event: "schedule",
    head_branch: "main",
    status: "completed",
    conclusion: "success",
    created_at: createdAt,
    run_started_at: createdAt,
    ...overrides,
  };
}

test("fails closed when the GitHub dispatch secret is absent", async (t) => {
  const previousFetch = globalThis.fetch;
  let calls = 0;
  globalThis.fetch = async () => { calls += 1; return jsonResponse({workflow_runs: []}); };
  t.after(() => { globalThis.fetch = previousFetch; });
  const result = await runScheduledRecoveryWatchdog({scheduledTime: Date.now()}, {});
  assert.equal(result.status, "disabled");
  assert.equal(calls, 0);
});

test("does not dispatch when the latest scheduled run is fresh", async (t) => {
  const previousFetch = globalThis.fetch;
  const now = Math.floor(Date.now() / 1000);
  let calls = 0;
  globalThis.fetch = async () => {
    calls += 1;
    return jsonResponse({workflow_runs: [runRecord(now, {
      created_at: new Date((now - 300) * 1000).toISOString(),
      run_started_at: new Date((now - 300) * 1000).toISOString(),
    })]});
  };
  t.after(() => { globalThis.fetch = previousFetch; });
  const result = await runScheduledRecoveryWatchdog({scheduledTime: now * 1000}, makeEnv(makeStorage()));
  assert.equal(result.status, "fresh_schedule");
  assert.equal(calls, 1);
});

test("dispatches recovery once for a stale schedule and carries deterministic event identity", async (t) => {
  const previousFetch = globalThis.fetch;
  const now = Math.floor(Date.now() / 1000);
  const storage = makeStorage();
  const requests = [];
  globalThis.fetch = async (url, init) => {
    requests.push({url: String(url), init});
    if (String(url).includes("/actions/workflows/")) {
      return jsonResponse({workflow_runs: [runRecord(now)]});
    }
    return new Response(null, {status: 204});
  };
  t.after(() => { globalThis.fetch = previousFetch; });
  const result = await runScheduledRecoveryWatchdog({scheduledTime: now * 1000}, makeEnv(storage));
  assert.equal(result.status, "dispatch_requested");
  assert.equal(requests.length, 2);
  const sent = JSON.parse(requests[1].init.body);
  assert.equal(sent.event_type, "orchestrator.recovery");
  assert.equal(sent.client_payload.event_id, "cloudflare-watchdog:12345");
  assert.equal(storage.values.get("dispatch_state").status, "sent");
});

test("does not recover a recent in-progress scheduled run", async (t) => {
  const previousFetch = globalThis.fetch;
  const now = Math.floor(Date.now() / 1000);
  let calls = 0;
  globalThis.fetch = async () => {
    calls += 1;
    return jsonResponse({workflow_runs: [runRecord(now, {
      status: "in_progress",
      created_at: new Date((now - 120) * 1000).toISOString(),
      run_started_at: new Date((now - 120) * 1000).toISOString(),
    })]});
  };
  t.after(() => { globalThis.fetch = previousFetch; });
  const result = await runScheduledRecoveryWatchdog({scheduledTime: now * 1000}, makeEnv(makeStorage()));
  assert.equal(result.status, "active_run");
  assert.equal(calls, 1);
});

test("backs off after a repository-dispatch failure instead of hammering GitHub", async (t) => {
  const previousFetch = globalThis.fetch;
  const now = Math.floor(Date.now() / 1000);
  const storage = makeStorage();
  let calls = 0;
  globalThis.fetch = async (url) => {
    calls += 1;
    if (String(url).includes("/actions/workflows/")) return jsonResponse({workflow_runs: [runRecord(now)]});
    return jsonResponse({message: "Resource not accessible by integration"}, 403);
  };
  t.after(() => { globalThis.fetch = previousFetch; });
  const env = makeEnv(storage);
  const result = await runScheduledRecoveryWatchdog({scheduledTime: now * 1000}, env);
  assert.equal(result.status, "dispatch_failed");
  assert.equal(storage.values.get("dispatch_state").status, "retryable");
  const second = await runScheduledRecoveryWatchdog({scheduledTime: (now + 60) * 1000}, env);
  assert.equal(second.status, "dispatch_suppressed");
  assert.equal(calls, 3);
});

test("durable claim blocks duplicate dispatches and releases after failure backoff", async () => {
  const storage = makeStorage();
  const watchdog = new RecoveryWatchdog({storage}, {});
  const post = (path, body) => watchdog.fetch(new Request("https://watchdog.internal" + path, {
    method: "POST",
    headers: {"content-type": "application/json"},
    body: JSON.stringify(body),
  }));
  const first = await (await post("/claim", {key: "schedule-gap:1", now: 1000})).json();
  assert.equal(first.allowed, true);
  const duplicateInflight = await (await post("/claim", {key: "schedule-gap:1", now: 1001})).json();
  assert.equal(duplicateInflight.allowed, false);
  const failed = await (await post("/finish", {
    key: "schedule-gap:1", claim_id: first.claim_id, ok: false, now: 1002,
  })).json();
  assert.equal(failed.status, "retryable");
  const inCooldown = await (await post("/claim", {key: "schedule-gap:1", now: 1100})).json();
  assert.equal(inCooldown.allowed, false);
  const afterCooldown = await (await post("/claim", {key: "schedule-gap:1", now: 1400})).json();
  assert.equal(afterCooldown.allowed, true);
});

test("a new scheduled-run identity is eligible after the previous dispatch", async () => {
  const storage = makeStorage();
  const watchdog = new RecoveryWatchdog({storage}, {});
  const post = (path, body) => watchdog.fetch(new Request("https://watchdog.internal" + path, {
    method: "POST",
    headers: {"content-type": "application/json"},
    body: JSON.stringify(body),
  }));
  const first = await (await post("/claim", {key: "schedule-gap:1", now: 1000})).json();
  await post("/finish", {
    key: "schedule-gap:1", claim_id: first.claim_id, ok: true, now: 1001, cooldown_seconds: 1800,
  });
  const second = await (await post("/claim", {key: "schedule-gap:2", now: 1002})).json();
  assert.equal(second.allowed, true);
});
