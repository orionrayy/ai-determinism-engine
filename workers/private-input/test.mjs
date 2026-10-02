import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

const worker = readFileSync(new URL("./src/index.js", import.meta.url), "utf8");
const durableObject = readFileSync(
  new URL("./src/private_input_object.js", import.meta.url),
  "utf8",
);
const leaseObject = readFileSync(
  new URL("./src/execution_lease_object.js", import.meta.url),
  "utf8",
);
const config = readFileSync(
  new URL("./wrangler.jsonc", import.meta.url),
  "utf8",
);

assert.match(worker, /X-Orchestrator-Protocol/);
assert.match(worker, /ai-orchestrator\.private-input\/v2/);
assert.match(worker, /getByName\(ref\)/);
assert.match(worker, /putInput\(envelope\)/);
assert.match(worker, /getInput\(ref\)/);
assert.match(worker, /deleteInput\(ref\)/);

assert.match(durableObject, /export class PrivateInput extends DurableObject/);
assert.match(durableObject, /CREATE TABLE IF NOT EXISTS inputs/);
assert.match(durableObject, /PRIMARY KEY/);
assert.match(durableObject, /setAlarm/);
assert.match(durableObject, /async alarm\(/);

assert.match(config, /"durable_objects"/);
assert.match(worker, /export \{ PrivateInput, ExecutionLease \}/);
assert.match(worker, /\/v1\/leases\/acquire/);
assert.match(worker, /\/v1\/leases\/reserve-attempt/);
assert.match(worker, /\/v1\/leases\/release/);

assert.match(leaseObject, /export class ExecutionLease extends DurableObject/);
assert.match(leaseObject, /retention_until/);
assert.match(leaseObject, /attempts INTEGER NOT NULL/);
assert.match(leaseObject, /async acquire\(/);
assert.match(leaseObject, /async reserve\(/);
assert.match(leaseObject, /async release\(/);
assert.match(leaseObject, /async alarm\(/);

assert.match(worker, /ExecutionLease/);
assert.match(worker, /\/v1\/leases\/acquire/);
assert.match(worker, /\/v1\/leases\/reserve-attempt/);
assert.match(worker, /\/v1\/leases\/release/);
assert.match(leaseObject, /export class ExecutionLease extends DurableObject/);
assert.match(leaseObject, /retention_until/);
assert.match(leaseObject, /attempts INTEGER NOT NULL/);
assert.match(leaseObject, /async acquire\(/);
assert.match(leaseObject, /async reserve\(/);
assert.match(leaseObject, /async release\(/);
assert.match(leaseObject, /async alarm\(/);
assert.match(config, /"class_name": "PrivateInput"/);
assert.match(config, /"class_name": "ExecutionLease"/);
assert.match(config, /"new_sqlite_classes"/);
assert.match(config, /"secrets"/);

console.log("private-input Worker static contract tests: OK");
