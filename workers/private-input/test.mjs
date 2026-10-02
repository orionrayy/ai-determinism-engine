import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

const worker = readFileSync(new URL("./src/index.js", import.meta.url), "utf8");
const durableObject = readFileSync(
  new URL("./src/private_input_object.js", import.meta.url),
  "utf8",
);
const config = readFileSync(new URL("./wrangler.jsonc", import.meta.url), "utf8");

assert.match(worker, /X-Orchestrator-Protocol/);
assert.match(worker, /ai-orchestrator\.private-input\/v2/);
assert.match(worker, /getByName\(envelope\.input_ref\)/);
assert.match(worker, /putInput\(envelope\)/);
assert.match(worker, /getByName\(match\[1\]\)\.getInput/);
assert.match(worker, /deleteInput/);
assert.doesNotMatch(worker, /ExecutionLease/);
assert.match(durableObject, /export class PrivateInput extends DurableObject/);
assert.match(durableObject, /CREATE TABLE IF NOT EXISTS inputs/);
assert.match(durableObject, /PRIMARY KEY/);
assert.match(durableObject, /setAlarm/);
assert.match(durableObject, /async alarm\(/);
assert.match(durableObject, /existing\.payload_json === payloadJson/);
assert.doesNotMatch(durableObject, /existing\.expires_at === envelope\.expires_at/);

assert.match(config, /"durable_objects"/);
assert.match(config, /"class_name": "PrivateInput"/);
assert.doesNotMatch(config, /ExecutionLease/);
assert.match(config, /"new_sqlite_classes"/);

console.log("private-input Worker static contract tests: OK");
