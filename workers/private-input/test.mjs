import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

const worker = readFileSync(new URL("./src/index.js", import.meta.url), "utf8");
const durableObject = readFileSync(
  new URL("./src/private_input_object.js", import.meta.url),
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
assert.match(config, /"class_name": "PrivateInput"/);
assert.match(config, /"storage": "sqlite"/);
assert.match(config, /"secrets"/);

console.log("private-input Worker static contract tests: OK");
