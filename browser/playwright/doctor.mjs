import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, "../..");
const packageJson = JSON.parse(fs.readFileSync(path.join(here, "package.json"), "utf8"));
const configs = [
  "mcp.config.json",
  "mcp.config.ephemeral.json",
  "mcp.config.full.example.json",
  "mcp-client.example.json",
];

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

const major = Number(process.versions.node.split(".")[0]);
assert(Number.isInteger(major) && major >= 20, "Node.js 20+ required; found " + process.version);
assert(packageJson.engines?.node === ">=20", "package.json must declare Node.js >=20");
assert(packageJson.scripts?.["browser:mcp"]?.includes("@playwright/mcp@0.0.86"), "MCP version must remain pinned");
assert(packageJson.scripts?.["browser:mcp:ephemeral"]?.includes("@playwright/mcp@0.0.86"), "ephemeral MCP version must remain pinned");

for (const file of configs) {
  const config = JSON.parse(fs.readFileSync(path.join(here, file), "utf8"));
  if (file === "mcp-client.example.json") {
    assert(config.mcpServers?.["playwright-ai-determinism-engine"], "MCP server config missing");
    continue;
  }
  assert(config.browser?.browserName === "chromium", file + ": chromium baseline missing");
  assert(Array.isArray(config.capabilities) && config.capabilities.includes("core"), file + ": core capability missing");
  assert(config.allowUnrestrictedFileAccess === false, file + ": unrestricted file access must remain disabled");
  assert(config.sharedBrowserContext === false, file + ": shared context must remain disabled");
}

const policyPath = path.join(root, "orchestrator", "browser_policy.py");
assert(fs.existsSync(policyPath), "browser policy module missing");
console.log(
  "PASS: Node " + process.version + "; " + configs.length +
  " JSON configs; pinned @playwright/mcp@0.0.86; policy module present."
);
