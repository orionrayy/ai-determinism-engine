import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, "../..");
const packageJson = JSON.parse(fs.readFileSync(path.join(here, "package.json"), "utf8"));
const configFiles = [
  "mcp.config.json",
  "mcp.config.ephemeral.json",
  "mcp.config.full.example.json",
];

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

const major = Number(process.versions.node.split(".")[0]);
assert(Number.isInteger(major) && major >= 20, "Node.js 20+ required; found " + process.version);
assert(packageJson.engines?.node === ">=20", "package.json must declare Node.js >=20");
assert(packageJson.scripts?.["browser:mcp"]?.includes("@playwright/mcp@0.0.86"), "MCP version must remain pinned");
assert(packageJson.scripts?.["browser:mcp:ephemeral"]?.includes("@playwright/mcp@0.0.86"), "ephemeral MCP version must remain pinned");

for (const file of configFiles) {
  const config = JSON.parse(fs.readFileSync(path.join(here, file), "utf8"));
  assert(config.browser?.browserName === "chromium", file + ": chromium baseline missing");
  assert(Array.isArray(config.capabilities) && config.capabilities.includes("core"), file + ": core capability missing");
  assert(config.allowUnrestrictedFileAccess === false, file + ": unrestricted file access must remain disabled");
  assert(config.sharedBrowserContext === false, file + ": shared context must remain disabled");
  assert(
    typeof config.outputDir === "string" && config.outputDir.startsWith("./.state/"),
    file + ": outputDir must remain inside the ignored local .state directory"
  );
}

const persistent = JSON.parse(fs.readFileSync(path.join(here, "mcp.config.json"), "utf8"));
assert(persistent.browser.isolated === false, "persistent profile must not be isolated");
assert(persistent.browser.userDataDir === "./.state/profile", "persistent profile must use the local .state directory");
assert(persistent.saveSession === true, "persistent profile must save session state");

const ephemeral = JSON.parse(fs.readFileSync(path.join(here, "mcp.config.ephemeral.json"), "utf8"));
assert(ephemeral.browser.isolated === true, "ephemeral profile must be isolated");
assert(!ephemeral.browser.userDataDir, "ephemeral profile must not use a persistent userDataDir");
assert(ephemeral.saveSession === false, "ephemeral profile must not save session state");

const windowsClient = JSON.parse(fs.readFileSync(path.join(here, "mcp-client.example.json"), "utf8"));
assert(
  windowsClient.mcpServers?.["playwright-ai-determinism-engine"]?.command === "powershell.exe",
  "Windows MCP client example must launch the PowerShell bootstrap script"
);
const posixClient = JSON.parse(fs.readFileSync(path.join(here, "mcp-client.posix.example.json"), "utf8"));
assert(
  posixClient.mcpServers?.["playwright-ai-determinism-engine"]?.command === "bash",
  "POSIX MCP client example must launch the shell bootstrap script"
);

const policyPath = path.join(root, "orchestrator", "browser_policy.py");
assert(fs.existsSync(policyPath), "browser policy module missing");
assert(fs.existsSync(path.join(here, ".gitignore")), "local browser state ignore rules missing");
console.log(
  "PASS: Node " + process.version + "; persistent/ephemeral profile policy; " +
  configFiles.length + " runtime configs; Windows/POSIX launch samples; pinned package."
);
