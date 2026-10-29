# Hallmark upstream and integration notes

- **Upstream:** https://github.com/Nutlope/hallmark
- **Imported directory:** `skills/hallmark/`
- **Pinned source commit:** `8f6969f8922caabde9de1784959eae11ddf68073`
- **Upstream commit date:** 2026-10-28
- **Skill version:** 2.0.0
- **Integration path:** `.agents/skills/hallmark/`
- **Included upstream files:** 182
- **License:** MIT; the original `LICENSE` is retained in this directory.

## Usage

Use in Codex as a project-level skill with `$hallmark <brief>`. For existing interfaces, only invoke the redesign/audit behavior when Hallmark is explicitly named, as documented by the upstream skill. The static linter can be run with:

```sh
node .agents/skills/hallmark/scripts/sloplint.mjs <file-or-directory>
```

The CI check invokes `scripts/install.mjs --codex --dry-run`, which validates the reference/theme package but does not install anything into the runner's home directory. Rendered browser checks require separately provisioned Chromium/Chrome and `puppeteer-core`.

## Operational and cost boundaries

- No external dependencies are installed during repository import or the CI validation job.
- The static linter and theme catalogue checks use Node.js and local files.
- Optional image generation calls Together AI and requires `TOGETHER_API_KEY`; this integration does not call that API or provision a key.
- The theme photo catalogue lists externally hosted Wikimedia Commons photographs with per-image licensing/credit information. Follow the upstream catalogue and retain required credits when a generated page uses them.
- The workspace skill does not bypass or auto-register with the orchestrator's runtime `SkillRegistry`; its normal trust, billing, digest, network, and permission gates remain separate.

The local copy preserves the upstream skill sources. For future upgrades, compare the upstream directory at the recorded commit with the intended newer version before updating the pin and rerunning the validation workflow.
