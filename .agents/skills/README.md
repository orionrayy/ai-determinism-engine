# Repository-scoped agent skills

Skills in this directory are checked into the repository so compatible coding agents can discover them when working in this project.

## Hallmark

- **Path:** [`hallmark/SKILL.md`](hallmark/SKILL.md)
- **Purpose:** Design and review interfaces with explicit visual direction, responsive behavior, accessibility and anti-generic-design checks.
- **Invoke (Codex):** `$hallmark <brief>`
- **Audit:** `$hallmark audit <path>`
- **Redesign:** `$hallmark redesign <path>` — use only when the request explicitly names Hallmark.
- **Source/version:** [`Nutlope/hallmark`](https://github.com/Nutlope/hallmark), upstream commit `8f6969f8922caabde9de1784959eae11ddf68073`; details in [`hallmark/UPSTREAM.md`](hallmark/UPSTREAM.md).

The validation workflow checks references and the theme catalogue without installing the skill into the runner's home directory. Browser-rendered checks need a compatible local Chromium/Chrome plus `puppeteer-core`; those are optional and are not installed by this repository workflow. Together AI image generation is also optional and requires an explicit API key, so it is not part of the free-first default path.

This directory does not grant runtime permissions to the orchestrator. Workflow execution remains subject to the canonical skill manifest and authorization controls.
