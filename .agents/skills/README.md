# Repository-scoped agent skills

Skills in this directory are checked into the repository so compatible coding agents can discover them when working in this project.

## Hallmark

- **Path:** [`hallmark/SKILL.md`](hallmark/SKILL.md)
- **Purpose:** Distinctive, responsive, accessible front-end/interface design and review.
- **Invoke in Codex:** `$hallmark <brief>`
- **Audit:** `$hallmark audit <path>`
- **Redesign:** `$hallmark redesign <path>` — use only when the request explicitly names Hallmark.
- **Origin:** [`Nutlope/hallmark`](https://github.com/Nutlope/hallmark), pinned source commit documented in `hallmark/UPSTREAM.md`.

## Document Craft

- **Path:** [`document-craft/SKILL.md`](document-craft/SKILL.md)
- **Purpose:** Academic papers, coursework, essays, lab/practicum and internship/PKL reports, proposals, and professional documents.
- **Routing:** Automatic for Indonesian/English document intents; verified guidance is attached to relevant drafting/review nodes after runtime authorization.
- **Writing standard:** Specific, natural prose; genuine author voice where supplied; evidence/citation integrity; no invented observations or personal experiences; no AI-detector-evasion tactics.

## Runtime catalogue

- **Explicit allowlist:** [`catalog.json`](catalog.json). A directory being present does not itself authorize a skill.
- **Authorization:** Runtime verifies each prompt file against its Git blob SHA, creates a `SkillManifest`, registers it with `SkillRegistry`, and authorizes trust, network and billing policy before prompt injection.
- **Scope:** Skill guidance is injected only into matching LLM nodes. Mixed document/UI tasks are routed per node. Both shipped skills are non-side-effecting and declared `free_public`; no network access is required for routing.

- **Path:** [`hallmark/SKILL.md`](hallmark/SKILL.md)
- **Purpose:** Design and review interfaces with explicit visual direction, responsive behavior, accessibility and anti-generic-design checks.
- **Invoke (Codex):** `$hallmark <brief>`
- **Audit:** `$hallmark audit <path>`
- **Redesign:** `$hallmark redesign <path>` — use only when the request explicitly names Hallmark.
- **Source/version:** [`Nutlope/hallmark`](https://github.com/Nutlope/hallmark), upstream commit `8f6969f8922caabde9de1784959eae11ddf68073`; details in [`hallmark/UPSTREAM.md`](hallmark/UPSTREAM.md).

The validation workflow checks references and the theme catalogue without installing the skill into the runner's home directory. Browser-rendered checks need a compatible local Chromium/Chrome plus `puppeteer-core`; those are optional and are not installed by this repository workflow. Together AI image generation is also optional and requires an explicit API key, so it is not part of the free-first default path.

The skill router is a prompt-context layer, not code execution. Existing safety, evidence, DAG, effect and repository contracts remain authoritative.
