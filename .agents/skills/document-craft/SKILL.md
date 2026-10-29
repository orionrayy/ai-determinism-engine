---
name: document-craft
description: Write, revise, research, and quality-check academic papers, essays, practicum/lab reports, internship reports, proposals, and professional documents in a natural, evidence-grounded voice.
version: 1.0.0
---

# Document Craft

Use this skill when a task produces or substantially edits a paper, assignment, essay, academic article, practical/lab report, internship/PKL report, research summary, proposal, memo, project report, or other long-form document. It complements Hallmark; it does not replace it. Hallmark is for interface design, not for writing a Word/PDF report.

## Operating rule

Produce writing that sounds like a careful person explaining something they understand. "Humanized" means clear, specific, natural, and sensitive to the author's actual voice—not intentional errors, fabricated personal experiences, or attempts to deceive AI-detection systems. Never promise an AI-detector score or optimize a document to bypass detectors.

Follow the user's assignment, rubric, source material, requested language, citation style, word/page limits, and formatting requirements first. Preserve the user's meaning and evidence. Do not overwrite a locked template or invent a required section just because a generic report template usually contains it.

## Before drafting

1. Read the full prompt, rubric, supplied images/files, existing draft, and relevant repository/document context. Build a compact requirement checklist and account for every question and deliverable.
2. Separate **known facts**, **verified source evidence**, **user-provided observations**, **reasonable analysis**, and **missing information**. Do not turn one category into another.
3. For research-based work, find sources that directly support the claims. Prefer primary sources, official documentation, peer-reviewed research, laws/standards from their issuing bodies, and original data where appropriate. Check title, author/organization, date, URL/DOI, and exact support. Never invent sources, quotations, page numbers, field observations, measurements, experimental results, interviewees, or reference entries.
4. Ask only for material that is genuinely essential. When a missing fact can be safely omitted, state the limitation or write around it. Never silently fill a missing observation with made-up details.

## Natural prose standard

- Use the language and register that fit the assignment and the author's supplied writing. In Indonesian, prefer direct, natural Bahasa Indonesia over inflated ceremonial prose; use technical terms when the subject requires them.
- Make each paragraph perform a job: explain a point, connect evidence to it, compare alternatives, interpret a result, or qualify a conclusion. Let paragraph length and sentence rhythm vary because the reasoning varies—not to simulate a human.
- Be concrete. Replace empty claims such as "sangat penting dalam era digital yang semakin berkembang pesat" with the actual problem, consequence, example, or evidence.
- Avoid repetitive paragraph openings and stock transitions. Do not mechanically start every paragraph with "Selain itu", "Di sisi lain", "Dengan demikian", or equivalent phrases. Use transitions only when the logical relationship calls for one.
- Avoid an automatic sequence of grand introduction, repeated restatement, generic advantages/disadvantages, and a conclusion that merely repeats the introduction. Use the structure that the actual question and evidence need.
- Avoid excessive colons, semicolons, em dashes, bold labels inside prose, nested bullet lists, symmetry for symmetry's sake, and overcompressed paragraphs. Use headings, lists, and tables only when they help the reader navigate or compare information.
- Do not add fabricated personal reflections, "I observed" statements, informal mistakes, or artificial quirks to make text look human. When the user provides genuine first-person notes, preserve their perspective and distinguish it from external evidence.
- Do not repeat the prompt as filler. Remove claims that sound polished but say nothing testable. Prefer precise verbs and nouns over adjective stacking.
- Keep a measured voice: not every result is "significant", every method "comprehensive", or every recommendation "strategic". Explain limits and counterevidence when they matter.

## Academic paper / coursework workflow

1. Extract the course question and each sub-question. Make a traceability checklist so no requested point is missed.
2. Establish the central argument or answer before writing the introduction. Organize sections around the reasoning and evidence, not a one-size-fits-all outline.
3. For each substantial factual or interpretive claim, make sure the source or supplied evidence actually supports it. Paraphrase faithfully and cite it. Direct quotations require exact wording and a location when the chosen style calls for one.
4. Keep citations and bibliography synchronized. Every in-text citation should map to a real reference; do not leave unused references unless the requested style requires them. Never fabricate DOI, URL, journal, publisher, page range, author, or year.
5. Make the analysis do more than summarize: explain why evidence matters, compare sources or cases, identify assumptions, note contradictions, and connect the finding back to the question.
6. In the final pass, check factual accuracy, prompt coverage, citation validity, continuity, terminology, word count, and the document's visual hierarchy.

## Practicum, laboratory, internship, and field reports

Use the requested institutional structure if provided. Otherwise derive only the sections needed from the actual task. Common elements may include objectives, date/location, tools/materials, method or activity sequence, results/observations, discussion, limitations, conclusion, and recommendations.

- Record what was actually done, observed, measured, or supplied. Do not manufacture lab values, dates, tools, screenshots, attendance, supervisor comments, work logs, or field results.
- If inputs are incomplete, separate documented facts from a proposed procedure or illustrative example. Never present a hypothetical example as completed practice.
- Describe the method clearly enough that a reader could understand or reproduce it. Match the analysis to the available data and state limitations rather than overselling the findings.
- Use a table when it makes measurements or event sequences easier to inspect; explain important patterns in prose. Label units, figure/table numbers, data origin, and any calculations.
- For internship/PKL reports, separate organization background, assigned duties, actual activities, outcomes, difficulties, and reflection. Do not invent company facts or achievements.

## Professional documents and formatting

Choose structure based on the job: persuade, explain, record, compare, instruct, or request. Keep conclusions and recommendations actionable and proportional to the evidence. For DOCX/PDF outputs, use coherent heading levels, readable body text, consistent spacing, page numbering where appropriate, accessible tables, figure captions, and a restrained cover page. Follow the user's required template and citation style. Do not add decorative title pages or boilerplate if they are not needed.

## Quality gate before delivery

Check these items against the source material and deliverable:

- [ ] Every requirement and sub-question is addressed.
- [ ] The main line of reasoning is understandable and the sections appear in a useful order.
- [ ] Claims, numbers, dates, quotes, observations, and references are grounded in supplied material or verified sources.
- [ ] No source, citation, quote, measurement, interview, or experience has been invented.
- [ ] Paragraphs are specific and non-repetitive; transitions and headings are not mechanically patterned.
- [ ] The prose fits the user's language, audience, and actual level of formality.
- [ ] The document follows requested structure, citation style, page/word limits, and file format.
- [ ] Limitations and uncertain facts are named instead of concealed.
- [ ] Any DOCX/PDF is opened or programmatically checked after generation when suitable tooling is available.

## Compact runtime guidance

For research nodes, prioritize verified source discovery and capture source metadata; do not draft unsupported conclusions. For drafting nodes, write from the requirement checklist and evidence notes. For critic/validation nodes, check omissions, fabricated claims, repetition, citation mismatches, formatting, and whether the prose actually answers the assignment. Keep the author’s actual voice where writing samples are supplied, and never claim the work reflects experiences the user did not provide.
