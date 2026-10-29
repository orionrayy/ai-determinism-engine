# Genre-Matched Original vs Generated Document Benchmark

## Purpose

This protocol evaluates whether document-generation changes improve genre fit, specificity, readability and visual organization when compared with genuine public reference documents. It is **not** an AI detector and does not assign human/AI authorship from stylistic cues. Institutional hosting, extraction success and natural-sounding prose are separate facts.

## Evidence classes

Store a source record with the minimum evidence necessary to reproduce its retrieval: normalized source URL, final artifact URL, SHA-256 of the retrieved artifact, response status, retrieval timestamp, document type, estimated genre, source host, record/byline date and provenance-review outcome. Do not copy source prose into Git. Keep licenses or terms-of-use unresolved until reviewed; public access does not automatically grant redistribution rights.

Use these provenance states:
- `unreviewed_candidate`: discovered by search only.
- `source_record_confirmed`: institution/issuer record and artifact link correspond.
- `metadata_consistent`: record, author/date, title and file agree on inspection.
- `manual_review_pass`: a reviewer checked the source record and artifact type.
- `excluded`: wrong document, template, duplicate, access restriction, insufficient content, privacy concern or unverifiable provenance.

Do not equate `manual_review_pass` with proof of every sentence's authorship. Where provenance remains uncertain, label it as such.

## Original-corpus sampling

1. Count a document toward the 500-document acceptance gate only when the public artifact was retrieved and enough content was extracted, not when a search result or repository landing page was found.
2. Deduplicate by normalized source URL and retrieved file SHA-256. If separate PDFs are only cover, abstract, contents or reference-list fragments, mark them as fragments, not full documents.
3. Report results by genre. The initial inventory uses separate classes for nursing case reports/LP askep, PKL or internship reports, theses, academic papers, laboratory reports, KKN/fieldwork, community-service reports, activity accountability/LPJ, government performance reports, annual/financial reports, meeting minutes, technical audits, QA, production/operations, project closeout and spreadsheets.
4. Preserve a per-genre count and publish uncertainty for small groups. Do not aggregate workbook styling with prose statistics or force one report structure onto every genre.
5. For clinical case documents, use only public, appropriately de-identified institutional or published material. Exclude records containing unnecessary identifiers or content whose public disclosure appears questionable. Keep extracted source text out of the repository.

## Generated comparison set

Create a separate control corpus; never mix it with the originals. For each selected genre, prepare a short neutral task brief that supplies the purpose, audience, factual inputs, constraints and expected deliverables without copying paragraphs, unique wording or private details from a source document.

For each brief, save:
- `brief_id`, genre, anonymized factual inputs and required sections;
- model/provider and model identifier if exposed, generation date, prompt version, decoding settings where available;
- condition label, such as `baseline`, `genre_guided` or `human_edited`;
- output checksum, format and word/page/sheet counts;
- whether a human edited it and the scope of that edit.

Keep baseline and revised outputs paired to the same brief. Use at least three independent briefs per genre before drawing genre-level conclusions; more are preferable. Generated samples must be labeled as generated or edited controls, never presented as original human work.

## Measurements

### Text structure
Measure word, paragraph and sentence counts; median and 90th-percentile paragraph length; median sentence length and variation; heading hierarchy; numbered-section patterns; section presence; tables/list markers; and document-specific structure such as assessment/diagnosis/intervention/evaluation in nursing reports or formulas/validation/freeze panes in spreadsheets.

### Language patterns
Measure vocabulary diversity and repeated n-grams with length-aware denominators. Count recurring transitions and contrast frames (including “bukan ..., melainkan ...”) per 1,000 words, not just raw totals. Track generic openings/conclusions, vague claims, unsupported generalizations, repeated sentence scaffolds and stock phrases only as review signals. Check the surrounding context before classifying a phrase as formulaic. Ordinary contrast or transitions can be appropriate and should not be penalized automatically.

### Visual and document design
For PDFs, compare page count, text coverage, orientation, image-bearing pages and approximate font-size hierarchy. For DOCX, compare actual heading styles, section breaks and table geometry. For XLSX, inspect formula prevalence, number formats, tables, charts, merged cells, validations, conditional formatting, sheet dimensions and freeze panes. Add a human visual review for alignment, whitespace, hierarchy, scanability, page breaks, table fit and consistency; extracted metrics alone cannot judge design quality.

## Blinded quality review

Reviewers should receive randomized document IDs and not see the source/control label during scoring. Whenever feasible, use two reviewers and reconcile large disagreements. Score the actual task and genre requirements rather than “how human” a document sounds.

Recommended dimensions:
1. Genre conventions and required sections.
2. Accuracy and traceability of facts.
3. Specificity to the brief and absence of invented details.
4. Paragraph and sentence readability.
5. Natural variation without forced informality.
6. Concision without omitted reasoning.
7. Useful, non-redundant headings and transitions.
8. Tables, figures and lists where they help the reader.
9. Layout hierarchy, whitespace and page flow.
10. Consistency of terminology, dates, numbers and references.

## Analysis and reporting

- Compare medians and distributions within the same genre and matched task condition.
- Report sample sizes and missingness beside every statistic. Do not compare a 3-page abstract fragment with a 70-page thesis.
- Use normalized counts, paired comparisons and bootstrap confidence intervals where practical. Report effect sizes; avoid declaring success from a single phrase count.
- Separate source validity, structural completeness, writing quality, visual quality and factual accuracy into separate dimensions.
- Summarize recurring weaknesses as actionable editing rules only when supported by repeated evidence and reviewer agreement.
- Do not optimize for AI-detector scores, conceal AI assistance, fabricate lived experience or mimic the distinctive wording of a named author. The goal is better documents, not false provenance.

## Acceptance gates

- [ ] At least 500 public artifacts pass the full-content extraction gate.
- [ ] Every counted artifact has source URL, final artifact URL, content checksum, extraction status and estimated genre.
- [ ] Excluded, partial and failed records remain auditable and are not counted toward the target.
- [ ] Provenance and licensing review statuses are explicit; no unsupported blanket “human-authored” claim.
- [ ] Results, distributions and limitations are reported by genre and format.
- [ ] Generated controls are stored separately, clearly labeled and paired by task brief.
- [ ] Style cues are contextual review signals, not authorship verdicts.
- [ ] High-impact conclusions are checked against representative documents and a blinded human review.
