# Preliminary source-structure spot checks

**Status:** exploratory notes from public-source extraction previews; not the final corpus analysis. These observations establish useful questions for the full audit. They do not claim that every listed document was manually inspected page-by-page or that authorship has been independently proven. Source bodies are not stored in this repository.

## Observed genre-specific structures

### Official performance report

The KKP-hosted performance report preview includes an institutional covering letter before the report itself. Within the report, front matter includes a foreword and table of contents, followed by chapters for organizational background, performance planning/indicators, performance accountability and closing material. A budget-realization section appears inside performance accountability. This is a **compound artifact**: a classifier should not mistake the lead-in letter for the whole report, and section metrics should operate on the report body where possible.

Source: [KKP — 2025 performance report PDF](https://kkp.go.id/download-pdf-akuntabilitas-kinerja/akuntabilitas-kinerja-pelaporan-kinerja-laporan-tahunan-kinerja-triwulan-iv-tahun-2025.pdf)

### Government meeting minutes

The BPOM minutes preview begins with document control fields and then identifies meeting subject, place, date, speakers and participants. The discussion is organized around the agenda and includes a record of remarks and the regulatory question being discussed. The extracted file also contains an electronic-signature certification note. Unlike a thesis, this genre does not need a generic research-methods chapter; metadata and chronological accountability are central.

Source: [BPOM — public consultation minutes PDF](https://simppu.pom.go.id/berkas_rancangan/o7kCsmte4TqzhbDiXBNCKOzsBObXpuKBQxmUl0Fk.pdf)

### Undergraduate thesis

Two university repository examples show that the front matter is a meaningful part of the document: title page, approval/sign-off, originality/authenticity statement, publication permission, abstract, contents and list pages can precede the main chapters. One preview includes a long acknowledgements section and religious/motto content. The PDF text extraction shows malformed glyphs in a non-Latin passage; treat extraction corruption separately from the author's writing quality. Do not let front-matter language dominate style measurements of the analytical body.

Sources: [UIN Maulana Malik Ibrahim — accounting thesis PDF](https://etheses.uin-malang.ac.id/67334/1/200502110101.pdf) · [Nusa Putra University — computer-science thesis PDF](https://repository.nusaputra.ac.id/id/eprint/1382/1/NOVA%20TEGUH%20KRISDWIYANTO%20.pdf)

### Nonprofit annual report

The BOS Foundation annual-report preview uses a bilingual contents structure that follows program areas, includes page references, and separates program activity from fundraising, communications, partner coordination and financial information. The FMCH Indonesia report is more narrative-led and foregrounds organization coverage, quantified outputs, executive summary, program descriptions, challenges and forward plans. These structures should be compared as **annual reports**, not flattened into a universal essay outline.

Sources: [BOS Foundation — 2024 annual report PDF](https://www.orangutan.or.id/cfind/source/files/annual-reports/bosf-annual-report-2024.pdf) · [FMCH Indonesia — 2024 annual report PDF](https://fmch-indonesia.org/wp-content/uploads/2025/01/Annual-Report-2024.pdf)

### Technical verification and project closure

The NRC-hosted verification and validation report preview shows a document identifier, revision, issue date, author/reviewer/approval fields and a change-history table, followed by a numbered technical contents list. Sections include purpose, scope, tasks performed, results, anomalies, quality metrics, safety metrics, risks, recommendations, references and attachments. This is a concrete record of traceability and revision control; it should not be judged by the conventions of a student paper.

The University of Edinburgh's project closure page uses a relatively concise project-specific structure: summary, background, scope and deliverables/activities. Its sections describe systems, responsibilities and migration/testing requirements rather than padding the page with broad introductory claims.

Sources: [NRC — V&V implementation phase summary report PDF](https://www.nrc.gov/docs/ML1606/ML16061A468.pdf) · [University of Edinburgh — project closure report](https://www.projects.ed.ac.uk/project/has002/closure-report)

## Implications for the analyser

1. **Separate front matter and body when measurable.** Cover pages, approval pages, abstract-only PDFs and chapter fragments must not count as full documents just because they have recognizable headings.
2. **Treat official boilerplate as genre-specific.** Formal minutes and government reports naturally use fixed field labels, standard salutations and recurring terms. Their raw frequency does not itself imply AI authorship.
3. **Score section fit by genre.** A minute record should be evaluated for participants, discussion, decisions and follow-up; a thesis for its research structure; an annual report for programs, outcomes and financial/accountability sections; a technical report for revision control, methods, results, risks and recommendations.
4. **Keep OCR/extraction quality separate from writing quality.** Broken character sequences, missing tables and reading-order errors can distort lexical and paragraph metrics. Such artifacts should be flagged and excluded from affected language measures.
5. **Use denominator-aware comparisons.** Compare repeated phrases, transitions, punctuation and n-grams per 1,000 words and within the same genre. A formal stock phrase can be conventional rather than a defect.
6. **Preserve concrete evidence rather than imitate voice.** The practical findings are structural: front matter has a purpose, headings track real work, tables preserve traceability, scope is specific, and evidence and outcomes are named. Do not copy source sentences or distinctive author style.

## Outstanding verification

All links above were retrievable through the research extraction path at spot-check time, and the extracted titles/structures align with their source records or issuer domains. A reviewer still needs to inspect full rendered pages, confirm content completeness, assess source terms/licensing and record the formal provenance state. These notes are not a substitute for the 500-document acceptance report or a genre-matched comparison against generated controls.
