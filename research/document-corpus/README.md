# Original Document Corpus Research

## Purpose

Build an auditable, genre-specific reference corpus for improving the structure, language, and visual organization of original Indonesian academic, professional, operational, and production documents. The target is at least 500 public source artifacts that pass access and content checks, with a report describing conventions by genre.

## Source inventory

The discovery batch JSONL files contain URL-level discovery records from targeted searches. At this checkpoint, each record is a candidate, not a fully validated document. Records preserve title, author/date when discovered, URL, estimated genre, host, candidate provenance class, artifact signal, and compact structural indicators. Search excerpts are deliberately not stored; source document text is not mirrored into this repository.

Candidate categories include nursing case reports (LP/asuhan keperawatan), internship/PKL reports, theses/final projects, academic papers, laboratory reports, fieldwork/KKN, event and activity accountability reports, community-service reports, government performance reports, annual reports, technical/QA reports, production/project reports, formal minutes, proposals and XLS/XLSX workbooks.

## Validation and analysis rule

A document is counted as fully analyzed only after a public artifact is retrieved and enough text or workbook structure is extracted. A repository landing page alone does not count as a complete document. The audit may follow one direct attachment link from an institutional record. At least 300 extracted words are required for text documents; spreadsheets require a readable workbook with nontrivial populated cells. Scanned documents without machine-readable text are flagged as partial.

The audit extracts:
- PDF page counts, page text coverage, page orientation, font-size distribution and image-bearing pages;
- DOCX paragraph counts, heading styles, sections and table dimensions;
- XLSX sheet dimensions, populated/formula/date cells, merged cells, tables, charts, conditional formatting, validation rules and freeze panes;
- section presence, paragraph/sentence length distributions, vocabulary diversity, repeated four-gram rate, heading/uppercase/numbered-section signals, table-like lines, punctuation profiles and recurring transitions/contrast phrases;
- stable URL and content checksums, access outcomes, format, estimated genre and source domain.

The report aggregates signals by genre. Phrase patterns (including contrast formulas such as “bukan ..., melainkan ...”) are descriptive review cues only; they do not establish AI authorship. The aim is higher quality, factuality, natural language, useful visual layout and genre fit, not detector evasion or fabricated human experiences.

## Provenance and rights

Sources prioritize official repositories, universities, government agencies, issuers and public professional organizations. Automatic URL/access validation cannot prove authorship by itself. The corpus still needs human spot-checking of repository records, bylines, dates, source rights and document type. The tool stores no source body text. It does not run OCR, bypass logins/paywalls, or use private/confidential patient records. Public clinical case reports can still contain sensitive details; structural research should not redistribute their text, and questionable sources must be excluded.

## Running the audit

Install Python 3.12, pymupdf and openpyxl, then run:

    python research/document-corpus/tools/audit_corpus.py --self-test
    python research/document-corpus/tools/audit_corpus.py --target 500 --max-attempts 850 --delay 0.15

The GitHub Actions workflow runs the same audit on the dedicated research branch, commits derived analysis files under results/, uploads an artifact and fails its acceptance gate if fewer than 500 documents pass the full-content extraction gate. That failure means more sources or extraction work is required.

## Current checkpoint

- 898 unique source candidates discovered across six batches.
- Candidate filter, downloader, PDF/DOCX/XLSX/HTML extractors and report aggregation are implemented in the audit tool.
- Acceptance target: 500 full-document analyses.
- Manual authenticity/licence review and visual quality control remain required; neither is inferred from successful extraction.
