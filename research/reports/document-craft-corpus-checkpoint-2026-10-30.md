# Laporan checkpoint riset Hallmark — 30 Oktober 2026

## Status yang dapat dibuktikan

Korpus saat ini memiliki **10 dokumen individual yang lolos gate provenance dan hak penggunaan** dari sasaran 500. Status target: `INSUFFICIENT_EVIDENCE`; sisa 490. Sampel masih kecil dan tidak cukup untuk menyimpulkan kualitas seluruh genre atau peningkatan Document Craft.

| Genre | Dokumen lolos gate |
|---|---:|
| LP/asuhan keperawatan | 0 |
| Laporan PKL/magang | 0 |
| Skripsi/tesis/disertasi | 2 |
| Makalah/artikel ilmiah | 3 |
| Laporan praktikum | 0 |
| Laporan penelitian | 0 |
| Proposal | 0 |
| Laporan kegiatan | 0 |
| Laporan kerja/profesional | 2 |
| Operasional/administratif | 0 |
| Workbook Excel | 2 |
| Genre lain — panduan profesional/operasional | 1 |

Registry sumber berisi 11 direktori penemuan (10 entri direktori jurnal dari DOAJ dan Portal Satu Data Indonesia). Jumlah itu bukan jumlah dokumen. Hak penggunaan diperiksa per dokumen; label lisensi tingkat jurnal dalam registry tidak otomatis diterapkan ke dokumen individual. Manifest memuat 15 catatan: 10 eligible, 2 partial (PKL dan KIAN), 1 artikel nursing case yang masih pending privacy review, 1 lead XLSX metadata-only, dan 1 laporan riset yang dikecualikan karena konflik syarat lisensi. Lima catatan non-eligible itu **tidak dihitung** ke angka 10/500.

## Sampel yang sudah diverifikasi

**The Effectiveness of Indonesian Government Assistance in Handling and Breaking Covid-19 Chains** — Tessa Septy Dynesia, *The Indonesian Journal of International Clinical Legal Education*, 2(3), 245–256 (2020), DOI [10.15294/ijicle.v2i3.38417](https://doi.org/10.15294/ijicle.v2i3.38417).

- Halaman penerbit menampilkan judul, penulis, tanggal publikasi, DOI dan pranala full text: https://journal.unnes.ac.id/sju/iccle/article/view/38417
- Kebijakan hak cipta jurnal menyebut penulis mempertahankan hak cipta dan artikel berada di bawah CC BY-SA 4.0: https://journal.unnes.ac.id/sju/iccle/copyrights
- PDF resmi 12 halaman menyebutkan lisensi CC BY-SA 4.0 dan pernyataan orisinalitas penulis. PDF diinspeksi secara visual pada halaman pertama, halaman isi awal, serta halaman terakhir; URL sumbernya tercatat di manifest.
- Tidak ada salinan PDF, teks penuh, alamat surel penulis, atau data pasien yang dimasukkan ke repositori. Yang disimpan hanya metadata bibliografis dan fitur struktur.
- Keaslian sebagai publikasi individual memiliki bukti kuat, tetapi **keakuratan setiap klaim faktual belum diverifikasi secara independen**.

## Analisis struktur manual: tiga artikel ilmiah dan satu panduan profesional

Bagian awal menempatkan label jenis artikel, judul, penulis, afiliasi, abstrak/uraian awal, kata kunci, DOI, riwayat pengiriman/revisi/penerimaan dan lisensi. Bagian isi menggunakan heading bernomor Romawi, paragraf utama dengan alignment rata kiri-kanan, butir kebijakan bernomor di dalam uraian, serta catatan sumber bernomor di bagian bawah halaman. Daftar referensi ditempatkan setelah kesimpulan dan memuat kombinasi entri bibliografis, URL berita, rincian jurnal, serta tahun. Bagian akhir mencakup pernyataan konflik kepentingan, pendanaan, orisinalitas/etika publikasi, dan format sitasi yang disarankan.

Secara visual, halaman pembuka menggunakan judul dan identitas penulis terpusat, huruf serif, margin cukup lebar, QR code, dan blok metadata publikasi di kaki halaman. Halaman isi memiliki running title/jurnal, nomor halaman, serta catatan kaki bertipe lebih kecil. Pada halaman akhir, label deklarasi dibedakan menggunakan teks tebal berwarna biru. Temuan ini dicatat sebagai karakteristik sampel ini, bukan ketentuan universal artikel ilmiah.

Pembacaan heading menemukan `I. INTRODUCTION`, `II. GOVERNMENT POLICY ON HANDLING COVID-19`, `III. GOVERNMENT ASSISTANCE THAT HAS BEEN FEATURED BY THE COMMUNITY`, `V. CONCLUSION`, dan `VI. REFERENCES`; pemindaian teks tidak menemukan heading IV. Ini merupakan **anomali kandidat yang perlu pemeriksaan editorial lebih lanjut**, bukan bukti bahwa semua artikel harus menggunakan penomoran tertentu.

Dokumen ini juga menunjukkan beberapa konstruksi kalimat dan pilihan istilah yang perlu ditinjau, dan ada klaim angka/tanggal yang patut dibandingkan dengan sumber primer. Temuan tersebut belum diperlakukan sebagai vonis faktual sebelum pemeriksaan sumber independen. Sampel dipakai untuk mempelajari struktur dokumen nyata serta membentuk kasus quality-control; bukan dijadikan model kualitas tulisan yang ideal.

Analisis fitur terstruktur berada di `research/corpus/features.jsonl`; metadata dan bukti provenance ada di `research/corpus/manifest.jsonl`.

## Batasan dan langkah berikutnya

1. Menambah dokumen berlisensi jelas dari sumber primer, tanpa menghitung halaman direktori, template, duplikat, atau dokumen yang hanya diketahui metadata-nya.
2. Mengumpulkan sampel beberapa keluarga sumber per genre agar pola institusional dapat dipisahkan dari kebiasaan penulis.
3. Menginspeksi struktur dan isi dokumen berikutnya secara independen; khusus fakta, sitasi, dan angka perlu pemeriksaan terhadap sumber pembanding.
4. Baru setelah cukup korpus per genre, membuat baseline Document Craft dengan input yang sebanding, membekukan rubrik, membandingkan secara blind, dan memasukkan cacat yang terbukti ke regression test.
5. Uji visual DOCX/PDF/Excel hasil generator masih belum dilakukan. Inspeksi satu PDF sumber tidak boleh dilaporkan sebagai validasi visual output Hallmark.

Belum ada klaim peningkatan empiris mutu Document Craft.


## Pipeline discovery metadata yang ditambahkan

- `research/corpus/harvest_doaj_candidates.py` menggunakan endpoint OAI-PMH artikel publik DOAJ dengan request berukuran terbatas, timeout, batas halaman, delay antar-request, filter ISSN dengan validasi checksum, resumption token, deduplikasi ID kandidat, dan penulisan atomik.
- Harvester menyimpan metadata bibliografis dan tautan artikel; tidak menyimpan abstrak atau mengunduh PDF. Setiap kandidat ditandai `corpus_eligible=false`, lisensi konten artikel belum diverifikasi, dan tidak pernah dipromosikan otomatis ke manifest penelitian.
- DOAJ menyatakan metadata artikel/jurnal yang diterbitkannya tersedia dengan waiver CC0, tetapi lisensi konten artikel yang ditunjuk harus diperiksa secara terpisah. Referensi: https://doaj.org/terms/ dan https://doaj.org/docs/oai-pmh/.
- Uji pipeline memakai fixture XML dan mock jaringan untuk memeriksa ISSN, parsing, hak penggunaan, token pagination, idempotensi dan mode dry-run. **Live harvest belum dijalankan**, sehingga jumlah metadata kandidat yang benar-benar diambil dari feed pada checkpoint ini = 0.
- CI memvalidasi skrip dan fixture, bukan availability feed saat ini. Sampel utama saat ini 4/500 (`INSUFFICIENT_EVIDENCE`).


### Sampel 2 — artikel hukum dan viktimologi, 2021

**Dating Violence: A Report from Legal and Victimological Perspective** — Salimi Muhammad Baindowi, *The Indonesian Journal of International Clinical Legal Education*, DOI [10.15294/ijicle.v3i3.48265](https://doi.org/10.15294/ijicle.v3i3.48265). Halaman penerbit menyatakan tanggal publikasi, lisensi CC BY-SA 4.0, dan pernyataan orisinalitas penulis: https://journal.unnes.ac.id/sju/iccle/article/view/48265. PDF 12 halaman diinspeksi pada halaman 1, 2 dan 12. Susunan heading meliputi Introduction, konsep/bentuk kekerasan dalam relasi, perspektif viktimologi, perlindungan hukum, Conclusion, dan References. Catatan kaki bernomor mendukung uraian; deklarasi konflik kepentingan, pendanaan, etika/orisinalitas, dan suggested citation diletakkan pada halaman akhir. Beberapa frasa janggal dan artefak tanda baca diperlakukan sebagai kekurangan khusus sumber, bukan gaya yang harus ditiru.

### Sampel 3 — analisis perlindungan kebebasan beragama, 2021

**Religious Tolerance: Does Indonesian Law Really Guarantee It?** — Mar’ie Muhammad Falah Akbar, *The Indonesian Journal of International Clinical Legal Education*, DOI [10.15294/ijicle.v3i1.43170](https://doi.org/10.15294/ijicle.v3i1.43170). Halaman penerbit dan PDF mencantumkan lisensi CC BY-SA 4.0: https://journal.unnes.ac.id/sju/iccle/article/view/43170. PDF 14 halaman diinspeksi pada halaman 1, 2 dan 14. Struktur menempatkan sejumlah bagian tematis hukum dan HAM; Recommendation muncul sebagai bagian terpisah setelah Conclusion. Pemindaian heading tampak meloncat dari V ke VII, sehingga heading VI yang tidak ditemukan dicatat sebagai anomali editorial kandidat dan butuh tindak lanjut, bukan pola umum. Artikel mencampur narasi legal, rujukan bernomor/footnote, dan klaim persentase yang perlu divalidasi independen.

### Sampel 4 — panduan teknis institusi profesional, 2019

**ID4D Practitioner’s Guide: Version 1.0 (October 2019)** — World Bank Group, Identification for Development (ID4D). PDF resmi: https://documents1.worldbank.org/curated/en/248371559325561562/pdf/ID4D-Practitioner-s-Guide.pdf; halaman lisensi/atribusi resmi: https://id4d.worldbank.org/guide/attribution. World Bank menyatakan CC BY 3.0 IGO, dengan syarat atribusi dan pengecualian bahwa sebagian komponen pihak ketiga mungkin memiliki hak berbeda. PDF 240 halaman (menurut viewer) diinspeksi secara visual pada sampul, halaman daftar isi, dan halaman terakhir; hanya metadata serta ciri struktur disimpan.

Berbeda dari artikel jurnal, panduan ini mengatur navigasi referensi dan pilihan operasional: halaman awal memiliki Version History, daftar tabel/gambar/boxes, About this Guide, abbreviations dan acknowledgements; isi utama terbagi menjadi I. Introduction, II. Designing an ID System, III. Topics, dan IV. Resources. Daftar isi menggunakan level heading bertingkat, indentasi, titik pemimpin menuju nomor halaman, dan perpindahan dari penomoran Romawi front matter ke halaman utama bernomor Arab. Sampul memakai blok navy/cyan/oranye, logo institusi, judul besar dan tanggal edisi. Pola ini cocok untuk panduan yang sering dirujuk, bukan template skripsi atau laporan PKL. Komponen pihak ketiga tidak diambil atau direpublikasikan.

## Sampel 5 — skripsi teknik lingkungan, 2020

**Laju Dekomposisi Karbon Pada Sampah Organik Pada Perencanaan Penutupan Tempat Pemrosesan Akhir (TPA) Kota Banda Aceh** — Muhammad Mefan Juansah, UIN Ar-Raniry, 2020. Official item: https://repository.ar-raniry.ac.id/17817/. The item-level page states Creative Commons Attribution 4.0; the PDF includes an originality declaration. The extracted PDF text layer spans 64 container pages and lists 53 thesis pages excluding front matter. No PDF or full text was retained; no student identifier or contact-like filename details were stored.

The front matter includes a title page, originality declaration, abstract and English abstract, preface, table of contents, and separate lists of tables, figures and appendices. The five-chapter structure is Introduction, Literature Review, Research Methodology, Results and Discussion, and Closing; references and appendices follow. The table of contents shows nested background/problem/objective/benefit/scope sections, methods subdivisions, and topic-based result subsections. Page references use dotted leaders; the extracted text shows Roman-numbered front matter and Arabic main-body pagination. A visible inconsistency labels the closing items as 4.1 and 4.2 under Chapter V; this is recorded as a source-specific QA finding, not a norm. This pass did not complete rendered-page visual inspection, so no font, color, margin, or image-quality claims are made.

## Sampel 6 — tugas akhir Teknologi Informasi, 2026

**Implementasi Sistem Penerjemah BISINDO Dua Arah Berbasis MediaPipe dan Artificial Intelligence** — Adea Marlinda, Universitas Islam Negeri Ar-Raniry Banda Aceh, 2026. Official item record: https://repository.ar-raniry.ac.id/58108/. The repository item declares Creative Commons Attribution 4.0; the full PDF text layer is public. No copy of the PDF or its full text is retained.

The cover identifies the document as a Tugas Akhir and includes author/program/faculty/institution/year fields. The front matter includes approval and validation sheets, Indonesian abstract, English abstract, preface, contents, list of figures and list of tables. The body has five chapters: Introduction; Literature Review; Research Methodology; Implementation and Research Results; Conclusions and Recommendations; followed by references. The contents uses nested decimal subsections, dotted leaders and page references; Roman-numbered front matter transitions to Arabic-numbered body pages.

The literature chapter is technology-specific (BISINDO, computer vision, MediaPipe, CNN, MobileNetV2, STT/TTS, Gradio, Colab, workflow automation, evaluation). The methods chapter subdivides system architecture and flow, data acquisition/dataset splitting, preprocessing, training and evaluation. The implementation chapter separates model/dataset processing, gesture/voice modes, webhook integration, functional testing, performance analysis and limitations. This is a useful example of a technical thesis linking literature, method, implementation and evaluation, not a universal thesis template.

A structural QA flag is preserved: the table of contents has a PDF bookmark artifact (“Error! Bookmark not defined”) next to the originality statement; the list of figures also appears to include a numbering/page-order mismatch. These source-specific issues should become QA checks, not patterns to imitate. Rendered visual inspection was not completed for this item, so no claims are made about font, color, margin, or image-rendering quality. Student ID and private acknowledgements are not retained; technical performance claims have not been independently reproduced.


## Sampel 7 — workbook data eksperimen baterai, CC BY 4.0

**24-hour Raw OCV Relaxation Dataset for Rechargeable Batteries with Different Chemistries (XLSX workbook)** — VOICILA Iulian-Teodor, ENACHE Bogdan-Adrian, VILCIU Irina, and SERITAN George-Calin; National University of Science and Technology POLITEHNICA Bucharest. Pinned repository workbook: https://github.com/BulyK47/battery-datasets/blob/ec8f3f3bc0898bb440432cb11a434250d6bcbb0c/ocv-relaxation-24h/ocv_relaxation_24h.xlsx. License at the same pinned source: https://github.com/BulyK47/battery-datasets/blob/ec8f3f3bc0898bb440432cb11a434250d6bcbb0c/LICENSE (CC BY 4.0); dataset DOI: https://doi.org/10.21227/651q-8v82. The workbook was downloaded by CI from the pinned commit and structurally parsed; binary size is 466,798 bytes and SHA-256 is `9b8c541675256c08c24cc6ebeb486a46d07c533a70bf4dabcca4955dba90e08a`.

The OOXML inspector found 17 visible worksheets: a metadata sheet `Header` plus 16 charge/discharge data sheets. All 16 data sheets have 1,442 row elements; the APR sheets have dimensions A1:G1442, the BSE sheets A1:D1442, and the other 12 sheets A1:B1442. `Header` has dimension A1:N101, 101 row elements, 546 non-empty cells and 23 merged ranges. The workbook has 0 hidden sheets, 0 formula cells and 0 table parts, plus 24 cellXfs and 8 borders. The publisher README describes Timestamp [h] and OCV voltage columns and 30 measurement traces across eight batteries and six chemistry families. Only structure/counts/hash were retained; no cell values were exported.

This sample broadens Excel evidence to a genuine experimental-data workbook. It is **not** evidence for formulas, print layout, chart quality or operational/business workbook conventions. Visual inspection in an Excel UI/print preview was not performed, and the scientific validity of the battery measurements was not independently reproduced.


## Sampel 8 — workbook belanja anggaran pemerintah, CC BY 4.0

**Budget 2020 — Expenditure Data: Estimates of Appropriations 2020/21 (XLSX workbook)** — The Treasury, New Zealand. Official page: https://www.treasury.govt.nz/publications/data/budget-2020-data-estimates-appropriations-2020-21. The Treasury page declares Crown Copyright, Attribution 4.0 International (CC BY 4.0) and identifies this release as superseded. Direct XLSX: https://www.treasury.govt.nz/sites/default/files/2020-05/b20-expenditure-data.xlsx.

GitHub Actions downloaded and inspected the 860,536-byte workbook, SHA-256 `d6a4f00122a05a86bf905f2c43f53cc065b8f033ee080647cf8033961c43aced`. The workbook has 8 visible worksheets: `Intro`, `Explanation`, `Raw Data`, `PivotAppnTypeSpendbyYear`, `GraphAppnTypeSpendbyYear`, `PivotAppnTypeByVote`, `GraphAppnTypeByVote`, and `PivotAppnTypeByPortfolio`. The overall structural tally is 6,512 row elements, 102,477 non-empty cells, two formula cells, two merged ranges, one worksheet with an auto-filter, four worksheets with frozen panes, one defined name, zero hidden sheets, and zero OOXML table parts. The `Raw Data` sheet has dimension A1:Q6304 and 101,427 non-empty cells. The two pivot-style sheets each contain one formula cell, one merged range and frozen panes; the two graph-named sheets have empty cell grids at A1 in the structural parser.

The workbook's separation between orientation, explanation/usage notes, raw rows, pivot summaries and graph-oriented sheets is useful evidence for workbook information architecture. A data filter and frozen panes support navigation in the main table. No cell values or financial figures were retained. The chart objects, screen appearance, print preview, formula outputs and fiscal figures were not visually checked or independently audited. Because the Treasury labels the release superseded, it is included only as a historical format exemplar, not as a source for current fiscal data.


## Sampel 9 — laporan tahunan profesional Department of Conservation NZ, CC BY 4.0

**Department of Conservation Annual Report 2024/25 — For the year ended 30 June 2025** — Department of Conservation Te Papa Atawhai, New Zealand. Official full PDF: https://www.doc.govt.nz/globalassets/documents/about-doc/annual-reports/annual-report-2025/annual-report-2025.pdf. The PDF itself declares Creative Commons Attribution 4.0 International (CC BY 4.0), gives Crown attribution terms, and contains a note that government emblems/logos have separate restrictions. Full PDF text layer was accessible; no full text or PDF copy is retained in Git.

The front matter contains a bilingual Māori/English report title, identifier C.13, reporting period, Crown copyright date, ISSNs, publishing/editing imprint, and the license notice. Contents use numbered major sections and bilingual headings paired with page references. The main architecture is:
1. Foreword including statement of responsibility.
2. Progress on strategic intentions.
3. Assessment of operations.
4. Organisational health and capability.
5. Annual and forecast financial statements.
6. Statements of expenses and capital expenditure for appropriations.
7. Independent auditor’s report.
8. Other matters, followed by six appendices/glossary and statutory references.

The hierarchy separates strategic outcomes and operational performance from organisational health/capability, departmental and non-departmental statements, appropriation reporting, the independent audit report, and appendices. Topics include governance/risk, outcome measures, funding/savings, performance information, culture/equity/safety, carbon-neutral reporting, accounting policies and notes. This is useful as a public-sector annual-report architecture, not as a universal report template. Rendered-page visual inspection was not performed; the text/contents review provides no evidence about typography, colors, charts, margins or print quality. Financial, environmental and performance claims were not independently re-audited.


## Sampel 10 — laporan tahunan Australian Bureau of Statistics, CC BY 4.0

**Australian Bureau of Statistics Annual Report 2024–25** — Australian Bureau of Statistics, Commonwealth of Australia. Official PDF: https://www.abs.gov.au/about/our-organisation/corporate-reporting/abs-annual-reports/ABS%20Annual%20Report%202024-25_final.pdf. The PDF states Creative Commons Attribution 4.0 International and asks for ABS attribution. The full text layer was accessible; no PDF or complete text was copied into GitHub.

The front matter includes the report title/year, copyright and ISSN, explicit license statement, contact metadata, letter of transmittal to the responsible Minister, acknowledgement of Country, and citation information. The table of contents divides the report into five major sections: Overview; Report on performance; Management and accountability; Appendices; and Reference material. The seven chapters cover the Australian Statistician’s review, ABS at a glance, special articles, annual performance statement, financial statements, corporate governance, and human-resource management. Three appendices address resource statements, climate/environmental sustainability, and errors/omissions; reference material includes abbreviations, glossary, statutory requirements and index.

The architecture places narrative/overview first, then performance and financial statements, followed by governance/workforce, appendices and reference material. The transmittal and statutory context establish accountability to the responsible Minister and Parliament. This differs from the NZ DOC annual report despite the shared annual-report genre: both separate performance and financial reporting, while their sectional hierarchy and governance context vary by jurisdiction. Rendered-page visual inspection was not performed; financial and performance claims have not been independently re-audited.

## Partial, metadata-only, privacy-pending, and excluded leads (not counted toward the target)

- **Menjadi Data Center Pada Program Amati Indonesia** — Politeknik Negeri Jember/SIPORA, 2022. The public summary and Chapter I are licensed CC BY-SA 4.0; the full-report PDF is restricted to registered users. The accessible chapter presents background, general/specific objectives and benefits, location/period, and implementation methods. The record uses `access_status=partial`, with privacy review pending, and contributes **zero** to the eligibility count until the full access/right/privacy gates can be satisfied.
- **Asuhan Keperawatan pada Klien Harga Diri Rendah dengan Penerapan Terapi Kognitif** — STIKes Muhammadiyah Ciamis repository, 2022. Public files include cover/approval/abstract/Chapter I/bibliography under a file-level Creative Commons Attribution notice; the clinical case and discussion chapters are restricted. Privacy and version-of-license review remain pending. No patient-level facts are retained. The record contributes **zero** to eligibility.
- **ABK Sekdin.xlsx — Analisis Beban Kerja Sekertaris Dinas Sosial Kabupaten Karanganyar** — official Portal Data Jawa Tengah resource declares CC BY at metadata level, but download requests to two listed hosts returned HTTP 403. No worksheet, formula, formatting, hidden-sheet, or cell-content claims have been made. It remains metadata-only and contributes **zero** to eligibility.
- **Studi Kasus: Asuhan Keperawatan pada Orang Dewasa dengan Isolasi Sosial** — Journal of Nursing Education and Practice, DOI [10.53801/jnep.v1i01.9](https://doi.org/10.53801/jnep.v1i01.9), CC BY-SA 4.0 and full text public. Only macro-headings are recorded (case-study label, abstract sections, Pendahuluan, Metode, Hasil dan Pembahasan, Kesimpulan, conflict/acknowledgement/funding/references). The full paper contains sensitive patient narrative, so the source is held at `privacy_review=pending`; no case facts, quotes, chronology, or contact details are retained. It contributes **zero** to eligibility until privacy review is resolved.
- **Seaweed in Indonesia: farming, utilization and research** — Wageningen Centre for Development Innovation, Report WCDI-22-220, DOI [10.18174/578007](https://doi.org/10.18174/578007). The source says CC BY 4.0 (Netherlands) but also explicitly prohibits commercial use. Because those conditions conflict and no authoritative clarification was obtained, the manifest record is `usage_scope=excluded`/`verification_status=excluded`; no feature record was created and the report contributes **zero** to eligibility.

## Apa yang dapat disimpulkan dari sepuluh sampel

Tiga artikel berasal dari penerbit/jurnal yang sama; karena itu, mereka dapat memperlihatkan variasi dalam satu tradisi publikasi tetapi belum menjadi tiga source family independen. Sebagian ciri berulang—metadata DOI dan lisensi di halaman awal, satu kolom serif, catatan bernomor, bagian referensi dan deklarasi penerbitan—masuk akal diuji sebagai hipotesis untuk artikel-artikel jurnal tersebut. Adanya bagian recommendation yang terpisah pada salah satu artikel serta perbedaan heading/topik memperlihatkan bahwa urutan bab tidak boleh dipaksakan sebagai template universal.

Panduan World Bank menambah satu genre fungsional yang berbeda: struktur modular yang mengutamakan lookup dan keputusan, indeks navigasi yang padat, dan hierarki heading dalam. Variasi antarartikel dan perbedaan artikel-vs-panduan harus dipertahankan. Sepuluh sampel tetap belum memberi bukti tentang norma LP/asuhan keperawatan, PKL lengkap, praktikum, laporan kegiatan, laporan kerja/profesional, atau workbook operasional/administratif berformula. Satu workbook Excel yang eligible adalah dataset eksperimental terukur dan tidak mewakili laporan keuangan, HR, anggaran atau spreadsheet operasional. Dua skripsi UIN Ar-Raniry berasal dari universitas yang sama tetapi program/topik berbeda; keduanya tidak menjadi dua institusi independen.

Analisis fitur per dokumen tersimpan sebagai JSONL pada `research/corpus/features.jsonl`; daftar sumber/hak dan status akses berada pada `research/corpus/manifest.jsonl`. Tidak ada PDF atau teks penuh yang disimpan ke GitHub.

Keterlacakan analisis diverifikasi oleh `research/corpus/validate_features.py`, dengan tes untuk referensi manifest, source family, URL bukti, analisis duplikat dan kelengkapan metode inspeksi. Workflow CI menyimpan ringkasan mesin `corpus-summary.json` dan `features-summary.json` sebagai artifact selama 14 hari. Ringkasan itu bersifat status inventaris, bukan hasil benchmark kualitas.

### Hasil live discovery DOAJ (metadata saja)

Pada run terbaru, pipeline mengakses DOAJ OAI-PMH melalui GitHub Actions, membaca 20 halaman (6.000 rekaman metadata), dan tidak menemukan metadata yang cocok dengan 11 ISSN sasaran. Status harvester: `NO_NEW_CANDIDATES`; kandidat baru 0. Tidak ada abstrak, PDF, atau teks artikel yang diunduh. Ini hanya kesimpulan dari rentang awal 20 halaman, karena harvester berhenti pada batas jumlah rekaman; bukan bukti bahwa jurnal sasaran tidak memiliki artikel. Parser kini memeriksa `dc:source` untuk ISSN selain `dc:identifier`/`dc:relation`, dengan regression test yang lulus.
