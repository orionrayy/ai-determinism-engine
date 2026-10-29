# Laporan checkpoint riset Hallmark — 30 Oktober 2026

## Status yang dapat dibuktikan

Korpus saat ini memiliki **7 dokumen individual yang lolos gate provenance dan hak penggunaan** dari sasaran 500. Status target: `INSUFFICIENT_EVIDENCE`; sisa 493. Sampel masih kecil dan tidak cukup untuk menyimpulkan kualitas seluruh genre atau peningkatan Document Craft.

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
| Laporan kerja/profesional | 0 |
| Operasional/administratif | 0 |
| Workbook Excel | 1 |
| Genre lain — panduan profesional/operasional | 1 |

Registry sumber berisi 11 direktori penemuan (10 entri direktori jurnal dari DOAJ dan Portal Satu Data Indonesia). Jumlah itu bukan jumlah dokumen. Hak penggunaan diperiksa per dokumen; label lisensi tingkat jurnal dalam registry tidak otomatis diterapkan ke dokumen individual. Manifest juga menyimpan dua catatan partial (laporan PKL dan KIAN) serta satu lead XLSX metadata-only; item parsial/metadata-only tersebut **tidak dihitung** ke angka 7/500.

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


**24-hour Raw OCV Relaxation Dataset for Rechargeable Batteries with Different Chemistries (XLSX workbook)** — VOICILA Iulian-Teodor, ENACHE Bogdan-Adrian, VILCIU Irina, and SERITAN George-Calin; National University of Science and Technology POLITEHNICA Bucharest. Pinned repository workbook: https://github.com/BulyK47/battery-datasets/blob/ec8f3f3bc0898bb440432cb11a434250d6bcbb0c/ocv-relaxation-24h/ocv_relaxation_24h.xlsx. License at the same pinned source: https://github.com/BulyK47/battery-datasets/blob/ec8f3f3bc0898bb440432cb11a434250d6bcbb0c/LICENSE (CC BY 4.0); dataset DOI: https://doi.org/10.21227/651q-8v82. The workbook was downloaded by CI from the pinned commit and structurally parsed; binary size is 466,798 bytes and SHA-256 is `9b8c541675256c08c24cc6ebeb486a46d07c533a70bf4dabcca4955dba90e08a`.

The OOXML inspector found 17 visible worksheets: a metadata sheet `Header` plus 16 charge/discharge data sheets. All 16 data sheets have 1,442 row elements; the APR sheets have dimensions A1:G1442, the BSE sheets A1:D1442, and the other 12 sheets A1:B1442. `Header` has dimension A1:N101, 101 row elements, 546 non-empty cells and 23 merged ranges. The workbook has 0 hidden sheets, 0 formula cells and 0 table parts, plus 24 cellXfs and 8 borders. The publisher README describes Timestamp [h] and OCV voltage columns and 30 measurement traces across eight batteries and six chemistry families. Only structure/counts/hash were retained; no cell values were exported.

This sample broadens Excel evidence to a genuine experimental-data workbook. It is **not** evidence for formulas, print layout, chart quality or operational/business workbook conventions. Visual inspection in an Excel UI/print preview was not performed, and the scientific validity of the battery measurements was not independently reproduced.

## Partial and metadata-only leads (not counted toward the target)

- **Menjadi Data Center Pada Program Amati Indonesia** — Politeknik Negeri Jember/SIPORA, 2022. The public summary and Chapter I are licensed CC BY-SA 4.0; the full-report PDF is restricted to registered users. The accessible chapter presents background, general/specific objectives and benefits, location/period, and implementation methods. The record uses `access_status=partial`, with privacy review pending, and contributes **zero** to the eligibility count until the full access/right/privacy gates can be satisfied.
- **Asuhan Keperawatan pada Klien Harga Diri Rendah dengan Penerapan Terapi Kognitif** — STIKes Muhammadiyah Ciamis repository, 2022. Public files include cover/approval/abstract/Chapter I/bibliography under a file-level Creative Commons Attribution notice; the clinical case and discussion chapters are restricted. Privacy and version-of-license review remain pending. No patient-level facts are retained. The record contributes **zero** to eligibility.
- **ABK Sekdin.xlsx — Analisis Beban Kerja Sekertaris Dinas Sosial Kabupaten Karanganyar** — official Portal Data Jawa Tengah resource declares CC BY at metadata level, but download requests to two listed hosts returned HTTP 403. No worksheet, formula, formatting, hidden-sheet, or cell-content claims have been made. It remains metadata-only and contributes **zero** to eligibility.

## Sampel 7 — workbook data eksperimen baterai, CC BY 4.0

Tiga artikel berasal dari penerbit/jurnal yang sama; karena itu, mereka dapat memperlihatkan variasi dalam satu tradisi publikasi tetapi belum menjadi tiga source family independen. Sebagian ciri berulang—metadata DOI dan lisensi di halaman awal, satu kolom serif, catatan bernomor, bagian referensi dan deklarasi penerbitan—masuk akal diuji sebagai hipotesis untuk artikel-artikel jurnal tersebut. Adanya bagian recommendation yang terpisah pada salah satu artikel serta perbedaan heading/topik memperlihatkan bahwa urutan bab tidak boleh dipaksakan sebagai template universal.

Panduan World Bank menambah satu genre fungsional yang berbeda: struktur modular yang mengutamakan lookup dan keputusan, indeks navigasi yang padat, dan hierarki heading dalam. Variasi antarartikel dan perbedaan artikel-vs-panduan harus dipertahankan. Tujuh sampel tetap belum memberi bukti tentang norma LP/asuhan keperawatan, PKL lengkap, praktikum, laporan kegiatan, laporan kerja/profesional, atau workbook operasional/administratif berformula. Satu workbook Excel yang eligible adalah dataset eksperimental terukur dan tidak mewakili laporan keuangan, HR, anggaran atau spreadsheet operasional. Dua skripsi UIN Ar-Raniry berasal dari universitas yang sama tetapi program/topik berbeda; keduanya tidak menjadi dua institusi independen.

Analisis fitur per dokumen tersimpan sebagai JSONL pada `research/corpus/features.jsonl`; daftar sumber/hak dan status akses berada pada `research/corpus/manifest.jsonl`. Tidak ada PDF atau teks penuh yang disimpan ke GitHub.

Keterlacakan analisis diverifikasi oleh `research/corpus/validate_features.py`, dengan tes untuk referensi manifest, source family, URL bukti, analisis duplikat dan kelengkapan metode inspeksi. Workflow CI menyimpan ringkasan mesin `corpus-summary.json` dan `features-summary.json` sebagai artifact selama 14 hari. Ringkasan itu bersifat status inventaris, bukan hasil benchmark kualitas.
