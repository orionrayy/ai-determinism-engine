# Laporan checkpoint riset Hallmark — 30 Oktober 2026

## Status yang dapat dibuktikan

Korpus saat ini memiliki **4 dokumen individual yang lolos gate provenance dan hak penggunaan** dari sasaran 500. Status target: `INSUFFICIENT_EVIDENCE`; sisa 496. Sampel masih kecil dan tidak cukup untuk menyimpulkan kualitas seluruh genre atau peningkatan Document Craft.

| Genre | Dokumen lolos gate |
|---|---:|
| LP/asuhan keperawatan | 0 |
| Laporan PKL/magang | 0 |
| Skripsi/tesis/disertasi | 0 |
| Makalah/artikel ilmiah | 3 |
| Laporan praktikum | 0 |
| Laporan penelitian | 0 |
| Proposal | 0 |
| Laporan kegiatan | 0 |
| Laporan kerja/profesional | 0 |
| Operasional/administratif | 0 |
| Workbook Excel | 0 |
| Genre lain — panduan profesional/operasional | 1 |

Registry sumber berisi 11 direktori penemuan (10 entri direktori jurnal dari DOAJ dan Portal Satu Data Indonesia). Jumlah itu bukan jumlah dokumen. Hak penggunaan diperiksa per dokumen; label lisensi tingkat jurnal dalam registry tidak otomatis diterapkan ke dokumen individual.

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

## Apa yang dapat disimpulkan dari empat sampel

Tiga artikel berasal dari penerbit/jurnal yang sama; karena itu, mereka dapat memperlihatkan variasi dalam satu tradisi publikasi tetapi belum menjadi tiga source family independen. Sebagian ciri berulang—metadata DOI dan lisensi di halaman awal, satu kolom serif, catatan bernomor, bagian referensi dan deklarasi penerbitan—masuk akal diuji sebagai hipotesis untuk artikel-artikel jurnal tersebut. Adanya bagian recommendation yang terpisah pada salah satu artikel serta perbedaan heading/topik memperlihatkan bahwa urutan bab tidak boleh dipaksakan sebagai template universal.

Panduan World Bank menambah satu genre fungsional yang berbeda: struktur modular yang mengutamakan lookup dan keputusan, indeks navigasi yang padat, dan hierarki heading dalam. Variasi antarartikel dan perbedaan artikel-vs-panduan harus dipertahankan. Empat sampel tidak memberi bukti tentang pola LP/asuhan keperawatan, laporan PKL, skripsi Indonesia, laporan kegiatan, maupun workbook Excel; seluruh genre tersebut masih bernilai nol.

Analisis fitur per dokumen tersimpan sebagai JSONL pada `research/corpus/features.jsonl`; daftar sumber/hak dan status akses berada pada `research/corpus/manifest.jsonl`. Tidak ada PDF atau teks penuh yang disimpan ke GitHub.

Keterlacakan analisis diverifikasi oleh `research/corpus/validate_features.py`, dengan tes untuk referensi manifest, source family, URL bukti, analisis duplikat dan kelengkapan metode inspeksi. Workflow CI menyimpan ringkasan mesin `corpus-summary.json` dan `features-summary.json` sebagai artifact selama 14 hari. Ringkasan itu bersifat status inventaris, bukan hasil benchmark kualitas.
