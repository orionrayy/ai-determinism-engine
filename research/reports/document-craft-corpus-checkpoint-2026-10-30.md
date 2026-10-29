# Laporan checkpoint riset Hallmark — 30 Oktober 2026

## Status yang dapat dibuktikan

Korpus memiliki **1 dokumen individual yang lolos gate provenance dan hak penggunaan** dari sasaran 500. Status target: `INSUFFICIENT_EVIDENCE`; sisa 499. Ini belum merupakan sampel representatif dan tidak cukup untuk menyimpulkan pola seluruh genre atau peningkatan kualitas Document Craft.

| Genre | Dokumen lolos gate |
|---|---:|
| LP/asuhan keperawatan | 0 |
| Laporan PKL/magang | 0 |
| Skripsi/tesis/disertasi | 0 |
| Makalah/artikel ilmiah | 1 |
| Laporan praktikum | 0 |
| Laporan penelitian | 0 |
| Proposal | 0 |
| Laporan kegiatan | 0 |
| Laporan kerja/profesional | 0 |
| Operasional/administratif | 0 |
| Workbook Excel | 0 |
| Genre lain | 0 |

Registry sumber berisi 11 direktori penemuan (10 entri direktori jurnal dari DOAJ dan Portal Satu Data Indonesia). Jumlah itu bukan jumlah dokumen. Hak penggunaan diperiksa per dokumen; label lisensi tingkat jurnal dalam registry tidak otomatis diterapkan ke dokumen individual.

## Sampel yang sudah diverifikasi

**The Effectiveness of Indonesian Government Assistance in Handling and Breaking Covid-19 Chains** — Tessa Septy Dynesia, *The Indonesian Journal of International Clinical Legal Education*, 2(3), 245–256 (2020), DOI [10.15294/ijicle.v2i3.38417](https://doi.org/10.15294/ijicle.v2i3.38417).

- Halaman penerbit menampilkan judul, penulis, tanggal publikasi, DOI dan pranala full text: https://journal.unnes.ac.id/sju/iccle/article/view/38417
- Kebijakan hak cipta jurnal menyebut penulis mempertahankan hak cipta dan artikel berada di bawah CC BY-SA 4.0: https://journal.unnes.ac.id/sju/iccle/copyrights
- PDF resmi 12 halaman menyebutkan lisensi CC BY-SA 4.0 dan pernyataan orisinalitas penulis. PDF diinspeksi secara visual pada halaman pertama, halaman isi awal, serta halaman terakhir; URL sumbernya tercatat di manifest.
- Tidak ada salinan PDF, teks penuh, alamat surel penulis, atau data pasien yang dimasukkan ke repositori. Yang disimpan hanya metadata bibliografis dan fitur struktur.
- Keaslian sebagai publikasi individual memiliki bukti kuat, tetapi **keakuratan setiap klaim faktual belum diverifikasi secara independen**.

## Analisis struktur manual, satu dokumen

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
- CI memvalidasi skrip dan fixture, bukan availability feed saat ini. Sampel utama tetap 1/500 (`INSUFFICIENT_EVIDENCE`).
