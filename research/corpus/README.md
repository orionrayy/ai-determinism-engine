# Hallmark — korpus referensi Document Craft

Baseline awal pipeline pada 2026-10-30 menggunakan manifest kosong. Manifest sekarang mencatat tujuh dokumen lolos gate eligibility, dua catatan akses parsial, satu kandidat nursing-case dengan privasi pending, dan satu lead spreadsheet metadata-only. Status dan distribusi terkini dilaporkan terpisah; target 500 tidak dipenuhi oleh metadata kandidat atau dokumen yang hanya diakses sebagian.

## Prinsip penerimaan

Satu entri hanya dihitung sebagai dokumen referensi orisinal terverifikasi jika seluruhnya benar: (1) sumber dokumen dapat ditautkan ke URL/identitas penerbit yang stabil; (2) isi benar-benar diakses secara sah untuk dianalisis; (3) keaslian dokumen individual, bukan sekadar reputasi situs, dibuktikan melalui halaman repositori/penerbit, metadata bibliografi, halaman judul, atau bukti sebanding; (4) provenance confidence `high`; (5) lisensi atau dasar izin tercatat dan meliputi penggunaan yang direncanakan; (6) status privasi lolos; (7) genre dan format tercatat; (8) tidak merupakan versi/duplikat dokumen lain. Template resmi, metadata-only, sumber yang lisensinya belum jelas, item tanpa akses isi, keluaran AI/sintetis, serta sumber yang belum diverifikasi tidak dihitung.

Akses publik tidak dengan sendirinya mengizinkan pengunduhan ulang atau redistribusi. Jangan menghindari login, paywall, CAPTCHA, robots/access controls, atau batas penggunaan. Bila teks penuh tidak boleh disimpan, gunakan hanya ekstraksi fitur/metadata yang diizinkan dan simpan sesedikit mungkin. Jangan masukkan rekam medis atau identitas pribadi ke Git. Untuk contoh klinis, gunakan materi yang secara eksplisit publik dan telah dideidentifikasi.

## Berkas

- `manifest.jsonl`: satu JSON object per baris; hanya inventaris dokumen individual. Baris kosong diabaikan. Tidak ada entri awal yang dimasukkan secara spekulatif.
- `manifest.schema.json`: kontrak field dan enum.
- `source_registry.json`: direktori penemuan sumber. Lisensi tingkat jurnal adalah petunjuk awal saja; tetap periksa lisensi setiap item.
- `validate_corpus.py`: validator stdlib-only, deduplikasi URL/hash, distribusi, serta status target.
- `validate_features.py`: validator keterlacakan analisis fitur ke manifest, source family, dan bukti provenance; melaporkan kekurangan analisis tanpa menyembunyikan batas generalisasi.
- `inspect_xlsx_structure.py`: pemeriksa struktur OOXML/XLSX berbasis standard library; menghitung sheet, dimensi, sel, formula, hidden sheet, filter, merged ranges, freeze panes, style counts dan SHA-256 tanpa mengekspor nilai sel.
- `evaluation-rubric.md`: rubrik benchmark yang harus dibekukan sebelum penilaian kandidat.
- `genre-taxonomy.json`: kelas dokumen dan fitur yang dipelajari sebagai hipotesis, bukan aturan universal.

## Penemuan metadata kandidat (DOAJ)

`harvest_doaj_candidates.py` mengambil **metadata artikel**, bukan PDF/teks artikel, melalui feed OAI-PMH publik DOAJ. DOAJ menyatakan metadata jurnal/artikel dapat digunakan kembali dengan waiver CC0; lisensi metadata ini tidak mengubah lisensi artikel yang dideskripsikannya. Lihat [DOAJ Terms](https://doaj.org/terms/) dan [OAI-PMH documentation](https://doaj.org/docs/oai-pmh/). Feed umum bisa tertinggal hingga 30 hari dan dapat memiliki volume besar; karena itu harvester membatasi jumlah halaman, ukuran respons, timeout, dan laju permintaan. Opsi ISSN memfilter hanya rekaman yang berhubungan dengan jurnal pilihan.

Contoh (maksimal dua halaman, tanpa mengambil konten penuh):

```sh
python research/corpus/harvest_doaj_candidates.py --dry-run --json --max-pages 2 --max-records 200
python research/corpus/harvest_doaj_candidates.py --json --max-pages 5 --max-records 500
```

Default output adalah `candidates.jsonl`. Harvester menggabungkan berdasarkan ID kandidat stabil, menulis secara atomik, tidak menyimpan abstrak, menandai `corpus_eligible=false`, dan tidak pernah memindahkan kandidat secara otomatis ke `manifest.jsonl`. Hak penggunaan, akses teks, autentisitas artikel, privasi, genre dan duplikasi harus diverifikasi per item sebelum promosi manual. Batas `--max-records` dievaluasi setelah satu halaman OAI selesai diproses, sehingga jumlah yang diproses dapat sedikit melewati ambang maksimal sebesar ukuran halaman terakhir; `--max-pages` dan ukuran respons tetap menjadi batas keras.

`candidates.jsonl` dan `manifest.jsonl` adalah dua inventaris berbeda. Metadata kandidat tidak dihitung dalam target 500.

Workflow CI juga menjalankan `validate_features.py` untuk memeriksa keterlacakan analisis ke manifest. Pada job CI, `corpus-summary.json` dan `features-summary.json` disimpan sebagai artifact inspeksi selama 14 hari; berkas itu berisi metadata, status gate, peringatan, dan distribusi, bukan teks penuh sumber.

## Format entri manifest

Lihat `manifest.schema.json`. Wajib menyimpan `record_id`, `source_family_id`, URL dan judul, penerbit/pencipta, genre/subgenre, tanggal dan format, hak penggunaan, provenance, cara memperoleh, status akses, hasil peninjauan privasi, status verifikasi, serta alasan pengecualian/duplikasi. Hash dan ukuran hanya dicatat bila pengukuran tidak melanggar ketentuan penggunaan atau kerahasiaan sumber.

`source_family_id` adalah pengenal stabil bagi karya/versi/repositori yang sama, dipakai untuk mencegah kebocoran antara set pengembangan, validasi, dan uji. Jangan membuat ID baru untuk salinan yang diformat ulang atau file yang diunduh dari cermin lain.

## Jalankan

```sh
python research/corpus/validate_corpus.py research/corpus/manifest.jsonl --json
python research/corpus/validate_corpus.py research/corpus/manifest.jsonl --require-target 500
python -m unittest discover -s research/corpus -p "test_*.py" -v
python research/corpus/validate_features.py research/corpus/manifest.jsonl research/corpus/features.jsonl --json
python research/corpus/inspect_xlsx_structure.py path/to/publicly-licensed-workbook.xlsx --json
```

Exit code `0` berarti bentuk manifest dapat diproses; status target akan tetap `INSUFFICIENT_EVIDENCE` jika kurang dari 500. Opsi `--require-target` sengaja gagal saat target belum tercapai. Error field/JSON menyebabkan exit non-zero agar data rusak tidak diam-diam diterima.

## Tahap benchmark

1. Catat semua kandidat terlebih dahulu; bedakan `pending`, `excluded`, `official_template`, `duplicate_or_version`, dan `synthetic_or_ai`.
2. Tinjau lisensi/izin dan bukti asal per item; jangan menyimpulkan lisensi dari domain saja.
3. Deduplikasi berdasarkan URL kanonis dan SHA-256 jika pencatatan hash diizinkan. Tinjau keluarga dokumen/versi secara manual untuk parafrasa, edisi, dan salinan yang berbeda hash.
4. Bekukan daftar sampel dan rubrik sebelum eksperimen. Pisahkan train/dev/test menurut `source_family_id`, bukan secara acak per file.
5. Simpan input, instruksi, generator/model/version, commit, konfigurasi, dan output baseline. Jangan menilai ulang baseline dengan aturan yang baru ditetapkan setelah melihat output.
6. Bandingkan kualitas struktur, substansi, bukti/sitasi, bahasa, konsistensi, dan format. Catat temuan negatif, ketidakpastian, serta variasi valid antarinstitusi.
7. Klaim peningkatan hanya jika hasil terukur dapat direproduksi dan tidak menunjukkan regresi bermakna pada genre lain.

## Status korpus pada checkpoint 2026-10-30

- Manifest memuat **11 catatan**: 7 eligible, 2 akses parsial, 1 kandidat artikel keperawatan yang belum lolos review privasi, dan 1 workbook metadata-only.
- Sampel yang memenuhi gate eligibility: **7 dari target 500**; sisa 493; status `INSUFFICIENT_EVIDENCE`.
- Distribusi eligible: `academic_paper` = 3, `other` (panduan profesional World Bank) = 1, `thesis_dissertation` = 2, `excel_workbook` = 1. Genre eligible lainnya = 0.
- Tujuh item eligible terdiri dari tiga artikel ICCLE CC BY-SA 4.0, panduan World Bank CC BY 3.0 IGO, dua skripsi UIN Ar-Raniry CC BY 4.0, dan workbook data eksperimen baterai CC BY 4.0 yang diperiksa dari binary XLSX terpin. Setiap lisensi dicek pada laman item/sumber resmi masing-masing.
- Dua partial items: satu laporan PKL Polije (summary + Chapter I publik berlisensi, laporan penuh dibatasi) dan satu KIAN/asuhan keperawatan (file publik awal; bab kasus dan pembahasan dibatasi). Keduanya disimpan sebagai catatan pending dan **tidak dihitung** sampai akses, privacy, dan hak penuh selesai ditinjau.
- Satu lead workbook pemerintah daerah mencatat metadata resmi CC BY untuk XLSX, tetapi download dari dua host mengembalikan HTTP 403. Sheet, formula, format, isi, dan risiko data pekerja belum diperiksa; status tetap metadata-only dan tidak dihitung.
- Satu artikel case-study keperawatan CC BY-SA 4.0 dengan teks penuh publik dicatat hanya sebagai structural-only candidate. Karena isinya berisi narasi klinis personal yang sensitif, privasi/anonymization review tetap `pending`; tidak ada fakta pasien disalin, dan item ini tidak ikut hitungan eligible.
- Satu workbook XLSX riset baterai berhasil diunduh pada runner GitHub dari commit sumber terpin dan diinspeksi secara struktural: 17 sheet terlihat (Header + 16 data), 0 formula, 0 sheet tersembunyi, 23 merged ranges pada Header. Binary SHA-256 dicatat dalam manifest; tidak ada nilai sel yang diekspor atau disimpan.
- Distribusi fitur: tujuh analisis struktur pada item eligible dan tiga catatan non-eligible (dua partial, satu nursing-case privacy-pending) ada di `features.jsonl`; satu spreadsheet metadata-only belum memiliki analisis isi.
- Registry penemuan berisi 11 direktori/katalog sumber, bukan dokumen. Harvester DOAJ dan fixture tests tersedia, tetapi live harvest dari OAI-PMH belum berhasil; kandidat feed live = 0.
- Analisis visual terhadap sumber tidak lengkap: beberapa PDF punya inspeksi visual terpilih, sedangkan kedua skripsi UIN Ar-Raniry, dua catatan partial, dan nursing case privacy-pending tidak mendapat rendering QA. Ini bukan validasi visual output generator.
- Belum ada eksperimen Document Craft baseline-vs-kandidat, penilaian reviewer blind, inter-rater agreement, ataupun bukti empiris peningkatan kualitas.

Korpus masih eksploratif dan jauh dari target. Klaim faktual sumber belum diverifikasi independen secara menyeluruh. Metadata dan fitur turunan disimpan tanpa salinan PDF atau teks penuh.
