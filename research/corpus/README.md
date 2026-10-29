# Hallmark — korpus referensi Document Craft

Baseline awal pipeline pada 2026-10-30 menggunakan manifest kosong. Setelah pemeriksaan sumber primer, manifest saat ini memuat satu artikel ilmiah akses terbuka yang terverifikasi untuk analisis fitur; detail dan distribusi terkini berada pada bagian status di bawah. Angka 500 adalah target penerimaan, bukan jumlah hasil pencarian. Registry sumber/kandidat metadata tidak dihitung sebagai sampel.

## Prinsip penerimaan

Satu entri hanya dihitung sebagai dokumen referensi orisinal terverifikasi jika seluruhnya benar: (1) sumber dokumen dapat ditautkan ke URL/identitas penerbit yang stabil; (2) isi benar-benar diakses secara sah untuk dianalisis; (3) keaslian dokumen individual, bukan sekadar reputasi situs, dibuktikan melalui halaman repositori/penerbit, metadata bibliografi, halaman judul, atau bukti sebanding; (4) provenance confidence `high`; (5) lisensi atau dasar izin tercatat dan meliputi penggunaan yang direncanakan; (6) status privasi lolos; (7) genre dan format tercatat; (8) tidak merupakan versi/duplikat dokumen lain. Template resmi, metadata-only, sumber yang lisensinya belum jelas, item tanpa akses isi, keluaran AI/sintetis, serta sumber yang belum diverifikasi tidak dihitung.

Akses publik tidak dengan sendirinya mengizinkan pengunduhan ulang atau redistribusi. Jangan menghindari login, paywall, CAPTCHA, robots/access controls, atau batas penggunaan. Bila teks penuh tidak boleh disimpan, gunakan hanya ekstraksi fitur/metadata yang diizinkan dan simpan sesedikit mungkin. Jangan masukkan rekam medis atau identitas pribadi ke Git. Untuk contoh klinis, gunakan materi yang secara eksplisit publik dan telah dideidentifikasi.

## Berkas

- `manifest.jsonl`: satu JSON object per baris; hanya inventaris dokumen individual. Baris kosong diabaikan. Tidak ada entri awal yang dimasukkan secara spekulatif.
- `manifest.schema.json`: kontrak field dan enum.
- `source_registry.json`: direktori penemuan sumber. Lisensi tingkat jurnal adalah petunjuk awal saja; tetap periksa lisensi setiap item.
- `validate_corpus.py`: validator stdlib-only, deduplikasi URL/hash, distribusi, serta status target.
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

## Format entri manifest

Lihat `manifest.schema.json`. Wajib menyimpan `record_id`, `source_family_id`, URL dan judul, penerbit/pencipta, genre/subgenre, tanggal dan format, hak penggunaan, provenance, cara memperoleh, status akses, hasil peninjauan privasi, status verifikasi, serta alasan pengecualian/duplikasi. Hash dan ukuran hanya dicatat bila pengukuran tidak melanggar ketentuan penggunaan atau kerahasiaan sumber.

`source_family_id` adalah pengenal stabil bagi karya/versi/repositori yang sama, dipakai untuk mencegah kebocoran antara set pengembangan, validasi, dan uji. Jangan membuat ID baru untuk salinan yang diformat ulang atau file yang diunduh dari cermin lain.

## Jalankan

```sh
python research/corpus/validate_corpus.py research/corpus/manifest.jsonl --json
python research/corpus/validate_corpus.py research/corpus/manifest.jsonl --require-target 500
python -m unittest discover -s research/corpus -p "test_*.py" -v
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

- Dokumen individual yang masuk manifest dengan provenance kuat: 1 artikel ilmiah akses terbuka, DOI `10.15294/ijicle.v2i3.38417`. Lisensi CC BY-SA 4.0 dicatat berdasarkan halaman artikel, halaman hak cipta jurnal, dan pernyataan pada PDF.
- Sampel yang memenuhi gate eligibility sesuai manifest: 1 dari target 500; sisa target 499; status `INSUFFICIENT_EVIDENCE`.
- Genre saat ini: `academic_paper` = 1. Semua genre lainnya = 0 sampel tervalidasi.
- Registry penemuan berisi 11 direktori/katalog sumber, tetapi tidak dihitung sebagai dokumen.
- Harvester metadata DOAJ dan tes fixture XML telah ditambahkan. Live harvest dari OAI-PMH belum dijalankan pada sesi ini; jumlah kandidat yang benar-benar diambil lewat feed = 0.
- Satu sampel telah diberi analisis struktur manual pada `features.jsonl`; analisis tersebut eksploratif dan tidak cukup untuk menggeneralisasi pola satu genre.
- Belum ada eksperimen Document Craft baseline-vs-kandidat atau validasi visual keluaran DOCX/PDF/Excel; visual yang diperiksa sejauh ini hanya PDF sumber satu artikel.

Sampel artikel ini bukan otomatis gold-standard kualitas. Klaim faktualnya belum diverifikasi independen; analisis menyimpan fitur struktur, bukan salinan PDF atau teks lengkap.
