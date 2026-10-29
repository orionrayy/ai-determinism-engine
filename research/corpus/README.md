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
- `validate_features.py`: validator keterlacakan analisis fitur ke manifest, source family, dan bukti provenance; melaporkan kekurangan analisis tanpa menyembunyikan batas generalisasi.
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
python research/corpus/validate_features.py research/corpus/manifest.jsonl research/corpus/features.jsonl --json
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

- Dokumen individual dengan provenance kuat dalam manifest: **4 dokumen unik** — tiga artikel ilmiah akses terbuka CC BY-SA 4.0 dari jurnal ICCLE dan satu panduan profesional World Bank CC BY 3.0 IGO. Hak dan sumber dicatat per item.
- Sampel yang memenuhi gate eligibility sesuai manifest: **4 dari target 500**; tersisa 496; status `INSUFFICIENT_EVIDENCE`.
- Distribusi: `academic_paper` = 3, `other` (panduan operasional/profesional) = 1; semua genre lainnya = 0 sampel tervalidasi.
- Registry penemuan berisi 11 direktori/katalog sumber; jumlah tersebut bukan dokumen. Harvester metadata DOAJ dan tes fixture XML tersedia, tetapi live harvest dari OAI-PMH belum berhasil dijalankan di sesi ini karena lingkungan tidak dapat mengakses endpoint; jumlah kandidat feed yang benar-benar diambil = 0.
- Keempat sampel memiliki fitur struktur manual di `features.jsonl`. Tiga artikel ICCLE berasal dari jurnal/penerbit yang sama, sehingga tidak dianggap tiga source family independen untuk membuktikan norma genre; satu panduan World Bank memperluas sampel ke genre dokumen profesional.
- Inspeksi visual sumber dilakukan secara terbatas: beberapa halaman per PDF, bukan seluruh halaman. Ini bukan validasi visual keluaran generator.
- Belum ada eksperimen Document Craft baseline-vs-kandidat, penilaian reviewer, atau bukti empiris peningkatan kualitas.

Korpus ini masih eksploratif. Klaim faktual sumber belum diverifikasi independen secara menyeluruh. Metadata dan ciri turunan disimpan tanpa salinan PDF atau teks penuh.
