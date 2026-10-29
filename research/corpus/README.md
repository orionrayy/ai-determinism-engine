# Hallmark — korpus referensi Document Craft

Status awal pada 2026-10-30: pipeline validasi disiapkan; manifest sampel utama sengaja kosong sampai dokumen individual melewati pemeriksaan provenance, hak penggunaan, privasi, dan duplikasi. Angka 500 adalah target penerimaan, bukan jumlah hasil pencarian. Berkas registry sumber hanya berisi tempat penemuan dan **tidak** dihitung sebagai sampel.

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

## Status awal yang terukur

- Dokumen manifest yang lolos validasi: 0 (manifest kosong pada checkpoint awal).
- Target: 500 dokumen unik yang terverifikasi.
- Status: `INSUFFICIENT_EVIDENCE`.
- Tidak ada klaim benchmark kualitas atau validasi visual DOCX/PDF/Excel pada checkpoint ini.
