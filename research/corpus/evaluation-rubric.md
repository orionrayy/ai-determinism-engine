# Rubrik benchmark Document Craft v1

Versi rubrik: `1.0.0`. Bekukan sebelum kandidat dibandingkan. Setiap dimensi diberi nilai ordinal 0–4 dengan bukti yang dicatat; jangan gabungkan semua dimensi menjadi satu skor tanpa melaporkan subskor. Reviewer tidak diberi label kondisi (baseline/kandidat) bila penilaian blind memungkinkan.

## Skala umum

- **0 — gagal:** cacat mendasar, tidak memenuhi tujuan, atau ada bukti penting yang direkayasa.
- **1 — lemah:** kekurangan besar yang membatasi kegunaan dan memerlukan revisi substansial.
- **2 — cukup:** dapat digunakan setelah revisi yang jelas; masih ada kelemahan bermakna.
- **3 — baik:** sesuai tujuan dan konvensi yang didukung sumber; hanya revisi minor.
- **4 — sangat baik:** akurat, lengkap, spesifik, koheren, dan hampir siap digunakan tanpa cacat bermakna.
- **N/A — tidak relevan:** hanya boleh dipakai dengan alasan tertulis; jangan mengubah N/A menjadi nol.

## Dimensi dan bobot evaluasi

| Dimensi | Bobot awal | Bukti yang harus dicatat |
|---|---:|---|
| Kesesuaian tujuan, instruksi, dan pembaca | 15% | Matriks pertanyaan/kriteria → lokasi jawaban |
| Struktur spesifik genre/institusi | 15% | Pedoman formal atau contoh lintas sumber; komponen hilang/tidak perlu |
| Ketepatan fakta dan keterlacakan bukti | 20% | Klaim → data/sumber pendukung; tandai klaim tanpa dasar |
| Substansi, argumentasi, dan sintesis | 15% | Hubungan data, penalaran, batasan, dan kesimpulan |
| Kekhususan serta konsistensi konteks | 10% | Istilah, angka, tanggal, nama bagian, dan konsistensi antarbagian |
| Naturalitas, register, dan koherensi bahasa | 10% | Kejelasan, pengulangan, transisi logis, variasi yang berfungsi; bukan “kesalahan manusia” |
| Sitasi dan daftar referensi | 5% | Validitas bibliografi dan kecocokan sitasi |
| Kualitas visual/teknis | 10% | Hanya dinilai bila file keluaran dan metode inspeksi tersedia |

Bobot adalah rancangan awal untuk benchmark umum. Sebelum studi per genre, reviewer boleh menandai dimensi yang tidak relevan; bobot tidak boleh diubah setelah melihat identitas kandidat. Untuk DOCX/PDF/Excel, tambahkan checklist teknis terpisah.

## Protokol eksperimen

1. Simpan kasus masukan, tugas, sumber yang diizinkan, versi instruksi/prompt, commit, konfigurasi/model/generator, tanggal, output asli, dan log.
2. Baseline harus dibuat dari kode Document Craft yang tercatat pada commit; jangan menyebut output generik sebagai perbandingan satu-banding-satu jika input/data aslinya tidak diberikan kepada generator.
3. Pisahkan kelompok berdasarkan `source_family_id` agar salinan/versi satu karya tidak bocor ke pengembangan, validasi, dan uji.
4. Nilai dokumen asli hanya sebagai deskripsi distribusi pola. Jangan memperlakukan satu dokumen sumber sebagai ground truth universal atau memberi nilai 4 hanya karena berasal dari manusia.
5. Uji factual/citation accuracy hanya bila tersedia sumber pembanding. Fitur panjang kalimat/keterbacaan bukan pengganti penilaian substansi.
6. Catat reviewer, konflik penilaian, alasan, dan kesepakatan antarpenilai bila jumlah penilai memungkinkan pengukuran yang bermakna.
7. Laporkan hasil per genre, rentang/ketidakpastian, sampel yang dikeluarkan, negative results, dan keterbatasan. Klaim peningkatan membutuhkan perbandingan berulang dan tidak boleh menyembunyikan regresi lintas genre.

## Template catatan satu temuan

- ID temuan:
- Genre/sumber keluarga:
- Bukti/referensi:
- Masalah teramati:
- Dimensi dan skor:
- Dampak pada pembaca/kebenaran/format:
- Perubahan yang diusulkan:
- Regression test terkait:
- Status verifikasi:
