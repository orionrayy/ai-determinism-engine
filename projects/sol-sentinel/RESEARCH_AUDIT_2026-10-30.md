# SOL Sentinel — audit riset dan checkpoint teknis

**Tanggal audit:** 30 Oktober 2026 (UTC; waktu laporan proyek mengikuti log GitHub Actions)  
**Status akhir riset:** \`INSUFFICIENT_EVIDENCE\`  
**Mode yang diizinkan:** \`RESEARCH_ONLY\` — data publik, backtest historis dan simulasi offline saja. Tidak ada order riil.

## Ringkasan eksekutif

Cabang kandidat SOL Sentinel telah memiliki scanner, perhitungan indikator multi-timeframe, pengujian historis, evaluasi kronologis 60/20/20, analisis biaya dan mesin simulasi dengan ledger SQLite. Pada commit yang diaudit, perbaikan validasi spread dan telemetry backtest telah lolos CI. Namun, baseline tetap tidak menghasilkan transaksi pada jendela historis 30 hari. Karena itu expectancy, win rate dan profit factor out-of-sample tidak dapat dinilai; P&L nol hanya mencerminkan tidak adanya posisi, bukan keberhasilan, keamanan atau profitabilitas strategi.

Implementasi kandidat berada di branch \`feat/sol-sentinel-paper-trader-v1\`, commit \`7d74015ecb99c0f1768baddd40b1643399c5bea2\`, melalui [PR #178](https://github.com/orionrayy/ai-determinism-engine/pull/178). PR masih draft/open dan belum digabung ke main. Laporan ini dan pembaruan canonical memory diajukan pada branch dokumentasi terpisah agar tidak menimpa salinan memori yang lebih lama di branch kandidat.

## 1. Bukti baseline dan artefak

| Item | Hasil terverifikasi |
|---|---|
| Main saat checkpoint | \`d5672c84a0ae184c3a95d3f925925887ece71aa4\` |
| Branch kandidat | \`feat/sol-sentinel-paper-trader-v1\` |
| SHA kandidat | \`7d74015ecb99c0f1768baddd40b1643399c5bea2\` |
| Status PR #178 | Draft, open, unmerged |
| CI unit test | [Run #268](https://github.com/orionrayy/ai-determinism-engine/actions/runs/40928041385), sukses |
| Hasil tes | 70 tes lulus; log melaporkan 0.085 detik |
| Workflow riset | [Run #25](https://github.com/orionrayy/ai-determinism-engine/actions/runs/40928032346), sukses |
| Artefak riset | [Artifact 12921441578](https://github.com/orionrayy/ai-determinism-engine/actions/runs/40928032346/artifacts/12921441578), 1,223,457 byte pada saat diperiksa, retensi tujuh hari |
| Dataset | 43.200 candle SOLUSDT 1 menit |
| SHA-256 dataset | \`ccc89cec753644f11248bc75ac8b7b13fd9042ef4eb1d5a72074a75cdb9e8b26\` |
| Jendela dataset | 2026-09-29 18:08:00 UTC – 2026-10-29 18:07:59.999 UTC |
| Kurs konversi | USD/IDR 18.158, tanggal 2026-10-29, Frankfurter; proxy denominasi USDT, bukan harga transaksi |

Kloning dan eksekusi test lokal tidak berhasil dilakukan di sesi ini karena container tidak dapat me-resolve \`github.com\`. Hasil tes yang dinyatakan lulus bersumber dari log GitHub Actions, bukan klaim eksekusi lokal.

## 2. Spread Reku: akar masalah dan perbaikan saat ini

Definisi spread konvensional terhadap harga tengah adalah:

\[
\mathrm{spread}_{bps} =
\frac{\mathrm{ask}-\mathrm{bid}}{(\mathrm{ask}+\mathrm{bid})/2}
\times 10{,}000
\]

Dua fixture pembeda penting:
- Bid 100, ask 102: mid 101, spread 2, hasil **198,01980198 bps**.
- Bid 99, ask 101: mid 100, spread 2, hasil **200 bps**.

Jadi output sekitar 198,0198 bps tidak berarti rumus midpoint salah. Ia benar untuk bid 100/ask 102, sedangkan 200 bps benar untuk bid 99/ask 101. Catatan historis menyebut test memperoleh 198,0198 ketika harapannya 200; secara matematis hal itu konsisten dengan fixture/ekspektasi yang tidak cocok. Fixture pre-fix asli tidak ditemukan sebagai artefak independen dalam tree saat audit, sehingga kita tidak mengklaim mengulang keadaan pra-perbaikan secara persis.

Di commit kandidat sekarang, fungsi **spread_metrics** menolak harga yang hilang, nonnumerik, boolean, nonfinite, nol/negatif dan kuotasi crossed. String numerik sengaja diubah menjadi float untuk kompatibilitas respons API. Midpoint dihitung dengan penjumlahan setengah harga untuk menghindari overflow pada harga finite yang sangat besar. Test mencakup spread nol, presisi sangat kecil/besar, quote invalid/crossed, serta formula midpoint. Ini sudah diverifikasi melalui tes CI 70/70, tetapi tidak berarti endpoint Reku selalu stabil atau spread satu snapshot mewakili riwayat.

## 3. Diagnosis nol transaksi

### Dataset dan split

| Segmen kronologis | Candle | Observasi sinyal | Filter-rejection events | Kandidat | Order simulasi dicoba | Fill | Closed trades | Net P&L IDR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Development 60% | 25.920 | 25.919 | 92.291 | 0 | 0 | 0 | 0 | 0 |
| Validation 20% | 8.640 | 8.639 | 29.232 | 0 | 0 | 0 | 0 | 0 |
| Out-of-sample 20% | 8.640 | 8.639 | 29.510 | 0 | 0 | 0 | 0 | 0 |

Periode development: 29 September – 17 Oktober 2026 UTC. Validation: 17 – 23 Oktober 2026 UTC. Out-of-sample: 23 – 29 Oktober 2026 UTC. Tiap segmen dimulai dengan ekuitas simulasi sendiri sebesar Rp300.000; pengembalian antarsplit tidak digabung atau dikompaun.

### Pola penolakan yang terukur

Counter pada mesin adalah **overlapping reasons**: satu candle dapat gagal di lebih dari satu filter. Karena itu total rejection events melebihi jumlah observasi dan tidak boleh dijumlahkan sebagai jumlah candle unik.

| Kombinasi filter gagal bersamaan | Development | Validation | Out-of-sample |
|---|---:|---:|---:|
| Tren 15m + momentum 5m + RSI 5m + breakout 1m + volume relatif 1m | 11.543 | 6.959 | 2.079 |
| Breakout 1m + volume relatif 1m | 4.760 | 2.760 | 919 |
| Tren 15m + momentum 5m + breakout 1m + volume relatif 1m | 4.120 | 2.562 | 748 |

Counter per filter pada full-window baseline: breakout 1m ditolak 40.141 kali; volume relatif 1m 34.800; momentum 5m 28.248; tren 15m 25.313; RSI 5m 23.648; momentum/exhaustion RSI 1m 1.787; dan edge setelah biaya di bawah ambang 198 kali. Counter filter bertumpang tindih; angka tersebut tidak independen.

**Kesimpulan yang terbukti:** tidak ada sinyal yang melewati seluruh rangkaian teknis ditambah filter biaya pada konfigurasi yang diuji; karenanya tidak ada order attempt/fill/posisi/closed trade. Sebanyak 198 evaluasi pada full-window ditolak oleh cost gate setelah persyaratan teknis sebelumnya lolos. Pada split yang dijalankan terpisah, cost gate tercatat 119 development, 43 validation, dan 34 out-of-sample; jumlah split 196 karena boundary/warm-up berbeda dari full-window yang mengandung dua observasi tambahan. Ini bukan bukti anomali.

**Hipotesis, bukan penyebab tunggal yang terbukti:** konjungsi tren 15m, momentum/RSI 5m, breakout 1m, volume relatif dan net-edge setelah biaya terlalu selektif untuk sampel ini. Telemetry kini menunjukkan kombinasi gagal bersama agar langkah berikutnya bisa diuji satu faktor per eksperimen; data ini tidak membenarkan pelonggaran banyak ambang sekaligus. Jangan memilih parameter berdasarkan out-of-sample yang sama.

## 4. Biaya dan kompatibilitas venue

Halaman biaya resmi Reku yang diperiksa pada 30 Oktober 2026 menampilkan Pro buy 0,1% + 0% PPN + 0,00555% CFX dan Pro sell 0,1% + 0,21% PPh + 0,00555% CFX. Jumlah asumsi fee dua sisi bernilai sekitar **42,11 bps**. Referensi resmi: https://reku.id/fees/crypto dan pengumuman penyesuaian CFX yang berlaku 1 Oktober 2026: https://reku.id/information/informasi-penyesuaian-biaya-cfx.

| Skenario yang telah ditentukan sebelum evaluasi | Biaya pulang-pergi | Ambang gross move termasuk buffer 25 bps | Transaksi | Net P&L |
|---|---:|---:|---:|---:|
| Reku Pro baseline: fee 42,11 + asumsi spread 10 + asumsi slippage 20 | 72,11 bps | 97,11 bps | 0 | Rp0 karena tidak ada posisi |
| Pro friction hipotetis lebih rendah: fee 42,11 + spread 5 + slippage 10 | 57,11 bps | 82,11 bps | 0 | Rp0 karena tidak ada posisi |
| Pro fee-only theoretical floor: fee saja, spread/slippage nol | 42,11 bps | 67,11 bps | 0 | Rp0 karena tidak ada posisi |
| Lightning fee-only theoretical floor: fee 22,11 bps, spread/slippage nol | 22,11 bps | 47,11 bps | 10 | **−Rp3.893,22** |

Lightning fee-only floor memiliki return neto −1,29774%, profit factor 0,2760 dan max drawdown 1,62433%. Skenario ini sengaja mengasumsikan spread/slippage nol dan **bukan** representasi kondisi eksekusi yang realistis. Ia juga menunjukkan bahwa memperkecil asumsi biaya secara tidak realistis tidak membuktikan edge.

Snapshot order book publik Reku pada run terbaru menunjukkan best bid Rp1.825.000, best ask Rp1.830.000 dan spread midpoint **27,359781 bps**. Skenario sensitivitas yang mempertahankan snapshot tersebut sepanjang jendela historis menghasilkan biaya model 89,469781 bps, ambang termasuk buffer 25 bps sebesar 114,469781 bps, dan tetap nol transaksi. Spread itu hanya satu snapshot; bukan time series spread historis.

Binance mendokumentasikan endpoint market data publik termasuk klines di https://github.com/binance/binance-spot-api-docs/blob/master/rest-api.md dan https://github.com/binance/binance-spot-api-docs/blob/master/faqs/market_data_only.md. Dataset Binance SOLUSDT yang dikonversi menggunakan referensi USD/IDR **bukan** feed candle eksekusi SOL/IDR Reku. Satuan timestamp, rule tick/lot/notional, dan order API tiap venue harus dipastikan terpisah.

Probe historis native Reku tidak lulus: legacy public \`/v2/activitytrans\` memberikan HTTP 404, dan kolektor sekarang menandai \`endpoint_not_found\`. Collector tidak mengubah data parsial menjadi riwayat penuh dan tidak membuat candle/trade sintetis. Karena bukti histori native Reku tidak tersedia, kompatibilitas harga eksekusi Reku belum tervalidasi.

## 5. Risiko, ledger, dan mode eksekusi

Kode kandidat memakai konfigurasi biaya eksplisit, sizing berdasarkan risk/exposure/cash, minimum notional per skenario, stop/target ATR, batas kerugian harian, fill entry pada open candle berikutnya, dan resolusi stop-first ketika stop serta target tersentuh pada candle yang sama. Laporan backtest menyediakan biaya, P&L neto, drawdown, losing streak, expectancy, win rate, profit factor, duration, rejection counters, order attempts, simulated fills, size rejections dan daily-lock telemetry.

Tes ledger yang ada menguji idempotensi run ID serta trade ID dengan SQLite \`INSERT OR IGNORE\`. Angka runtime untuk paper engine masih berasal dari simulator/replay, bukan akun atau saldo live. Tidak ada private endpoint, order placement, cancel, transfer, deposit atau withdrawal yang digunakan dalam checkpoint ini; adapter perdagangan riil tetap dinonaktifkan.

## 6. DAG milestone dan status penerimaan

| ID | Milestone | Status | Bukti / hambatan |
|---|---|---|---|
| P0 | Pulihkan keadaan dan lindungi perubahan | PASSED | Main SHA, branch, PR, CI dan artefak diidentifikasi; tidak ada reset/force-push |
| P1 | Reproduksi dan perbaiki spread | PASSED | Rumus midpoint, validasi invalid/extreme, 70 tes CI lulus |
| P2 | Rekonsiliasi branch/CI | BLOCKED | PR #178 masih draft; kandidat di belakang main dua commit; tidak dilakukan merge/rebase/force-push |
| P3 | Diagnosis nol transaksi | PASSED | Split metrics, rejection combinations, cost gate dan zero-fill telemetry tersedia |
| P4 | Integritas data/indikator | BLOCKED | Data publik dan hash terverifikasi, tetapi tidak tersedia native Reku history untuk validasi silang |
| P5 | Eksperimen strategi terkontrol | PASSED | Baseline + skenario biaya yang dideklarasikan; tidak ada optimisasi otomatis |
| P6 | Venue dan biaya | BLOCKED | Fee publik terverifikasi dan spread snapshot diukur; slippage diasumsikan, histori/price matching Reku belum terbukti |
| P7 | Backtest, risiko, ledger | BLOCKED | CI dan tes sintetis lulus; zero OOS trades gagal memenuhi bukti edge dan sampel yang memadai |
| P8 | Kompatibilitas Termux | BLOCKED | Perangkat Android tidak dapat diakses; belum ada eksekusi diagnostik di perangkat |
| P9 | Persistensi dan artefak | PASSED | Laporan, canonical memory checkpoint dan log tambahan diajukan untuk review di branch/PR terpisah |
| P10 | Status kesiapan | PASSED | \`RESEARCH_ONLY\` + \`INSUFFICIENT_EVIDENCE\`; bukan \`PAPER_READY\` |

## 7. Kriteria minimum sebelum promosi mode

1. Rekonsiliasi branch terhadap main dengan commit/tree diff yang ditinjau. Jangan merge PR atau mengubah branch protection otomatis.
2. Verifikasi sumber native Reku, atau dokumentasikan batasan tanpa menyebut backtest Binance sebagai Reku-native.
3. Tidak ada kerusakan data, mismatch timeframe atau kebocoran masa depan; validasi indikator dengan fixture deterministik dan referensi independen.
4. Bekukan parameter kandidat sebelum out-of-sample yang independen. Jangan menyesuaikan threshold berkali-kali pada periode yang sama.
5. Minimum evidence riset yang telah ditetapkan: expectancy neto OOS positif setelah biaya; profit factor minimal 1,2 bila terdefinisi; target 100 closed trades jika bisa didapat tanpa mengubah metodologi; drawdown di bawah batas risiko; dan hasil tidak runtuh dalam skenario biaya yang realistis. Saat jumlah transaksi nol, metrik terkait harus tetap null/tidak terdefinisi.
6. Jalankan replay dan restart ledger dengan input sama; pastikan tidak ada duplicate order/trade/event dan posisi/ekuitas konsisten.
7. Jalankan RUNBOOK_TERMUX.md secara aktual pada perangkat Android; sesi ini tidak mengklaim validasi tersebut.
8. Paper mode harus tetap terpisah dan tidak mengaktifkan real orders. Penggunaan real order memerlukan otorisasi terpisah, mekanisme kill switch, rekonsiliasi serta peninjauan risiko yang belum menjadi sasaran audit ini.

Target ekuitas Rp3.500.000 adalah aspirasi proyek, bukan bukti saldo, keuntungan yang telah dicapai atau proyeksi terjamin. Tidak ada hasil kuantitatif pada checkpoint ini yang mendukung klaim target tersebut akan tercapai.

## 8. Sumber dan tautan

- [PR #178 — implementasi kandidat](https://github.com/orionrayy/ai-determinism-engine/pull/178)
- [SOL Sentinel Tests #268](https://github.com/orionrayy/ai-determinism-engine/actions/runs/40928041385)
- [SOL Sentinel Research #25](https://github.com/orionrayy/ai-determinism-engine/actions/runs/40928032346)
- [Artefak riset](https://github.com/orionrayy/ai-determinism-engine/actions/runs/40928032346/artifacts/12921441578)
- [Reku fee schedule](https://reku.id/fees/crypto)
- [Reku CFX fee update](https://reku.id/information/informasi-penyesuaian-biaya-cfx)
- [Binance Spot REST API](https://github.com/binance/binance-spot-api-docs/blob/master/rest-api.md)
- [Binance market-data-only endpoints](https://github.com/binance/binance-spot-api-docs/blob/master/faqs/market_data_only.md)
- [Frankfurter USD/IDR rate source](https://api.frankfurter.dev/v2/rate/USD/IDR)

**Batas interpretasi:** hasil adalah bukti untuk dataset, konfigurasi, commit, dan waktu run yang dicantumkan. Data publik dapat berubah; artefak GitHub Actions dapat kedaluwarsa; hasil riset bukan rekomendasi transaksi atau janji keuntungan.
