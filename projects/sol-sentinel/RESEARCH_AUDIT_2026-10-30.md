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


## Follow-up after ledger and timestamp hardening — 2026-10-30

This section supersedes the earlier run #25 capture metrics as the latest research observation while preserving the original run history above.

### Follow-up code and test commits
- \`2cac2368e2a6fa992335a07926a4994770be1e7f\`: fixed SQLite replay conflicts. The same \`run_id\` is accepted as idempotent only when the canonical serialized report and its trade rows match exactly; conflicting report data or corrupted/incomplete rows fail closed rather than being silently ignored. Added regression tests for identical run with trades, conflicting report content, and missing rows.
- \`833e57a13022c3efe6ff5115df2b2e0697072817\`: enforced candle close-time alignment for 1m CSV/direct backtest input and direct OHLCV validity. The loader/backtest rejects a close time other than \`open_time_ms + 59,999\`, non-finite OHLCV, invalid prices, negative volume, and impossible high/low.
- \`8fb11f7081a355958add549052b1ecec90464423\`: added independent hand-calculated fixtures for EMA, RSI and ATR.
- Latest branch head test workflow [#278](https://github.com/orionrayy/ai-determinism-engine/actions/runs/40930666859): **79 passed, 0 failed**. Research run [#27](https://github.com/orionrayy/ai-determinism-engine/actions/runs/40930361106) succeeded on the production-code SHA \`833e57a...\`, with 76 tests in its compile/test step. Artifact [12922686248](https://github.com/orionrayy/ai-determinism-engine/actions/runs/40930361106/artifacts/12922686248), 1,223,413 bytes.

### Latest captured dataset and outcome
- Window: \`2026-09-29T18:23:00+00:00\` through \`2026-10-29T18:22:59.999000+00:00\`; 43,200 1m candles; dataset SHA-256 \`6e140e14be2e05d30819b96e90e37372ee13df55bb7d4f31fc9c9382bef86c14\`.
- Full baseline: 43,199 signal evaluations; 154,149 multi-label rejection events; zero signal candidates/order attempts/fills/trades; equity IDR 300,000 unchanged; net P&L and return 0 because no trade; expectancy, win rate and profit factor null/undefined.
- Split outcomes: development 25,920 bars / 92,282 rejection events; validation 8,640 / 29,263; out-of-sample 8,640 / 29,513. All splits zero closed trades.
- Highest-count joint failure (15m trend, 5m momentum, 5m RSI, 1m breakout, 1m relative-volume) was 11,544 development, 2,215 validation and 2,081 OOS observations. Full-window reason counters include 40,141 1m breakout failures, 34,801 missing 1m volume confirmations and 196 cost-gate rejections; reasons overlap, so totals are not unique candle counts.
- Latest quote snapshot: best bid IDR 1,825,000, ask IDR 1,830,000, midpoint spread 27.359781 bps, public orderbook/chart probes OK. Native Reku activity history remains unavailable due to a documented endpoint 404; no historical fills or candles were fabricated.
- Predeclared modeled costs are unchanged. Reku Pro modeled round-trip cost 72.11 bps, 97.11 bps including net-edge buffer; snapshot-spread sensitivity 89.469781 bps, 114.469781 bps including buffer. The theoretical Lightning fee-only case still loses IDR 3,893.219 over 10 trades at -1.29774%, PF 0.2760 and 1.62433% max drawdown under zero-spread/zero-slippage assumptions.
- Final readiness: \`RESEARCH_ONLY\` / \`INSUFFICIENT_EVIDENCE\`. Zero OOS trades means no independent evidence for positive net expectancy. The Rp3,500,000 target is aspirational. Real orders were not used, and Android/Termux remains untested.


## Final reconciliation and persistence verification — 2026-10-30

- Confirmed latest implementation/test chain: ledger commit \`2cac2368e2a6fa992335a07926a4994770be1e7f\`; candle timestamp/OHLCV commit \`833e57a13022c3efe6ff5115df2b2e0697072817\`; hand-calculated indicator fixtures \`8fb11f7081a355958add549052b1ecec90464423\`; latest completed tests [#280](https://github.com/orionrayy/ai-determinism-engine/actions/runs/40931067383) with 79 passed/0 failed. Research [#27](https://github.com/orionrayy/ai-determinism-engine/actions/runs/40930361106) succeeded on code SHA \`833e57a...\`, artifact #12922686248. Keep this source SHA distinct from later tests/docs-only commits.
- Branch compare main \`d5672c84a0ae184c3a95d3f925925887ece71aa4\` versus PR #178: diverged, merge base \`eaa864a554c2c995ce73bd9f6b4d07206ecc56c5\`, 167 commits ahead and 2 behind, with 27 files in the PR diff. Although GitHub marks it mergeable, the branch has not been reconciled or approved for merge; PR #178 remains draft.
- The canonical memory first write had a stale-blob mistake due to checking idempotency after appending the marker. Read-after-write validation caught it; a new blob was created, verified and then extended with the latest code/research follow-ups. The current memory file and this report are both read back from the documentation branch. The docs PR #180 is draft/open and based on current main.
- The latest source dataset from research #27 is SHA-256 \`6e140e14be2e05d30819b96e90e37372ee13df55bb7d4f31fc9c9382bef86c14\` over 43,200 Binance SOLUSDT 1m candles from \`2026-09-29T18:23:00+00:00\` to \`2026-10-29T18:22:59.999000+00:00\`. All 60/20/20 segments had zero trades, fills, and P&L from absent positions; expectancy, win rate and PF are null. Read the prior follow-up for split event counts and cost sensitivities.
- Readiness remains \`RESEARCH_ONLY\` / \`INSUFFICIENT_EVIDENCE\`. No live orders or account actions; Termux not tested; native Reku activity history blocked by HTTP 404. Target Rp3,500,000 remains aspirational.


## Latest compare refresh

A final fresh compare after all current branch documentation commits reports PR #178 as diverged from main: merge base \`eaa864a554c2c995ce73bd9f6b4d07206ecc56c5\`, 168 commits ahead, 2 behind, 27 changed files; current main SHA \`d5672c84a0ae184c3a95d3f925925887ece71aa4\`, feature branch head \`3d2054da51ebb34f98380c50257e61735e650475\`. This supersedes the earlier point-in-time count of 167 ahead. Keep the PR draft/unmerged; no history rewriting or integration was attempted.


## Latest fixed-head comparison

At the final feature journal head \`7e1b62df19a950f79f58eea2a0a1a118fdb03d2d\`, GitHub reports the PR #178 branch is diverged from main \`d5672c84a0ae184c3a95d3f925925887ece71aa4\`: merge base \`eaa864a554c2c995ce73bd9f6b4d07206ecc56c5\`, 169 commits ahead, 2 behind, 27 files in the PR diff. This supersedes earlier point-in-time 167/168 ahead counts. Keep the implementation PR draft and do not rewrite/merge its history without a dedicated review. No such integration action was performed in this session.


## Addendum — current infrastructure probe and CI reconciliation (2026-10-30)

This addendum supersedes earlier branch-head and CI fields where they differ; the historical dataset/run findings below remain the last reproducible research evidence, not a newly rerun backtest.

### Current repository state

- Main: `d5672c84a0ae184c3a95d3f925925887ece71aa4`.
- Implementation branch current head at this checkpoint: `0b5dac9f3ac64c1753437fcc4afe96b8171cc079`.
- PR #178 remains open and draft, not merged: https://github.com/orionrayy/ai-determinism-engine/pull/178.
- Current compare against main: `diverged`, 175 commits ahead / 2 behind, common base `eaa864a554c2c995ce73bd9f6b4d07206ecc56c5`, 29 changed files. Do not rebase or merge blindly; this branch includes a large project history.
- Current feature-branch CI: SOL Sentinel Tests runs #294/#295 on head `0b5dac9f...`, both passed: https://github.com/orionrayy/ai-determinism-engine/actions/runs/40936825849 and https://github.com/orionrayy/ai-determinism-engine/actions/runs/40936815027.
- The suite comprises 88 discovered `test_*` methods by source count at this head. GitHub Actions confirms the compile and unittest-discovery steps succeeded. The decoded GitHub Actions job log was retrieved via the GitHub connector and confirms `Ran 88 tests in 0.181s`, `OK` (0 failures) on the exact head. This matches an independent count of test methods across eight test modules.
- The previous 79-test run #284 is historical; #294/#295 are the latest verification for the quote-probe patch and updated Termux runbook/workflow.
- Latest research workflow #27 remains on earlier source commit `833e57a13022c3efe6ff5115df2b2e0697072817`; no fresh backtest on the newly changed head is claimed. The dataset SHA remains `6e140e14be2e05d30819b96e90e37372ee13df55bb7d4f31fc9c9382bef86c14` for that historical run.

### New read-only Jupiter probe

- Added `projects/sol-sentinel/jupiter_quote_probe.py` at source commit `ed203fcae1ed0176c27959330865e69e2fb282a7`.
- Added 9 fixture-based tests in `projects/sol-sentinel/test_jupiter_quote_probe.py` at commit `a0849839638c509cc5289c1c3d84f16c95eb08d7`.
- Added compile coverage for both files at commit `88918e8fe666e26406b1939bda6ad94c9ea9a16d`; documented a Termux read-only command at commit `0b5dac9f3ac64c1753437fcc4afe96b8171cc079`.
- Probe creates a Jupiter Swap V2 `/order` request without `taker`, so the code path is quote-only. It does not sign or submit; it rejects an unexpected transaction field, validates amount/mints and response fields, bounds timeout/response size, and never prints a supplied API key.
- Source docs: https://developers.jup.ag/docs/swap/order-and-execute and https://developers.jup.ag/pricing. The actual Jupiter live quote endpoint was **not** called in this audit because a real response is not required for fixture-based CI; live API schema/rate limits remain to be verified with a controlled read-only invocation from the user's environment.
- Latest quote-probe CI succeeded, but CI is only implementation and regression evidence, not trading strategy evidence.

### Infrastructure and target economics

See `projects/sol-sentinel/INFRASTRUCTURE_DECISION_2026-10-30.md` for the full official-source comparison, fee arithmetic, unknowns and preregistered gates. Key decision: use Reku as an on-ramp candidate, a dedicated self-custody Solana wallet as custody/signing boundary, and Jupiter V2 only for quote collection initially. Do not yet route live funds: current SOL external withdrawal charge/minimum and supported on-ramp/off-ramp path are not verified from a current official UI snapshot. Pintu Pro fee schedule was checked as a comparator, but official automated API support was not confirmed and it is not selected as a bot venue.

Published fee component arithmetic for same nominal Rp300,000 buy and sell:
- Reku Pro 0.4211% ≈ Rp1,263.30, before spread/slippage/withdrawal.
- Reku Lightning 0.2211% ≈ Rp663.30, before spread/slippage/withdrawal.
- Pintu Pro maker 0.4611% ≈ Rp1,383.30; taker 0.5611% ≈ Rp1,683.30, before spread/slippage/transfer. This is a fee table comparison, not a claim that Pintu is cheaper or an API-compatible execution venue.
- Solana base fee reference is 5,000 lamports/signature, with optional priority fee; failed transactions still pay network fees.
- Rp300,000 -> Rp3,500,000 equity is 11.6667x starting capital, equal to +1,066.67% cumulative return. Target remains aspirational and is not a forecast.

### Decision and readiness

Current mode remains `RESEARCH_ONLY`; strategy status remains `INSUFFICIENT_EVIDENCE`. The last strategy run has 0 trades in development, validation and OOS, so win rate, profit factor and expectancy cannot be estimated. No parameter changes were made to force a trade. No exchange order, wallet transfer, deposit, withdrawal, production credential, private key, transaction signature or submission was used. Do not advance to paper/live execution solely because CI passes.


## SOL Sentinel adapter hardening checkpoint — 2026-10-30 (session follow-up)
<!-- SOL_SENTINEL_SESSION_2026-10-30_ADAPTER_HARDENING_V3 -->

### Verified code state
- Implementation branch: `feat/sol-sentinel-paper-trader-v1`; verified code head `70b1734f260202fad70a60b31730c5097fb8bb95`. PR #178 remains draft/open/unmerged: https://github.com/orionrayy/ai-determinism-engine/pull/178.
- Latest CI for that exact head: SOL Sentinel Tests run #40942942848, conclusion `success`; runner log: `Ran 116 tests in 0.121s`, `OK`. https://github.com/orionrayy/ai-determinism-engine/actions/runs/40942942848
- Changes include `projects/sol-sentinel/execution_adapters.py`, `test_execution_adapters.py`, `PLATFORM_ADAPTER_DECISION_2026-10-30.md`, scanner/history guards, expanded CI compile coverage, and updated Termux runbook.
- A first run of the initial adapter patch failed two newly authored tests; root causes were an incorrect test fee arithmetic expectation (10 bps on Rp180,100 notional = Rp18.01, not Rp1.801) and a transfer fixture that supplied a deposit minimum while expecting it missing. Test fixture/math were corrected; no validation was weakened. Subsequent run #40941545790 passed 112 tests, and final head run #40942942848 passed 116.
- GitHub compare remains `diverged`, 196 commits ahead / 2 behind, 32 changed files at the latest reconciliation. Do not merge/rebase/force-push this branch blindly. Main has not been changed.

### Implemented adapter contracts
- Platform-neutral protocols now separate market data, capital source, custody/balance lookup, transfer planning, transaction signer, chain submitter, execution and ledger.
- Implemented `IndodaxPublicMarketDataAdapter` for public-only depth and pair catalogue reads; no API key is sent, no account balance endpoint is called, no orders are sent. It validates public metadata, exact market ID, bid/ask levels, bounded response size, network timeout, retry policy and a 0.35s minimum spacing (max theoretical about 171 calls/minute, under INDODAX's documented public cap of 180/minute). Current dynamic pair fields still need a direct real read in the deployment environment before treating their current values as operationally verified.
- Venue metadata maps `base_currency` as quote currency and `traded_currency` as base asset, reading `price_precision`/`pricescale`, `quantity_increment`, `trade_min_base_currency`, and `trade_min_traded_currency`; unit tests cover missing/ambiguous metadata. Both base quantity minimum and quote notional minimum are gated.
- Paper execution uses Decimal amounts, venue/symbol identity, tick/step, minimum quantity/notional, quote identity/age, future clock skew, spread cap, and marketability. Non-marketable limits return `NO_FILL`; repeated identical idempotency intents return the first receipt; changed payload under a reused key fails closed.
- Transfer assessment only checks source/destination network name, explicit verification flags, fee/minimum knowledge, net amount and stale/missing cost evidence. It always reports `transfer_sent=false`. `ExecutionPolicy(LIVE)` continues to reject even when approval flags are set because no live executor exists.
- Explicitly no private key/seed handling, no wallet signer, no signed exchange API, no production credentials, no transfer, no exchange order, and no mainnet execution.

### Input validation fixes
- `sol_sentinel.py`: identical duplicate candles deduplicate; same timestamp with conflicting OHLCV now raises `ValueError` instead of silent last-write-wins.
- `capture_history.py`: malformed kline rows, non-finite quote volume, negative quote volume, fractional/negative trade count, misaligned interval timestamps, nonstandard close time and conflicting duplicates are rejected. Capture still fails closed on insufficient/gapped history.
- `reku_history.py`: activity rows reject fractional side codes and zero/negative notional or quantity, in addition to existing finite/range checks. The legacy Reku activity history endpoint remains known to return HTTP 404; this code path remains unavailable until an official currently supported endpoint is verified.
- These changes distinguish malformed-data errors from strategy no-signal outcomes; they do not alter strategy thresholds.

### Strategy and venue decision — still blocked
- Research result #27 remains the latest verified full backtest, on engine SHA `833e57a13022c3efe6ff5115df2b2e0697072817`, dataset SHA `6e140e14be2e05d30819b96e90e37372ee13df55bb7d4f31fc9c9382bef86c14`, 43,200 Binance SOLUSDT 1m bars; all chronological 60/20/20 segments have 0 trades. New adapter tests are not new evidence of strategy profitability.
- The rejection counter totals are overlapping rejection events, not independent blocked candles. Root cause therefore remains multi-filter rejection plus economic gate, and no single threshold is proven to be the sole culprit. Do not lower thresholds to manufacture entries. Costs remain scenario assumptions, not measured fill friction.
- INDODAX SOL/IDR is the first exchange-adapter candidate because its official market page lists an active SOL/IDR pair, and official Trade API docs distinguish View, Spot Trading, and Deposit & Withdraw permissions. Any eventual automation should use the least permission required; never combine withdrawal permission into the execution key.
- Official INDODAX API PDF (copyright 2026) documents public `/api/pairs`, `/api/depth/$pair_id`, `/api/server_time`, and a public rate cap of 180 requests/minute; examples list pair increment/minimum fields. We reviewed the rendered Public REST page (PDF page 84) and the documented schema. Direct API JSON for `/api/pairs` and `/api/depth/solidr` could not be accessed through the current web connector, and the GitHub CI used mocks, so actual endpoint JSON compatibility is not yet verified in this session.
- Official INDODAX SOL/IDR page snapshot at audit time showed price Rp1,822,124 and 24h volume about Rp5.9bn; this is volatile display data, not a quote used for the backtest or proof of executable spread. https://indodax.com/trade/SOLIDR
- Reku remains an on-ramp candidate only. Its official fee page lists Pro buy 0.1% + 0.00555% CFX; Pro sell 0.1% + 0.21% PPh + 0.00555% CFX; Lightning buy 0% + 0.00555% CFX and sell 0% + 0.21% PPh + 0.00555% CFX. External send fees/minimums vary by network; the current SOL-specific withdrawal amount/minimum was not verified in parsed official content. https://reku.id/fees/crypto
- Jupiter Swap V2 remains a self-custody alternative with quote-only probe implemented, but not an execution adapter. The official Developer Platform states keyless is 0.5 RPS and free API key is 1 RPS; programmatic signing, swap submission, wallet custody, priority-fee and IDR off-ramp costs remain unimplemented/unverified. https://developers.jup.ag/
- Current official Solana docs state base fee 5,000 lamports/signature plus optional priority fees; failed transactions still pay fees. This is not the complete cost of a routed swap. https://solana.com/docs/core/fees
- Indonesian platform legal eligibility and current SOL network deposit/withdrawal availability must be checked per current official account/app before a separately approved transfer. OJK's September 2026 update notes two crypto exchanges and 34 licensed crypto-asset ecosystem entities; do not infer every exchange, token or network route is supported. https://iru.ojk.go.id/iru/WebSite/ArticleList/View/1084_September_2026_Board-of-Commissioners-Meeting-Financial-Services-Sector-Stability-Remains-Resilient-Amid-Global-Economic-Pressures

### Readiness
- Technical contract readiness: improved, with 116 offline tests passing on latest implementation head.
- Strategy readiness: `INSUFFICIENT_EVIDENCE` (zero historical trades).
- Operational readiness: `RESEARCH_ONLY`; public INDODAX endpoints not directly verified from this session; Termux still not tested.
- Live readiness: blocked. No live executor, no approval gate completed, no actual order/transfer or wallet signing.
- Next: run an isolated real *public read-only* INDODAX metadata/depth smoke test from an allowed network; inspect response fields and server-time freshness; then add read-only contract check. In parallel, diagnose signal rejection combinations and preserve fixed parameters for validation/OOS. Update the two PRs only after checking their current head/base and CI; no merge or base rewrite without reviewed merge plan.


## Execution follow-up — adapter contracts and fresh CI proof (2026-10-30)

### Verified change and test state

Implementation branch: `feat/sol-sentinel-paper-trader-v1`, current code head `5f2acac4bddfa3dc72c4dea3bc429e14de9d6172`.

- Latest branch-head CI: [run #40945972566](https://github.com/orionrayy/ai-determinism-engine/actions/runs/40945972566), successful. Decoded Actions log reports `Ran 120 tests ... OK`. This is offline compile/unit-test evidence, not a live endpoint probe or strategy profitability result.
- New module: `projects/sol-sentinel/execution_adapters.py`; tests: `projects/sol-sentinel/test_execution_adapters.py`; venue decision: `projects/sol-sentinel/PLATFORM_ADAPTER_DECISION_2026-10-30.md`.
- The adapter layer isolates market data, capital-source metadata, custody balance interface, transfer planning, paper execution and ledger contracts. The INDODAX adapter reads the public pair catalogue and depth only; its test transport is mocked. The paper executor is in-memory and quote-crossing only. Live policy remains unconditionally blocked.
- Current feature branch is still diverged from current `main`: 207 ahead / 2 behind; merge base `eaa864a554c2c995ce73bd9f6b4d07206ecc56c5`; 32 changed files. PR #178 remains draft/open/unmerged. Do not blindly rebase/merge.

### Error register

| Priority | Finding | Root cause / proof | Current state |
|---|---|---|---|
| P1 | A route whose withdrawal fee consumes the entire requested amount could be represented as a nominal plan with zero remainder | A regression was added for fee == requested amount. The planner returns zero received amount and adds `no_transfer_value_remaining`; the route is not approved. | Fixed/guarded; regression is in the passing suite. |
| P1 | Transfer verification flags could accept non-boolean truthy input | A string such as `"false"` is truthy in Python if consumed as an ordinary bool. `TransferRoute` now rejects non-boolean values at construction. | Fixed; regression is in the passing suite. |
| P1 | Reusing an idempotency key for a different order could cause paper-state ambiguity | Paper adapter fingerprint and ledger conflict tests require identical intent/receipt content on replay. | Guarded by current unit tests; durable cross-process exchange idempotency is not implemented. |
| P2 | Conflicting duplicate source candles were silently overwriteable by a last-row-wins map | Same timestamp but differing OHLCV was not distinguishable from an identical duplicate. Scanner/capture now deduplicate identical rows and fail closed for conflicting history rows. | Fixed with regression tests; no live-source smoke test performed here. |
| P2 | Historical activity rows could include fractional side identifiers or empty/zero-quantity trades | Side values were cast directly to int and zero values could enter the parsed tape. The parser now rejects non-integral side values, unsupported side values and non-positive trade economics. | Fixed with regression coverage; native Reku activity endpoint still is not verified and the last research evidence for legacy activity history returned HTTP 404. |
| P2 | Market-data requests required an explicit documented venue boundary | Earlier engine contract did not define a portable venue/precision/routing interface. Added venue-neutral protocols and INDODAX public `/api/pairs` / `/api/depth/{pair_id}` adapter, finite timeouts, bounded retries, capped response size, public rate pacing, Decimal tick/lot/minimum checks and bid/ask validation. | Mock-tested and CI green; public endpoint behavior has not been smoke-tested from this session. |
| P2 | Venue-specific minimum and fees can be confused with generic market metadata | Official INDODAX help states that current all-in fee detail is displayed in Profile/Trading Fees, effective 1 October 2026; Lite min is IDR 10,000 and direct Pro min is IDR 25,000 (orders IDR 10,000–24,999 route via Lite). The per-market pair metadata alone is not a substitute for the account/mode rule. | Documented; add an explicit mode-specific simulator floor before using the adapter in venue-specific paper tests. |
| P1 | Strategy still has no demonstrated tradable edge | Last full 60/20/20 historical research run #27 had 0 trades in development, validation and OOS on dataset SHA `6e140e14be2e05d30819b96e90e37372ee13df55bb7d4f31fc9c9382bef86c14`. A new complete research run on the latest adapter branch was not completed. | Open blocker; `INSUFFICIENT_EVIDENCE`. |

### Platform decision and remaining proof

Reku is treated only as the on-ramp. Reku's current public coin-status page lists SOL deposit/withdrawal as Active and its official notice documents the Solana network, but the current account-specific withdrawal amount/fee/minimum still must be confirmed in the official app. See https://reku.id/status and https://reku.id/information/reku-support-sol-solana-network-dan-ftm-ethereum-network.

INDODAX is the initial centralized-exchange adapter candidate because the official 2026 API documentation exposes a no-key public pair catalogue and depth route, with a stated 180 requests/min public limit. Its current per-account all-in fee and exact SOL deposit-network acceptance are not certified by this test suite. Official references: https://indodax.com/downloads/INDODAXCOM-API-DOCUMENTATION.pdf and https://help.indodax.com/hc/en-us/articles/4416646599705-Details-of-Transaction-Fees-on-INDODAX. The latter says minimum order value IDR 25,000 for direct Pro and IDR 10,000 for Lite; exact current fee components are shown in the account UI.

Jupiter remains quote-only; free API key limit is 1 RPS, keyless is 0.5 RPS. Solana base fee is documented as 5,000 lamports per signature plus optional priority fee; failed transactions can still pay fees. Sources: https://developers.jup.ag/pricing, https://developers.jup.ag/changelog/developer-platform, and https://solana.com/docs/core/fees.

### Safety / gate status

No transfer, order, wallet connection, signature, private key, production credential, or secret was used. No live execution adapter exists. `RESEARCH_ONLY / INSUFFICIENT_EVIDENCE` remains mandatory until strategy evidence, direct public endpoint validation, network and fee proof, Termux verification, paper-forward observation, and all required readiness gates pass.


## Current reconciliation update — native INDODAX candles and source-aware costs (2026-10-30 WIB)

### Latest verified CI and changed contracts

- Code head: `39d093b6e44635431475ea635f018327310793fd`, branch `feat/sol-sentinel-paper-trader-v1`.
- Exact-head push test run [#40949432502](https://github.com/orionrayy/ai-determinism-engine/actions/runs/40949432502) succeeded with 135 offline tests and 0 failures. The PR-triggered duplicate run was still running when this update was created.
- Native IDR data adapter/capture exists in `execution_adapters.py` and `capture_indodax_history.py`; regression files include `test_execution_adapters.py`, `test_capture_indodax_history.py`, and updates to paper/evaluator tests. A read-only adapter has no order/sign/transfer method. The live policy gate remains closed.

### Newly verified public endpoint observation

Successful [research run #40948530609](https://github.com/orionrayy/ai-determinism-engine/actions/runs/40948530609) captured 30 days of native INDODAX SOL/IDR history: 43,200 contiguous 1m bars, `data_status=OK`, no FX conversion, dataset SHA-256 `4297c660ed1c3485bb5641babb5a286b8738290e88f8bb18c9f58bb903fa687b`; range 2026-09-29 20:24:00 UTC to 2026-10-29 20:23:59.999 UTC; pair ID `solidr`. This was a public, unauthenticated request from a GitHub Actions runner. It does not verify account eligibility, SOL deposit network, withdrawal fee/minimum, order permissions, or Android/Termux behavior.

The same research run showed:
- Reku public market snapshot returned `data_status=OK`.
- Legacy Reku public activity history returned HTTP 404; native Reku transaction tape is not available through that route.
- The Binance SOLUSDT strategy baseline produced 0 signal candidates and 0 trades. The fixed 60/20/20 splits also produced 0 candidates/trades.
- The Reku Lightning fee-only floor (zero spread/slippage, explicitly theoretical) generated 10 simulated trades and roughly IDR -3,893.22 net. Do not treat this as an executable strategy or forecast.

### Error/mitigation register — current

| Priority | Finding | Root cause | Fix / state |
|---|---|---|---|
| P1 | Native IDR prices could be multiplied by USD/IDR if reused through the old Binance-centric backtest path | Quote currency/source was not modeled as a first-class input; native IDR and USDT were not contractually distinct | Source venue/pair/currency now comes from capture sidecar or explicit declaration; native IDR uses rate 1.0 and report records no FX conversion. Regression covers IDR identity and USDT single conversion. |
| P1 | Reku cost sensitivity could have been run against a native INDODAX IDR file | Preset scenarios encoded Reku fees but were not venue-gated | `cost_sensitivity.py` now rejects native IDR data, explaining that a separate venue-specific fee model is required. |
| P1 | A native IDR backtest could proceed with a declared but zero order minimum or non-finite fee assumptions | Explicit non-null checks did not alone enforce viable numeric bounds | `resolve_fee_inputs()` now rejects non-finite/negative fees and spread/slippage and rejects a non-positive minimum order notional before backtest config creation. A regression test was added; latest CI was still queued for that exact last patch at the time of writing. |
| P1 | Bad capture provenance could be consumed as if valid | Sidecar source/status/hash are external inputs | Backtest source resolver rejects non-OK sidecar status and mismatched CSV SHA-256 when a digest is supplied. Regression covers error status/hash mismatch. |
| P2 | Public native market/history endpoint was undocumented at code boundary and had not been smoke-tested | Previous adapter was mocked only | Actual GitHub Actions public-readonly capture succeeded for 43,200 bars. Current code includes strict history range, completeness, timestamp, duplicate and OHLC integrity checks. This proves endpoint/data capture only, not execution. |
| P2 | Reku activity history endpoint still HTTP 404 | Legacy `/v2/activitytrans` route is not currently available from the public research call | Collector remains read-only, reports `endpoint_not_found`, and does not fake/substitute a complete tape. |
| P1 | Strategy edge remains unproven | Zero trade candidates on current completed baseline; no closed-trade statistics are estimable | Keep `INSUFFICIENT_EVIDENCE` / `RESEARCH_ONLY`; no thresholds were relaxed. |

### Fee and network compatibility decision

- Reku remains the acquisition/on-ramp candidate. It is not a mandatory trading venue and its fee values must not be copied into the INDODAX simulator. Exact current account SOL withdrawal network, fee and minimum must be checked in the official app before a separate, explicitly authorized transfer.
- INDODAX public SOL/IDR historical capture works from the runner. The same point-in-time pair metadata reported price tick IDR 1, quantity step 1e-8 SOL, API quote minimum IDR 10,000 and minimum base quantity 0.00559952 SOL. These are per-market filters, not selected account-mode minimums. Official help describes Lite minimum IDR 10,000 and direct Pro minimum IDR 25,000 (orders IDR 10,000–24,999 in Pro interface route via Lite); exact current all-in fee components are displayed in the account Profile/Trading Fees UI: https://help.indodax.com/hc/en-us/articles/4416646599705-Details-of-Transaction-Fees-on-INDODAX.
- Do not use placeholder fee inputs as actual costs. Native INDODAX backtest now requires explicit `--fee-model indodax-custom`, entry/exit rates, spread, slippage, a positive mode-specific minimum and a source label. A successful simulator run with guessed costs is not a valid profitability claim.
- Jupiter remains a quote-only self-custody/DEX candidate. No signing/submission or wallet access was implemented.

### Branch/PR state and next action

PR #178 remains draft/open/unmerged. PR #180 remains draft/open/unmerged and carries the canonical memory/audit docs. Branch divergence is substantial, so compare/review the entire delta before any eventual integration; no merge/rebase/force-push was performed. No transfer, order, wallet signing, secret, or account action occurred. Android/Termux remains untested. Strategy readiness and live readiness remain blocked.


## Full research reconciliation — run #57, 30-day dual-source capture (2026-10-30 WIB)

Latest complete current-logic research: [SOL Sentinel Research #40949409224](https://github.com/orionrayy/ai-determinism-engine/actions/runs/40949409224), code engine commit `cb3ebab0b22a82f5ed1c9a6e748cfb5930d1439a`. Engine/backtest code is unchanged at the current implementation head `39d093b6e44635431475ea635f018327310793fd`; the intervening commit was a test-only adjustment. Exact-head test CI [#40949432502](https://github.com/orionrayy/ai-determinism-engine/actions/runs/40949432502) succeeded with 135 tests / 0 failures.

### Datasets observed

| Source | Status | Coverage | Dataset SHA-256 | Notes |
|---|---|---:|---|---|
| Binance SOLUSDT 1m | Capture/backtest success | 43,200 bars; 2026-09-29 20:29:00 UTC to 2026-10-29 20:28:59.999 UTC | `cb685aa993bbde9e2abec8c9c350e98097ba97d6b7c50fa323de1921ed1a7a47` | USDT reference converted with dated USD/IDR proxy, not venue-executable prices |
| INDODAX SOL/IDR 1m | Public read-only capture `OK` | 43,200 bars; 2026-09-29 20:30:00 UTC to 2026-10-29 20:29:59.999 UTC | `08f20992fed30c0921dee547e4cb4623c4dbf3b3c74ef732ce895ebc1fcf5633` | Native IDR, no FX conversion; captured but not backtested with INDODAX-specific costs |

### Strategy diagnosis, separated from code execution

For the completed Binance baseline, `signal_candidates=0`, entry attempts/fills=0, closed trades=0. Final equity equals starting equity only because no position was opened; expectancy, win rate and profit factor are undefined, not zero-return performance evidence. All fixed chronological 60/20/20 segments also have zero trades.

Full-window overlapping rejection counters were: 25,424 `15m_trend_not_confirmed`; 40,161 `1m_breakout_not_confirmed`; 1,786 `1m_momentum_exhaustion_or_missing`; 34,801 `1m_volume_confirmation_missing`; 28,329 `5m_momentum_not_confirmed`; 23,759 `5m_rsi_outside_entry_range`; 195 `estimated_net_edge_below_threshold`. These counts overlap because one bar can fail multiple checks; they cannot be summed to infer unique failed bars or used to claim a single threshold caused all zero trades. Since the engine recorded no candidates before order-size/fill, there is no evidence here of a broken order fill or ledger causing the zero-trade result. Do not loosen the rules just to manufacture trade count.

The predeclared Reku Lightning fee-only floor (zero spread/slippage) yielded 10 simulated trades and approximately IDR -3,893.22 net with profit factor about 0.276. It is an intentionally unrealistic mathematical lower bound, not an executable venue cost or forecast. Reku Pro/other normal cost scenarios yielded no candidates/trades. The native INDODAX data captured in run #57 have not been backtested because current all-in fees and selected mode must be explicitly verified and supplied; the Reku cost presets now reject native IDR input.

### Error status refresh

- **Fixed and regression-tested:** conflicting duplicate candles, malformed/fractional/zero-value Reku activity trades, source currency/venue identity, sidecar status/hash checking, native-IDR identity conversion, cross-venue fee preset rejection, and zero/non-finite native venue minimum/fee input protection.
- **Public-readonly verified:** native INDODAX `/api/pairs`, `/api/depth/{pair_id}` and TradingView history call path were exercised by successful GitHub Actions public requests. The history artifact confirms full contiguous candles. This is not account/credential/withdrawal/deposit verification.
- **Still unavailable:** legacy Reku `/v2/activitytrans` continues to return HTTP 404. Do not substitute incomplete legacy data as a complete tape.
- **Not yet proven:** native INDODAX strategy P&L with explicit verified venue costs; current user account fees/network support; Termux; paper-forward behavior; live executor/signing boundary.
- **Current-head research:** run #58 [#40949432867](https://github.com/orionrayy/ai-determinism-engine/actions/runs/40949432867) was still in progress at the last check. Its engine code matches run #57; do not claim it is completed until the run state and artefacts confirm that.

### Integration state

PR #178 remains draft/open/unmerged; latest pre-checkpoint comparison to `main` was diverged (239 ahead / 2 behind, merge base `eaa864a554c2c995ce73bd9f6b4d07206ecc56c5`). PR #180 remains draft/open/unmerged and contains this audit and canonical memory addendum. No merge, rebase, reset, force-push, wallet connection, signing, transfer or live order was performed. Readiness remains `RESEARCH_ONLY / INSUFFICIENT_EVIDENCE`.


## Current final-source validation — INDODAX depth probe and run #60 (2026-10-30 WIB)

### Latest exact-read evidence

- Research workflow [#40950218320](https://github.com/orionrayy/ai-determinism-engine/actions/runs/40950218320) completed successfully on workflow/code head `89a8d7181954a1524379d3b03fa77e703eb9331d`. Latest branch-head unit CI after doc-only changes [#40950649418](https://github.com/orionrayy/ai-determinism-engine/actions/runs/40950649418) succeeded: 137 tests, 0 failures.
- INDODAX native history: 43,200 contiguous 1m candles; SHA-256 `c5b9a370dd42ece13b63de139f9e2638bc9259262d02e3d8e950ae66fe4f7a30`; period 2026-09-29 20:36:00 UTC to 2026-10-29 20:35:59.999 UTC; IDR source currency; no FX conversion; status OK.
- The same run's public depth probe returned `market_quote_status=OK`, `market_quote_error=null`, `orders_enabled=false`. Snapshot timestamp 2026-10-29T20:37:47.062073Z; bid IDR 1,777,113; ask IDR 1,779,923; midpoint IDR 1,778,518; spread 15.799671411815905 bps. This is a public-read-only book observation, not a guaranteed fill or historical spread series.
- Earlier run [#40950150440](https://github.com/orionrayy/ai-determinism-engine/actions/runs/40950150440) recorded a second valid public depth snapshot (bid Rp1,775,886 / ask Rp1,780,450 / spread ~25.6669 bps) and another native history dataset. Different point-in-time spreads are evidence not to use a single quote as a 30-day constant except in clearly labeled sensitivity calculations.
- Native metadata in run #60: API market minimum IDR 10,000, minimum base quantity 0.00563136 SOL, price tick IDR 1, quantity step 1e-8 SOL. Account/mode minimum and fee remain separate, and not verified for the user's account.

### Most recent 30-day strategy result

In run #60 the Binance SOLUSDT input had 43,200 1m bars and dataset SHA-256 `bcc67c412280453fc5c499f9e0179ab8a118b331766470f0b59a8a18f62305e7`. The baseline and every fixed chronological 60/20/20 segment still had zero signal candidates and zero closed trades; expectancy, win rate and profit factor were undefined. The hypothetical Reku Lightning fee-only floor still produced 10 simulations and net P&L roughly IDR -3,893.22 with spread and slippage assumed zero. Thus strategy viability remains unproven; no thresholds were loosened.

The research workflow printed an exit-code-2 error for the optional legacy Reku activity-history collector due HTTP 404, but that step is configured `continue-on-error: true`; the overall research run succeeded, and the metadata marks the Reku history source ERROR. The Reku public market snapshot remained OK. This must not be read as a transfer outage finding or as a failed strategy backtest.

### Clarification of prior live endpoint scope

Earlier checkpoints said that live smoke testing was still required for INDODAX depth. That was true before #40950150440. As of #40950218320, the public unauthenticated `/api/pairs`, native TradingView OHLC history and `/api/depth/solidr` quote path have each been exercised successfully from GitHub Actions. They are now public market-data integration evidence; none implies authentication, order submission, user-account permissions, deposit-network support or Android/Termux validation.

### Remaining P1 gates

1. Verify selected INDODAX mode, actual all-in fee components from Profile/Trading Fees and the correct mode minimum. Public market metadata (IDR 10,000 in this snapshot) is not the same as the direct-Pro minimum IDR 25,000 (Lite IDR 10,000) described by the current official help page: https://help.indodax.com/hc/en-us/articles/4416646599705-Details-of-Transaction-Fees-on-INDODAX.
2. Supply measured/documented spread and explicitly labeled slippage assumptions before native IDR paper backtest/evaluation. Never pass Reku fees into native INDODAX.
3. Confirm the precise SOL network on the official Reku withdrawal preview and destination deposit screen, without sending a transfer.
4. Test the runbook on actual Termux and conduct a forward paper period with the chosen venue's quotes/fill assumptions.
5. Keep live execution blocked until all gates and separate explicit approval exist.

PR #178 remains draft/open/unmerged and diverged from main (last checked 239 ahead / 2 behind; merge base `eaa864a554c2c995ce73bd9f6b4d07206ecc56c5`). PR #180 remains draft/open/unmerged. No real order, transfer, wallet signature, private key, or account credential was used.


## Latest verification snapshot — 2026-10-30 WIB

Implementation head `479f2adc5bc4d401ca107ce1c044610e0a45c0d2` passes [CI #40951133963](https://github.com/orionrayy/ai-determinism-engine/actions/runs/40951133963): 137 offline unit tests. No live exchange execution is covered by this suite.

Research run [#40950218320](https://github.com/orionrayy/ai-determinism-engine/actions/runs/40950218320), artifact [#12931446637](https://github.com/orionrayy/ai-determinism-engine/actions/runs/40950218320/artifacts/12931446637), completed successfully on code checkpoint `89a8d7181954a1524379d3b03fa77e703eb9331d`. It captured 43,200 native INDODAX SOL/IDR 1m candles across 30 completed days, SHA-256 `c5b9a370dd42ece13b63de139f9e2638bc9259262d02e3d8e950ae66fe4f7a30`, with `data_status=OK` and `fx_conversion_applied=false`. Public depth at 2026-10-29 20:37:47 UTC was bid Rp1,777,113 / ask Rp1,779,923 / midpoint Rp1,778,518 / spread 15.7996714 bps. The preceding point snapshot was about 25.6669 bps. Two point observations are insufficient for a spread-time-series model.

The completed strategy evaluation in this run still used the Binance SOLUSDT reference dataset SHA-256 `bcc67c412280453fc5c499f9e0179ab8a118b331766470f0b59a8a18f62305e7`. Baseline and fixed 60/20/20 segments produced zero candidates/trades, therefore no estimable expectancy, win rate or profit factor. The fee-only Reku Lightning floor produced 10 hypothetical trades at roughly -Rp3,893.22 net under zero spread/slippage assumptions; this is not an executable route estimate. Native INDODAX history was captured but was not backtested with verified INDODAX mode fees/spread/slippage.

PR #178 remains draft/open/unmerged. Current compare to `main` is diverged, 246 ahead / 2 behind, 34 changed files, merge base `eaa864a554c2c995ce73bd9f6b4d07206ecc56c5`. PR #180 remains draft/open/unmerged; the canonical memory/audit handoff is not merged into `main`. Reku's legacy native activity history endpoint still returns 404; Reku public quote snapshot is separate and reports OK. Termux and wallet/account operations remain untested/not performed. Status: `RESEARCH_ONLY / INSUFFICIENT_EVIDENCE`.

## Follow-up: native INDODAX fee-separated replay and operations hardening — 2026-10-30

### Verified repository and CI evidence

- Feature PR #178 remains open/draft/unmerged; docs PR #180 remains open/draft/unmerged. No merge/rebase/force push or main modification occurred.
- Feature PR head at this audit: 91e421e8fa9460e7d86bbdf760b928ce0cb3ab34. Main at audit: d5672c84a0ae184c3a95d3f925925887ece71aa4. An ancestry compare captured at the immediately prior feature CI commit showed diverged history, 261 ahead / 2 behind, 37 changed files. Recompute against the exact latest head before any future integration.
- Frozen native INDODAX fee-separated replay: [workflow #40993527618](https://github.com/orionrayy/ai-determinism-engine/actions/runs/40993527618), artifact #12949927917. Original dataset SHA-256 c5b9a370dd42ece13b63de139f9e2638bc9259262d02e3d8e950ae66fe4f7a30, 43,200 contiguous 1m SOL/IDR candles, no FX. All three Pro scenarios had zero candidates/trades on dev/validation/OOS.
- Recurring same-run native replay: [workflow #40994319663](https://github.com/orionrayy/ai-determinism-engine/actions/runs/40994319663), artifact #12950538188. It no longer downloads a fixed prior artifact with seven-day retention. SHA-256 909b24977537110d2bd0ca114d5715d29646eb36b6ed9a268bf9470b64a05c59, 43,200 contiguous bars, no FX. Same-run data identity and OHLCV/gap guards passed.
- The current-run replay produced: Pro taker snapshot 75.030003 bps modeled round-trip friction; Pro taker stress 107.7769 bps; Pro maker snapshot 55.030003 bps. Taker scenarios produced zero candidates/trades across all splits. Maker sensitivity produced one development-only simulated trade (+Rp466.818 net, fees Rp574.698, friction Rp175.991) and zero validation/OOS trades. Single n=1 development trade does not validate an edge; no OOS metrics are estimable.
- One current-run spread snapshot was 2.920003 bps. Frozen earlier snapshot 15.799671 bps; earlier point 25.6669 bps. These few snapshots are not a historical spread distribution. Slippage assumptions remain 5/10 bps per side, not calibrated. Maker queue and post-only behavior remain unmodelled.
- Current-run overlapping reason counts on taker baseline: breakout 40,510; volume 35,882; 5m momentum 28,299; 15m trend 25,989; 5m RSI 23,415; 1m exhaustion/missing 3,142; cost-edge gate 143. Counts overlap; do not add them as unique candle counts or tune OOS to reduce them.
- Offline test suite [#40994692715](https://github.com/orionrayy/ai-determinism-engine/actions/runs/40994692715) passed 146 tests. Explicit Termux doctor smoke step [#40994852707](https://github.com/orionrayy/ai-determinism-engine/actions/runs/40994852707) passed with network disabled. This is GitHub-host validation only, not Android device validation.

### Fee source and implementation updates

- Official INDODAX help page checked 2026-10-30: https://help.indodax.com/hc/id/articles/4416646599705-Rincian-Biaya-Transaksi-di-INDODAX . Pro IDR all-in: maker buy 10.555 bps, taker buy 20.555, maker sell 31.555, taker sell 41.555. Direct Pro min Rp25,000; Lite min Rp10,000; Pro-interface Rp10,000–24,999 orders route through Lite. The exact user's profile fee page and Lite all-in fee have not been checked. All-in rates already include CFX and sell tax; do not double count.
- A dedicated fee audit was added at projects/sol-sentinel/INDODAX_FEE_MODEL_2026-10-30.md. Pair metadata is volatile: price tick Rp1, quantity step 1e-8 SOL, public API notional floor Rp10,000; the effective mode minimum still controls simulated eligibility.
- A lightweight standard-library Termux diagnostics script and tests were added; it includes read-only public pair metadata GET and streamed native dataset verification. It has not been run on the user's handset.
- Paper adapter now retains best-level bid/ask quantities and supports deterministic top-level partial-fill receipts. It does not sweep deeper levels or manage residual orders; maker queue and fill probability remain outside model.
- Jupiter remains quote-only. Full expiry/slippage/routing/fee schema validation is outstanding. Reku native activity-history route remains HTTP 404, and its purchase/on-ramp account API is not verified.

### Status and outstanding limitations

Strategy state remains RESEARCH_ONLY / INSUFFICIENT_EVIDENCE. There is no OOS trade. No live orders, signers, deposits, withdrawals or transfers occurred. No private keys/API secrets were handled. The scanner JSONL watch process is a read-only observation loop rather than a completed persistent forward-paper account and risk monitor. Next gates are repeated depth sampling, calibrated slippage/depth walk, account-specific mode fee verification, Jupiter schema validation, both-side network route evidence, actual Termux handset diagnostics, and a persistent forward-paper sample without parameter changes.

## Latest repeated native SOL/IDR rolling window — workflow #40994685989

- Successful current-run native INDODAX capture and fee-separated evaluation: [workflow #40994685989](https://github.com/orionrayy/ai-determinism-engine/actions/runs/40994685989), artifact #12951295388, code commit 50c0c00dc4bc5a78ab212dc20bd4894fb99783e6.
- Dataset SHA-256 d76434d4c9019249e924e72c5b05d0b200c0dfe3c4709c02a6d821fa0b22327c; 43,200 contiguous 1m SOL/IDR bars from 2026-09-30T03:01:00Z through 2026-10-30T03:00:59.999Z; no FX conversion.
- Depth quote at 2026-10-30T03:02:06.143046Z: bid Rp1,737,289 / ask Rp1,739,906; spread 15.0523626084 bps. Public filters: tick Rp1, quantity step 1e-8 SOL, API min notional Rp10,000, minimum quantity 0.00576894 SOL.
- Modeled friction: Pro taker/current snapshot 87.162363 bps; Pro taker/stress 107.7769 bps; Pro maker/current snapshot 67.162363 bps. Sparse snapshots and assumed slippage (5/10 bps per side) remain uncalibrated.
- All scenarios now report zero candidates, order attempts/fills, positions and closed trades in development, validation and OOS. P&L zero is only no-trade accounting; expectancy/win rate/profit factor are undefined.
- Overlapping taker-current filter reason counts: 1m breakout 40,506; 1m relative volume 35,873; 5m momentum 28,306; 15m trend 25,981; 5m RSI 23,415; 1m momentum exhaustion/missing 3,135; cost edge 143.
- The prior same-run window #40994319663 generated one maker-only development trade at +Rp466.818 net, without validation/OOS trades. This newest rolling window did not reproduce that trade; do not use it as an edge/profitability claim.
- 146-test suite passed in #40994692715. Explicit offline Termux-doctor smoke succeeded in #40994852707. Device-level Android validation remains unperformed.
- Latest status remains RESEARCH_ONLY / INSUFFICIENT_EVIDENCE; no live orders, signer, purchase, transfer, deposit or withdrawal occurred.
## Chronological leakage review — 2026-10-30

- Inspected evaluate_splits.py and paper_engine.py after the latest native replay. The evaluator keeps the 60/20/20 slices chronological and disjoint, checks intra-segment one-minute continuity, evaluates each segment independently and performs no automatic parameter tuning.
- The strategy constructs 5m/15m aggregates only from complete contiguous 1m buckets. It maps higher-timeframe features using close_time <= the decision candle close; an incomplete later 5m/15m bar is not used. Current-bar relative volume uses the preceding 20 volumes, breakout threshold uses preceding 20 highs, and a signal at bar close is only entered at the next 1m bar open.
- OHLC exit handling explicitly assumes stop-first if stop and target are both touched in one candle; open positions at end-of-data are force-closed at the last close. No look-ahead defect was identified in the inspected signal/split path.
- Residual limits: indicators/warmup are rebuilt separately inside each split, discarding the initial warmup observations; OHLC bars do not disclose exact intrabar path; spread/slippage and maker/queue assumptions are not calibrated. This is a code-path review, not a formal proof.

## Follow-up engineering checkpoint — public INDODAX depth collector (2026-10-30)

- Feature implementation: commit `708fef5d043113e68a9d0b2d43e21e1586167a92`, PR #178 (draft/open/unmerged).
- CI #40999102295 and #40999106258 both succeeded on this exact SHA. Entire offline suite: 168 tests, 0 failures. Termux doctor smoke: 6 PASS, 1 WARN because the runner is Ubuntu, 0 FAIL; network probe skipped.
- Added `indodax_microstructure.py`: standard-library-only, read-only `GET https://indodax.com/api/depth/solidr`, ordered multi-level bid/ask validation, spread bps, midpoint-band depth notional, SQLite observation/level/attempt journal, idempotency, finite collection, bounded retries and SQL-based distribution summary.
- New collector tests locally: 22 passed. The suite uses mocked payloads and SQLite temporary files.
- No live order book was successfully captured in this session. Consequently there is no empirical spread/depth time series yet and no market microstructure statistics to report. The first run needs to happen on a real network/device.
- Latest OHLC research remains unchanged: workflow #40994685989, 43,200 native SOL/IDR one-minute bars, SHA `d76434d4c9019249e924e72c5b05d0b200c0dfe3c4709c02a6d821fa0b22327c`, all latest reported scenarios zero trades through OOS.
- Execution limitations unchanged: top-of-book-only partial fill in the previous simulator; no multi-level depth walk/residual order management yet. Forward-paper orchestration is incomplete.
- Android Termux is still not device-validated. The CI doctor's 1 WARN explicitly records non-Termux Linux runner and must not be described as handset evidence.
- Readiness remains `RESEARCH_ONLY / INSUFFICIENT_EVIDENCE`.


## Verified v1.2 paper execution checkpoint — 2026-10-30

- Latest feature commit: `6e9424e9b8ad322e94f898d9de86324680ca0ae1` on `feat/sol-sentinel-paper-trader-v1`; PR #178 stays draft/open/unmerged.
- Added `depth_execution.py` and `test_depth_execution.py`: multi-level bid/ask book walking, per-level Decimal fills and cumulative VWAP, limit boundaries, cash/fee and inventory constraints, GTC/IOC residual lifecycle, cancellation/expiration, append-only SQLite fills/events, order/snapshot idempotency, transaction atomicity, recovery and fill-based portfolio reconciliation.
- Liquidity pool rule: each bid/ask side has a content fingerprint (venue, symbol, side and ordered price/quantity levels). Same side-book across later timestamps shares the same consumed pool; changes to bid do not reset identical ask liquidity. This avoids repeatedly counting unchanged displayed depth as fresh. Changed side-book creates a new pool, which remains an approximation absent trade-flow.
- SQLite has separate snapshot audit and shared per-side pool records. Startup backfills old snapshot-consumption records into these pools and refuses over-consumed historical aggregation.
- Local engine regression: 38 passed with `ResourceWarning` treated as error.
- Exact-head CI [#41002769671](https://github.com/orionrayy/ai-determinism-engine/actions/runs/41002769671): 206 tests passed, 0 failed, on the latest commit. Compile and Termux doctor smoke passed; doctor reported 6 PASS / 1 WARN because GitHub's host was Ubuntu and skipped its public network probe. No Android handset validation is claimed.
- Updated docs and Termux doctor compile coverage. No source-level changes were made to the previous market-data adapter or old top-of-book paper adapter.
- This is still a standalone paper execution engine, not an end-to-end forward-paper system: no unattended market-data loop, strategy/risk integration, maker queue model, daily-loss circuit breaker or auto-ingestion from collector SQLite yet.
- No live orders, account operations, signers, transfers, deposits, withdrawals or chain transactions. Research remains zero OOS trades / no positive edge evidence; overall readiness `RESEARCH_ONLY / INSUFFICIENT_EVIDENCE`.

 
## Integrated forward-paper v1.3 activation evidence — 2026-10-30

Implementation and activation evidence is recorded in canonical continuity marker `SOL_SENTINEL_V1_3_FORWARD_PAPER_ACTIVE_2026_10_30`.

Exact-head code/test SHA: `e827a1c2008c5cb0c1` (`e827a1c2008c5cb0c1` is the recorded prefix; full SHA `e827a1c2008c5cb0cabf04d79a53188646460c84`). Tests [#41006225151](https://github.com/orionrayy/ai-determinism-engine/actions/runs/41006225151): 216 passed, 0 failed. Termux Doctor: 6 PASS / 1 WARN / 0 FAIL on Ubuntu CI; no Android handset test is claimed.

The integrated forward-paper activation job succeeded on the same run. Artifact [#12955632037](https://github.com/orionrayy/ai-determinism-engine/actions/runs/41006225151/artifacts/12955632037), cycle `92f2cb5c750ddf6d648407cb75368fd14aedf9ad`, 603 native INDODAX SOL/IDR 1m candles, public depth observation `e5fbfc98f14a2ed57ac75a7888a4d4982dbe26d3d41b4fa543f92bb61ba79570`, execution snapshot `f72481320b8b56c512ffbdc238e7e51de90944339915072e6faf44241f35d18a`.

Output: `FORWARD_PAPER_ACTIVE`, `NO_TRADE`, risk `NORMAL`, entry permission true, IDR 300,000 cash, zero SOL, zero orders/fills/fees, and consistent ledger/portfolio reconciliation. Collector summary: 12 actual public depth observations, zero request errors, mean spread 8.9996 bps, median 9.1719 bps and P95 12.3240 bps. The sample is small and newly collected, so it is telemetry, not calibrated market microstructure evidence.

A second workflow cycle on the same candle correctly suppressed the repeated strategy signal (`signal_slot_already_processed`). No duplicate order, real exchange order, wallet operation, transfer, or credential access occurred. The persistent SQLite checkpoint is stored on `sol-sentinel-paper-state`.

The five-minute scheduled workflow file exists on the implementation branch but is not active on the repository schedule until merged to the default branch. PR #178 is still draft and diverged from main; do not merge it directly. A clean main-based integration PR is being prepared. Keep strategy research as `INSUFFICIENT_EVIDENCE` and live execution disabled.

 
## Clean main integration and post-merge paper checkpoint — 2026-10-30

Clean main-based PR [#181](https://github.com/orionrayy/ai-determinism-engine/pull/181) merged via squash as `58ec3df095c60ccc6868bf0cbccb3c18384419eb`. It carried the integrated forward-paper runner, serialized scheduled workflow, test coverage, operating guide, audit, infrastructure decision and canonical memory. The old PR #178 was not merged because its history diverged; #181 was 1 commit ahead / 0 behind of main with 48 changed files.

Exact PR tests passed: [Orchestrator Tests #41006890245](https://github.com/orionrayy/ai-determinism-engine/actions/runs/41006890245) and [SOL Sentinel Tests #41006890361](https://github.com/orionrayy/ai-determinism-engine/actions/runs/41006890361). Sentinel suite: 216 passed, 0 failed. Termux Doctor on Ubuntu: 6 PASS / 1 WARN / 0 FAIL, with the warning correctly noting no physical Android handset execution.

The same run activated forward-paper and stored artifact [#12955034631](https://github.com/orionrayy/ai-determinism-engine/actions/runs/41006890361/artifacts/12955034631). Cycle `7e08efadca684497f5341b3bdb760ad18cb62857`, source SHA `5bf6c2cc9598d49c4ecec4458373538b3afa18f5`, 610 native 1m candles, depth observation `62ffba97d99af94e33ad7b8d7d154dbce11a22d583c4cdf24859a5597b605c8f`, execution snapshot `e7f3dd4c7860e65be38cec14a32171c57741714a997b349ce8d51a97b145fb38`. Signal: NO_TRADE; reasons: 15m trend not confirmed, 1m momentum exhaustion/missing. Risk NORMAL; entries permitted; equity IDR 300,000; cash IDR 300,000; SOL 0; orders/fills/fees/P&L 0; ledger reconciliation consistent.

Collector summary: 17 actual public depth observations, 0 request failures, mean spread 9.634764 bps, median 11.105374 bps, P95 16.071871 bps. Small live sample only; not a calibrated spread/slippage/queue distribution.

State DB branch latest checkpoint at this review: `sol-sentinel-paper-state` commit `c3f259242d450d64b4baeb01ae083c178fd4d8b0`. Five-minute scheduled workflow now resides on `main`, so GitHub's default-branch scheduling condition is met; a separate post-merge scheduled run is not yet confirmed.

Readiness remains: ENGINEERING_VALIDATED / FORWARD_PAPER_ACTIVE; RESEARCH_EVIDENCE_INSUFFICIENT; LIVE_AUTHORIZATION_READY=false. No live order or asset movement was attempted.
