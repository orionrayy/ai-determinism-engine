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
