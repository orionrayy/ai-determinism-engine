# Aurum v1.3.1 — temporal calibration checkpoint

Local release v1.3.1: 276,128 bytes; SHA-256 b63af5ab3412e7cb0bac14deffdb0d8cf07e024b0e46364d6fe5e1329d831d5. Local clean checks: 185 pytest, 109 unittest, compileall, Mini App JS/HTML and ZIP extraction/retest pass.

The evaluator fits Platt scaling on earlier OOS forecasts only when labels mature before the later chronological holdout. It reports Brier/log loss/ECE/MCE/reliability and moving-block bootstrap intervals. Eight focused tests and 240/240 synthetic contracts passed. The 720-row illustration in the full package is synthetic, not market evidence.

The bundled market sample remains 60 bars and correctly yields zero OOS TP-before-SL predictions. Separate hosted replay on 35,954 public rows found fixed-rule baselines net-negative under assumed costs and no meaningful Brier skill. NO_TRADE remains active. The full app ZIP is downloadable, not tracked as a GitHub binary.
