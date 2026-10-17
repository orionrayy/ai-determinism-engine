# Temporal calibration protocol

Fits regularized monotone Platt scaling only on earlier out-of-sample forecasts whose target labels have matured before a later chronological holdout starts. Reports Brier score, log loss, equal-width ECE/MCE, reliability bins, a training base-rate reference and moving-block bootstrap intervals.

Run `python -m unittest -v` then `python run_calibration_contracts.py`. The 240 cases are synthetic code-contract checks only. Even on a successful holdout, deployment_calibrated, validated and live_signal_enabled remain false. Production requires licensed broker Bid/Ask/tick data, independent chronological regimes, untouched holdout, measured costs, paper trading and separate security/risk review.
