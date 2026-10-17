#!/usr/bin/env python3
"""Online, read-only XAUUSD dataset audit and deterministic benchmark; stdlib only."""
import argparse, csv, hashlib, io, json, math, random, statistics, sys, time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.error import URLError, HTTPError
from urllib.request import Request, urlopen

import evaluation_protocol_v1_4 as eval_protocol
import source_concordance_v1_0 as source_concordance

SOURCE = "https://raw.githubusercontent.com/getdata-finance/xauusd-5m-ohlcv-metals-historical-data/main/XAUUSD_5m.csv"
SOURCE_REPO = "getdata-finance/xauusd-5m-ohlcv-metals-historical-data"
LICENSE = "https://github.com/getdata-finance/xauusd-5m-ohlcv-metals-historical-data/blob/main/LICENSE"
SECONDARY_SOURCE = "https://raw.githubusercontent.com/Sai310421/xauusd-data/main/csv/XAUUSD/XAUUSD_M5_2026Q1Q2.csv"
SECONDARY_SOURCE_REPO = "Sai310421/xauusd-data"
SECONDARY_LICENSE = "https://github.com/Sai310421/xauusd-data/blob/main/LICENSE"
SECONDARY_README = "https://github.com/Sai310421/xauusd-data/blob/main/README.md"
MAX_BYTES = 16 * 1024 * 1024
RULES = ("last_return_sign", "ema5_vs_ema10", "mean_reversion_z10", "prior20_range_breakout", "flat_no_trade")
ARCHETYPES = ("normal", "high_volatility", "low_volatility", "upside_gap", "downside_gap", "long_wicks", "compressed_wicks", "volume_spike", "zero_volume", "missing_bar", "duplicate_timestamp", "invalid_ohlc")

class DatasetError(ValueError):
    pass

def parse_time(raw):
    raw = raw.strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        for fmt in ("%Y.%m.%d %H:%M", "%Y.%m.%d %H:%M:%S", "%Y-%m-%d %H:%M"):
            try:
                return datetime.strptime(raw, fmt)
            except ValueError:
                pass
    raise DatasetError("unparseable timestamp")

def parse_csv(payload, min_rows=1000):
    if len(payload) > MAX_BYTES:
        raise DatasetError("source exceeded 16 MiB cap")
    try:
        reader = csv.DictReader(io.StringIO(payload.decode("utf-8-sig"), newline=""))
    except UnicodeDecodeError as exc:
        raise DatasetError("CSV is not UTF-8") from exc
    if not reader.fieldnames:
        raise DatasetError("missing header")
    headers = [x.strip().lower() for x in reader.fieldnames]
    if len(set(headers)) != len(headers):
        raise DatasetError("duplicate headers")
    alias = {"timestamp": ("timestamp", "datetime", "date", "time", "date_time"), "open": ("open", "o"), "high": ("high", "h"), "low": ("low", "l"), "close": ("close", "c"), "volume": ("volume", "vol", "tickvol", "tick_volume")}
    actual = {x.strip().lower(): x for x in reader.fieldnames}
    col = {k: next((actual[a] for a in aliases if a in actual), None) for k, aliases in alias.items()}
    missing = [k for k, v in col.items() if not v]
    if missing:
        raise DatasetError("missing columns: " + ",".join(missing))
    rows = []
    for line, raw in enumerate(reader, 2):
        if None in raw or any(raw.get(col[k]) is None for k in alias):
            raise DatasetError("malformed row " + str(line))
        r = {"timestamp": parse_time(raw[col["timestamp"]])}
        for k in ("open", "high", "low", "close", "volume"):
            try:
                r[k] = float(raw[col[k]])
            except (TypeError, ValueError) as exc:
                raise DatasetError("nonnumeric " + k + " at row " + str(line)) from exc
            if not math.isfinite(r[k]):
                raise DatasetError("non-finite " + k + " at row " + str(line))
        o, h, l, c, v = (r[k] for k in ("open", "high", "low", "close", "volume"))
        if min(o, h, l, c) <= 0 or h < max(o, c) or l > min(o, c) or h < l or v < 0:
            raise DatasetError("invalid OHLCV geometry at row " + str(line))
        rows.append(r)
    if len(rows) < min_rows:
        raise DatasetError("insufficient rows: " + str(len(rows)) + " < " + str(min_rows))
    aware = {x["timestamp"].utcoffset() is not None for x in rows}
    if len(aware) != 1:
        raise DatasetError("mixed timezone-aware and naive timestamps")
    if any(rows[i]["timestamp"] <= rows[i-1]["timestamp"] for i in range(1, len(rows))):
        raise DatasetError("timestamps not strictly increasing")
    return rows

def fetch():
    req = Request(SOURCE, headers={"User-Agent": "Aurum-Intelligence/1.3.0 online-benchmark"})
    last = None
    for attempt in range(3):
        try:
            with urlopen(req, timeout=40) as res:
                if not res.geturl().startswith("https://raw.githubusercontent.com/getdata-finance/"):
                    raise DatasetError("redirect outside approved origin")
                size = res.headers.get("Content-Length")
                if size and int(size) > MAX_BYTES:
                    raise DatasetError("Content-Length exceeded cap")
                body = res.read(MAX_BYTES + 1)
                if len(body) > MAX_BYTES:
                    raise DatasetError("download exceeded cap")
                return body
        except DatasetError:
            raise
        except (OSError, URLError, HTTPError) as exc:
            last = exc
            if attempt < 2:
                time.sleep(.3 * (attempt + 1))
    raise RuntimeError("source download failed after bounded retries: " + type(last).__name__)


def fetch_secondary():
    """Fetch a second public community OHLCV source under the same size/retry safety limits."""
    req = Request(SECONDARY_SOURCE, headers={"User-Agent": "Aurum-Intelligence/1.4.0 source-concordance"})
    last = None
    for attempt in range(3):
        try:
            with urlopen(req, timeout=40) as res:
                if not res.geturl().startswith("https://raw.githubusercontent.com/Sai310421/"):
                    raise DatasetError("secondary redirect outside approved origin")
                size = res.headers.get("Content-Length")
                if size and int(size) > MAX_BYTES:
                    raise DatasetError("secondary Content-Length exceeded cap")
                body = res.read(MAX_BYTES + 1)
                if len(body) > MAX_BYTES:
                    raise DatasetError("secondary download exceeded cap")
                return body
        except DatasetError:
            raise
        except (OSError, URLError, HTTPError) as exc:
            last = exc
            if attempt < 2:
                time.sleep(.3 * (attempt + 1))
    raise RuntimeError("secondary source download failed after bounded retries: " + type(last).__name__)

def choose_indices(n, count=50):
    lo, hi = 29, n - 2
    if count < 30 or hi < lo or count > hi - lo + 1:
        raise ValueError("need at least 30 distinct cases and sufficient history")
    out = [lo + round(i * (hi - lo) / (count - 1)) for i in range(count)]
    if len(set(out)) != count:
        raise ValueError("could not select unique chronological cases")
    return out

def ema(xs, span):
    alpha, value = 2 / (span + 1), xs[0]
    for x in xs[1:]:
        value = alpha * x + (1 - alpha) * value
    return value

def get_signal(rows, i, rule):
    if i < 1 or i >= len(rows):
        raise ValueError("decision index out of range")
    p = rows[i]["close"]
    if rule == "flat_no_trade":
        return 0
    if rule == "last_return_sign":
        previous = rows[i - 1]["close"]
        return int(p > previous) - int(p < previous)
    if rule == "ema5_vs_ema10":
        xs = [x["close"] for x in rows[max(0, i - 59):i + 1]]
        short, long = ema(xs, 5), ema(xs, 10)
        return int(short > long) - int(short < long)
    if rule == "mean_reversion_z10":
        xs = [x["close"] for x in rows[max(0, i - 9):i + 1]]
        sd = statistics.pstdev(xs)
        z = (p - statistics.mean(xs)) / sd if sd else 0
        return -1 if z > 1 else 1 if z < -1 else 0
    previous = rows[max(0, i - 20):i]
    return 1 if p > max(x["high"] for x in previous) else -1 if p < min(x["low"] for x in previous) else 0

def case_replay(rows, n=50):
    out = []
    for i in choose_indices(len(rows), n):
        target = rows[i+1]
        move = target["close"] - target["open"]
        for rule in RULES:
            s = get_signal(rows, i, rule)
            gross = s * move if s else 0.0
            cost = .5 if s else 0.0
            out.append({"decision_index": i, "decision_time": rows[i]["timestamp"].isoformat(), "outcome_time": target["timestamp"].isoformat(), "rule": rule, "signal": s, "next_body_direction": int(move > 0) - int(move < 0), "gross_pnl_units": round(gross, 8), "assumed_cost_units": cost, "net_pnl_units": round(gross-cost, 8), "cost_is_assumption": True})
    return out

def _max_drawdown(values):
    equity = peak = drawdown = 0.0
    for value in values:
        equity += value
        peak = max(peak, equity)
        drawdown = max(drawdown, peak - equity)
    return drawdown


def _trade_summary(values, correct, opportunities):
    wins = sum(v for v in values if v > 0)
    losses = -sum(v for v in values if v < 0)
    trades = len(values)
    return {
        "opportunities": opportunities,
        "trades": trades,
        "signal_coverage": trades / opportunities if opportunities else None,
        "directional_accuracy": correct / trades if trades else None,
        "net_pnl_units": round(sum(values), 8),
        "mean_net_pnl_per_trade": round(statistics.mean(values), 8) if values else None,
        "profit_factor": round(wins / losses, 8) if losses else None,
        "profit_factor_status": "defined" if losses else ("no_trades" if not values else "no_losing_trades"),
        "max_drawdown_units": round(_max_drawdown(values), 8),
    }


def run_full_history_replay(rows, warmup_bars=60, holdout_fraction=0.20,
                            round_trip_spread=0.30, slippage_per_fill=0.10,
                            round_trip_commission=0.0):
    """Evaluate every valid sequential next-bar opportunity with fixed rules.

    Rules are not fitted/tuned here. The last chronological segment is reported separately;
    full-history results are descriptive and are not an independent out-of-sample claim.
    """
    if not isinstance(warmup_bars, int) or warmup_bars < 30 or len(rows) < warmup_bars + 10:
        raise ValueError("insufficient_history_or_invalid_warmup")
    if not 0.05 <= holdout_fraction <= 0.50:
        raise ValueError("holdout_fraction_out_of_range")
    costs = (round_trip_spread, slippage_per_fill, round_trip_commission)
    if not all(math.isfinite(float(v)) and float(v) >= 0 for v in costs):
        raise ValueError("invalid_cost_assumption")
    n = len(rows)
    split = max(warmup_bars + 1, min(n - 2, int(n * (1 - holdout_fraction))))
    ranges = {
        "full_history": (warmup_bars, n - 1),
        "early_segment": (warmup_bars, split),
        "chronological_tail_holdout": (split, n - 1),
    }
    opportunities_total = n - warmup_bars - 1
    development_opportunities = split - warmup_bars
    folds = []
    for fold_index in range(4):
        start = warmup_bars + (fold_index * development_opportunities) // 4
        end = warmup_bars + ((fold_index + 1) * development_opportunities) // 4
        name = "chronological_fold_" + str(fold_index + 1)
        ranges[name] = (start, end)
        folds.append({
            "name": name, "start_index": start, "end_index_exclusive": end,
            "start_time": rows[start]["timestamp"].isoformat(),
            "end_time": rows[end - 1]["timestamp"].isoformat(),
            "opportunities": end - start,
        })
    # Fold five is exactly the final chronological holdout; the five folds are disjoint.
    name = "chronological_fold_5"
    start, end = ranges["chronological_tail_holdout"]
    ranges[name] = (start, end)
    folds.append({
        "name": name, "start_index": start, "end_index_exclusive": end,
        "start_time": rows[start]["timestamp"].isoformat(),
        "end_time": rows[end - 1]["timestamp"].isoformat(),
        "opportunities": end - start,
    })
    cost_per_trade = round_trip_spread + 2 * slippage_per_fill + round_trip_commission
    summary = {}
    for rule in RULES:
        segments = {}
        for segment, (start, end) in ranges.items():
            pnls, correct, opportunities = [], 0, max(0, end - start)
            for i in range(start, end):
                side = get_signal(rows, i, rule)
                target = rows[i + 1]
                move = target["close"] - target["open"]
                actual = int(move > 0) - int(move < 0)
                if side:
                    pnls.append(side * move - cost_per_trade)
                    correct += int(side == actual)
            segments[segment] = _trade_summary(pnls, correct, opportunities)
        summary[rule] = segments
    return {
        "kind": "fixed_rule_full_history_and_chronological_tail_diagnostic",
        "n_candles": n,
        "decision_opportunities": opportunities_total,
        "warmup_bars": warmup_bars,
        "tail_holdout_fraction_requested": holdout_fraction,
        "tail_holdout_start_index": split,
        "tail_holdout_start_time": rows[split]["timestamp"].isoformat(),
        "chronological_folds": folds,
        "signal_contract": "signal_after_close_i_enter_next_open_exit_next_close",
        "rules_fitted_or_tuned": False,
        "cost_model": {
            "round_trip_spread_units_assumed": round_trip_spread,
            "slippage_per_fill_units_assumed": slippage_per_fill,
            "round_trip_commission_units_assumed": round_trip_commission,
            "total_cost_per_trade_units": cost_per_trade,
            "status": "assumptions_not_observed_broker_costs",
        },
        "rules": summary,
        "limitations": [
            "Community OHLCV is not bid/ask quotes, ticks, or actual broker fills.",
            "Rules and costs are diagnostic; no rule was tuned on this dataset.",
            "Full-history metrics are descriptive; the last chronological segment is not an independent external dataset.",
            "Next-candle body direction and close-open PnL are not equivalent to TP-before-SL outcomes or executable returns.",
        ],
    }


def iter_prequential_probability_rows(rows, warmup_bars=60, rolling_window=288):
    """Generate causal next-bar up/not-up probability baselines using closed bars only."""
    if warmup_bars < 2 or rolling_window < 2:
        raise ValueError("invalid_probability_baseline_policy")
    outcomes = [int(row["close"] > row["open"]) for row in rows]
    for i in range(warmup_bars, len(rows) - 1):
        history = outcomes[1:i + 1]
        recent = history[-rolling_window:]
        yield {
            "decision_index": i,
            "target_index": i + 1,
            "target_time": rows[i + 1]["timestamp"].isoformat(),
            "actual_up": outcomes[i + 1],
            "p_constant_50pct": 0.5,
            "p_expanding_laplace": (sum(history) + 1.0) / (len(history) + 2.0),
            "p_rolling_laplace": (sum(recent) + 1.0) / (len(recent) + 2.0),
            "training_labels_available": len(history),
        }


def _binary_log_loss(probability, actual, epsilon=1e-12):
    p = min(1.0 - epsilon, max(epsilon, float(probability)))
    return -(actual * math.log(p) + (1 - actual) * math.log(1 - p))


def _reliability_summary(probabilities, outcomes, bins=10):
    n = len(probabilities)
    rows, ece, gaps = [], 0.0, []
    for index in range(bins):
        lower, upper = index / bins, (index + 1) / bins
        selected = [(p, y) for p, y in zip(probabilities, outcomes)
                    if lower <= p < upper or (index == bins - 1 and p == 1.0)]
        mean_probability = statistics.mean(p for p, _ in selected) if selected else None
        observed_rate = statistics.mean(y for _, y in selected) if selected else None
        gap = abs(mean_probability - observed_rate) if selected else None
        if selected:
            ece += len(selected) / n * gap
            gaps.append(gap)
        rows.append({"lower": lower, "upper": upper, "n": len(selected),
                     "mean_probability": mean_probability, "observed_up_rate": observed_rate,
                     "absolute_calibration_gap": gap})
    return {
        "bins": rows,
        "ece_descriptive": ece if n else None,
        "mce_descriptive": max(gaps) if gaps else None,
        "calibration_status": "prequential_baseline_diagnostic_not_a_calibrated_trading_model",
    }


def _block_bootstrap_ci(values, block_size=288, replicates=200, seed=20261016):
    """Deterministic moving-block bootstrap for paired loss improvements; approximate only."""
    n = len(values)
    if n < 30:
        return {"lower_95": None, "upper_95": None, "status": "insufficient_samples", "n": n}
    block = min(max(2, int(block_size)), n)
    rng = random.Random(seed)
    estimates = []
    for _ in range(replicates):
        total, sampled = 0.0, 0
        while sampled < n:
            start = rng.randrange(0, n - block + 1)
            take = min(block, n - sampled)
            total += sum(values[start:start + take])
            sampled += take
        estimates.append(total / n)
    estimates.sort()
    return {
        "lower_95": estimates[int(0.025 * (replicates - 1))],
        "upper_95": estimates[int(0.975 * (replicates - 1))],
        "status": "approximate_moving_block_bootstrap",
        "block_size_bars": block, "replicates": replicates, "seed": seed, "n": n,
    }


def run_prequential_probability_diagnostics(rows, warmup_bars=60, rolling_window=288,
                                            bootstrap_replicates=200):
    """Evaluate causal direction-rate baselines; none is claimed calibrated or production-ready."""
    forecasts = list(iter_prequential_probability_rows(rows, warmup_bars, rolling_window))
    if len(forecasts) < 30:
        return {"status": "insufficient_history", "predictions": len(forecasts),
                "validated_model": False, "calibrated_model": False, "methods": {}}
    actuals = [row["actual_up"] for row in forecasts]
    rolling_name = "rolling_" + str(rolling_window) + "_bar_laplace_base_rate"
    fields = {
        "constant_50pct": "p_constant_50pct",
        "expanding_laplace_base_rate": "p_expanding_laplace",
        rolling_name: "p_rolling_laplace",
    }
    probabilities = {name: [row[field] for row in forecasts] for name, field in fields.items()}
    briers = {name: [(p - y) ** 2 for p, y in zip(ps, actuals)]
              for name, ps in probabilities.items()}
    log_losses = {name: [_binary_log_loss(p, y) for p, y in zip(ps, actuals)]
                  for name, ps in probabilities.items()}
    base_brier = briers["constant_50pct"]
    base_log_loss = log_losses["constant_50pct"]
    metrics = {}
    base_brier_mean = statistics.mean(base_brier)
    for name, ps in probabilities.items():
        brier_values, log_values = briers[name], log_losses[name]
        metrics[name] = {
            "n": len(ps),
            "brier_score": statistics.mean(brier_values),
            "log_loss": statistics.mean(log_values),
            "mean_probability": statistics.mean(ps),
            "observed_up_rate": statistics.mean(actuals),
            "brier_skill_vs_constant_50pct": 1.0 - statistics.mean(brier_values) / base_brier_mean
                if base_brier_mean else None,
            "mean_brier_improvement_vs_constant_50pct": statistics.mean(
                [base - value for base, value in zip(base_brier, brier_values)]),
            "mean_log_loss_improvement_vs_constant_50pct": statistics.mean(
                [base - value for base, value in zip(base_log_loss, log_values)]),
            "reliability": _reliability_summary(ps, actuals),
        }
    out = {
        "status": "DIAGNOSTIC_ONLY",
        "target": "next_bar_candle_body_up_vs_not_up",
        "predictions": len(forecasts),
        "first_decision_index": forecasts[0]["decision_index"],
        "last_decision_index": forecasts[-1]["decision_index"],
        "forecast_timing": "after close[i], predict body direction of i+1 with outcomes through i only",
        "base_rate_window_bars": {"expanding": "all previously closed candles", "rolling": rolling_window},
        "methods": metrics,
        "uncertainty": {
            name: _block_bootstrap_ci(
                [base - value for base, value in zip(base_brier, briers[name])],
                block_size=min(288, max(2, len(forecasts) // 5)),
                replicates=bootstrap_replicates, seed=20261016 + idx,
            )
            for idx, name in enumerate(("expanding_laplace_base_rate", rolling_name), 1)
        },
        "validated_model": False,
        "calibrated_model": False,
        "live_signal_enabled": False,
        "limitations": [
            "These are causal historical-rate baselines, not a trained trading model.",
            "The target is candle-body up vs not-up, not TP-before-SL or profitable execution.",
            "ECE/MCE are descriptive; temporal dependence and market regime shifts remain.",
            "Moving-block bootstrap intervals are approximate; the block length is heuristic.",
            "Data is community OHLCV, not broker Bid/Ask/tick/fill data.",
        ],
    }
    return out


def valid_path(rows):
    if not rows: return False
    try:
        for i, r in enumerate(rows):
            ts = r["timestamp"] if isinstance(r["timestamp"], datetime) else parse_time(str(r["timestamp"]))
            prev = rows[i-1]["timestamp"] if i else None
            if i and ts <= (prev if isinstance(prev, datetime) else parse_time(str(prev))): return False
            o, h, l, c, v = (float(r[k]) for k in ("open","high","low","close","volume"))
            if not all(math.isfinite(x) for x in (o,h,l,c,v)) or min(o,h,l,c) <= 0 or h < max(o,c) or l > min(o,c) or h < l or v < 0: return False
        return True
    except (KeyError, TypeError, ValueError, OverflowError):
        return False

def market_scenarios(rows, anchors=5):
    max_start = max(1, len(rows)-31)
    starts = [round(i*(max_start-1)/max(1, anchors-1)) for i in range(anchors)] if anchors > 1 else [max_start//2]
    out = []
    for aid, start in enumerate(starts):
        for category in ARCHETYPES:
            for variation in range(20):
                x = [dict(r) for r in rows[start:start+30]]
                k, f = variation % len(x), .01 + (variation+1)*.001
                invalid = category in ("duplicate_timestamp", "invalid_ohlc")
                if category == "high_volatility":
                    r=x[k]; p=r["close"]; span=(r["high"]-r["low"])*(2+f); r.update(open=p,high=max(p,r["open"])+span,low=max(.01,p-span),close=p*(1+f/3)); r["high"]=max(r["high"],r["open"],r["close"]); r["low"]=min(r["low"],r["open"],r["close"])
                elif category == "low_volatility":
                    r=x[k]; p=r["close"]; r.update(open=p,high=p*(1+f/100),low=p*(1-f/100),close=p*(1+f/400))
                elif category in ("upside_gap","downside_gap"):
                    delta=f if category=="upside_gap" else -f
                    for r in x[k:]:
                        for key in ("open","high","low","close"): r[key]*=1+delta
                elif category == "long_wicks": x[k]["high"]*=1+f; x[k]["low"]*=1-f
                elif category == "compressed_wicks":
                    r=x[k]; r["low"]=min(r["open"],r["close"])*(1-f/10); r["high"]=max(r["open"],r["close"])*(1+f/10)
                elif category == "volume_spike": x[k]["volume"]*=10+variation
                elif category == "zero_volume": x[k]["volume"]=0
                elif category == "missing_bar": del x[k]
                elif category == "duplicate_timestamp": x[k]["timestamp"]=x[k-1 if k else 1]["timestamp"]
                elif category == "invalid_ohlc": x[k]["high"]=min(x[k]["open"],x[k]["close"])*.5
                out.append({"anchor_id":aid,"anchor_start_index":start,"archetype":category,"variation":variation,"passed":valid_path(x)==(not invalid),"synthetic_counterfactual":True})
    return out

def software_scenarios(rows):
    categories=("valid","duplicate_time","reverse_time","mixed_timezone","nan","negative_volume","bad_high","bad_low","missing_close","bad_timestamp","empty","zero_volume","flat_bar","single_row","extreme_valid","gap_valid","string_time","extra_field","copy_valid","negative_price")
    out=[]
    for cat in categories:
        for v in range(12):
            x=[dict(r) for r in rows[:60]]
            k=v%len(x); expected=True
            if cat=="duplicate_time": x[k+1]["timestamp"]=x[k]["timestamp"]; expected=False
            elif cat=="reverse_time": x[k],x[k+1]=x[k+1],x[k]; expected=False
            elif cat=="mixed_timezone": x[-1]["timestamp"]=x[-1]["timestamp"].replace(tzinfo=None) if x[-1]["timestamp"].tzinfo else x[-1]["timestamp"].replace(tzinfo=timezone.utc); expected=False
            elif cat=="nan": x[k]["close"]=float("nan"); expected=False
            elif cat=="negative_volume": x[k]["volume"]=-1; expected=False
            elif cat=="bad_high": x[k]["high"]=.01; expected=False
            elif cat=="bad_low": x[k]["low"]=x[k]["high"]+1; expected=False
            elif cat=="missing_close": del x[k]["close"]; expected=False
            elif cat=="bad_timestamp": x[k]["timestamp"]="invalid"; expected=False
            elif cat=="empty": x=[]; expected=False
            elif cat=="zero_volume": x[k]["volume"]=0
            elif cat=="flat_bar":
                p=x[k]["close"]; x[k].update(open=p,high=p,low=p,close=p)
            elif cat=="single_row": x=x[:1]
            elif cat=="extreme_valid":
                for field in ("open","high","low","close"): x[k][field]*=(3+v/10)
            elif cat=="gap_valid":
                for r in x[k+1:]: r["timestamp"]+=timedelta(minutes=31+v)
            elif cat=="string_time": x[k]["timestamp"]=x[k]["timestamp"].isoformat()
            elif cat=="extra_field": x[k]["diagnostic_note"]="ignored"
            elif cat=="negative_price": x[k]["open"]=-1; expected=False
            observed=valid_path(x)
            out.append({"category":cat,"variation":v,"expected_valid":expected,"observed_valid":observed,"passed":observed==expected})
    return out

def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        if rows:
            w=csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def run(payload, output, min_rows=1000, cases=50, anchors=5, source_url=SOURCE, secondary_payload=None, secondary_source_url=SECONDARY_SOURCE):
    t=time.perf_counter(); rows=parse_csv(payload,min_rows)
    intervals=[(rows[i]["timestamp"]-rows[i-1]["timestamp"]).total_seconds() for i in range(1,len(rows))]
    digest=hashlib.sha256(payload).hexdigest(); replay=case_replay(rows,cases); market=market_scenarios(rows,anchors); software=software_scenarios(rows); full_history=run_full_history_replay(rows); probability=run_prequential_probability_diagnostics(rows); quality=eval_protocol.audit_dataset_quality(rows); tp_sl_protocol,tp_sl_events=eval_protocol.run_tp_sl_protocol(rows,get_signal,RULES); secondary_rows=parse_csv(secondary_payload,min_rows) if secondary_payload is not None else []; secondary_crosscheck,overlap_sample=source_concordance.compare_source_datasets(rows,secondary_rows) if secondary_rows else ({"status":"NOT_RUN","reason":"secondary source was not supplied; local fixture mode","reference_only":True,"decision_output":"NO_TRADE"},[]); secondary_crosscheck.update({"source_repository":SECONDARY_SOURCE_REPO,"source_url":secondary_source_url,"license_reference":SECONDARY_LICENSE,"readme_reference":SECONDARY_README,"dataset_sha256":hashlib.sha256(secondary_payload).hexdigest() if secondary_payload is not None else None,"dataset_bytes":len(secondary_payload) if secondary_payload is not None else None,"repository_license_label":"MIT per repository LICENSE; upstream data rights not independently verified","intended_use_rights_verified":False,"timezone_claim_from_readme":"UTC; not independently verified","reference_only":True}); secondary_quality=eval_protocol.audit_dataset_quality(secondary_rows) if secondary_rows else None
    baselines=[]
    for rule in RULES:
        xs=[r for r in replay if r["rule"]==rule]; trades=[r for r in xs if r["signal"]]; pnl=[r["net_pnl_units"] for r in trades]; equity=peak=dd=0.0
        for value in pnl: equity+=value; peak=max(peak,equity); dd=max(dd,peak-equity)
        correct=sum(r["signal"]==r["next_body_direction"] for r in trades)
        baselines.append({"rule":rule,"cases":len(xs),"trades":len(trades),"direction_accuracy":correct/len(trades) if trades else None,"net_pnl_units":round(sum(pnl),8),"max_drawdown_units":round(dd,8)})
    fail=sum(not r["passed"] for r in market)+sum(not r["passed"] for r in software)
    status="PASS" if fail==0 and len(replay)==cases*5 and len(market)>=200 and len(software)>=200 and (secondary_payload is None or secondary_crosscheck.get("status")=="CROSS_SOURCE_DIAGNOSTIC_ONLY") else "FAIL"
    output.mkdir(parents=True,exist_ok=True); write_csv(output/"real_data_case_results.csv",replay); write_csv(output/"market_scenario_results.csv",market); write_csv(output/"software_scenario_results.csv",software); write_csv(output/"tp_sl_barrier_outcomes.csv",tp_sl_events); write_csv(output/"secondary_source_overlap_sample.csv",overlap_sample); (output/"tp_sl_evaluation_protocol.json").write_text(json.dumps(tp_sl_protocol,indent=2,allow_nan=False)+"\n",encoding="utf-8"); (output/"secondary_source_crosscheck.json").write_text(json.dumps(secondary_crosscheck,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    report={"status":status,"product":"Aurum Intelligence","version":"1.4.0-p0-evaluation-protocol","application_version":"1.3.1","source":{"repository":SOURCE_REPO,"url":source_url,"license_reference":LICENSE,"license_verified_for_intended_use":False,"sha256":digest,"bytes":len(payload)},"dataset":{"rows":len(rows),"start":rows[0]["timestamp"].isoformat(),"end":rows[-1]["timestamp"].isoformat(),"source_timezone_verified":False,"median_interval_seconds":statistics.median(intervals) if intervals else None,"intervals_over_600s":sum(x>600 for x in intervals)},"dataset_quality":quality,"secondary_source_quality":secondary_quality,"secondary_source_crosscheck":secondary_crosscheck,"tp_sl_evaluation_protocol":tp_sl_protocol,"real_case_benchmark":{"cases":cases,"evaluations":len(replay),"case_sampling":"timestamp-spread sample across complete dataset","cost_model":"assumed 0.30 spread plus 0.10 slippage per fill; not broker observed","rules":baselines},"full_history_replay":full_history,"prequential_probability_diagnostics":probability,"scenarios":{"market_paths":len(market),"market_passed":sum(r["passed"] for r in market),"market_failed":sum(not r["passed"] for r in market),"market_anchors":anchors,"market_archetypes":len(ARCHETYPES),"software_scenarios":len(software),"software_passed":sum(r["passed"] for r in software),"software_failed":sum(not r["passed"] for r in software)},"runtime_ms":round((time.perf_counter()-t)*1000,3),"limitations":["community OHLCV is not broker Bid/Ask or actual fills","source timezone and license scope require independent verification","costs are assumptions, not measured broker values","generated paths are counterfactuals, not historical events","scenario passes do not establish predictive skill"],"decision_output":"NO_TRADE","predictive_model_validated":False,"live_trading_enabled":False}
    (output/"online_dataset_benchmark.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    md=["# Aurum online dataset benchmark","","Result: "+status+"; decision gate: NO_TRADE.","","- Dataset SHA-256: "+digest,"- Rows: "+str(len(rows)),"- Range: "+report["dataset"]["start"]+" through "+report["dataset"]["end"],"- Historical cases: "+str(cases)+"; baseline evaluations: "+str(len(replay)),"- Market scenarios: "+str(len(market))+" ("+str(report["scenarios"]["market_passed"])+" passed)","- Software stress cases: "+str(len(software))+" ("+str(report["scenarios"]["software_passed"])+" passed)","","| Rule | Trades | Direction accuracy | Net PnL units | Max drawdown |","|---|---:|---:|---:|---:|"]
    for r in baselines:
        accuracy = "n/a" if r["direction_accuracy"] is None else format(r["direction_accuracy"], ".4f")
        md.append("| {} | {} | {} | {:.4f} | {:.4f} |".format(r["rule"], r["trades"], accuracy, r["net_pnl_units"], r["max_drawdown_units"]))
    md += ["", "## Full-history fixed-rule replay", "",
        "Every eligible next-bar opportunity is evaluated after a 60-bar warm-up. Rules are fixed and not fitted/tuned. Folds 1–4 partition the first 80%; fold 5 is exactly the final 20% tail holdout. The five folds are disjoint; metrics are descriptive, not proof of generalization.", "",
        "| Rule | Segment | Opportunities | Trades | Coverage | Direction accuracy | Net PnL units | Max drawdown |",
        "|---|---|---:|---:|---:|---:|---:|---:|"]
    for rule_name, segments in full_history["rules"].items():
        for segment_name in ("chronological_fold_1", "chronological_fold_2", "chronological_fold_3", "chronological_fold_4", "chronological_fold_5"):
            m = segments[segment_name]
            coverage = "n/a" if m["signal_coverage"] is None else format(m["signal_coverage"], ".4f")
            accuracy = "n/a" if m["directional_accuracy"] is None else format(m["directional_accuracy"], ".4f")
            md.append("| {} | {} | {} | {} | {} | {} | {:.4f} | {:.4f} |".format(rule_name, segment_name, m["opportunities"], m["trades"], coverage, accuracy, m["net_pnl_units"], m["max_drawdown_units"]))
    md += ["", "## Causal prequential probability diagnostics", "",
        "Predictions: {}. Target: next-bar candle body up vs not-up. These are historical-rate baselines, not a calibrated trading model.".format(probability.get("predictions", 0)), "",
        "| Baseline | N | Brier | Log loss | Brier skill vs 0.5 | ECE descriptive | MCE descriptive |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for method_name, m in probability.get("methods", {}).items():
        reliability = m["reliability"]
        md.append("| {} | {} | {:.6f} | {:.6f} | {:.6f} | {:.6f} | {:.6f} |".format(method_name, m["n"], m["brier_score"], m["log_loss"], m["brier_skill_vs_constant_50pct"], reliability["ece_descriptive"], reliability["mce_descriptive"]))
    md += ["", "Brier-improvement uncertainty uses a deterministic moving-block bootstrap; intervals are approximate. ECE/MCE are descriptive and sensitive to regime change/temporal dependence.",
        "", "This is a structure and probability-baseline diagnostic, not a broker-executable backtest or validated model. Verify dataset rights, timezone, quote side, execution costs and fills before production use.", "License reference: " + LICENSE, ""]
        md += ["", "## Cross-source data concordance (reference-only)", "",
        "- Secondary source: {} ({} rows); repository label is MIT, but upstream data rights and broker provenance remain unverified.".format(secondary_source_url, len(secondary_rows) if secondary_rows else 0),
        "- Exact matched timestamps: {}. Status: {}. Median close difference: {} bps; p95: {} bps; one-bar return correlation: {}.".format(secondary_crosscheck.get("common_timestamps", 0), secondary_crosscheck.get("status"), secondary_crosscheck.get("absolute_close_difference_basis_points", {}).get("median"), secondary_crosscheck.get("absolute_close_difference_basis_points", {}).get("p95"), secondary_crosscheck.get("one_bar_return_correlation")),
        "- Datasets are kept separate; comparison uses exact overlapping timestamps only. Neither source is treated as executable ground truth.", ""]
    md += ["", "## P0 TP-before-SL chronological evaluation protocol", "",
        "- Protocol: {}. Candidate trade events: {}. Binary-scored events: {}. Same-bar ambiguous events: {}.".format(tp_sl_protocol.get("protocol_version"), tp_sl_protocol.get("candidate_trade_events", 0), tp_sl_protocol.get("scored_binary_events", 0), tp_sl_protocol.get("ambiguous_same_bar_events", 0)),
        "- Partitions: development 70%, calibration 10%, final chronological holdout 20%; calibration labels that exit after holdout start are purged.",
        "- Same-bar TP/SL collisions are excluded rather than optimistically labeled. Cost values are hypothetical R-unit sensitivities, not observed broker costs.", "",
        "| Fixed rule | Side | Signals | Resolved labels | TP-first rate | Holdout status | Raw Brier | Calibrated Brier |",
        "|---|---|---:|---:|---:|---|---:|---:|"]
    for rule_name, side_results in tp_sl_protocol.get("rules", {}).items():
        for side_name, detail in side_results.items():
            if not isinstance(detail, dict) or "overall_event_summary" not in detail:
                continue
            summary = detail["overall_event_summary"]; oos = detail["oos_probability_evaluation"]
            raw = oos.get("raw_holdout_metrics", {}); calibrated = oos.get("calibrated_holdout_metrics", {})
            raw_brier = "n/a" if raw.get("brier_score") is None else "{:.6f}".format(raw["brier_score"])
            cal_brier = "n/a" if calibrated.get("brier_score") is None else "{:.6f}".format(calibrated["brier_score"])
            tp_rate = "n/a" if summary.get("tp_first_rate") is None else "{:.4f}".format(summary["tp_first_rate"])
            md.append("| {} | {} | {} | {} | {} | {} | {} | {} |".format(rule_name, side_name, summary["candidate_signals"], summary["resolved_binary_labels"], tp_rate, oos.get("status"), raw_brier, cal_brier))
    md += ["", "This report scores a causal expanding base-rate comparator for the exact TP-before-SL target. Platt calibration is fitted only on the designated calibration segment and only from labels matured by holdout start. These are historical single-source OHLC diagnostics—not a validated trading model or broker-executable result.", "",
        "Dataset quality profile: {}".format(json.dumps(quality, sort_keys=True)), "Secondary-source quality profile: {}".format(json.dumps(secondary_quality, sort_keys=True)) if secondary_quality else "Secondary-source quality profile: not run in local fixture mode", ""]
    (output/"online_dataset_benchmark.md").write_text("\n".join(md),encoding="utf-8")
    return {"status":status,"dataset_rows":len(rows),"dataset_sha256":digest,"real_data_cases":cases,"real_data_evaluations":len(replay),"full_history_decision_opportunities":full_history["decision_opportunities"],"prequential_probability_predictions":probability.get("predictions",0),"market_scenarios":len(market),"market_passed":report["scenarios"]["market_passed"],"software_scenarios":len(software),"software_passed":report["scenarios"]["software_passed"],"runtime_ms":report["runtime_ms"],"decision_output":"NO_TRADE","secondary_source_status":secondary_crosscheck.get("status"),"secondary_common_timestamps":secondary_crosscheck.get("common_timestamps",0),"secondary_close_difference_p95_bps":secondary_crosscheck.get("absolute_close_difference_basis_points",{}).get("p95"),"secondary_return_correlation":secondary_crosscheck.get("one_bar_return_correlation"),"tp_sl_protocol_status":tp_sl_protocol.get("status"),"tp_sl_candidate_trade_events":tp_sl_protocol.get("candidate_trade_events",0),"tp_sl_scored_binary_events":tp_sl_protocol.get("scored_binary_events",0),"tp_sl_ambiguous_same_bar_events":tp_sl_protocol.get("ambiguous_same_bar_events",0)}

def main():
    p=argparse.ArgumentParser(); p.add_argument("--input",type=Path); p.add_argument("--secondary-input",type=Path); p.add_argument("--output-dir",type=Path,default=Path("benchmark-output")); p.add_argument("--min-rows",type=int,default=1000); p.add_argument("--cases",type=int,default=50); p.add_argument("--scenario-anchors",type=int,default=5); a=p.parse_args()
    if a.cases<30 or a.scenario_anchors<1: p.error("cases must be >=30 and anchors >=1")
    try:
        payload=a.input.read_bytes() if a.input else fetch(); url="local-file:"+a.input.name if a.input else SOURCE
        if a.secondary_input: secondary_payload=a.secondary_input.read_bytes(); secondary_url="local-file:"+a.secondary_input.name
        elif not a.input: secondary_payload=fetch_secondary(); secondary_url=SECONDARY_SOURCE
        else: secondary_payload=None; secondary_url=SECONDARY_SOURCE
        result=run(payload,a.output_dir,a.min_rows,a.cases,a.scenario_anchors,url,secondary_payload,secondary_url); print(json.dumps(result,indent=2)); return 0 if result["status"]=="PASS" else 2
    except Exception as exc:
        print(json.dumps({"status":"FAIL","error_type":type(exc).__name__,"error":str(exc)[:300]},indent=2),file=sys.stderr); return 1
if __name__ == "__main__":
    raise SystemExit(main())
