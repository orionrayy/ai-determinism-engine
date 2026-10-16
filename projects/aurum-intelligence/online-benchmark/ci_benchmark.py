#!/usr/bin/env python3
"""Online, read-only XAUUSD dataset audit and deterministic benchmark; stdlib only."""
import argparse, csv, hashlib, io, json, math, statistics, sys, time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.error import URLError, HTTPError
from urllib.request import Request, urlopen

SOURCE = "https://raw.githubusercontent.com/getdata-finance/xauusd-5m-ohlcv-metals-historical-data/main/XAUUSD_5m.csv"
SOURCE_REPO = "getdata-finance/xauusd-5m-ohlcv-metals-historical-data"
LICENSE = "https://github.com/getdata-finance/xauusd-5m-ohlcv-metals-historical-data/blob/main/LICENSE"
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
    req = Request(SOURCE, headers={"User-Agent": "Aurum-Intelligence/1.1.0 online-benchmark"})
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
    closes = [x["close"] for x in rows[:i+1]]
    p = closes[-1]
    if rule == "flat_no_trade": return 0
    if rule == "last_return_sign": return int(p > closes[-2]) - int(p < closes[-2])
    if rule == "ema5_vs_ema10":
        xs = closes[-60:]
        return int(ema(xs, 5) > ema(xs, 10)) - int(ema(xs, 5) < ema(xs, 10))
    if rule == "mean_reversion_z10":
        xs = closes[-10:]
        sd = statistics.pstdev(xs)
        z = (p - statistics.mean(xs)) / sd if sd else 0
        return -1 if z > 1 else 1 if z < -1 else 0
    prev = rows[max(0, i-20):i]
    return 1 if p > max(x["high"] for x in prev) else -1 if p < min(x["low"] for x in prev) else 0

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

def run(payload, output, min_rows=1000, cases=50, anchors=5, source_url=SOURCE):
    t=time.perf_counter(); rows=parse_csv(payload,min_rows)
    intervals=[(rows[i]["timestamp"]-rows[i-1]["timestamp"]).total_seconds() for i in range(1,len(rows))]
    digest=hashlib.sha256(payload).hexdigest(); replay=case_replay(rows,cases); market=market_scenarios(rows,anchors); software=software_scenarios(rows)
    baselines=[]
    for rule in RULES:
        xs=[r for r in replay if r["rule"]==rule]; trades=[r for r in xs if r["signal"]]; pnl=[r["net_pnl_units"] for r in trades]; equity=peak=dd=0.0
        for value in pnl: equity+=value; peak=max(peak,equity); dd=max(dd,peak-equity)
        correct=sum(r["signal"]==r["next_body_direction"] for r in trades)
        baselines.append({"rule":rule,"cases":len(xs),"trades":len(trades),"direction_accuracy":correct/len(trades) if trades else None,"net_pnl_units":round(sum(pnl),8),"max_drawdown_units":round(dd,8)})
    fail=sum(not r["passed"] for r in market)+sum(not r["passed"] for r in software)
    status="PASS" if fail==0 and len(replay)==cases*5 and len(market)>=200 and len(software)>=200 else "FAIL"
    output.mkdir(parents=True,exist_ok=True); write_csv(output/"real_data_case_results.csv",replay); write_csv(output/"market_scenario_results.csv",market); write_csv(output/"software_scenario_results.csv",software)
    report={"status":status,"product":"Aurum Intelligence","version":"1.1.0","source":{"repository":SOURCE_REPO,"url":source_url,"license_reference":LICENSE,"license_verified_for_intended_use":False,"sha256":digest,"bytes":len(payload)},"dataset":{"rows":len(rows),"start":rows[0]["timestamp"].isoformat(),"end":rows[-1]["timestamp"].isoformat(),"source_timezone_verified":False,"median_interval_seconds":statistics.median(intervals) if intervals else None,"intervals_over_600s":sum(x>600 for x in intervals)},"real_case_benchmark":{"cases":cases,"evaluations":len(replay),"cost_model":"assumed 0.30 spread plus 0.10 slippage per fill; not broker observed","rules":baselines},"scenarios":{"market_paths":len(market),"market_passed":sum(r["passed"] for r in market),"market_failed":sum(not r["passed"] for r in market),"market_anchors":anchors,"market_archetypes":len(ARCHETYPES),"software_scenarios":len(software),"software_passed":sum(r["passed"] for r in software),"software_failed":sum(not r["passed"] for r in software)},"runtime_ms":round((time.perf_counter()-t)*1000,3),"limitations":["community OHLCV is not broker Bid/Ask or actual fills","source timezone and license scope require independent verification","costs are assumptions, not measured broker values","generated paths are counterfactuals, not historical events","scenario passes do not establish predictive skill"],"decision_output":"NO_TRADE","predictive_model_validated":False,"live_trading_enabled":False}
    (output/"online_dataset_benchmark.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    md=["# Aurum online dataset benchmark","","Result: "+status+"; decision gate: NO_TRADE.","","- Dataset SHA-256: "+digest,"- Rows: "+str(len(rows)),"- Range: "+report["dataset"]["start"]+" through "+report["dataset"]["end"],"- Historical cases: "+str(cases)+"; baseline evaluations: "+str(len(replay)),"- Market scenarios: "+str(len(market))+" ("+str(report["scenarios"]["market_passed"])+" passed)","- Software stress cases: "+str(len(software))+" ("+str(report["scenarios"]["software_passed"])+" passed)","","| Rule | Trades | Direction accuracy | Net PnL units | Max drawdown |","|---|---:|---:|---:|---:|"]
    for r in baselines: md.append("| "+r["rule"]+" | "+str(r["trades"])+" | "+("n/a" if r["direction_accuracy"] is None else f'{r["direction_accuracy"]:.4f}')+" | "+f'{r["net_pnl_units"]:.4f}'+" | "+f'{r["max_drawdown_units"]:.4f}'+" |')
    md += ["","This is a public-data structure and diagnostic benchmark, not a broker-executable backtest or a validated prediction. Recheck the exact license, source timezone, spread, commission, slippage and fill data before production use.","License reference: "+LICENSE,""]
    (output/"online_dataset_benchmark.md").write_text("\n".join(md),encoding="utf-8")
    return {"status":status,"dataset_rows":len(rows),"dataset_sha256":digest,"real_data_cases":cases,"real_data_evaluations":len(replay),"market_scenarios":len(market),"market_passed":report["scenarios"]["market_passed"],"software_scenarios":len(software),"software_passed":report["scenarios"]["software_passed"],"runtime_ms":report["runtime_ms"],"decision_output":"NO_TRADE"}

def main():
    p=argparse.ArgumentParser(); p.add_argument("--input",type=Path); p.add_argument("--output-dir",type=Path,default=Path("benchmark-output")); p.add_argument("--min-rows",type=int,default=1000); p.add_argument("--cases",type=int,default=50); p.add_argument("--scenario-anchors",type=int,default=5); a=p.parse_args()
    if a.cases<30 or a.scenario_anchors<1: p.error("cases must be >=30 and anchors >=1")
    try:
        payload=a.input.read_bytes() if a.input else fetch(); url="local-file:"+a.input.name if a.input else SOURCE
        result=run(payload,a.output_dir,a.min_rows,a.cases,a.scenario_anchors,url); print(json.dumps(result,indent=2)); return 0 if result["status"]=="PASS" else 2
    except Exception as exc:
        print(json.dumps({"status":"FAIL","error_type":type(exc).__name__,"error":str(exc)[:300]},indent=2),file=sys.stderr); return 1
if __name__ == "__main__":
    raise SystemExit(main())
