import os
import tempfile
import unittest
from magic_engine import (
    DataIntegrityError, EvidenceLedger, LedgerConflict, SymbolRules, analyze_snapshot,
    ema, floor_step, parse_candles, parse_symbol_rules, rsi_wilder, size_order,
    _funding_interval_hours,
)

def rules():
    return SymbolRules("MAGICUSDT","TRADING","PERPETUAL","USDT","USDT",0.00001,1.0,1.0,1.0,1.0,5.0,None)

def make_bars(interval_ms,n=90,drift=0.00015,start=1_790_000_000_000,price0=0.08):
    out=[]; p=price0
    for i in range(n):
        p *= 1.0+drift+((i%7)-3)*0.00003
        o=p*0.9995; c=p; h=max(o,c)*1.001; l=min(o,c)*0.999; op=start+i*interval_ms
        out.append([op,str(o),str(h),str(l),str(c),str(1000+i),op+interval_ms-1])
    return out

class MagicEngineTests(unittest.TestCase):
    def test_floor_step(self):
        self.assertEqual(floor_step(1.239,0.01),1.23)

    def test_symbol_filters_and_min_notional(self):
        info={"symbols":[{"symbol":"MAGICUSDT","status":"TRADING","contractType":"PERPETUAL","quoteAsset":"USDT","marginAsset":"USDT","filters":[
            {"filterType":"PRICE_FILTER","tickSize":"0.00001"},{"filterType":"LOT_SIZE","minQty":"1","stepSize":"1"},
            {"filterType":"MARKET_LOT_SIZE","minQty":"1","stepSize":"1"},{"filterType":"MIN_NOTIONAL","notional":"5"}]}]}
        parsed=parse_symbol_rules(info,"MAGICUSDT"); self.assertEqual(parsed.min_notional,5)
        check=size_order(1,0.0825,parsed); self.assertFalse(check["valid"])
        self.assertIn("NOTIONAL_BELOW_EXCHANGE_MINIMUM",check["reasons"])

    def test_candle_validation_excludes_forming_and_checks_gaps(self):
        interval=60_000; raw=make_bars(interval,n=61,start=1_790_000_000_000); now=raw[-1][6]+1
        parsed=parse_candles(raw,"1m",now_ms=now,min_bars=60); self.assertEqual(len(parsed),61)
        bad=list(raw); bad[10],bad[11]=bad[11],bad[10]
        with self.assertRaises(DataIntegrityError): parse_candles(bad,"1m",now_ms=now,min_bars=60)

    def test_candle_validator_rejects_impossible_ohlc(self):
        raw=make_bars(60_000,n=60); raw[5]=list(raw[5]); raw[5][2]="0.001"
        with self.assertRaises(DataIntegrityError): parse_candles(raw,"1m",now_ms=raw[-1][6]+1,min_bars=60)

    def test_indicators_are_finite(self):
        values=[float(i) for i in range(1,80)]
        self.assertAlmostEqual(ema(values,20),69.5,places=1); self.assertGreater(rsi_wilder(values,14),99)

    def test_ledger_identical_replay_and_integrity(self):
        with tempfile.TemporaryDirectory() as td:
            ledger=EvidenceLedger(os.path.join(td,"x.sqlite3"))
            report={"created_at_utc":"2026-10-31T00:00:00+00:00","symbol":"MAGICUSDT","status":"NO_TRADE","metrics":{"x":1}}
            a=ledger.record(report); b=ledger.record(report)
            self.assertTrue(a["inserted"]); self.assertTrue(b["replay"]); self.assertEqual(ledger.status()["integrity"],"ok")

    def test_minimum_notional_never_auto_increases_size(self):
        check=size_order(1.0,0.0825,rules())
        self.assertEqual(check["requested_notional_usdt"],1.0); self.assertEqual(check["actual_notional_usdt"],0.99)
        self.assertFalse(check["valid"])

    def test_analyzer_always_disables_live_execution(self):
        report=analyze_snapshot({"symbol":"MAGICUSDT"})
        self.assertEqual(report["status"],"NO_TRADE"); self.assertIn("LIVE_TRADING_DISABLED",report["decision_reasons"])
        self.assertFalse(report["policy"]["live_trading_enabled"])

    def test_funding_interval_is_derived_not_hardcoded(self):
        snapshot={"premium_index":{"nextFundingTime":1_800_000_000_000},"optional":{"funding_history":[{"fundingTime":1_799_985_600_000}]}}
        self.assertAlmostEqual(_funding_interval_hours(snapshot),4.0)

    def test_analyzer_validates_multitimeframe_candidate_but_stays_no_trade(self):
        now=1_800_000_000_000; candles={}
        for tf,ms in (("5m",300_000),("15m",900_000),("1h",3_600_000),("4h",14_400_000)):
            rows=[]; price=0.0825
            for i in range(90):
                price *= 1.0+0.00001+((i%6)-2.5)*0.00015
                op=now-90*ms+i*ms; o=price*0.9998; c=price
                rows.append([op,str(o),str(max(o,c)*1.0005),str(min(o,c)*0.9995),str(c),str(1000+i),op+ms-1])
            prev_high=max(float(row[2]) for row in rows[-21:-1]); prev_close=float(rows[-2][4])
            close=max(prev_high*1.00005,prev_close*1.0002); op=now-ms
            rows[-1]=[op,str(prev_close),str(close*1.001),str(prev_close*0.999),str(close),"5000",now-1]
            candles[tf]=rows
        snapshot={"symbol":"MAGICUSDT","collected_start_ms":now-2000,"collected_end_ms":now,"exchange_server_time_ms":now,
            "quote_received_ms":{"ticker":now-1000,"premium_index":now-900,"depth":now-800,"ticker_24h":now-700},
            "rules":rules().__dict__,"ticker":{"price":"0.0825"},"ticker_24h":{"quoteVolume":"10000000"},
            "premium_index":{"markPrice":"0.0825","lastFundingRate":"0.00005","nextFundingTime":now+3*3_600_000},
            "depth":{"bids":[["0.08249","100"]],"asks":[["0.08251","100"]]},
            "candles":candles,"optional":{"funding_history":[{"fundingTime":now-3_600_000}]}}
        report=analyze_snapshot(snapshot)
        self.assertTrue(report["market_data_quality"]["valid"],report["market_data_quality"])
        self.assertEqual(report["metrics"]["funding_interval_hours"],4.0)
        self.assertEqual(report["metrics"]["candidate_direction"],"LONG")
        self.assertEqual(report["status"],"NO_TRADE")
        self.assertIn("RESEARCH_ONLY_NO_ORDER_ROUTE",report["decision_reasons"])
        self.assertIn("ACCOUNT_TAKER_FEE_UNVERIFIED",report["decision_reasons"])

    def test_symbol_missing_min_notional_fails_closed(self):
        info={"symbols":[{"symbol":"MAGICUSDT","status":"TRADING","contractType":"PERPETUAL","quoteAsset":"USDT","marginAsset":"USDT","filters":[]} ]}
        with self.assertRaises(DataIntegrityError): parse_symbol_rules(info,"MAGICUSDT")

    def test_ledger_refuses_conflicting_same_identity(self):
        with tempfile.TemporaryDirectory() as td:
            ledger=EvidenceLedger(os.path.join(td,"x.sqlite3"))
            report={"created_at_utc":"2026-10-31T00:00:00+00:00","symbol":"MAGICUSDT","status":"NO_TRADE"}
            ledger.record(report)
            with ledger._connect() as conn: conn.execute("UPDATE analysis_runs SET payload_sha256=?,payload_json=?",("x","{}"))
            with self.assertRaises(LedgerConflict): ledger.record(report)

if __name__=="__main__":
    unittest.main(verbosity=2)
