import math
import unittest
from datetime import datetime, timedelta, timezone

import external_context_benchmark_v1 as bench
from point_in_time_features_v1 import THEME_WHITELIST


def make_rows(n=1200):
    start=datetime(2024,1,1,tzinfo=timezone.utc)
    rows=[]
    for i in range(n):
        idx=100+i
        label=int(math.sin(i*0.23)>0)
        row={
            "decision_id":f"d-{i:05d}","decision_time":start+timedelta(minutes=i),
            "decision_index":idx,"target_exit_index":idx+5,"label":label,"label_mature":True,
            "price_return_1":math.sin(i*0.17),"price_return_3":math.cos(i*0.07),
            "coverage_cot":"available","feature_cot_net_position":float((i%17)-8),
            "feature_cot_net_position_change":math.sin(i*0.031),
            "coverage_macro":"available","feature_macro_level":100+math.sin(i*0.013),
            "feature_macro_change":math.cos(i*0.021),
            "coverage_news":"available_zero_events" if i%7==0 else "available",
            "feature_news_count_1h":float(i%5),"feature_news_count_24h":float(i%13),
            "feature_news_tone_mean_24h":math.sin(i*0.051),
        }
        for theme in THEME_WHITELIST:
            row[f"feature_news_{theme.lower()}_count_24h"]=float((i+len(theme))%4)
        rows.append(row)
    return rows


class ExternalContextBenchmarkTests(unittest.TestCase):
    def test_all_ablations_use_same_decision_rows_and_labels(self):
        report=bench.run_context_ablation_benchmark(make_rows(),context_observation_count=120_000,
            price_feature_keys=("price_return_1","price_return_3"),bootstrap_replicates=40)
        results=[report["ablations"][k] for k in ("price_only","price_cot","price_macro","price_news","price_combined")
            if report["ablations"][k]["status"]=="SCORED"]
        self.assertGreaterEqual(len(results),4)
        self.assertEqual(len({r["scored_row_count"] for r in results}),1)
        self.assertEqual(len({r["scored_rows_sha256"] for r in results}),1)
        self.assertEqual(len({r["labels_sha256"] for r in results}),1)

    def test_final_holdout_is_not_used_for_feature_or_threshold_selection(self):
        rows=make_rows()
        a=bench.run_context_ablation_benchmark(rows,context_observation_count=120_000,
            price_feature_keys=("price_return_1","price_return_3"),bootstrap_replicates=40)
        mutated=make_rows()
        for row in mutated[960:]:
            row["feature_cot_net_position"]=9999999.0
        b=bench.run_context_ablation_benchmark(mutated,context_observation_count=120_000,
            price_feature_keys=("price_return_1","price_return_3"),bootstrap_replicates=40)
        self.assertEqual(a["ablations"]["price_cot"]["scaler_sha256"],b["ablations"]["price_cot"]["scaler_sha256"])
        self.assertEqual(a["ablations"]["price_cot"]["model_sha256"],b["ablations"]["price_cot"]["model_sha256"])
        self.assertFalse(a["protocol"]["holdout_used_for_selection"])
        self.assertFalse(a["protocol"]["holdout_used_for_calibration"])

    def test_matured_labels_only_update_causal_baseline(self):
        rows=make_rows(150)
        predictions=bench.prequential_base_rate_predictions(rows)
        target=predictions[-1]
        decision_index=rows[-1]["decision_index"]
        past=[r["label"] for r in rows[:-1] if r["target_exit_index"]<=decision_index]
        expected=(sum(past)+1)/(len(past)+2)
        self.assertAlmostEqual(target["probability"],expected)
        for r in rows:
            r["label_mature"]=False
        empty=bench.prequential_base_rate_predictions(rows)
        self.assertTrue(all(p["matured_history_count"]==0 for p in empty))

    def test_future_feature_mutation_does_not_change_holdout_predictions(self):
        rows=make_rows()
        boundary=rows[960]["decision_time"]
        for i,row in enumerate(rows):
            if i>=960:
                row["context_available_at_by_family"]={"news":row["decision_time"]+timedelta(minutes=1)}
        report_a=bench.run_context_ablation_benchmark(rows,context_observation_count=120_000,
            price_feature_keys=("price_return_1","price_return_3"),bootstrap_replicates=40)
        changed=make_rows()
        for i,row in enumerate(changed):
            if i>=960:
                row["context_available_at_by_family"]={"news":row["decision_time"]+timedelta(minutes=1)}
                row["feature_news_count_24h"]=999999.0
        report_b=bench.run_context_ablation_benchmark(changed,context_observation_count=120_000,
            price_feature_keys=("price_return_1","price_return_3"),bootstrap_replicates=40)
        self.assertEqual(report_a["ablations"]["price_news"]["prediction_sha256"],
                         report_b["ablations"]["price_news"]["prediction_sha256"])

    def test_block_bootstrap_is_deterministic(self):
        p=[0.2,0.8,0.4,0.7]*30
        y=[0,1,1,0]*30
        self.assertEqual(bench.paired_moving_block_brier_ci(p,y,[0.3]*len(p),y,
            block_size=4,replicates=40,seed=7),bench.paired_moving_block_brier_ci(p,y,[0.3]*len(p),y,
            block_size=4,replicates=40,seed=7))

    def test_low_count_calibration_is_flagged(self):
        report=bench.run_context_ablation_benchmark(make_rows(500),context_observation_count=50,
            price_feature_keys=("price_return_1","price_return_3"),bootstrap_replicates=40)
        self.assertIn(report["ablations"]["price_only"]["calibration_status"],
            ("INSUFFICIENT_CALIBRATION_SAMPLES","SINGLE_CLASS_CALIBRATION_DATA"))
        self.assertEqual(report["scale_gate"]["status"],"NOT_MET")

    def test_context_scale_below_100k_is_not_met_not_synthetic_pass(self):
        report=bench.run_context_ablation_benchmark(make_rows(600),context_observation_count=99_999,
            price_feature_keys=("price_return_1","price_return_3"),bootstrap_replicates=40)
        self.assertEqual(report["scale_gate"]["status"],"NOT_MET")
        self.assertEqual(report["scale_gate"]["observations"],99_999)
        self.assertFalse(report["scale_gate"]["passed"])

    def test_report_never_enables_trading(self):
        report=bench.run_context_ablation_benchmark(make_rows(),context_observation_count=120_000,
            price_feature_keys=("price_return_1","price_return_3"),bootstrap_replicates=40)
        self.assertEqual(report["decision_gate"],"NO_TRADE")
        self.assertFalse(report["deployment_calibrated"])
        self.assertFalse(report["validated"])
        self.assertFalse(report["live_signal_enabled"])


if __name__=="__main__":
    unittest.main()
