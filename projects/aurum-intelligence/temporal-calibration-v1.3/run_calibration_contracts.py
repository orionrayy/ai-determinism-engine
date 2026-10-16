"""240 deterministic temporal-calibration contracts; synthetic forecasts only."""
import json,math,random,time
from datetime import datetime,timezone
from calibration_diagnostics import evaluate_temporal_calibration,probability_metrics
def main():
 start=time.perf_counter(); counts={}
 for case in range(240):
  rng=random.Random(20261016+case); rows=[]
  for i in range(40+case%25):
   z=rng.uniform(-2.7,2.7); truth=1/(1+math.exp(-z)); p=1/(1+math.exp(-((.35+case%9*.17)*z+((case%13)-6)*.055))); d=100+5*i
   rows.append({"decision_index":d,"target_exit_index":d+3,"p_tp_uncalibrated":p,"actual_tp":int(rng.random()<truth)})
  m=probability_metrics([r["p_tp_uncalibrated"] for r in rows],[r["actual_tp"] for r in rows],bins=5,reference_probability=.5)
  assert m["n"]==len(rows) and math.isfinite(m["brier_score"]) and 0<=m["brier_score"]<=1
  e=evaluate_temporal_calibration(rows,min_calibration_examples=20,min_evaluation_examples=20,bins=5,bootstrap_replicates=20,block_size=4,seed=case)
  assert not e["deployment_calibrated"] and not e["live_signal_enabled"]
  if e.get("calibrator_fitted"): assert e["max_calibration_exit_index"]<=e["first_evaluation_decision_index"]
  counts[e["status"]]=counts.get(e["status"],0)+1
 print(json.dumps({"status":"PASS","scenario_count":240,"passed":240,"failed":0,"status_counts":counts,"synthetic_only":True,
  "runtime_ms":round((time.perf_counter()-start)*1000,3),"created_at_utc":datetime.now(timezone.utc).isoformat(timespec="seconds"),
  "deployment_calibrated":False,"validated":False,"live_signal_enabled":False,"scope":"contract validation only; not market-performance evidence"},indent=2))
if __name__=="__main__": main()
