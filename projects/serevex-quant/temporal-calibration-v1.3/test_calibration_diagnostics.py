import math, random, unittest
from calibration_diagnostics import apply_platt_scaler,evaluate_temporal_calibration,fit_platt_scaler,moving_block_bootstrap_intervals,probability_metrics

def rows(n=500,seed=78):
 rng=random.Random(seed); out=[]
 for i in range(n):
  z=rng.uniform(-2.2,2.2); truth=1/(1+math.exp(-z)); p=1/(1+math.exp(-(.58*z+.45))); d=60+5*i
  out.append({"decision_index":d,"target_exit_index":d+6,"p_tp_uncalibrated":p,"actual_tp":int(rng.random()<truth)})
 return out
class Tests(unittest.TestCase):
 def test_metrics(self):
  m=probability_metrics([.1,.8,.7,.2],[0,1,1,0],reference_probability=.5); self.assertEqual(m["n"],4); self.assertGreater(m["brier_skill_vs_reference"],0)
 def test_short_single_class_and_fit(self):
  self.assertEqual(fit_platt_scaler([.2]*19,[0,1]*9+[1],min_samples=20)["status"],"insufficient_calibration_samples")
  self.assertEqual(fit_platt_scaler([.2]*30,[1]*30,min_samples=20)["status"],"single_class_calibration_data")
  x=rows(240); f=fit_platt_scaler([r["p_tp_uncalibrated"] for r in x],[r["actual_tp"] for r in x],min_samples=100)
  self.assertTrue(f["fitted"]); self.assertGreaterEqual(f["slope"],0); self.assertLessEqual(apply_platt_scaler(.7,slope=f["slope"],intercept=f["intercept"]),1)
 def test_temporal_holdout(self):
  r=evaluate_temporal_calibration(rows(600),min_calibration_examples=100,min_evaluation_examples=50,bootstrap_replicates=30,seed=22)
  self.assertEqual(r["status"],"research_calibration_evaluated_on_later_holdout"); self.assertLessEqual(r["max_calibration_exit_index"],r["first_evaluation_decision_index"])
  self.assertFalse(r["deployment_calibrated"]); self.assertFalse(r["validated"]); self.assertFalse(r["live_signal_enabled"])
 def test_unmatured_fail_closed(self):
  x=rows(300); split=150
  for r in x[:split]: r["target_exit_index"]=x[split]["decision_index"]+100
  y=evaluate_temporal_calibration(x,min_calibration_examples=100,min_evaluation_examples=50,bootstrap_replicates=20)
  self.assertEqual(y["status"],"insufficient_matured_calibration_or_holdout_rows"); self.assertFalse(y["calibrator_fitted"]); self.assertIsNone(y["max_calibration_exit_index"])
 def test_invalid_inputs_and_order(self):
  with self.assertRaisesRegex(ValueError,"probability_out_of_range"): probability_metrics([1.1],[0])
  with self.assertRaisesRegex(ValueError,"outcome_must_be_binary"): probability_metrics([.2],[2])
  with self.assertRaisesRegex(ValueError,"probability_0_must_be_finite_number"): probability_metrics([float("nan")],[0])
  x=rows(180); x[3],x[4]=x[4],x[3]
  with self.assertRaisesRegex(ValueError,"strictly_chronological"): evaluate_temporal_calibration(x,min_calibration_examples=20,min_evaluation_examples=20)
 def test_holdout_label_mutation_not_leaked(self):
  x=rows(500,4321); kw=dict(min_calibration_examples=100,min_evaluation_examples=50,bootstrap_replicates=20,seed=5)
  a=evaluate_temporal_calibration(x,**kw); y=[dict(z) for z in x]
  for z in y[250:]: z["actual_tp"]=1-z["actual_tp"]
  b=evaluate_temporal_calibration(y,**kw)
  self.assertAlmostEqual(a["platt_fit"]["slope"],b["platt_fit"]["slope"],places=12); self.assertAlmostEqual(a["platt_fit"]["intercept"],b["platt_fit"]["intercept"],places=12)
 def test_bootstrap_determinism(self):
  x=rows(220); p=[r["p_tp_uncalibrated"] for r in x]; y=[r["actual_tp"] for r in x]
  self.assertEqual(moving_block_bootstrap_intervals(p,y,replicates=30,seed=123),moving_block_bootstrap_intervals(p,y,replicates=30,seed=123))
 def test_240_contract_scenarios(self):
  rng=random.Random(20261016)
  for case in range(240):
   data=[]
   for i in range(40+case%25):
    truth=rng.random(); p=max(.01,min(.99,.5+(truth-.5)*(.25+(case%7)*.22)+((case%11)-5)*.008)); d=100+i*5
    data.append({"decision_index":d,"target_exit_index":d+3,"p_tp_uncalibrated":p,"actual_tp":int(rng.random()<truth)})
   m=probability_metrics([r["p_tp_uncalibrated"] for r in data],[r["actual_tp"] for r in data],bins=5); self.assertEqual(m["n"],len(data))
   e=evaluate_temporal_calibration(data,min_calibration_examples=20,min_evaluation_examples=20,bins=5,bootstrap_replicates=20,block_size=4,seed=case)
   self.assertFalse(e["deployment_calibrated"]); self.assertFalse(e["live_signal_enabled"])
   if e.get("calibrator_fitted"): self.assertLessEqual(e["max_calibration_exit_index"],e["first_evaluation_decision_index"])
if __name__=="__main__": unittest.main(verbosity=2)
