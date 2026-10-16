"""Leakage-aware temporal calibration diagnostics; never enables trading."""
from __future__ import annotations
import math, random
from statistics import mean
from typing import Any, Mapping, Sequence
EPS=1e-6

def _number(v: Any, name: str) -> float:
    if isinstance(v,bool): raise ValueError(f"{name}_must_be_finite_number")
    try: x=float(v)
    except (TypeError,ValueError,OverflowError) as e: raise ValueError(f"{name}_must_be_finite_number") from e
    if not math.isfinite(x): raise ValueError(f"{name}_must_be_finite_number")
    return x

def _pairs(ps: Sequence[Any], ys: Sequence[Any]):
    if len(ps)!=len(ys): raise ValueError("probability_outcome_length_mismatch")
    if not ps: raise ValueError("forecasts_must_not_be_empty")
    p2=[]; y2=[]
    for i,(p,y) in enumerate(zip(ps,ys)):
        p=_number(p,f"probability_{i}")
        if not 0<=p<=1: raise ValueError(f"probability_out_of_range_at_{i}")
        if isinstance(y,bool) or y not in (0,1): raise ValueError(f"outcome_must_be_binary_at_{i}")
        p2.append(p); y2.append(int(y))
    return p2,y2

def _logit(p):
    p=min(1-EPS,max(EPS,p)); return math.log(p/(1-p))
def _sigmoid(z): return 1/(1+math.exp(-max(-35,min(35,z))))

def apply_platt_scaler(probability: Any, *, slope: float, intercept: float) -> float:
    p=_number(probability,"probability"); a=_number(slope,"slope"); b=_number(intercept,"intercept")
    if not 0<=p<=1: raise ValueError("probability_out_of_range")
    if a<0: raise ValueError("platt_slope_must_be_non_negative")
    return _sigmoid(a*_logit(p)+b)

def fit_platt_scaler(probabilities: Sequence[Any], outcomes: Sequence[Any], *, min_samples=100, iterations=500, l2=.01):
    ps,ys=_pairs(probabilities,outcomes)
    if isinstance(min_samples,bool) or not isinstance(min_samples,int) or min_samples<20: raise ValueError("min_samples_must_be_integer_at_least_20")
    if isinstance(iterations,bool) or not isinstance(iterations,int) or not 1<=iterations<=5000: raise ValueError("iterations_must_be_integer_between_1_and_5000")
    penalty=_number(l2,"l2")
    if not 0<=penalty<=100: raise ValueError("l2_out_of_range")
    base={"n":len(ps),"method":"regularized_monotone_platt_scaling","fitted":False}
    if len(ps)<min_samples: return {**base,"status":"insufficient_calibration_samples","minimum_required":min_samples}
    positives=sum(ys)
    if positives in (0,len(ys)): return {**base,"status":"single_class_calibration_data","positive_count":positives}
    xs=[_logit(p) for p in ps]; n=len(xs); a,b=1.,0.
    def loss(aa,bb):
        s=.5*penalty*((aa-1)**2+bb**2)
        for x,y in zip(xs,ys):
            z=aa*x+bb; s+=(max(z,0)+math.log1p(math.exp(-abs(z)))-y*z)/n
        return s
    current=loss(a,b); stopped=False
    for _ in range(iterations):
        ga,gb=penalty*(a-1),penalty*b
        for x,y in zip(xs,ys):
            err=_sigmoid(a*x+b)-y; ga+=err*x/n; gb+=err/n
        norm=math.hypot(ga,gb)
        if norm<1e-8: stopped=True; break
        step=1.; accepted=False
        for _ in range(24):
            na=max(0.,a-step*ga); nb=max(-15.,min(15.,b-step*gb)); nl=loss(na,nb)
            if nl<=current-1e-4*step*norm*norm or nl<current:
                a,b,current,accepted=na,nb,nl,True; break
            step*=.5
        if not accepted: stopped=True; break
    if not all(math.isfinite(x) for x in (a,b,current)): return {**base,"status":"non_finite_fit_rejected"}
    return {**base,"status":"fit_complete","fitted":True,"slope":a,"intercept":b,"training_log_loss_regularized":current,
        "positive_count":positives,"negative_count":n-positives,"converged_or_step_limited":stopped,"regularization_l2":penalty}

def probability_metrics(probabilities: Sequence[Any], outcomes: Sequence[Any], *, bins=10, reference_probability=None):
    ps,ys=_pairs(probabilities,outcomes)
    if isinstance(bins,bool) or not isinstance(bins,int) or not 2<=bins<=50: raise ValueError("bins_must_be_integer_between_2_and_50")
    clipped=[min(1-1e-12,max(1e-12,p)) for p in ps]
    brier=mean((p-y)**2 for p,y in zip(clipped,ys))
    ll=mean(-(y*math.log(p)+(1-y)*math.log(1-p)) for p,y in zip(clipped,ys))
    bins_out=[]; ece=mce=0.
    for i in range(bins):
        lo,hi=i/bins,(i+1)/bins
        part=[(p,y) for p,y in zip(ps,ys) if lo<=p<hi or (i==bins-1 and p==1)]
        if not part:
            bins_out.append({"lower":lo,"upper":hi,"n":0,"mean_predicted":None,"observed_rate":None,"absolute_gap":None}); continue
        pred=mean(p for p,_ in part); obs=mean(y for _,y in part); gap=abs(pred-obs)
        ece+=len(part)/len(ps)*gap; mce=max(mce,gap)
        bins_out.append({"lower":lo,"upper":hi,"n":len(part),"mean_predicted":pred,"observed_rate":obs,"absolute_gap":gap})
    out={"n":len(ps),"positive_count":sum(ys),"negative_count":len(ys)-sum(ys),"event_rate":mean(ys),
         "mean_probability":mean(ps),"brier_score":brier,"log_loss":ll,"ece_equal_width_descriptive":ece,
         "mce_equal_width_descriptive":mce,"reliability_bins":bins_out,"empty_reliability_bins":sum(x["n"]==0 for x in bins_out),
         "interpretation":"Descriptive only; ECE/MCE depend on binning and sample size."}
    if reference_probability is not None:
        ref=_number(reference_probability,"reference_probability")
        if not 0<=ref<=1: raise ValueError("reference_probability_out_of_range")
        ref=min(1-1e-12,max(1e-12,ref)); rb=mean((ref-y)**2 for y in ys)
        rll=mean(-(y*math.log(ref)+(1-y)*math.log(1-ref)) for y in ys)
        out.update({"reference_probability":reference_probability,"reference_brier_score":rb,"reference_log_loss":rll,
          "brier_skill_vs_reference":1-brier/rb if rb>0 else None,"log_loss_improvement_vs_reference":rll-ll})
    return out

def moving_block_bootstrap_intervals(probabilities: Sequence[Any], outcomes: Sequence[Any], *, bins=10, block_size=12, replicates=250, seed=20261016):
    ps,ys=_pairs(probabilities,outcomes)
    if isinstance(block_size,bool) or not isinstance(block_size,int) or block_size<1: raise ValueError("block_size_must_be_positive_integer")
    if isinstance(replicates,bool) or not isinstance(replicates,int) or not 20<=replicates<=5000: raise ValueError("replicates_must_be_integer_between_20_and_5000")
    if isinstance(seed,bool) or not isinstance(seed,int): raise ValueError("seed_must_be_integer")
    rng=random.Random(seed); n=len(ps); size=min(block_size,n)
    values={k:[] for k in ("brier_score","log_loss","ece_equal_width_descriptive","mce_equal_width_descriptive")}
    for _ in range(replicates):
        bp=[]; by=[]
        while len(bp)<n:
            start=rng.randrange(0,n-size+1); bp.extend(ps[start:start+size]); by.extend(ys[start:start+size])
        m=probability_metrics(bp[:n],by[:n],bins=bins)
        for key in values: values[key].append(float(m[key]))
    ci={}
    for key,x in values.items():
        x.sort(); ci[key]={"lower_95":x[max(0,int(.025*(len(x)-1)))],"upper_95":x[min(len(x)-1,int(.975*(len(x)-1)))],"replicates":len(x)}
    return {"method":"moving_block_bootstrap_percentile","block_size":size,"seed":seed,"intervals":ci,
      "warning":"Sensitivity diagnostic; block length and regime drift affect coverage."}

def evaluate_temporal_calibration(rows: Sequence[Mapping[str,Any]], *, probability_key="p_tp_uncalibrated", outcome_key="actual_tp",
    decision_index_key="decision_index", exit_index_key="target_exit_index", calibration_fraction=.5, min_calibration_examples=100,
    min_evaluation_examples=50, bins=10, bootstrap_replicates=250, block_size=12, seed=20261016):
    base={"status":"insufficient_evaluation_history","calibrator_fitted":False,"holdout_evaluated":False,"deployment_calibrated":False,"validated":False,"live_signal_enabled":False}
    if isinstance(calibration_fraction,bool) or not isinstance(calibration_fraction,(int,float)) or not .2<=calibration_fraction<=.8: raise ValueError("calibration_fraction_must_be_between_0_2_and_0_8")
    for name,v in (("min_calibration_examples",min_calibration_examples),("min_evaluation_examples",min_evaluation_examples)):
        if isinstance(v,bool) or not isinstance(v,int) or v<20: raise ValueError(f"{name}_must_be_at_least_20")
    if not rows: return {**base,"status":"no_out_of_sample_predictions","n_predictions":0}
    data=[]; previous=-1
    for i,row in enumerate(rows):
        if not isinstance(row,Mapping): raise ValueError(f"prediction_row_must_be_mapping_at_{i}")
        d,x=row.get(decision_index_key),row.get(exit_index_key)
        if isinstance(d,bool) or not isinstance(d,int) or d<0: raise ValueError(f"invalid_decision_index_at_{i}")
        if isinstance(x,bool) or not isinstance(x,int) or x<d: raise ValueError(f"invalid_exit_index_at_{i}")
        if d<=previous: raise ValueError("prediction_rows_must_be_strictly_chronological")
        previous=d; p=_number(row.get(probability_key),f"probability_{i}")
        if not 0<=p<=1: raise ValueError(f"probability_out_of_range_at_{i}")
        y=row.get(outcome_key)
        if isinstance(y,bool) or y not in (0,1): raise ValueError(f"outcome_must_be_binary_at_{i}")
        data.append({"decision_index":d,"target_exit_index":x,"probability":p,"outcome":int(y)})
    split=int(len(data)*float(calibration_fraction))
    if split<1 or split>=len(data): return {**base,"status":"insufficient_split_size","n_predictions":len(data)}
    candidates,holdout=data[:split],data[split:]; first=holdout[0]["decision_index"]
    train=[r for r in candidates if r["target_exit_index"]<=first]
    detail={"n_predictions":len(data),"n_calibration_candidates":len(candidates),"n_calibration_matured":len(train),"n_evaluation":len(holdout),
      "first_evaluation_decision_index":first,"max_calibration_exit_index":max((r["target_exit_index"] for r in train),default=None),"calibration_fraction":float(calibration_fraction)}
    if len(train)<min_calibration_examples or len(holdout)<min_evaluation_examples:
        return {**base,**detail,"status":"insufficient_matured_calibration_or_holdout_rows"}
    tp,ty=_pairs([r["probability"] for r in train],[r["outcome"] for r in train]); hp,hy=_pairs([r["probability"] for r in holdout],[r["outcome"] for r in holdout])
    fit=fit_platt_scaler(tp,ty,min_samples=min_calibration_examples)
    if not fit.get("fitted"): return {**base,**detail,"status":fit["status"],"platt_fit":fit}
    cal=[apply_platt_scaler(p,slope=fit["slope"],intercept=fit["intercept"]) for p in hp]; ref=(sum(ty)+1)/(len(ty)+2)
    raw=probability_metrics(hp,hy,bins=bins,reference_probability=ref); adj=probability_metrics(cal,hy,bins=bins,reference_probability=ref)
    raw_ci=moving_block_bootstrap_intervals(hp,hy,bins=bins,block_size=block_size,replicates=bootstrap_replicates,seed=seed)
    cal_ci=moving_block_bootstrap_intervals(cal,hy,bins=bins,block_size=block_size,replicates=bootstrap_replicates,seed=seed+1)
    return {**base,**detail,"status":"research_calibration_evaluated_on_later_holdout","calibrator_fitted":True,"holdout_evaluated":True,
      "calibration_method":"regularized monotone Platt scaling","platt_fit":fit,"training_reference_probability":ref,"raw_holdout_metrics":raw,
      "calibrated_holdout_metrics":adj,"raw_holdout_block_bootstrap_ci":raw_ci,"calibrated_holdout_block_bootstrap_ci":cal_ci,
      "comparison":{"brier_improvement_calibrated_minus_raw":raw["brier_score"]-adj["brier_score"],"log_loss_improvement_calibrated_minus_raw":raw["log_loss"]-adj["log_loss"],
       "ece_reduction_calibrated_minus_raw":raw["ece_equal_width_descriptive"]-adj["ece_equal_width_descriptive"],
       "interpretation":"Positive improvement favors calibration on this holdout only; not production validation."},
      "limitations":["Bootstrap coverage depends on block size and regime drift.","Training excludes outcomes not mature at holdout start.","OHLC labels and assumed costs are not broker fills.","No execution or decision policy is altered."]}
