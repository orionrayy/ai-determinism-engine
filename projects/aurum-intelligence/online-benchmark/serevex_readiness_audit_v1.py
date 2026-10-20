"""Fail-closed evidence audit for Serevex benchmark artifacts."""
from __future__ import annotations
import argparse, hashlib, json, math
from pathlib import Path
from typing import Any, Mapping

BASELINE = "0.1b"
MODEL_SOURCE_FIELDS = (
    "provenance_verified", "data_rights_verified", "timestamp_semantics_verified",
    "historical_feature_availability_verified", "broker_quote_equivalence_verified",
)
SCORES = ("brier", "log_loss", "ece", "positive_rate", "rows")
MIN_CONTEXT_COVERAGE = 0.95

def mp(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}

def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))

def same(a: Any, b: Any, tol: float = 1e-12) -> bool:
    if finite(a) and finite(b):
        return abs(float(a) - float(b)) <= tol
    return a == b

def _hashlike(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(ch in "0123456789abcdef" for ch in value.lower())

def audit_reports(
    reports: Mapping[str, Any],
    input_sha256: Mapping[str, str] | None = None,
    input_errors: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    blockers = {"model": [], "price_data": [], "external_context": [], "production": []}
    violations: list[str] = []
    error_area = {
        "model": "model",
        "online_dataset": "price_data",
        "tick_bidask": "price_data",
        "secondary_crosscheck": "price_data",
        "external_context": "external_context",
        "execution_replay": "production",
    }
    for artifact, reason in (input_errors or {}).items():
        area = error_area.get(artifact, "production")
        blockers[area].append(f"benchmark_artifact_unreadable:{artifact}:{reason}")
    model = mp(reports.get("model"))
    if not model:
        blockers["model"].append("required_hierarchical_shrinkage_v2_report_missing")
    else:
        source = mp(model.get("source_qualification"))
        for field in MODEL_SOURCE_FIELDS:
            if source.get(field) is not True:
                blockers["price_data"].append(f"model_source_evidence_unverified:{field}")
        base_audit = mp(model.get("baseline_reproduction_audit"))
        if base_audit.get("verified") is not True:
            blockers["model"].append("causal_baseline_reproduction_not_verified")
        err = base_audit.get("maximum_absolute_error")
        if not finite(err) or float(err) > 1e-12:
            blockers["model"].append("causal_baseline_reproduction_error_exceeds_tolerance")
        selection = mp(model.get("model_selection_gate"))
        calibration = mp(model.get("calibration_gate"))
        select_ok = selection.get("passed") is True
        calibration_ok = calibration.get("passed") is True
        if not select_ok:
            blockers["model"].append("model_selection_gate_failed:" + str(selection.get("status", "MISSING")))
        if not calibration_ok:
            blockers["model"].append("calibration_gate_not_passed:" + str(calibration.get("status", "MISSING")))
        if not select_ok and calibration.get("status") not in (None, "NOT_RUN_MODEL_SELECTION_GATE_FAILED"):
            violations.append("calibration_gate_ran_after_model_selection_failure")
        final = mp(model.get("final_holdout"))
        if final.get("final_holdout_can_change_candidate_choice") is not False:
            violations.append("final_holdout_is_not_locked_against_candidate_selection")
        baseline_metrics, policy_metrics = mp(final.get("baseline_metrics")), mp(final.get("policy_metrics"))
        policy_is_baseline = final.get("policy") == BASELINE
        if not (select_ok and calibration_ok) and not policy_is_baseline:
            violations.append("candidate_policy_promoted_without_both_research_gates")
        if policy_is_baseline or not (select_ok and calibration_ok):
            for key in SCORES:
                if key not in baseline_metrics or key not in policy_metrics:
                    violations.append(f"fallback_metric_missing:{key}")
                elif key == "rows" and (
                    not isinstance(policy_metrics[key], int)
                    or isinstance(policy_metrics[key], bool)
                    or not isinstance(baseline_metrics[key], int)
                    or isinstance(baseline_metrics[key], bool)
                ):
                    violations.append("fallback_row_count_not_integer")
                elif key != "rows" and (
                    not finite(policy_metrics[key]) or not finite(baseline_metrics[key])
                ):
                    violations.append(f"fallback_metric_non_numeric:{key}")
                elif not same(policy_metrics[key], baseline_metrics[key]):
                    violations.append(f"fallback_policy_differs_from_baseline:{key}")
        required_safety = {
            "decision_output": "NO_TRADE", "model_validated": False,
            "deployment_calibrated": False, "live_signal_enabled": False,
            "live_order_execution_enabled": False,
        }
        for key, expected in required_safety.items():
            if key not in model or model.get(key) != expected:
                violations.append(f"unsafe_or_missing_authority_state:{key}")
    online = mp(reports.get("online_dataset"))
    if not online:
        blockers["price_data"].append("online_dataset_benchmark_missing")
    model_hash = mp(mp(model.get("source_qualification"))).get("dataset_sha256") if model else None
    online_hash = mp(online.get("source")).get("sha256")
    tick_primary_hash = mp(tick.get("primary_dataset")).get("sha256") if tick else None
    if not _hashlike(model_hash):
        blockers["price_data"].append("model_dataset_sha256_missing_or_invalid")
    if not _hashlike(online_hash):
        blockers["price_data"].append("online_dataset_sha256_missing_or_invalid")
    if _hashlike(model_hash) and _hashlike(online_hash) and model_hash != online_hash:
        violations.append("model_ledger_dataset_hash_differs_from_online_benchmark")
    if _hashlike(tick_primary_hash) and _hashlike(online_hash) and tick_primary_hash != online_hash:
        violations.append("tick_audit_primary_dataset_hash_differs_from_online_benchmark")
    online_gates = mp(online.get("source_qualification_gates"))
    if not online_gates:
        blockers["price_data"].append("online_source_qualification_gates_missing")
    for required_name in ("primary_price", "secondary_price"):
        if required_name not in online_gates:
            blockers["price_data"].append(f"online_required_source_gate_missing:{required_name}")
    for name, raw in online_gates.items():
        gate = mp(raw)
        if gate.get("research_qualification_status") != "QUALIFIED":
            blockers["price_data"].append(f"online_source_not_qualified:{name}:{gate.get('research_qualification_status', 'MISSING')}")
        if gate.get("source_promotion_allowed") is not True:
            blockers["price_data"].append(f"online_source_promotion_not_authorized:{name}")
    tick = mp(reports.get("tick_bidask"))
    if not tick:
        blockers["price_data"].append("tick_bidask_benchmark_missing")
    tick_gates = mp(tick.get("source_qualification_gates"))
    if not tick_gates:
        blockers["price_data"].append("tick_source_qualification_gates_missing")
    for required_name in ("primary_price", "secondary_price", "tick_bidask"):
        if required_name not in tick_gates:
            blockers["price_data"].append(f"tick_required_source_gate_missing:{required_name}")
    for name, raw in tick_gates.items():
        gate = mp(raw)
        if gate.get("research_qualification_status") != "QUALIFIED":
            blockers["price_data"].append(f"tick_source_not_qualified:{name}:{gate.get('research_qualification_status', 'MISSING')}")
        if gate.get("source_promotion_allowed") is not True:
            blockers["price_data"].append(f"tick_source_promotion_not_authorized:{name}")
    lineage = mp(tick.get("data_lineage_diagnostic"))
    if lineage.get("source_independence_verified") is not True:
        blockers["price_data"].append("tick_source_independence_unverified:" + str(lineage.get("status", "MISSING")))
    secondary = mp(reports.get("secondary_crosscheck"))
    if not secondary:
        blockers["price_data"].append("secondary_source_crosscheck_missing")
    elif (secondary.get("status") != "PASS" or secondary.get("timestamp_semantics_verified") is not True
          or secondary.get("provider_provenance_independently_verified") is not True
          or secondary.get("intended_use_rights_verified") is not True):
        blockers["price_data"].append("secondary_source_disagreement_or_provenance_unresolved:" + str(secondary.get("status", "MISSING")))
    context = mp(reports.get("external_context"))
    if not context:
        blockers["external_context"].append("external_context_benchmark_missing")
    for key in ("news_1h", "news_24h", "macro"):
        feature = mp(mp(context.get("temporal_feature_coverage")).get(key))
        coverage, eligible, total = feature.get("coverage_fraction"), feature.get("eligible_rows"), feature.get("total_rows")
        if not finite(coverage) or float(coverage) < MIN_CONTEXT_COVERAGE or not finite(eligible) or not finite(total) or float(total) <= 0:
            blockers["external_context"].append(f"temporal_coverage_insufficient:{key}:{eligible}/{total}")
    execution = mp(reports.get("execution_replay"))
    if not execution:
        blockers["production"].append("execution_replay_report_missing")
    elif execution.get("validated_for_profitability") is not True or execution.get("validated_model") is not True:
        blockers["production"].append("model_or_profitability_not_validated:" + str(execution.get("status", "MISSING")))
    if execution.get("live_execution_authorized") is True:
        violations.append("research_report_claims_live_execution_authority")
    if model and mp(mp(model.get("final_holdout"))).get("policy") != BASELINE and blockers["price_data"]:
        violations.append("candidate_policy_promoted_while_price_sources_unqualified")
    for area in blockers:
        blockers[area] = sorted(set(blockers[area]))
    violations = sorted(set(violations))
    all_blockers = [f"{area}:{item}" for area, items in blockers.items() for item in items]
    select_ok = mp(model.get("model_selection_gate")).get("passed") is True if model else False
    calibration_ok = mp(model.get("calibration_gate")).get("passed") is True if model else False
    review_ok = bool(model) and select_ok and calibration_ok and not blockers["model"] and not blockers["price_data"] and not violations
    overall = "INTEGRITY_VIOLATION" if violations else ("RESEARCH_ONLY_BLOCKED" if all_blockers else "RESEARCH_CANDIDATE_ELIGIBLE_FOR_INDEPENDENT_REVIEW")
    return {
        "schema_version": "serevex-readiness-audit-v1",
        "overall_status": overall,
        "model_candidate_status": "CANDIDATE_PASSED_RESEARCH_GATES" if select_ok and calibration_ok else "BASELINE_ONLY",
        "source_status": "QUALIFIED" if not blockers["price_data"] else "REFERENCE_ONLY",
        "external_context_status": "ADEQUATE_FOR_RESEARCH_REVIEW" if not blockers["external_context"] else "INSUFFICIENT_TEMPORAL_COVERAGE",
        "execution_status": "ECONOMICALLY_VALIDATED" if not blockers["production"] else "NOT_VALIDATED",
        "research_candidate_review_eligible": review_ok,
        "production_authorized": False, "decision_output": "NO_TRADE",
        "live_signal_enabled": False, "live_order_execution_enabled": False,
        "blockers_by_area": blockers, "blocker_count": len(all_blockers),
        "integrity_violations": violations, "integrity_violation_count": len(violations),
        "input_sha256": dict(sorted((input_sha256 or {}).items())),
        "input_errors": dict(sorted((input_errors or {}).items())),
        "policy_note": "Passing permits independent research review only; it never authorizes deployment, live signals, or orders.",
    }

def render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Serevex Readiness Audit", "",
        f"- Overall status: **{report.get('overall_status')}**",
        f"- Model candidate: {report.get('model_candidate_status')}",
        f"- Price sources: {report.get('source_status')}",
        f"- External context: {report.get('external_context_status')}",
        f"- Execution: {report.get('execution_status')}",
        f"- Research candidate review eligible: {report.get('research_candidate_review_eligible')}",
        "- Production authorized: False",
        "- Decision output: NO_TRADE", "", "## Blockers", "",
    ]
    for area in ("model", "price_data", "external_context", "production"):
        entries = mp(report.get("blockers_by_area")).get(area, [])
        if entries:
            lines.append("### " + area.replace("_", " ").title())
            lines.extend("- " + item for item in entries)
            lines.append("")
    lines.extend(["## Integrity violations", ""])
    violations = report.get("integrity_violations", [])
    lines.extend(("- " + x for x in violations) if violations else ["None."])
    lines.extend(["", "> This audits evidence readiness, not predictive accuracy or profitability. It never authorizes trading.", ""])
    return "\n".join(lines)

def run_audit(benchmark_dir: Path, output_dir: Path) -> dict[str, Any]:
    spec = {
        "model": "serevex-stage123/serevex_hierarchical_shrinkage_v2_report.json",
        "online_dataset": "online_dataset_benchmark.json",
        "tick_bidask": "tick_bidask_benchmark.json",
        "secondary_crosscheck": "secondary_source_crosscheck.json",
        "external_context": "external-context/external_context_benchmark.json",
        "execution_replay": "large-scale/execution_replay_report.json",
    }
    reports, hashes, errors = {}, {}, {}
    output_dir.mkdir(parents=True, exist_ok=True)
    for key, relative in spec.items():
        path = benchmark_dir / relative
        if not path.is_file():
            reports[key] = None
            errors[key] = "missing_artifact"
            continue
        raw = path.read_bytes()
        hashes[key] = hashlib.sha256(raw).hexdigest()
        try:
            reports[key] = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            reports[key] = None
            errors[key] = "invalid_json_or_encoding"
    result = audit_reports(reports, hashes, errors)
    (output_dir / "serevex_readiness_audit.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    (output_dir / "readiness_summary.md").write_text(render_markdown(result), encoding="utf-8")
    return result

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run_audit(args.benchmark_dir, args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 2 if result["integrity_violations"] or result["input_errors"] else 0

if __name__ == "__main__":
    raise SystemExit(main())
