"""Emit an evidence-gated readiness report from actual benchmark outputs."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any, Mapping
import research_readiness_gate_v1 as gate


def read_json(path: Path, errors: list[str]) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        errors.append("missing_input:" + str(path))
        return None
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        errors.append("unreadable_input:" + str(path) + ":" + type(exc).__name__)
        return None
    if not isinstance(value, dict):
        errors.append("input_root_must_be_object:" + str(path))
        return None
    return value


def status_of(report: Mapping[str, Any] | None) -> str | None:
    value = report.get("status") if report else None
    return value if isinstance(value, str) else None


def data_status(report: Mapping[str, Any] | None) -> str:
    gates = report.get("source_qualification_gates") if report else None
    if not isinstance(gates, dict) or not gates:
        return "NOT_ASSESSED"
    statuses = [item.get("research_qualification_status") for item in gates.values() if isinstance(item, dict)]
    if not statuses:
        return "NOT_ASSESSED"
    if any(value in {"BLOCKED", "REJECTED"} for value in statuses):
        return "BLOCKED"
    # Review eligibility and CI success are not final source approval.
    return "REFERENCE_ONLY"


def statistical_status(
    large: Mapping[str, Any] | None, context: Mapping[str, Any] | None
) -> str:
    if not large or not context:
        return "NOT_ASSESSED"
    if (
        large.get("validated_model") is True
        and large.get("deployment_calibrated") is True
        and context.get("validated") is True
        and context.get("deployment_calibrated") is True
    ):
        return "SUPPORTED"
    if large.get("validation_decision") == "REJECTED" or context.get("validation_decision") == "REJECTED":
        return "REJECTED"
    return "INCONCLUSIVE"


def economic_status(
    large: Mapping[str, Any] | None, context: Mapping[str, Any] | None
) -> str:
    if not large or not context:
        return "NOT_ASSESSED"
    for report in (large, context):
        evidence = report.get("economic_validation")
        if not isinstance(evidence, dict):
            continue
        if evidence.get("measured_broker_costs") is not True or evidence.get("independent_holdout") is not True:
            continue
        lower = evidence.get("net_utility_ci95_lower")
        if isinstance(lower, bool) or not isinstance(lower, (int, float)):
            continue
        if evidence.get("status") == "NEGATIVE" or lower <= 0:
            return "NEGATIVE"
        if evidence.get("status") == "SUPPORTED":
            return "SUPPORTED"
    return "INCONCLUSIVE"


def operations_status(root: Path) -> str:
    path = root / "paper-trading" / "acceptance.json"
    if not path.is_file():
        return "NOT_RUN"
    errors: list[str] = []
    item = read_json(path, errors)
    if not item:
        return "FAIL"
    return "PASS" if (
        item.get("status") == "PASS"
        and item.get("human_review_approved") is True
        and isinstance(item.get("review_evidence_reference"), str)
        and item["review_evidence_reference"].strip()
    ) else "FAIL"


def build_report(root: Path, contracts_path: Path, run_url: str) -> dict[str, Any]:
    errors: list[str] = []
    reports = {
        "online_dataset": read_json(root / "online_dataset_benchmark.json", errors),
        "tick_bidask": read_json(root / "tick_bidask_benchmark.json", errors),
        "large_scale": read_json(root / "large-scale" / "large_scale_bidask_benchmark.json", errors),
        "external_context": read_json(root / "external-context" / "external_context_benchmark.json", errors),
    }
    contracts = read_json(contracts_path, errors)
    benchmarks_pass = all(report is not None and status_of(report) == "PASS" for report in reports.values())
    contracts_pass = bool(
        contracts and contracts.get("status") == "PASS"
        and contracts.get("scenario_count") == 240 and contracts.get("passed") == 240
        and contracts.get("failed") == 0 and contracts.get("synthetic_only") is True
    )
    pipeline = "PASS" if benchmarks_pass and contracts_pass and not errors else "FAIL"
    refs = {name: run_url for name in (
        "pipeline_status", "data_status", "statistical_status", "economic_status", "operations_status"
    )} if isinstance(run_url, str) and run_url.startswith("https://") else {}
    result = gate.evaluate_research_readiness(
        pipeline_status=pipeline,
        data_status=data_status(reports["tick_bidask"]),
        statistical_status=statistical_status(reports["large_scale"], reports["external_context"]),
        economic_status=economic_status(reports["large_scale"], reports["external_context"]),
        operations_status=operations_status(root),
        evidence_refs=refs,
        release_review_status="PENDING",
    )

    online = reports["online_dataset"] or {}
    large = reports["large_scale"] or {}
    context = reports["external_context"] or {}
    tick = reports["tick_bidask"] or {}
    primary = online.get("dataset", {}) if isinstance(online.get("dataset"), dict) else {}
    source = online.get("source", {}) if isinstance(online.get("source"), dict) else {}
    large_data = large.get("dataset", {}) if isinstance(large.get("dataset"), dict) else {}
    scenario = large.get("scenario_contract", {}) if isinstance(large.get("scenario_contract"), dict) else {}
    probability = large.get("prequential_probability_evaluation", {})
    aggregate = probability.get("aggregate", {}) if isinstance(probability, dict) else {}
    scale = context.get("scale_gate", {}) if isinstance(context.get("scale_gate"), dict) else {}
    coverage = context.get("gdelt_time_coverage", {}) if isinstance(context.get("gdelt_time_coverage"), dict) else {}
    gates = tick.get("source_qualification_gates")

    result.update({
        "project": "Serevex Quant Research",
        "run_url": run_url,
        "inputs": {name: {"present": item is not None, "status": status_of(item)} for name, item in reports.items()},
        "calibration_contracts": {
            "present": contracts is not None,
            "status": status_of(contracts),
            "scenario_count": contracts.get("scenario_count") if contracts else None,
            "passed": contracts.get("passed") if contracts else None,
            "synthetic_only": contracts.get("synthetic_only") if contracts else None,
        },
        "observed_data_summary": {
            "community_primary_rows": primary.get("rows"),
            "community_primary_sha256": source.get("sha256"),
            "large_scale_bid_ask_rows": large_data.get("paired_valid_bid_ask_candles"),
            "large_scale_months": large_data.get("months_requested"),
            "unique_real_anchor_timestamps": scenario.get("selected_unique_anchor_timestamps"),
            "real_contiguous_atr_windows": scenario.get("eligible_real_contiguous_windows_with_valid_atr"),
            "tp_sl_path_scenarios": scenario.get("tp_sl_path_scenarios"),
            "brier_skill_vs_matured_label_baseline": aggregate.get("brier_skill_vs_global_matured_label_baseline") if isinstance(aggregate, dict) else None,
            "gdelt_observations": scale.get("observations"),
            "gdelt_scale_gate": scale.get("status"),
            "external_context_completeness": context.get("source_completeness_status") or coverage.get("status"),
            "all_sources_forbid_auto_promotion": all(
                isinstance(item, dict) and item.get("source_promotion_allowed") is False
                for item in gates.values()
            ) if isinstance(gates, dict) and gates else None,
        },
        "input_errors": errors,
        "limitations": [
            "A passing pipeline is not a validated forecasting model.",
            "Source acquisition may pass while provenance, intended-use rights, timestamps, bar convention or quote basis remain unresolved.",
            "Public tick spread is not a measured user-broker fill cost.",
            "External context can pass acquisition while temporal completeness or feature eligibility remains partial.",
            "This report never authorizes live signals or order routing.",
        ],
    })
    return result


def write_report(report: Mapping[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "research_readiness.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    dimensions = [
        ("Pipeline", report["checks"]["pipeline_status"]["status"]),
        ("Data eligibility", report["checks"]["data_status"]["status"]),
        ("Statistical evidence", report["checks"]["statistical_status"]["status"]),
        ("Economic evidence", report["checks"]["economic_status"]["status"]),
        ("Operational readiness", report["checks"]["operations_status"]["status"]),
    ]
    lines = [
        "# Serevex Research Readiness", "",
        f"- Overall gate: **{report['status']}**",
        f"- Promotion status: **{report['promotion_status']}**",
        f"- Decision: **{report['decision_output']}**",
        f"- Live signal enabled: **{report['live_signal_enabled']}**",
        f"- Live order execution enabled: **{report['live_order_execution_enabled']}**",
        f"- Evidence run: {report.get('run_url') or 'not supplied'}", "",
        "| Dimension | Status |", "|---|---|",
    ]
    lines.extend(f"| {name} | {status} |" for name, status in dimensions)
    lines.extend(["", "## Blockers"])
    lines.extend(f"- `{item}`" for item in report["blockers"])
    if not report["blockers"]:
        lines.append("- None recorded.")
    lines.extend(["", "## Observed benchmark scale", "", "| Metric | Observation |", "|---|---:|"])
    lines.extend(
        f"| {name} | {value if value is not None else 'n/a'} |"
        for name, value in report["observed_data_summary"].items()
    )
    lines.extend([
        "", "## Interpretation", "", report["interpretation"], "",
        "This report is diagnostic and does not validate alpha, profitability, licensing or production fitness.", "",
    ])
    (output_dir / "research_readiness.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", required=True, type=Path)
    parser.add_argument("--calibration-contracts", required=True, type=Path)
    parser.add_argument("--run-url", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    report = build_report(args.benchmark_dir, args.calibration_contracts, args.run_url)
    write_report(report, args.output_dir)
    print(json.dumps({
        "status": report["status"],
        "promotion_status": report["promotion_status"],
        "statuses": {key: value["status"] for key, value in report["checks"].items()},
        "decision_output": report["decision_output"],
        "blockers": report["blockers"],
        "report": str(args.output_dir / "research_readiness.json"),
    }, ensure_ascii=False, indent=2))
    # A blocked research gate is an expected result, not a CI execution failure.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
