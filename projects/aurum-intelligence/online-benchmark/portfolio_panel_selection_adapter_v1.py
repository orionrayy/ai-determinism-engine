"""Fail-closed adapter from synchronized portfolio panels to White's Reality Check.

This is inference wiring only. It never selects live trades or enables execution.
A valid panel is necessary but not sufficient: the full registered strategy set,
source/cost/replay evidence, an untouched holdout, and stable hashes/references
must all be supplied again at the inference boundary.
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Mapping, Sequence

from selection_bias_guard_v1 import (
    DECISION_OUTPUT,
    LIVE_ORDER_EXECUTION_ENABLED,
    LIVE_SIGNAL_ENABLED,
    evaluate_white_reality_check,
)

ADAPTER_VERSION = "portfolio-panel-to-selection-guard-v1.0.0"
REQUIRED_GATES = (
    "trial_registry_complete",
    "locked_holdout_unseen",
    "source_data_qualified",
    "execution_costs_verified",
    "portfolio_replay_verified",
)


def _sha256(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{64}", value) is not None


def _valid_evidence_ref(value: object) -> bool:
    if not isinstance(value, str) or not value.strip() or len(value) > 2048:
        return False
    raw = value.strip()
    if raw.startswith("https://"):
        from urllib.parse import urlparse
        parsed = urlparse(raw)
        return bool(parsed.hostname) and parsed.username is None and parsed.password is None
    if raw.startswith(("repo://", "evidence://")):
        scheme, identifier = raw.split("://", 1)
        return scheme in {"repo", "evidence"} and bool(identifier.strip("/")) and not any(c.isspace() for c in identifier)
    return False


def _recompute_panel_sha256(panel: Mapping[str, Any]) -> str:
    manifest = {
        "schema_version": panel.get("schema_version"),
        "expected_interval_seconds": panel.get("expected_interval_seconds"),
        "timestamps": panel.get("timestamps"),
        "session_ids": panel.get("session_ids"),
        "strategy_ledger_sha256": panel.get("strategy_ledger_sha256"),
        "strategy_returns": panel.get("strategy_returns"),
        "benchmark_returns": panel.get("benchmark_returns"),
    }
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _blocked(reason: str, *, panel_sha256: object = None) -> dict[str, Any]:
    return {
        "adapter_version": ADAPTER_VERSION,
        "adapter_status": "BLOCKED",
        "blocker": reason,
        "input_panel_sha256": panel_sha256 if _sha256(panel_sha256) else None,
        "selection_guard": None,
        "decision_output": DECISION_OUTPUT,
        "live_signal_enabled": LIVE_SIGNAL_ENABLED,
        "live_order_execution_enabled": LIVE_ORDER_EXECUTION_ENABLED,
        "limitations": [
            "This adapter does not create portfolio returns or authenticate evidence.",
            "A structurally valid panel is not proof of market edge.",
            "No result from this adapter authorizes signal generation or order execution.",
        ],
    }


def evaluate_synchronized_portfolio_panel(
    panel: Mapping[str, Any],
    *,
    registered_strategy_ids: Sequence[str],
    expected_interval_seconds: int,
    trial_registry_sha256: str,
    source_dataset_sha256: str,
    gate_flags: Mapping[str, bool],
    gate_evidence_refs: Mapping[str, str],
    block_lengths: Sequence[int] = (12, 24, 60),
    bootstrap_replicates: int = 1000,
    seed: int = 20261019,
    alpha: float = 0.05,
    evidence_kind: str = "real_market_research",
) -> dict[str, Any]:
    """Validate the full panel/registry contract and call the existing selection guard.

    registered_strategy_ids must be the full pre-registered family, including
    candidates that performed poorly. Extra or missing IDs block inference.
    source_dataset_sha256 is the frozen market dataset/manifest digest, not the
    panel hash; both are reported and kept separate.
    """
    if not isinstance(panel, Mapping):
        return _blocked("panel_must_be_mapping")
    panel_digest = panel.get("panel_sha256")
    if not _sha256(panel_digest):
        return _blocked("panel_sha256_invalid")
    if panel.get("time_alignment_status") != "PASS":
        return _blocked("panel_time_alignment_not_pass", panel_sha256=panel_digest)
    if panel.get("selection_guard_input_eligible") is not True or panel.get("panel_status") != "EVIDENCE_ELIGIBLE":
        return _blocked("one_or_more_ledger_evidence_gates_blocked", panel_sha256=panel_digest)

    raw_returns = panel.get("strategy_returns")
    if not isinstance(raw_returns, Mapping) or not raw_returns:
        return _blocked("strategy_return_map_missing", panel_sha256=panel_digest)
    raw_names = list(raw_returns.keys())
    if any(not isinstance(name, str) or not name.strip() for name in raw_names):
        return _blocked("strategy_ids_invalid", panel_sha256=panel_digest)
    if len(set(raw_names)) != len(raw_names):
        return _blocked("strategy_ids_duplicate", panel_sha256=panel_digest)

    if isinstance(registered_strategy_ids, (str, bytes)) or not isinstance(registered_strategy_ids, Sequence) or not registered_strategy_ids:
        return _blocked("registered_strategy_ids_must_be_nonempty_sequence", panel_sha256=panel_digest)
    if any(not isinstance(name, str) or not name.strip() for name in registered_strategy_ids):
        return _blocked("registered_strategy_ids_invalid", panel_sha256=panel_digest)
    registered = list(registered_strategy_ids)
    if len(set(registered)) != len(registered):
        return _blocked("registered_strategy_ids_duplicate", panel_sha256=panel_digest)
    if set(registered) != set(raw_names):
        return _blocked("registered_strategy_family_does_not_match_panel", panel_sha256=panel_digest)

    ledger_hashes = panel.get("strategy_ledger_sha256")
    if not isinstance(ledger_hashes, Mapping) or set(ledger_hashes.keys()) != set(raw_names):
        return _blocked("strategy_ledger_hash_family_mismatch", panel_sha256=panel_digest)
    if any(not _sha256(value) for value in ledger_hashes.values()):
        return _blocked("strategy_ledger_sha256_invalid", panel_sha256=panel_digest)

    timestamps = panel.get("timestamps")
    session_ids = panel.get("session_ids")
    benchmark_returns = panel.get("benchmark_returns")
    if not isinstance(timestamps, list) or not isinstance(session_ids, list) or not isinstance(benchmark_returns, list):
        return _blocked("panel_clock_or_benchmark_missing", panel_sha256=panel_digest)
    n = len(timestamps)
    if n < 500 or len(session_ids) != n or len(benchmark_returns) != n:
        return _blocked("panel_clock_or_benchmark_length_invalid", panel_sha256=panel_digest)
    if panel.get("row_count") != n:
        return _blocked("panel_row_count_mismatch", panel_sha256=panel_digest)
    if panel.get("strategy_count") != len(raw_names):
        return _blocked("panel_strategy_count_mismatch", panel_sha256=panel_digest)
    if isinstance(expected_interval_seconds, bool) or not isinstance(expected_interval_seconds, int) or expected_interval_seconds <= 0:
        return _blocked("expected_interval_seconds_must_be_positive_integer", panel_sha256=panel_digest)
    if panel.get("schema_version") != "synchronized-portfolio-return-panel-v1.0.0":
        return _blocked("panel_schema_version_mismatch", panel_sha256=panel_digest)
    if panel.get("expected_interval_seconds") != expected_interval_seconds:
        return _blocked("expected_interval_mismatch", panel_sha256=panel_digest)
    if isinstance(panel.get("row_count"), bool) or not isinstance(panel.get("row_count"), int):
        return _blocked("panel_row_count_type_invalid", panel_sha256=panel_digest)
    if isinstance(panel.get("strategy_count"), bool) or not isinstance(panel.get("strategy_count"), int):
        return _blocked("panel_strategy_count_type_invalid", panel_sha256=panel_digest)
    try:
        recomputed_digest = _recompute_panel_sha256(panel)
    except (TypeError, ValueError, OverflowError):
        return _blocked("panel_manifest_not_canonical_json", panel_sha256=panel_digest)
    if recomputed_digest != panel_digest.lower():
        return _blocked("panel_sha256_mismatch", panel_sha256=panel_digest)

    if not isinstance(gate_flags, Mapping) or set(gate_flags.keys()) != set(REQUIRED_GATES):
        return _blocked("gate_flags_must_cover_exact_required_gates", panel_sha256=panel_digest)
    if any(not isinstance(gate_flags.get(key), bool) for key in REQUIRED_GATES):
        return _blocked("gate_flags_must_be_boolean", panel_sha256=panel_digest)
    if any(gate_flags[key] is not True for key in REQUIRED_GATES):
        return _blocked("one_or_more_inference_gate_flags_false", panel_sha256=panel_digest)
    if not isinstance(gate_evidence_refs, Mapping) or set(gate_evidence_refs.keys()) != set(REQUIRED_GATES):
        return _blocked("gate_evidence_refs_must_cover_exact_required_gates", panel_sha256=panel_digest)
    if not _sha256(trial_registry_sha256):
        return _blocked("trial_registry_sha256_invalid", panel_sha256=panel_digest)
    if not _sha256(source_dataset_sha256):
        return _blocked("source_dataset_sha256_invalid", panel_sha256=panel_digest)
    if trial_registry_sha256.lower() == source_dataset_sha256.lower():
        return _blocked("trial_registry_and_source_dataset_digests_must_be_distinct", panel_sha256=panel_digest)
    if panel_digest.lower() in {trial_registry_sha256.lower(), source_dataset_sha256.lower()}:
        return _blocked("panel_registry_source_digest_collision", panel_sha256=panel_digest)
    if any(not _valid_evidence_ref(gate_evidence_refs[key]) for key in REQUIRED_GATES):
        return _blocked("gate_evidence_reference_invalid", panel_sha256=panel_digest)
    if evidence_kind not in {"synthetic_contract_test", "real_market_research"}:
        return _blocked("evidence_kind_invalid", panel_sha256=panel_digest)

    try:
        report = evaluate_white_reality_check(
            timestamps=timestamps,
            session_ids=session_ids,
            strategy_returns=raw_returns,
            benchmark_returns=benchmark_returns,
            expected_interval_seconds=expected_interval_seconds,
            block_lengths=block_lengths,
            bootstrap_replicates=bootstrap_replicates,
            seed=seed,
            alpha=alpha,
            returns_basis="net_return",
            evidence_kind=evidence_kind,
            trial_registry_complete=gate_flags["trial_registry_complete"],
            locked_holdout_unseen=gate_flags["locked_holdout_unseen"],
            source_data_qualified=gate_flags["source_data_qualified"],
            execution_costs_verified=gate_flags["execution_costs_verified"],
            portfolio_replay_verified=gate_flags["portfolio_replay_verified"],
            trial_registry_sha256=trial_registry_sha256,
            dataset_sha256=source_dataset_sha256,
            gate_evidence_refs=gate_evidence_refs,
        )
    except (ValueError, TypeError, OverflowError):
        return _blocked("selection_guard_rejected_input", panel_sha256=panel_digest)
    return {
        "adapter_version": ADAPTER_VERSION,
        "adapter_status": "PASS",
        "input_panel_sha256": panel_digest.lower(),
        "input_strategy_ledger_sha256": {key: ledger_hashes[key].lower() for key in sorted(ledger_hashes)},
        "registered_strategy_ids_sha256_ordered": list(registered),
        "registered_strategy_count": len(registered),
        "source_dataset_sha256": source_dataset_sha256.lower(),
        "selection_guard": report,
        "decision_output": DECISION_OUTPUT,
        "live_signal_enabled": LIVE_SIGNAL_ENABLED,
        "live_order_execution_enabled": LIVE_ORDER_EXECUTION_ENABLED,
        "interpretation": "Adapter and statistical evaluation completed; any PASS is only a prerequisite for independent replication, not trading permission.",
        "limitations": [
            "The full registry is compared by exact strategy IDs; completeness still depends on the independently reviewed registry evidence.",
            "Dataset and ledger hashes provide lineage links but do not authenticate upstream provenance or licensing.",
            "A selection-adjusted statistical result does not establish net executable profitability or deployment readiness.",
            "No result from this adapter authorizes signal generation or order execution.",
        ],
    }
