"""Selection-bias guard for synchronized strategy-return panels.

Implements a non-studentized White Reality Check max-mean statistic with a
segmented moving-block bootstrap. It is inferential-only: it does not select a winner,
authorize a trade, or validate a live strategy.

A valid production input must be a portfolio-level, time-aligned, net-of-cost return
panel with a complete trial registry, qualified data, and an untouched lockbox.
Per-signal overlapping event rows and evenly-spaced sampled anchors are not substitutes.
"""
from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse

import numpy as np

PROTOCOL_VERSION = "white-reality-check-segmented-block-bootstrap-v1"
LIVE_SIGNAL_ENABLED = False
LIVE_ORDER_EXECUTION_ENABLED = False
DECISION_OUTPUT = "NO_TRADE"


def _timestamp(value: object) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        raw = value.strip()
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        parsed = datetime.fromisoformat(raw)
    else:
        raise ValueError("timestamp_must_be_aware_datetime_or_iso8601")
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("timestamp_must_be_timezone_aware")
    return parsed.astimezone(timezone.utc)


def _finite_vector(values: Any, name: str, n_expected: int | None = None) -> np.ndarray:
    if isinstance(values, (str, bytes)) or not hasattr(values, "__iter__"):
        raise ValueError(f"{name}_must_be_sequence")
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name}_must_be_numeric") from exc
    if array.ndim != 1 or (n_expected is not None and len(array) != n_expected):
        raise ValueError(f"{name}_length_or_shape_invalid")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name}_contains_non_finite_values")
    return array


def _valid_sha256(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{64}", value) is not None


def _valid_evidence_ref(value: object) -> bool:
    if not isinstance(value, str) or not value.strip() or len(value) > 2048:
        return False
    raw = value.strip()
    if raw.startswith("https://"):
        parsed = urlparse(raw)
        return bool(parsed.hostname) and parsed.username is None and parsed.password is None
    if raw.startswith(("repo://", "evidence://")):
        scheme, identifier = raw.split("://", 1)
        return scheme in {"repo", "evidence"} and bool(identifier.strip("/")) and not any(c.isspace() for c in identifier)
    return False


def _session_blocks(session_ids: Sequence[str], block_length: int) -> tuple[np.ndarray, ...]:
    if len(session_ids) == 0:
        raise ValueError("session_ids_empty")
    if any(not isinstance(item, str) or not item.strip() for item in session_ids):
        raise ValueError("session_ids_must_be_nonempty_strings")
    spans: list[tuple[int, int]] = []
    seen: set[str] = set()
    start = 0
    current = session_ids[0]
    for index, item in enumerate(session_ids[1:], start=1):
        if item != current:
            seen.add(current)
            if item in seen:
                raise ValueError("session_ids_must_be_contiguous")
            spans.append((start, index))
            start, current = index, item
    spans.append((start, len(session_ids)))
    blocks: list[np.ndarray] = []
    for start, end in spans:
        if end - start >= block_length:
            blocks.extend(np.arange(i, i + block_length, dtype=np.int64)
                          for i in range(start, end - block_length + 1))
    if not blocks:
        raise ValueError("no_session_long_enough_for_block_length")
    return tuple(blocks)


def evaluate_white_reality_check(
    *,
    timestamps: Sequence[datetime | str],
    session_ids: Sequence[str],
    strategy_returns: Mapping[str, Any],
    benchmark_returns: Any,
    expected_interval_seconds: int,
    block_lengths: Sequence[int] = (12, 24, 60),
    bootstrap_replicates: int = 1000,
    seed: int = 20261019,
    alpha: float = 0.05,
    returns_basis: str = "net_r",
    evidence_kind: str = "synthetic_contract_test",
    trial_registry_complete: bool = False,
    locked_holdout_unseen: bool = False,
    source_data_qualified: bool = False,
    execution_costs_verified: bool = False,
    portfolio_replay_verified: bool = False,
    trial_registry_sha256: str | None = None,
    dataset_sha256: str | None = None,
    gate_evidence_refs: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Evaluate a preregistered candidate family against a synchronized benchmark.

    The null is that no candidate has positive expected differential vs benchmark.
    Shared time blocks preserve cross-strategy dependence and approximate temporal
    dependence. This is not a complete strategy-selection or profitability certificate.
    """
    if not isinstance(strategy_returns, Mapping) or not strategy_returns:
        raise ValueError("strategy_returns_mapping_required")
    if len(strategy_returns) > 1000:
        raise ValueError("strategy_family_exceeds_1000_candidates")
    if any(not isinstance(key, str) or not key.strip() for key in strategy_returns):
        raise ValueError("strategy_ids_must_be_nonempty_strings")
    if isinstance(expected_interval_seconds, bool) or not isinstance(expected_interval_seconds, int) or expected_interval_seconds <= 0:
        raise ValueError("expected_interval_seconds_must_be_positive_integer")
    if isinstance(bootstrap_replicates, bool) or not isinstance(bootstrap_replicates, int) or not 200 <= bootstrap_replicates <= 10000:
        raise ValueError("bootstrap_replicates_must_be_between_200_and_10000")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed_must_be_integer")
    if not isinstance(alpha, (int, float)) or isinstance(alpha, bool) or not math.isfinite(float(alpha)) or not 0 < float(alpha) < 0.2:
        raise ValueError("alpha_must_be_between_0_and_0_2")
    if returns_basis not in {"net_r", "net_return", "net_account_currency_pnl"}:
        raise ValueError("returns_basis_must_be_net_of_costs")
    if evidence_kind not in {"synthetic_contract_test", "real_market_research"}:
        raise ValueError("evidence_kind_invalid")
    if not isinstance(block_lengths, Sequence) or isinstance(block_lengths, (str, bytes)) or not block_lengths:
        raise ValueError("block_lengths_required")
    normalized_blocks: list[int] = []
    for value in block_lengths:
        if isinstance(value, bool) or not isinstance(value, int) or value < 2:
            raise ValueError("block_lengths_must_be_integers_at_least_two")
        if value not in normalized_blocks:
            normalized_blocks.append(value)
    if len(normalized_blocks) > 8:
        raise ValueError("too_many_block_length_sensitivity_values")

    n = len(timestamps)
    if n < 500:
        raise ValueError("at_least_500_aligned_return_observations_required")
    parsed_times = [_timestamp(item) for item in timestamps]
    if any(parsed_times[i] <= parsed_times[i - 1] for i in range(1, n)):
        raise ValueError("timestamps_must_be_strictly_increasing")
    if len(session_ids) != n:
        raise ValueError("session_ids_length_mismatch")
    for index in range(1, n):
        if session_ids[index] == session_ids[index - 1]:
            delta = int((parsed_times[index] - parsed_times[index - 1]).total_seconds())
            if delta != expected_interval_seconds:
                raise ValueError("within_session_timestamps_must_match_expected_cadence")

    benchmark = _finite_vector(benchmark_returns, "benchmark_returns", n)
    strategy_names = sorted(strategy_returns)
    columns = [_finite_vector(strategy_returns[name], f"strategy_{name}", n) for name in strategy_names]
    panel = np.column_stack(columns)
    differential = panel - benchmark[:, None]
    means = np.mean(differential, axis=0)
    observed_best_mean = float(np.max(means))
    observed_stat = float(math.sqrt(n) * max(0.0, observed_best_mean))
    family_blocks = {block_length: _session_blocks(session_ids, block_length) for block_length in normalized_blocks}

    required_gates = {
        "trial_registry_complete": trial_registry_complete,
        "locked_holdout_unseen": locked_holdout_unseen,
        "source_data_qualified": source_data_qualified,
        "execution_costs_verified": execution_costs_verified,
        "portfolio_replay_verified": portfolio_replay_verified,
    }
    refs = gate_evidence_refs if isinstance(gate_evidence_refs, Mapping) else {}
    valid_refs = {key: refs[key].strip() for key in required_gates if key in refs and _valid_evidence_ref(refs[key])}
    hash_gates_pass = _valid_sha256(trial_registry_sha256) and _valid_sha256(dataset_sha256)
    gates_pass = all(value is True for value in required_gates.values()) and set(required_gates).issubset(valid_refs) and hash_gates_pass

    results = []
    rng = np.random.default_rng(seed)
    centered = differential - means[None, :]
    for block_length in normalized_blocks:
        starts = family_blocks[block_length]
        stats = np.empty(bootstrap_replicates, dtype=np.float64)
        for replicate in range(bootstrap_replicates):
            index_parts = []
            sampled_n = 0
            while sampled_n < n:
                block = starts[int(rng.integers(0, len(starts)))]
                take = min(block_length, n - sampled_n)
                index_parts.append(block[:take])
                sampled_n += take
            sampled_indices = np.concatenate(index_parts)
            sampled_means = np.mean(centered[sampled_indices], axis=0)
            stats[replicate] = math.sqrt(n) * max(0.0, float(np.max(sampled_means)))
        p_value = float((1 + np.count_nonzero(stats >= observed_stat)) / (bootstrap_replicates + 1))
        results.append({
            "block_length_observations": block_length,
            "bootstrap_replicates": bootstrap_replicates,
            "seed": seed,
            "observed_max_mean_excess": observed_best_mean,
            "observed_max_statistic": observed_stat,
            "familywise_p_value": p_value,
            "reject_null_at_alpha": bool(p_value <= float(alpha) and observed_best_mean > 0),
            "bootstrap_statistic_q95": float(np.quantile(stats, 0.95)),
        })

    all_adjusted_tests_pass = all(item["reject_null_at_alpha"] for item in results)
    base = {
        "protocol_version": PROTOCOL_VERSION,
        "method": "White Reality Check; non-studentized max-mean statistic; segmented moving-block bootstrap",
        "n_aligned_observations": n,
        "strategy_count_in_return_panel": len(strategy_names),
        "strategy_ids": strategy_names,
        "registered_trial_registry_sha256": trial_registry_sha256,
        "dataset_sha256": dataset_sha256,
        "returns_basis": returns_basis,
        "benchmark": "provided synchronized benchmark returns; commonly no-trade zeros, but caller must declare actual benchmark",
        "mean_excess_by_strategy": {name: float(means[i]) for i, name in enumerate(strategy_names)},
        "observed_best_mean_excess_statistical_only": observed_best_mean,
        "bootstrap_sensitivity": results,
        "all_block_lengths_reject_null": all_adjusted_tests_pass,
        "alpha": float(alpha),
        "evidence_kind": evidence_kind,
        "gate_status": {key: ("PASS" if value is True and key in valid_refs else "BLOCKED") for key, value in required_gates.items()},
        "gate_evidence_refs": valid_refs,
        "candidate_selection_status": "PASS_FOR_INDEPENDENT_REPLICATION" if all_adjusted_tests_pass else "NO_SELECTION_ADJUSTED_EDGE_EVIDENCE",
        "promotion_status": "DIAGNOSTIC_ONLY",
        "decision_output": DECISION_OUTPUT,
        "live_signal_enabled": LIVE_SIGNAL_ENABLED,
        "live_order_execution_enabled": LIVE_ORDER_EXECUTION_ENABLED,
        "limitations": [
            "Input must be an aligned, regular-cadence portfolio-level return series; event-level overlapping trade rows are invalid.",
            "No-trade periods must be present as zero returns; missing timestamps must not be silently dropped.",
            "The trial registry must include every materially tested strategy/configuration, including failures and superseded variants.",
            "A passed selection-adjusted test is only a prerequisite for independent replication, never proof of deployable profitability.",
            "Cost, source, lockbox and portfolio gates require independent evidence review; evidence references are not self-authenticating.",
            "Block-length sensitivity values must be registered before holdout inspection.",
        ],
    }
    if evidence_kind != "real_market_research":
        base["promotion_status"] = "SYNTHETIC_ONLY_NO_MARKET_PROMOTION"
    elif not gates_pass:
        base["promotion_status"] = "BLOCKED_MISSING_OR_UNVERIFIED_EVIDENCE_GATES"
    elif not all_adjusted_tests_pass:
        base["promotion_status"] = "BLOCKED_SELECTION_ADJUSTED_SIGNIFICANCE"
    elif observed_best_mean <= 0:
        base["promotion_status"] = "BLOCKED_NO_POSITIVE_MEAN_EXCESS"
    else:
        base["promotion_status"] = "ELIGIBLE_FOR_INDEPENDENT_REPLICATION_ONLY"
    return base
