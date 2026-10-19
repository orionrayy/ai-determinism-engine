"""Serevex delayed-feedback, regime-aware fixed-share expert mixer.

Research only. Consumes pre-generated causal probabilities; does not fit models,
authorize trades, or establish predictive alpha. Updates use only matured labels.
Standard library only.
"""
from __future__ import annotations

import hashlib
import heapq
import json
import math
from typing import Any, Mapping, Sequence

PROTOCOL_VERSION = "serevex-delayed-fixed-share-expert-v1"
BASELINE_ID = "__baseline__"
AUTHORITY = {
    "research_only": True,
    "market_promotion_allowed": False,
    "model_validated": False,
    "deployment_calibrated": False,
    "live_signal_enabled": False,
    "live_order_execution_enabled": False,
    "decision_output": "NO_TRADE",
}


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _probability(value: object, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field}_must_be_finite_probability")
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field}_must_be_finite_probability") from exc
    if not math.isfinite(out) or not 0.0 <= out <= 1.0:
        raise ValueError(f"{field}_must_be_finite_probability")
    return out


def _sha(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _normalize_records(
    records: Sequence[Mapping[str, Any]],
    *,
    registered_experts: Sequence[str],
    registered_regimes: Sequence[str],
    as_of_index: int,
) -> tuple[list[dict[str, Any]], int]:
    if not registered_experts or any(not isinstance(x, str) or not x.strip() or x == BASELINE_ID for x in registered_experts):
        raise ValueError("registered_expert_id_invalid_or_reserved")
    if len(set(registered_experts)) != len(registered_experts):
        raise ValueError("registered_experts_must_be_unique")
    if not registered_regimes or any(not isinstance(x, str) or not x.strip() for x in registered_regimes):
        raise ValueError("registered_regime_id_invalid")
    if len(set(registered_regimes)) != len(registered_regimes):
        raise ValueError("registered_regimes_must_be_unique")
    experts, regimes = set(registered_experts), set(registered_regimes)
    expected = {
        "decision_id", "decision_index", "label_available_index", "label",
        "label_mature", "regime_id", "regime_available_index", "segment_id",
        "baseline_probability", "expert_probabilities",
    }
    seen_ids: set[str] = set()
    prior_index = -1
    out: list[dict[str, Any]] = []
    ignored_after_cutoff = 0
    for pos, source in enumerate(records):
        if not isinstance(source, Mapping):
            raise ValueError(f"record_must_be_mapping_at_{pos}")
        missing = sorted(expected - set(source))
        if missing:
            raise ValueError(f"missing_fields_at_{pos}:{','.join(missing)}")
        row = dict(source)
        did = row["decision_id"]
        if not isinstance(did, str) or not did.strip() or did in seen_ids:
            raise ValueError(f"missing_or_duplicate_decision_id_at_{pos}")
        seen_ids.add(did)
        di, li, ri = row["decision_index"], row["label_available_index"], row["regime_available_index"]
        if not _is_int(di) or di < 0 or di <= prior_index:
            raise ValueError(f"decision_indices_must_be_strictly_increasing_at_{pos}")
        if not _is_int(li) or li <= di:
            raise ValueError(f"label_must_mature_after_decision_at_{pos}")
        if not _is_int(ri) or not 0 <= ri <= di:
            raise ValueError(f"regime_not_available_as_of_decision_at_{pos}")
        if not isinstance(row["regime_id"], str) or row["regime_id"] not in regimes:
            raise ValueError(f"unregistered_regime_at_{pos}")
        segment = row["segment_id"]
        if not isinstance(segment, str) or not segment.strip():
            raise ValueError(f"segment_id_invalid_at_{pos}")
        if not isinstance(row["label_mature"], bool):
            raise ValueError(f"label_mature_must_be_boolean_at_{pos}")
        label = row["label"]
        if row["label_mature"] and (isinstance(label, bool) or label not in (0, 1)):
            raise ValueError(f"mature_label_must_be_binary_at_{pos}")
        if label is not None and (isinstance(label, bool) or label not in (0, 1)):
            raise ValueError(f"label_must_be_binary_or_none_at_{pos}")
        pbase = _probability(row["baseline_probability"], "baseline_probability")
        ps = row["expert_probabilities"]
        if not isinstance(ps, Mapping) or set(ps) != experts:
            raise ValueError(f"expert_probability_keys_must_match_registry_at_{pos}")
        normalized_ps = {expert: _probability(ps[expert], f"expert_probability:{expert}") for expert in registered_experts}
        row["baseline_probability"] = pbase
        row["expert_probabilities"] = normalized_ps
        row["probability_map"] = {**normalized_ps, BASELINE_ID: pbase}
        row["_input_position"] = pos
        if di >= as_of_index:
            ignored_after_cutoff += 1
        else:
            out.append(row)
        prior_index = di
    if not out:
        raise ValueError("no_decisions_before_as_of_index")
    return out, ignored_after_cutoff


def _uniform_weights(option_ids: Sequence[str]) -> dict[str, float]:
    value = 1.0 / len(option_ids)
    return {key: value for key in option_ids}


def _update_weights(
    weights: Mapping[str, float],
    rows: Sequence[Mapping[str, Any]],
    *,
    option_ids: Sequence[str],
    eta: float,
    fixed_share_rate: float,
) -> dict[str, float]:
    if not rows:
        return dict(weights)
    mean_deltas: dict[str, float] = {}
    for option in option_ids:
        losses = []
        for row in rows:
            y = int(row["label"])
            p = float(row["probability_map"][option])
            pb = float(row["baseline_probability"])
            loss = (p - y) ** 2
            base_loss = (pb - y) ** 2
            # Baseline-relative loss is in [-1, 1] for Brier loss.
            losses.append(loss - base_loss)
        mean_deltas[option] = sum(losses) / len(losses)

    # Log-domain update avoids overflow/underflow for long streams.
    logw = {key: math.log(max(float(weights[key]), 1e-300)) - eta * mean_deltas[key] for key in option_ids}
    peak = max(logw.values())
    scaled = {key: math.exp(max(-700.0, logw[key] - peak)) for key in option_ids}
    total = sum(scaled.values())
    posterior = {key: value / total for key, value in scaled.items()}
    if fixed_share_rate:
        n = len(option_ids)
        posterior = {
            key: (1.0 - fixed_share_rate) * posterior[key] + fixed_share_rate / n
            for key in option_ids
        }
    # Normalize once more to absorb floating-point drift.
    total = sum(posterior.values())
    return {key: posterior[key] / total for key in option_ids}


def _metric_pack(probabilities: Sequence[float], labels: Sequence[int]) -> dict[str, Any]:
    if len(probabilities) != len(labels):
        raise ValueError("metric_lengths_must_match")
    n = len(labels)
    if not n:
        return {"n": 0, "brier": None, "log_loss": None, "positive_rate": None}
    eps = 1e-15
    brier = sum((float(p) - int(y)) ** 2 for p, y in zip(probabilities, labels)) / n
    ll = 0.0
    for p, y in zip(probabilities, labels):
        p = min(1 - eps, max(eps, float(p)))
        ll += -(int(y) * math.log(p) + (1 - int(y)) * math.log(1 - p))
    return {
        "n": n,
        "brier": brier,
        "log_loss": ll / n,
        "positive_rate": sum(labels) / n,
    }


def _best_switching_loss(
    rows: Sequence[Mapping[str, Any]],
    *,
    option_ids: Sequence[str],
    max_switches: int,
) -> float | None:
    """Hindsight-only DP comparator; never used to update or select deployed weights."""
    if not rows:
        return None
    losses = [
        [(float(row["probability_map"][option]) - int(row["label"])) ** 2 for option in option_ids]
        for row in rows
    ]
    k_count = len(option_ids)
    # dp[s][k]: minimum loss ending on k with at most s switches.
    dp = [[losses[0][k] for k in range(k_count)] for _ in range(max_switches + 1)]
    for t in range(1, len(losses)):
        nxt: list[list[float]] = [[0.0] * k_count for _ in range(max_switches + 1)]
        for switches in range(max_switches + 1):
            for k in range(k_count):
                stay = dp[switches][k]
                if switches == 0:
                    best = stay
                else:
                    change = min(dp[switches - 1][j] for j in range(k_count) if j != k) if k_count > 1 else math.inf
                    best = min(stay, change)
                nxt[switches][k] = best + losses[t][k]
        dp = nxt
    return min(dp[max_switches])


def _validate_evidence_manifest(
    manifest: Mapping[str, Any] | None,
    registered_experts: Sequence[str],
) -> dict[str, Any]:
    """Validate manifest shape only; hashes do not independently prove source authenticity."""
    if not isinstance(manifest, Mapping):
        return {"status": "MISSING_MANIFEST_NO_MARKET_CLAIM", "manifest_sha256": None}
    required = {
        "target_id", "target_definition_sha256", "dataset_sha256",
        "source_lineage_sha256", "forecast_ledger_sha256", "model_artifact_sha256_by_expert",
    }
    missing = sorted(required - set(manifest))
    if missing:
        return {"status": "INCOMPLETE_MANIFEST_NO_MARKET_CLAIM", "missing_fields": missing, "manifest_sha256": _sha(dict(manifest))}
    def is_digest(value: object) -> bool:
        return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value.lower())
    bad_hashes = [key for key in ("target_definition_sha256", "dataset_sha256", "source_lineage_sha256", "forecast_ledger_sha256") if not is_digest(manifest.get(key))]
    model_hashes = manifest.get("model_artifact_sha256_by_expert")
    if not isinstance(manifest.get("target_id"), str) or not str(manifest.get("target_id")).strip():
        missing.append("target_id_nonempty_string")
    if not isinstance(model_hashes, Mapping) or set(model_hashes) != set(registered_experts):
        missing.append("model_artifact_sha256_by_expert_must_match_registry")
    elif any(not is_digest(model_hashes.get(key)) for key in registered_experts):
        bad_hashes.append("model_artifact_sha256_by_expert")
    if missing or bad_hashes:
        return {
            "status": "INVALID_MANIFEST_SHAPE_NO_MARKET_CLAIM",
            "missing_or_invalid_fields": missing,
            "invalid_hash_fields": bad_hashes,
            "manifest_sha256": _sha(dict(manifest)),
        }
    return {
        "status": "MANIFEST_SHAPE_VALID_SOURCE_NOT_AUTHENTICATED",
        "target_id": manifest["target_id"],
        "manifest_sha256": _sha(dict(manifest)),
        "source_authenticity_independently_verified": False,
    }


def run_delayed_fixed_share_expert_mixer(
    records: Sequence[Mapping[str, Any]],
    *,
    as_of_index: int,
    registered_experts: Sequence[str],
    registered_regimes: Sequence[str],
    eta: float = 1.0,
    fixed_share_rate: float = 0.02,
    min_global_matured: int = 30,
    min_regime_matured: int = 40,
    max_hindsight_switches: int = 3,
    evidence_kind: str = "real_market_research",
    evidence_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Produce causal mixtures with labels released only after their maturity index.

    At each decision index t, first process only labels with label_available_index < t.
    Weights update using Brier-loss difference vs the shared causal baseline, then
    Fixed-Share redistributes a small amount of mass to all experts to retain the
    ability to recover after a regime change. Per-regime experts are used only after
    that regime has enough matured history; otherwise global weights or the baseline
    are used. This is an experiment policy, not a claimed regret guarantee.
    """
    if not _is_int(as_of_index) or as_of_index < 0:
        raise ValueError("as_of_index_must_be_nonnegative_integer")
    if isinstance(eta, bool) or not isinstance(eta, (int, float)) or not math.isfinite(float(eta)) or not 0 < eta <= 20:
        raise ValueError("eta_must_be_finite_and_in_(0,20]")
    if isinstance(fixed_share_rate, bool) or not isinstance(fixed_share_rate, (int, float)) or not math.isfinite(float(fixed_share_rate)) or not 0 <= fixed_share_rate < 0.5:
        raise ValueError("fixed_share_rate_must_be_in_[0,0.5)")
    if not _is_int(min_global_matured) or min_global_matured < 0:
        raise ValueError("min_global_matured_must_be_nonnegative_integer")
    if not _is_int(min_regime_matured) or min_regime_matured < 0:
        raise ValueError("min_regime_matured_must_be_nonnegative_integer")
    if not _is_int(max_hindsight_switches) or max_hindsight_switches < 0 or max_hindsight_switches > 50:
        raise ValueError("max_hindsight_switches_must_be_integer_0_to_50")
    if evidence_kind not in {"real_market_research", "synthetic_contract_test", "replay_fixture"}:
        raise ValueError("evidence_kind_unrecognized")
    rows, ignored_after_cutoff = _normalize_records(
        records,
        registered_experts=registered_experts,
        registered_regimes=registered_regimes,
        as_of_index=as_of_index,
    )
    manifest_validation = _validate_evidence_manifest(evidence_manifest, registered_experts)
    options = [*registered_experts, BASELINE_ID]
    global_weights = _uniform_weights(options)
    regime_weights = {regime: _uniform_weights(options) for regime in registered_regimes}
    global_matured_count = 0
    regime_matured_counts = {regime: 0 for regime in registered_regimes}
    pending: list[tuple[int, int, int, dict[str, Any]]] = []
    pending_sequence = 0
    predictions: list[dict[str, Any]] = []
    update_events: list[dict[str, Any]] = []

    for row in rows:
        current_index = int(row["decision_index"])
        # Release feedback strictly before the current decision index.
        ready: list[tuple[int, int, int, dict[str, Any]]] = []
        while pending and pending[0][0] < current_index:
            ready.append(heapq.heappop(pending))
        # Preserve maturity-time order and update ties as one batch, avoiding arbitrary
        # outcome ordering when labels share the same availability index.
        grouped: dict[int, list[dict[str, Any]]] = {}
        for maturity_index, _decision_index, _seq, pending_row in ready:
            grouped.setdefault(maturity_index, []).append(pending_row)
        for maturity_index in sorted(grouped):
            batch = sorted(grouped[maturity_index], key=lambda x: x["decision_index"])
            global_weights = _update_weights(
                global_weights, batch, option_ids=options,
                eta=float(eta), fixed_share_rate=float(fixed_share_rate),
            )
            regime_batch: dict[str, list[dict[str, Any]]] = {}
            for old in batch:
                regime_batch.setdefault(old["regime_id"], []).append(old)
            for regime, regime_rows in regime_batch.items():
                regime_weights[regime] = _update_weights(
                    regime_weights[regime], regime_rows, option_ids=options,
                    eta=float(eta), fixed_share_rate=float(fixed_share_rate),
                )
                regime_matured_counts[regime] += len(regime_rows)
            global_matured_count += len(batch)
            update_events.append({
                "label_available_index": maturity_index,
                "applied_before_decision_index": current_index,
                "batch_count": len(batch),
                "global_matured_count_after": global_matured_count,
                "regime_matured_counts_after": dict(regime_matured_counts),
                "global_weights_sha256_after": _sha(global_weights),
                "regime_weights_sha256_after": {key: _sha(value) for key, value in regime_weights.items()},
            })

        regime = row["regime_id"]
        if regime_matured_counts[regime] >= min_regime_matured:
            weights_used = dict(regime_weights[regime])
            source = "REGIME_FIXED_SHARE"
        elif global_matured_count >= min_global_matured:
            weights_used = dict(global_weights)
            source = "GLOBAL_FIXED_SHARE"
        else:
            weights_used = None
            source = "BASELINE_FALLBACK_INSUFFICIENT_MATURED_HISTORY"
        if weights_used is None:
            p_mix = float(row["baseline_probability"])
            weights_for_ledger: dict[str, float] = {}
        else:
            p_mix = sum(weights_used[key] * float(row["probability_map"][key]) for key in options)
            p_mix = min(1.0, max(0.0, p_mix))
            weights_for_ledger = weights_used
        pred = {
            "decision_id": row["decision_id"],
            "decision_index": current_index,
            "label_available_index": row["label_available_index"],
            "regime_id": regime,
            "segment_id": row["segment_id"],
            "forecast_source": source,
            "forecast_frozen_before_outcome": True,
            "weight_updates_use_only_previously_matured_labels": True,
            "probability": p_mix,
            "baseline_probability": float(row["baseline_probability"]),
            "expert_probabilities": dict(row["expert_probabilities"]),
            "weights_used": weights_for_ledger,
            "weights_sha256": _sha(weights_for_ledger),
            "matured_labels_seen_globally_before_decision": global_matured_count,
            "matured_labels_seen_in_regime_before_decision": regime_matured_counts[regime],
            "label": row["label"],
            "label_mature": row["label_mature"],
            "_row": row,
        }
        predictions.append(pred)

        # The outcome is enqueued only after the prediction has been frozen.
        # Mature feedback cannot affect the current row's probability.
        if row["label_mature"]:
            heapq.heappush(
                pending,
                (int(row["label_available_index"]), current_index, pending_sequence, {**row, "probability_map": dict(row["probability_map"])}),
            )
            pending_sequence += 1

    # Bring the state forward to the declared as-of boundary even if there is no
    # prediction row exactly at as_of_index. Labels with availability < as_of are
    # usable for the next forecast; this final flush cannot alter already-issued rows.
    ready_final: list[tuple[int, int, int, dict[str, Any]]] = []
    while pending and pending[0][0] < as_of_index:
        ready_final.append(heapq.heappop(pending))
    grouped_final: dict[int, list[dict[str, Any]]] = {}
    for maturity_index, _decision_index, _seq, pending_row in ready_final:
        grouped_final.setdefault(maturity_index, []).append(pending_row)
    for maturity_index in sorted(grouped_final):
        batch = sorted(grouped_final[maturity_index], key=lambda x: x["decision_index"])
        global_weights = _update_weights(
            global_weights, batch, option_ids=options,
            eta=float(eta), fixed_share_rate=float(fixed_share_rate),
        )
        regime_batch: dict[str, list[dict[str, Any]]] = {}
        for old in batch:
            regime_batch.setdefault(old["regime_id"], []).append(old)
        for regime, regime_rows in regime_batch.items():
            regime_weights[regime] = _update_weights(
                regime_weights[regime], regime_rows, option_ids=options,
                eta=float(eta), fixed_share_rate=float(fixed_share_rate),
            )
            regime_matured_counts[regime] += len(regime_rows)
        global_matured_count += len(batch)
        update_events.append({
            "label_available_index": maturity_index,
            "applied_before_decision_index": as_of_index,
            "batch_count": len(batch),
            "global_matured_count_after": global_matured_count,
            "regime_matured_counts_after": dict(regime_matured_counts),
            "global_weights_sha256_after": _sha(global_weights),
            "regime_weights_sha256_after": {key: _sha(value) for key, value in regime_weights.items()},
            "final_as_of_state_flush": True,
        })

    scored = [
        pred for pred in predictions
        if pred["label_mature"] is True
        and pred["label"] in (0, 1)
        and int(pred["label_available_index"]) < as_of_index
    ]
    labels = [int(x["label"]) for x in scored]
    mix_metrics = _metric_pack([x["probability"] for x in scored], labels)
    baseline_metrics = _metric_pack([x["baseline_probability"] for x in scored], labels)
    mix_brier = mix_metrics["brier"]
    base_brier = baseline_metrics["brier"]
    expert_metrics: dict[str, Any] = {}
    for expert in registered_experts:
        m = _metric_pack([x["expert_probabilities"][expert] for x in scored], labels)
        m["brier_skill_vs_baseline"] = (
            float(base_brier - m["brier"]) if base_brier is not None and m["brier"] is not None else None
        )
        expert_metrics[expert] = m
    mix_metrics["brier_skill_vs_baseline"] = (
        float(base_brier - mix_brier) if base_brier is not None and mix_brier is not None else None
    )
    per_regime: dict[str, Any] = {}
    for regime in registered_regimes:
        cohort = [x for x in scored if x["regime_id"] == regime]
        ys = [int(x["label"]) for x in cohort]
        mm = _metric_pack([x["probability"] for x in cohort], ys)
        bm = _metric_pack([x["baseline_probability"] for x in cohort], ys)
        mm["brier_skill_vs_baseline"] = (
            float(bm["brier"] - mm["brier"]) if bm["brier"] is not None and mm["brier"] is not None else None
        )
        per_regime[regime] = {"mixture": mm, "baseline": bm, "forecast_source_counts": {
            source: sum(1 for x in predictions if x["regime_id"] == regime and x["forecast_source"] == source)
            for source in sorted({x["forecast_source"] for x in predictions if x["regime_id"] == regime})
        }}
    scored_rows = [x["_row"] for x in scored]
    comparator_losses = {
        opt: sum((float(row["probability_map"][opt]) - int(row["label"])) ** 2 for row in scored_rows)
        for opt in options
    } if scored_rows else {}
    best_fixed_option = min(comparator_losses, key=comparator_losses.get) if comparator_losses else None
    mix_total_loss = sum((float(x["probability"]) - int(x["label"])) ** 2 for x in scored)
    best_switch_loss = _best_switching_loss(scored_rows, option_ids=options, max_switches=max_hindsight_switches)
    return {
        "protocol_version": PROTOCOL_VERSION,
        "status": "DIAGNOSTIC_ONLY",
        "as_of_index": as_of_index,
        "evidence_kind": evidence_kind,
        "evidence_manifest_validation": manifest_validation,
        "market_evidence_eligible": False,
        "input_decision_count_before_cutoff": len(rows),
        "ignored_decisions_at_or_after_cutoff": ignored_after_cutoff,
        "registered_experts": list(registered_experts),
        "registered_regimes": list(registered_regimes),
        "baseline_expert_id": BASELINE_ID,
        "update_rule": "baseline-relative Brier exponential weights + fixed-share; update only on labels with availability index strictly earlier than current decision",
        "eta": float(eta),
        "fixed_share_rate": float(fixed_share_rate),
        "min_global_matured": min_global_matured,
        "min_regime_matured": min_regime_matured,
        "scored_matured_rows": len(scored),
        "unmatured_or_unavailable_outcomes_excluded": len(predictions) - len(scored),
        "online_mixture": mix_metrics,
        "causal_baseline": baseline_metrics,
        "experts": expert_metrics,
        "by_regime": per_regime,
        "hindsight_comparators": {
            "status": "DIAGNOSTIC_HINDSIGHT_ONLY_NOT_USABLE_FOR_DECISIONS",
            "best_fixed_expert_or_baseline_id": best_fixed_option,
            "best_fixed_total_brier_loss": comparator_losses.get(best_fixed_option) if best_fixed_option else None,
            "mixture_total_brier_loss": mix_total_loss,
            "regret_vs_best_fixed_hindsight": mix_total_loss - comparator_losses[best_fixed_option] if best_fixed_option else None,
            "max_switches": max_hindsight_switches,
            "best_switching_total_brier_loss": best_switch_loss,
            "regret_vs_best_switching_hindsight": mix_total_loss - best_switch_loss if best_switch_loss is not None else None,
            "caveat": "These comparators use outcomes from the entire scored cohort and are diagnostics only. They are not online predictions, training targets, or deployment decisions.",
        },
        "forecast_ledger": [
            {k: v for k, v in row.items() if k != "_row"}
            for row in predictions
        ],
        "update_events": update_events,
        "pending_label_count_after_as_of": len(pending),
        "final_global_weights": dict(global_weights),
        "final_regime_weights": {key: dict(value) for key, value in regime_weights.items()},
        **AUTHORITY,
    }