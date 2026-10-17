"""Evidence-gated research readiness; this module never authorizes trading.

Pipeline execution, data admissibility, statistical evidence, economic evidence, and
operational readiness are separate dimensions. A passing test suite is not market
validation. Production still requires a separate, explicit human release process.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from urllib.parse import urlparse

SCHEMA_VERSION = "1.0.0"

_ALLOWED = {
    "pipeline_status": {"NOT_RUN", "PASS", "FAIL"},
    "data_status": {"NOT_ASSESSED", "QUALIFIED", "REFERENCE_ONLY", "BLOCKED"},
    "statistical_status": {"NOT_ASSESSED", "SUPPORTED", "INCONCLUSIVE", "REJECTED"},
    "economic_status": {"NOT_ASSESSED", "SUPPORTED", "INCONCLUSIVE", "NEGATIVE"},
    "operations_status": {"NOT_RUN", "PASS", "FAIL"},
}
_PASS_VALUES = {
    "pipeline_status": "PASS",
    "data_status": "QUALIFIED",
    "statistical_status": "SUPPORTED",
    "economic_status": "SUPPORTED",
    "operations_status": "PASS",
}


def _valid_reference(value: object) -> bool:
    """Accept only stable evidence references; names and booleans are not evidence."""
    if not isinstance(value, str):
        return False
    value = value.strip()
    if not value or len(value) > 2048 or any(ch.isspace() for ch in value):
        return False
    if value.startswith("https://"):
        parsed = urlparse(value)
        return bool(parsed.hostname) and parsed.username is None and parsed.password is None
    if value.startswith(("evidence://", "repo://")):
        scheme, identifier = value.split("://", 1)
        return scheme in {"evidence", "repo"} and bool(identifier.strip("/"))
    return False


def evaluate_research_readiness(
    *,
    pipeline_status: str,
    data_status: str,
    statistical_status: str,
    economic_status: str,
    operations_status: str,
    evidence_refs: Mapping[str, object] | None = None,
    release_review_status: str = "PENDING",
    release_review_reference: object = None,
) -> dict[str, Any]:
    """Return a fail-closed evidence summary without enabling any trading path.

    A favorable status is counted only when a stable evidence reference is attached.
    The caller is responsible for recording reviewer identity and the exact evidence
    artifact; this function deliberately does not infer market truth from CI status.
    """
    values = {
        "pipeline_status": pipeline_status,
        "data_status": data_status,
        "statistical_status": statistical_status,
        "economic_status": economic_status,
        "operations_status": operations_status,
    }
    for field, allowed in _ALLOWED.items():
        value = values[field]
        if not isinstance(value, str) or value not in allowed:
            raise ValueError(f"{field}_invalid")

    if evidence_refs is not None and not isinstance(evidence_refs, Mapping):
        raise ValueError("evidence_refs_must_be_mapping")
    refs = dict(evidence_refs or {})
    unknown = sorted(set(refs) - set(_ALLOWED))
    if unknown:
        raise ValueError("unknown_evidence_reference_fields:" + ",".join(unknown))
    if not isinstance(release_review_status, str) or release_review_status not in {"PENDING", "APPROVED", "REJECTED"}:
        raise ValueError("release_review_status_invalid")

    checks: dict[str, dict[str, object]] = {}
    blockers: list[str] = []
    for field, expected_pass in _PASS_VALUES.items():
        status_ok = values[field] == expected_pass
        ref = refs.get(field)
        ref_ok = _valid_reference(ref)
        qualified = status_ok and ref_ok
        checks[field] = {
            "status": values[field],
            "passed": qualified,
            "evidence_reference_attached": ref_ok,
            "evidence_reference": ref.strip() if ref_ok else None,
        }
        if not status_ok:
            blockers.append(f"{field}_{values[field].lower()}")
        elif not ref_ok:
            blockers.append(f"{field}_evidence_reference_missing_or_invalid")

    technical_gates_passed = not blockers
    review_ref_ok = _valid_reference(release_review_reference)
    if release_review_status == "APPROVED" and not review_ref_ok:
        blockers.append("release_review_approval_reference_missing_or_invalid")

    if blockers:
        promotion_status = "BLOCKED"
    elif release_review_status == "REJECTED":
        promotion_status = "REJECTED_BY_RELEASE_REVIEW"
    elif release_review_status == "APPROVED" and review_ref_ok:
        promotion_status = "RELEASE_REVIEW_APPROVED_DEPLOYMENT_STILL_DISABLED"
    else:
        promotion_status = "ELIGIBLE_FOR_SEPARATE_RELEASE_REVIEW"

    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "evidence_gated_research_readiness",
        "status": "PASS" if not blockers else "BLOCKED",
        "checks": checks,
        "technical_gates_passed": technical_gates_passed,
        "release_review_status": release_review_status,
        "release_review_reference_attached": review_ref_ok,
        "promotion_status": promotion_status,
        "blockers": blockers,
        "validated_model": False,
        "deployment_calibrated": False,
        "live_signal_enabled": False,
        "live_order_execution_enabled": False,
        "decision_output": "NO_TRADE",
        "interpretation": (
            "This report separates evidence readiness from model validation. It cannot "
            "prove predictive skill or profitability and never authorizes live execution."
        ),
    }
