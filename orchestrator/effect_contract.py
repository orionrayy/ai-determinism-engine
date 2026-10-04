from __future__ import annotations

from dataclasses import dataclass
from typing import Any

RETRY_SAFE = "safe"
RETRY_RECONCILE = "reconcile"
RETRY_BLOCKED = "blocked"

RECONCILE_DETERMINISTIC = "deterministic"
RECONCILE_PROVIDER = "provider"
RECONCILE_NONE = "none"

IDENTITY_PROVIDER = "provider"
IDENTITY_ENGINE = "engine"
IDENTITY_NONE = "none"

FENCING_ENGINE = "engine"
FENCING_PROVIDER = "provider"
FENCING_NONE = "none"


@dataclass(frozen=True)
class EffectContract:
    identity: str = IDENTITY_NONE
    retry: str = RETRY_BLOCKED
    reconciliation: str = RECONCILE_NONE
    fencing: str = FENCING_NONE
    notes: str = ""

    @property
    def can_reconcile(self) -> bool:
        return self.reconciliation != RECONCILE_NONE

    @property
    def retry_allowed_after_uncertain(self) -> bool:
        return self.retry == RETRY_RECONCILE

    @property
    def has_effect_identity(self) -> bool:
        return self.identity != IDENTITY_NONE


DEFAULT_EFFECT_CONTRACT = EffectContract()


def _normalize(value: Any, allowed: set[str], fallback: str) -> str:
    candidate = str(value or "").strip().lower()
    return candidate if candidate in allowed else fallback


def resolve_effect_contract(
    node: Any,
    registry: dict[str, dict[str, Any]],
) -> EffectContract:
    tool = str(getattr(node, "tool", "") or "").strip()
    action = str(
        getattr(node, "input", {}).get("action")
        if isinstance(getattr(node, "input", {}), dict)
        else ""
    ).strip()
    spec = registry.get(tool, {})
    contracts = spec.get("effect_contracts")
    if not isinstance(contracts, dict):
        return DEFAULT_EFFECT_CONTRACT

    raw = contracts.get(action)
    if raw is None:
        raw = contracts.get("*")
    if not isinstance(raw, dict):
        return DEFAULT_EFFECT_CONTRACT

    return EffectContract(
        identity=_normalize(
            raw.get("identity"),
            {IDENTITY_PROVIDER, IDENTITY_ENGINE, IDENTITY_NONE},
            IDENTITY_NONE,
        ),
        retry=_normalize(
            raw.get("retry"),
            {RETRY_SAFE, RETRY_RECONCILE, RETRY_BLOCKED},
            RETRY_BLOCKED,
        ),
        reconciliation=_normalize(
            raw.get("reconciliation"),
            {
                RECONCILE_DETERMINISTIC,
                RECONCILE_PROVIDER,
                RECONCILE_NONE,
            },
            RECONCILE_NONE,
        ),
        fencing=_normalize(
            raw.get("fencing"),
            {FENCING_ENGINE, FENCING_PROVIDER, FENCING_NONE},
            FENCING_NONE,
        ),
        notes=str(raw.get("notes") or "").strip(),
    )


def require_live_effect_contract(
    node: Any,
    registry: dict[str, dict[str, Any]],
    *,
    dry_run: bool,
) -> EffectContract:
    contract = resolve_effect_contract(node, registry)
    if dry_run:
        return contract

    if contract.identity == IDENTITY_NONE:
        raise RuntimeError(
            f"live side effect {getattr(node, 'tool', '')}.{getattr(node, 'input', {}).get('action', '')} "
            "has no effect identity contract"
        )

    if contract.retry == RETRY_BLOCKED and contract.reconciliation == RECONCILE_NONE:
        # A non-retryable opaque effect is valid. It must simply remain fail-closed
        # on uncertain outcome; this guard rejects only malformed configurations.
        return contract

    if contract.retry == RETRY_RECONCILE and not contract.can_reconcile:
        raise RuntimeError(
            f"effect contract for {getattr(node, 'tool', '')} allows reconcile-based retry "
            "but declares no reconciliation mechanism"
        )

    return contract
