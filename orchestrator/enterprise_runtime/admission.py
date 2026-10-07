from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

from .contracts import PolicySnapshot, TaskEnvelope


@dataclass(frozen=True, slots=True)
class AdmissionDecision:
    status: str
    reason: str
    retry_at: int | None = None
    scope: str = ""


class FairAdmissionController:
    """Deterministic pre-queue admission and weighted-fairness reference policy.

    This class is intentionally process-local. It defines stable semantics for
    tenant/workflow/capability/provider budgets without claiming distributed
    coordination; production deployments must back the counters with an
    authoritative shared store.
    """

    STATUSES = {
        "accepted",
        "delayed",
        "quota_blocked",
        "provider_limited",
        "capacity_blocked",
        "cancelled",
        "dead_lettered",
    }

    def __init__(self) -> None:
        self._limits: dict[tuple[str, str], int] = {}
        self._weights: dict[str, int] = {}
        self._inflight: dict[tuple[str, str], int] = {}
        self._provider_limited_until: dict[str, int] = {}
        self._tenant_virtual_finish: dict[str, float] = {}
        self._clock = time.time

    def configure_limit(self, scope_type: str, scope_id: str, maximum: int) -> None:
        maximum = int(maximum)
        if maximum < 0:
            raise ValueError("maximum must be non-negative")
        self._limits[(str(scope_type), str(scope_id))] = maximum

    def configure_weight(self, tenant_id: str, weight: int) -> None:
        weight = int(weight)
        if weight < 1:
            raise ValueError("weight must be >= 1")
        self._weights[str(tenant_id)] = weight

    def block_provider(self, provider: str, *, until: int) -> None:
        self._provider_limited_until[str(provider)] = int(until)

    def release(self, envelope: TaskEnvelope) -> None:
        keys = self._scope_keys(envelope)
        for key in keys:
            current = self._inflight.get(key, 0)
            self._inflight[key] = max(0, current - 1)

    def admit(
        self,
        envelope: TaskEnvelope,
        *,
        policy: PolicySnapshot | None = None,
        estimated_cost: int = 1,
        now: int | None = None,
    ) -> AdmissionDecision:
        now = int(self._clock() if now is None else now)
        estimated_cost = max(1, int(estimated_cost))
        if policy is not None:
            if policy.tenant_id != envelope.tenant_id:
                return AdmissionDecision("quota_blocked", "tenant_identity_mismatch", scope="tenant")
            if policy.free_only:
                billing = str(envelope.skill.get("billing_class", "unknown")).lower()
                if billing in {"paid", "credentialed_optional", "unknown"}:
                    return AdmissionDecision("quota_blocked", "hard_free_policy_rejects_nonfree_skill", scope="policy")
        if bool(envelope.metadata.get("cancelled")):
            return AdmissionDecision("cancelled", "task_cancelled", scope="task")

        provider = str(envelope.routing.get("provider", "")).strip()
        if provider and now < self._provider_limited_until.get(provider, 0):
            return AdmissionDecision(
                "provider_limited", "provider_permit_window_blocked",
                retry_at=self._provider_limited_until[provider], scope=provider,
            )

        scopes = self._scope_keys(envelope)
        for scope_type, scope_id in scopes:
            maximum = self._limits.get((scope_type, scope_id))
            if maximum is None:
                continue
            used = self._inflight.get((scope_type, scope_id), 0)
            if used + estimated_cost > maximum:
                return AdmissionDecision(
                    "quota_blocked",
                    f"{scope_type}_concurrency_limit",
                    retry_at=now + 1,
                    scope=f"{scope_type}:{scope_id}",
                )

        tenant = envelope.tenant_id
        weight = self._weights.get(tenant, 1)
        floor = min(self._tenant_virtual_finish.values(), default=0.0)
        current = self._tenant_virtual_finish.get(tenant, floor)
        if current > floor + (1.0 / weight):
            return AdmissionDecision(
                "delayed", "weighted_fairness_turn", retry_at=now + 1, scope=f"tenant:{tenant}"
            )

        for key in scopes:
            self._inflight[key] = self._inflight.get(key, 0) + estimated_cost
        self._tenant_virtual_finish[tenant] = max(current, floor) + (estimated_cost / weight)
        return AdmissionDecision("accepted", "admitted", scope=f"tenant:{tenant}")

    @staticmethod
    def _scope_keys(envelope: TaskEnvelope) -> tuple[tuple[str, str], ...]:
        provider = str(envelope.routing.get("provider", "")).strip()
        capability = envelope.capability
        return tuple(
            item for item in (
                ("tenant", envelope.tenant_id),
                ("workflow", envelope.workflow_id),
                ("capability", capability),
                (("provider", provider) if provider else None),
            )
            if item is not None
        )

    def snapshot(self) -> dict[str, Any]:
        return {
            "limits": {f"{k[0]}:{k[1]}": v for k, v in sorted(self._limits.items())},
            "inflight": {f"{k[0]}:{k[1]}": v for k, v in sorted(self._inflight.items())},
            "weights": dict(sorted(self._weights.items())),
            "provider_limited_until": dict(sorted(self._provider_limited_until.items())),
        }
