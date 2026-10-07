from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .contracts import ContractError, SkillManifest, TaskEnvelope, canonical_digest


class SkillAuthorizationError(PermissionError):
    pass


@dataclass(frozen=True, slots=True)
class SkillRecord:
    manifest: SkillManifest
    healthy: bool = True
    quarantined: bool = False
    health_reason: str = ""


class SkillRegistry:
    """Versioned skill registry and execution authorization boundary."""

    def __init__(self) -> None:
        self._skills: dict[tuple[str, str], SkillRecord] = {}

    @staticmethod
    def expected_digest(manifest: SkillManifest) -> str:
        payload = manifest.to_dict()
        payload["digest"] = ""
        return canonical_digest(payload)

    def register(self, manifest: SkillManifest, *, verify_digest: bool = True) -> None:
        if verify_digest and manifest.digest != self.expected_digest(manifest):
            raise ContractError("skill manifest digest mismatch")
        key = (manifest.skill_id, manifest.version)
        existing = self._skills.get(key)
        if existing is not None and existing.manifest.digest != manifest.digest:
            raise ContractError("skill version already registered with a different digest")
        self._skills[key] = SkillRecord(manifest)

    def get(self, skill_id: str, version: str) -> SkillManifest:
        try:
            record = self._skills[(str(skill_id).lower(), str(version))]
        except KeyError as exc:
            raise SkillAuthorizationError("skill_not_registered") from exc
        if record.quarantined or not record.healthy:
            raise SkillAuthorizationError("skill_unavailable")
        return record.manifest

    def quarantine(self, skill_id: str, version: str, reason: str) -> None:
        key = (str(skill_id).lower(), str(version))
        record = self._skills[key]
        self._skills[key] = SkillRecord(record.manifest, healthy=False, quarantined=True, health_reason=str(reason)[:512])

    def restore(self, skill_id: str, version: str) -> None:
        key = (str(skill_id).lower(), str(version))
        record = self._skills[key]
        self._skills[key] = SkillRecord(record.manifest)

    def authorize(
        self,
        envelope: TaskEnvelope,
        *,
        trust_level: str,
        network_class: str,
        free_only: bool,
        granted_permissions: Mapping[str, Any] | None = None,
    ) -> SkillManifest:
        skill_id = str(envelope.skill.get("skill_id", "")).strip().lower()
        version = str(envelope.skill.get("version", "")).strip()
        if not skill_id or not version:
            raise SkillAuthorizationError("task_missing_skill_identity")
        manifest = self.get(skill_id, version)

        expected = str(envelope.skill.get("digest", "")).strip().lower()
        if expected and expected != manifest.digest:
            raise SkillAuthorizationError("skill_digest_mismatch")

        worker_trust = str(trust_level).lower()
        levels = {"untrusted": 0, "standard": 1, "trusted": 2, "isolated": 3}
        if levels.get(worker_trust, -1) < levels.get(manifest.trust_level, 0):
            raise SkillAuthorizationError("worker_trust_insufficient")

        required_network = str(manifest.permissions.get("network_class", ""))
        if required_network and required_network != network_class:
            raise SkillAuthorizationError("network_policy_mismatch")

        if free_only and manifest.billing_class not in {"free_public", "free_allowance"}:
            raise SkillAuthorizationError("hard_free_rejects_skill")

        granted = dict(granted_permissions or {})
        required = dict(manifest.permissions)
        for key, value in required.items():
            if key == "network_class":
                continue
            if granted.get(key) != value:
                raise SkillAuthorizationError(f"permission_not_granted:{key}")

        return manifest

    def list(self) -> tuple[SkillManifest, ...]:
        return tuple(self._skills[k].manifest for k in sorted(self._skills))

    def health(self) -> dict[str, Any]:
        return {
            f"{skill_id}@{version}": {
                "healthy": record.healthy,
                "quarantined": record.quarantined,
                "reason": record.health_reason,
            }
            for (skill_id, version), record in sorted(self._skills.items())
        }
