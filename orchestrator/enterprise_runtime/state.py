from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any, Callable, Iterable


class StateIntegrityError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class EventRecord:
    sequence: int
    event_type: str
    payload: dict[str, Any]
    event_digest: str

    @classmethod
    def make(cls, sequence: int, event_type: str, payload: dict[str, Any]) -> "EventRecord":
        body = {"sequence": sequence, "event_type": event_type, "payload": payload}
        digest = "sha256:" + hashlib.sha256(
            json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        ).hexdigest()
        return cls(sequence, event_type, payload, digest)

    def to_dict(self) -> dict[str, Any]:
        return {"sequence": self.sequence, "event_type": self.event_type,
                "payload": self.payload, "event_digest": self.event_digest}


@dataclass(frozen=True, slots=True)
class StateManifest:
    workflow_id: str
    tenant_id: str
    generation: int
    state_digest: str
    watermark: int
    snapshot_file: str
    event_file: str
    artifact_refs_file: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "workflow_id": self.workflow_id, "tenant_id": self.tenant_id,
            "generation": self.generation, "state_digest": self.state_digest,
            "watermark": self.watermark, "snapshot_file": self.snapshot_file,
            "event_file": self.event_file, "artifact_refs_file": self.artifact_refs_file,
        }


class FileObjectStore:
    """Atomic filesystem object store for $0 reference deployments."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def put_bytes(self, key: str, data: bytes) -> str:
        target = self.root / key
        target.parent.mkdir(parents=True, exist_ok=True)
        digest = "sha256:" + hashlib.sha256(data).hexdigest()
        with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as tmp:
            tmp.write(data)
            temp_name = Path(tmp.name)
        temp_name.replace(target)
        return digest

    def get_bytes(self, key: str) -> bytes:
        return (self.root / key).read_bytes()

    def exists(self, key: str) -> bool:
        return (self.root / key).is_file()

    def delete(self, key: str) -> None:
        path = self.root / key
        if path.exists():
            path.unlink()


class SegmentedWorkflowStore:
    """Separates compact state/header, event history, and artifact references."""

    def __init__(self, object_store: FileObjectStore) -> None:
        self.objects = object_store

    @staticmethod
    def _json(value: Any) -> bytes:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()

    @staticmethod
    def _state_digest(state: dict[str, Any]) -> str:
        return "sha256:" + hashlib.sha256(SegmentedWorkflowStore._json(state)).hexdigest()

    def write_snapshot(self, workflow_id: str, tenant_id: str, state: dict[str, Any],
                       *, generation: int, watermark: int,
                       artifact_refs: Iterable[dict[str, Any]] = ()) -> StateManifest:
        prefix = f"workflows/{workflow_id}"
        snapshot_key = f"{prefix}/snapshot-{generation}.json"
        event_key = f"{prefix}/events.jsonl"
        refs_key = f"{prefix}/artifact-refs-{generation}.json"
        digest = self._state_digest(state)
        self.objects.put_bytes(snapshot_key, self._json(state))
        self.objects.put_bytes(refs_key, self._json(list(artifact_refs)))
        if not self.objects.exists(event_key):
            self.objects.put_bytes(event_key, b"")
        manifest = StateManifest(workflow_id, tenant_id, generation, digest, int(watermark),
                                 snapshot_key, event_key, refs_key)
        self.objects.put_bytes(f"{prefix}/manifest.json", self._json(manifest.to_dict()))
        return manifest

    def append_event(self, workflow_id: str, event_type: str, payload: dict[str, Any],
                     *, sequence: int) -> EventRecord:
        key = f"workflows/{workflow_id}/events.jsonl"
        existing = self.objects.get_bytes(key).decode() if self.objects.exists(key) else ""
        expected = 1 if not existing.strip() else max(
            int(json.loads(line)["sequence"]) for line in existing.splitlines()) + 1
        if sequence != expected:
            raise StateIntegrityError(f"event sequence gap: expected {expected}, got {sequence}")
        record = EventRecord.make(sequence, event_type, payload)
        combined = (existing + json.dumps(record.to_dict(), sort_keys=True,
                                          separators=(",", ":"), ensure_ascii=False) + "\n").encode()
        self.objects.put_bytes(key, combined)
        return record

    def load_manifest(self, workflow_id: str) -> StateManifest:
        raw = json.loads(self.objects.get_bytes(f"workflows/{workflow_id}/manifest.json"))
        required = {"workflow_id","tenant_id","generation","state_digest","watermark",
                    "snapshot_file","event_file","artifact_refs_file"}
        if set(raw) != required:
            raise StateIntegrityError("manifest fields mismatch")
        if raw["workflow_id"] != workflow_id:
            raise StateIntegrityError("manifest workflow identity mismatch")
        return StateManifest(**raw)

    def reconstruct(self, workflow_id: str,
                    reducer: Callable[[dict[str, Any], EventRecord], dict[str, Any]]) -> dict[str, Any]:
        manifest = self.load_manifest(workflow_id)
        state = json.loads(self.objects.get_bytes(manifest.snapshot_file))
        if self._state_digest(state) != manifest.state_digest:
            raise StateIntegrityError("snapshot digest mismatch")
        lines = self.objects.get_bytes(manifest.event_file).decode().splitlines()
        expected = 1
        for line in lines:
            if not line.strip():
                continue
            raw = json.loads(line)
            record = EventRecord(
                int(raw["sequence"]), str(raw["event_type"]), dict(raw["payload"]), str(raw["event_digest"])
            )
            if record.sequence != expected:
                raise StateIntegrityError("event sequence mismatch")
            if EventRecord.make(record.sequence, record.event_type, record.payload).event_digest != record.event_digest:
                raise StateIntegrityError("event digest mismatch")
            state = reducer(state, record)
            expected += 1
        return state

    def verify(self, workflow_id: str,
               reducer: Callable[[dict[str, Any], EventRecord], dict[str, Any]]) -> bool:
        state = self.reconstruct(workflow_id, reducer)
        manifest = self.load_manifest(workflow_id)
        return isinstance(state, dict) and self._state_digest(state) == self._state_digest(state) and manifest.workflow_id == workflow_id
