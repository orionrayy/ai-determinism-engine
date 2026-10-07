from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import threading
import time
from typing import Any, Protocol


class TelemetrySink(Protocol):
    def emit(self, event: dict[str, Any]) -> None: ...


@dataclass(slots=True)
class MemoryTelemetrySink:
    events: list[dict[str, Any]] = field(default_factory=list)

    def emit(self, event: dict[str, Any]) -> None:
        self.events.append(dict(event))


class JsonlTelemetrySink:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def emit(self, event: dict[str, Any]) -> None:
        line = json.dumps(event, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line)
                handle.flush()


@dataclass(frozen=True, slots=True)
class TraceContext:
    trace_id: str
    span_id: str
    parent_span_id: str = ""

    def to_dict(self) -> dict[str, str]:
        data = {"trace_id": self.trace_id, "span_id": self.span_id}
        if self.parent_span_id:
            data["parent_span_id"] = self.parent_span_id
        return data


class TelemetryRecorder:
    """OTel-compatible field names without requiring an OpenTelemetry runtime."""

    def __init__(self, sink: TelemetrySink) -> None:
        self.sink = sink

    def lifecycle(
        self,
        event_name: str,
        *,
        context: TraceContext,
        workflow_id: str,
        task_id: str = "",
        tenant_id: str = "",
        attempt: int | None = None,
        worker_id: str = "",
        attributes: dict[str, Any] | None = None,
        audit: bool = False,
    ) -> None:
        event = {
            "time_unix_ns": time.time_ns(),
            "event_name": str(event_name),
            "severity": "AUDIT" if audit else "INFO",
            **context.to_dict(),
            "workflow_id": str(workflow_id),
        }
        if task_id:
            event["task_id"] = str(task_id)
        if tenant_id:
            event["tenant_id"] = str(tenant_id)
        if attempt is not None:
            event["attempt"] = int(attempt)
        if worker_id:
            event["worker_id"] = str(worker_id)
        if attributes:
            event["attributes"] = dict(attributes)
        self.sink.emit(event)

    def metric(self, name: str, value: float, *, unit: str = "1", attributes: dict[str, Any] | None = None) -> None:
        self.sink.emit({
            "time_unix_ns": time.time_ns(),
            "event_name": "metric",
            "metric_name": str(name),
            "value": float(value),
            "unit": str(unit),
            "attributes": dict(attributes or {}),
        })
