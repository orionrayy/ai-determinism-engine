from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from bridge_runtime import (
    BridgeRuntimeError,
    BridgeUpstreamError,
    describe_routes,
    handle_reconciliation,
    handle_request,
    verify_signature,
)

HOST = "0.0.0.0"
PORT = int(os.environ.get("PORT", "10000"))


class Handler(BaseHTTPRequestHandler):
    server_version = "AIOrchestratorBridge/1.0"

    def _send(self, code: int, payload: dict):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        route = self.path.split("?", 1)[0]
        if route == "/capabilities":
            self._send(200, {
                "ok": True,
                "protocol": "ai-orchestrator.connector/v1",
                "connectors": describe_routes(),
            })
            return
        if route == "/health":
            self._send(200, {
                "ok": True,
                "service": "ai-orchestrator-connector-bridge",
                "protocol": "ai-orchestrator.connector/v1",
            })
            return
        self._send(404, {"ok": False, "error": "not_found"})

    def do_POST(self):
        route = self.path.split("?", 1)[0]
        if route not in {"/bridge", "/api/bridge", "/bridge/reconcile", "/api/bridge/reconcile"}:
            self._send(404, {"ok": False, "error": "not_found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 64 * 1024:
                raise BridgeRuntimeError("invalid request size")
            raw = self.rfile.read(length)
            secret = os.environ.get("ORCHESTRATOR_CONNECTOR_BRIDGE_SECRET", "")
            if not secret:
                raise BridgeRuntimeError("bridge secret is not configured")
            headers = {key.lower(): value.strip() for key, value in self.headers.items()}
            if not verify_signature(headers, raw, secret, method="POST", path=route):
                self._send(401, {"ok": False, "error": "invalid_signature"})
                return
            try:
                payload = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise BridgeRuntimeError("invalid JSON body") from exc
            if not isinstance(payload, dict):
                raise BridgeRuntimeError("request body must be an object")
            if route in {"/bridge/reconcile", "/api/bridge/reconcile"}:
                self._send(200, handle_reconciliation(payload))
            else:
                self._send(200, handle_request(payload, secret))
        except BridgeUpstreamError as exc:
            self._send(
                exc.status_code,
                {"ok": False, "error": str(exc), "uncertain": exc.uncertain},
            )
        except BridgeRuntimeError as exc:
            self._send(400, {"ok": False, "error": str(exc)})
        except Exception:
            self._send(500, {"ok": False, "error": "bridge_internal_error"})


if __name__ == "__main__":
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()