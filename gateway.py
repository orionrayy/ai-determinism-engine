#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import secrets
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import urllib.parse
import urllib.request

HOST = "0.0.0.0"
PORT = int(os.environ.get("PORT", "10000"))

def github_dispatch(goal: str, metadata: dict) -> dict:
    token = os.environ.get("GITHUB_GATEWAY_TOKEN")
    repository = os.environ.get("GITHUB_REPOSITORY", "orionrayy/ai-determinism-engine")
    if not token:
        raise RuntimeError("GITHUB_GATEWAY_TOKEN is not configured")
    url = f"https://api.github.com/repos/{repository}/dispatches"
    payload = json.dumps({
        "event_type": "orchestrator.event",
        "client_payload": {"goal": goal, "metadata": metadata},
    }).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
            "Content-Type": "application/json",
            "User-Agent": "ai-orchestrator-gateway/1.0",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return {"github_status": response.status}

def authorized(headers: dict[str, str]) -> bool:
    configured = os.environ.get("GATEWAY_SHARED_SECRET")
    if not configured:
        return False
    supplied = headers.get("Authorization", "")
    expected = "Bearer " + configured
    return secrets.compare_digest(supplied, expected)

class Handler(BaseHTTPRequestHandler):
    server_version = "AIOrchestratorGateway/1.0"

    def _send(self, code: int, data: dict):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args):
        print("%s - %s" % (self.address_string(), format % args), flush=True)

    def do_GET(self):
        path = urllib.parse.urlsplit(self.path).path
        if path == "/health":
            self._send(200, {
                "ok": True,
                "service": "ai-orchestrator-gateway",
                "configured": {
                    "github_token": bool(os.environ.get("GITHUB_GATEWAY_TOKEN")),
                    "shared_secret": bool(os.environ.get("GATEWAY_SHARED_SECRET")),
                },
            })
            return
        self._send(404, {"ok": False, "error": "not_found"})

    def do_POST(self):
        path = urllib.parse.urlsplit(self.path).path
        if path != "/event":
            self._send(404, {"ok": False, "error": "not_found"})
            return

        if not authorized({k: v for k, v in self.headers.items()}):
            self._send(401, {"ok": False, "error": "unauthorized"})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length > 128 * 1024:
                raise ValueError("payload too large")
            raw = self.rfile.read(length)
            payload = json.loads(raw.decode("utf-8") if raw else "{}")
            goal = str(payload.get("goal", "")).strip()
            if not goal:
                raise ValueError("goal is required")
            metadata = payload.get("metadata", {})
            if not isinstance(metadata, dict):
                metadata = {"value": str(metadata)}
            result = github_dispatch(goal, metadata)
            self._send(202, {"ok": True, "queued": True, **result})
        except ValueError as exc:
            self._send(400, {"ok": False, "error": str(exc)})
        except Exception as exc:
            print(f"dispatch error: {exc}", flush=True)
            self._send(503, {"ok": False, "error": "dispatch_unavailable"})

if __name__ == "__main__":
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
