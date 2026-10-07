from __future__ import annotations

from dataclasses import dataclass
import json
import sqlite3
import time
import uuid
from typing import Any, Callable, Protocol


@dataclass(frozen=True, slots=True)
class QueueClaimRecord:
    task_id: str
    tenant_id: str
    payload: dict[str, Any]
    worker_id: str
    claim_id: str
    attempt: int
    lease_expires_at: int
    priority: int
    partition: str


class TaskQueue(Protocol):
    def enqueue(self, task_id: str, tenant_id: str, payload: dict[str, Any], *,
                dedupe_key: str, available_at: int | None = None,
                priority: int = 0, partition: str = "default",
                max_attempts: int = 8) -> bool: ...
    def claim(self, worker_id: str, *, lease_seconds: int = 120,
              partition: str | None = None) -> QueueClaimRecord | None: ...
    def ack(self, claim: QueueClaimRecord) -> bool: ...
    def nack(self, claim: QueueClaimRecord, error: str, *,
             retry_at: int | None = None, dead_letter: bool = False) -> str: ...


class SQLiteTaskQueue:
    """Reference queue with transactional claim/ack/nack semantics."""

    def __init__(self, database: str = ":memory:") -> None:
        self.db = sqlite3.connect(database, timeout=30, isolation_level=None,
                                  check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self._init()

    def _init(self) -> None:
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS tasks (
            task_id TEXT PRIMARY KEY,
            tenant_id TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            dedupe_key TEXT NOT NULL UNIQUE,
            status TEXT NOT NULL CHECK(status IN ('queued','processing','completed','failed','dead')),
            attempt INTEGER NOT NULL DEFAULT 0,
            max_attempts INTEGER NOT NULL DEFAULT 8,
            available_at INTEGER NOT NULL,
            priority INTEGER NOT NULL DEFAULT 0,
            partition TEXT NOT NULL DEFAULT 'default',
            worker_id TEXT,
            claim_id TEXT,
            lease_expires_at INTEGER,
            last_error TEXT NOT NULL DEFAULT '',
            created_at INTEGER NOT NULL,
            updated_at INTEGER NOT NULL
        );
        CREATE INDEX IF NOT EXISTS tasks_ready_idx
          ON tasks(status, available_at, priority DESC, created_at);
        CREATE INDEX IF NOT EXISTS tasks_lease_idx
          ON tasks(status, lease_expires_at);
        CREATE INDEX IF NOT EXISTS tasks_tenant_idx
          ON tasks(tenant_id, status);
        """)

    def close(self) -> None:
        self.db.close()

    def enqueue(self, task_id: str, tenant_id: str, payload: dict[str, Any], *,
                dedupe_key: str, available_at: int | None = None,
                priority: int = 0, partition: str = "default",
                max_attempts: int = 8) -> bool:
        now = int(time.time())
        try:
            self.db.execute(
                """INSERT INTO tasks(task_id,tenant_id,payload_json,dedupe_key,status,
                   attempt,max_attempts,available_at,priority,partition,created_at,updated_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
                (task_id, tenant_id, json.dumps(payload, sort_keys=True, separators=(",", ":")),
                 dedupe_key, "queued", 0, max(1, int(max_attempts)),
                 now if available_at is None else int(available_at), int(priority), partition,
                 now, now),
            )
            return True
        except sqlite3.IntegrityError as exc:
            if "dedupe_key" in str(exc) or "tasks.task_id" in str(exc):
                return False
            raise

    def reclaim_expired(self, *, now: int | None = None) -> int:
        now = int(time.time() if now is None else now)
        cur = self.db.execute(
            """UPDATE tasks SET
                   status=CASE WHEN attempt >= max_attempts THEN 'dead' ELSE 'queued' END,
                   worker_id=NULL,claim_id=NULL,lease_expires_at=NULL,
                   available_at=?,last_error=CASE
                     WHEN attempt >= max_attempts THEN 'lease_expired_max_attempts'
                     ELSE last_error END,
                   updated_at=?
               WHERE status='processing' AND lease_expires_at IS NOT NULL
                 AND lease_expires_at <= ?""",
            (now, now, now),
        )
        return cur.rowcount

    def claim(self, worker_id: str, *, lease_seconds: int = 120,
              partition: str | None = None) -> QueueClaimRecord | None:
        now = int(time.time())
        lease_seconds = max(1, min(int(lease_seconds), 3600))
        self.db.execute("BEGIN IMMEDIATE")
        try:
            self.reclaim_expired(now=now)
            where = ["status='queued'", "available_at<=?"]
            args: list[Any] = [now]
            if partition is not None:
                where.append("partition=?")
                args.append(partition)
            row = self.db.execute(
                f"""SELECT * FROM tasks WHERE {' AND '.join(where)}
                    ORDER BY priority DESC, created_at ASC LIMIT 1""", args
            ).fetchone()
            if row is None:
                self.db.execute("COMMIT")
                return None
            claim_id = str(uuid.uuid4())
            attempt = int(row["attempt"]) + 1
            expires = now + lease_seconds
            cur = self.db.execute(
                """UPDATE tasks SET status='processing',attempt=?,worker_id=?,
                   claim_id=?,lease_expires_at=?,updated_at=?
                   WHERE task_id=? AND status='queued'""",
                (attempt, worker_id, claim_id, expires, now, row["task_id"]),
            )
            if cur.rowcount != 1:
                self.db.execute("ROLLBACK")
                return None
            self.db.execute("COMMIT")
            return QueueClaimRecord(
                task_id=str(row["task_id"]), tenant_id=str(row["tenant_id"]),
                payload=json.loads(row["payload_json"]), worker_id=worker_id,
                claim_id=claim_id, attempt=attempt, lease_expires_at=expires,
                priority=int(row["priority"]), partition=str(row["partition"]),
            )
        except Exception:
            self.db.execute("ROLLBACK")
            raise

    def ack(self, claim: QueueClaimRecord) -> bool:
        now = int(time.time())
        cur = self.db.execute(
            """UPDATE tasks SET status='completed',worker_id=NULL,claim_id=NULL,
                   updated_at=?,lease_expires_at=NULL
               WHERE task_id=? AND status='processing' AND worker_id=? AND claim_id=?""",
            (now, claim.task_id, claim.worker_id, claim.claim_id),
        )
        return cur.rowcount == 1

    def nack(self, claim: QueueClaimRecord, error: str, *,
             retry_at: int | None = None, dead_letter: bool = False) -> str:
        now = int(time.time())
        row = self.db.execute(
            "SELECT attempt,max_attempts FROM tasks WHERE task_id=? AND status='processing' "
            "AND worker_id=? AND claim_id=?",
            (claim.task_id, claim.worker_id, claim.claim_id),
        ).fetchone()
        if row is None:
            return "stale"
        status = "dead" if dead_letter or int(row["attempt"]) >= int(row["max_attempts"]) else "queued"
        available = now if retry_at is None else int(retry_at)
        cur = self.db.execute(
            """UPDATE tasks SET status=?,worker_id=NULL,claim_id=NULL,lease_expires_at=NULL,
               available_at=?,last_error=?,updated_at=?
               WHERE task_id=? AND status='processing' AND worker_id=? AND claim_id=?""",
            (status, available, str(error)[:2048], now, claim.task_id, claim.worker_id, claim.claim_id),
        )
        return status if cur.rowcount == 1 else "stale"

    def status(self, task_id: str) -> str | None:
        row = self.db.execute("SELECT status FROM tasks WHERE task_id=?", (task_id,)).fetchone()
        return None if row is None else str(row["status"])

    def depth(self, tenant_id: str | None = None) -> int:
        query = "SELECT COUNT(*) AS n FROM tasks WHERE status='queued'"
        args: tuple[Any, ...] = ()
        if tenant_id is not None:
            query += " AND tenant_id=?"
            args = (tenant_id,)
        row = self.db.execute(query, args).fetchone()
        return int(row["n"])


class PostgresTaskQueue:
    """PostgreSQL adapter using an injected DB-API connection.

    No PostgreSQL driver is imported by the core. A deployment can supply its
    own psycopg/psycopg2/DB-API-compatible connection factory.
    """

    def __init__(self, connection_factory: Callable[[], Any]) -> None:
        self.connection_factory = connection_factory

    def enqueue(self, task_id: str, tenant_id: str, payload: dict[str, Any], *,
                dedupe_key: str, available_at: int | None = None, priority: int = 0,
                partition: str = "default", max_attempts: int = 8) -> bool:
        now = int(time.time())
        conn = self.connection_factory()
        try:
            cur = conn.cursor()
            cur.execute(
                """INSERT INTO orchestrator_tasks
                (task_id,tenant_id,payload_json,dedupe_key,status,attempt,max_attempts,
                 available_at,priority,partition,created_at,updated_at)
                VALUES (%s,%s,%s,%s,'queued',0,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (dedupe_key) DO NOTHING""",
                (task_id, tenant_id, json.dumps(payload, sort_keys=True, separators=(",", ":")),
                 dedupe_key, max(1, int(max_attempts)), now if available_at is None else int(available_at),
                 int(priority), partition, now, now),
            )
            ok = cur.rowcount == 1
            conn.commit()
            return ok
        finally:
            conn.close()

    def claim(self, worker_id: str, *, lease_seconds: int = 120,
              partition: str | None = None) -> QueueClaimRecord | None:
        now = int(time.time())
        conn = self.connection_factory()
        try:
            cur = conn.cursor()
            cur.execute("BEGIN")
            cur.execute(
                """UPDATE orchestrator_tasks SET
                       status=CASE WHEN attempt >= max_attempts THEN 'dead' ELSE 'queued' END,
                       worker_id=NULL,claim_id=NULL,lease_expires_at=NULL,available_at=%s,
                       last_error=CASE WHEN attempt >= max_attempts
                         THEN 'lease_expired_max_attempts' ELSE last_error END,
                       updated_at=%s
                   WHERE status='processing' AND lease_expires_at IS NOT NULL
                     AND lease_expires_at <= %s""",
                (now, now, now),
            )
            params: list[Any] = [now]
            clause = "status='queued' AND available_at<=%s"
            if partition is not None:
                clause += " AND partition=%s"
                params.append(partition)
            cur.execute(
                f"""SELECT task_id,tenant_id,payload_json,attempt,priority,partition
                    FROM orchestrator_tasks
                    WHERE {clause}
                    ORDER BY priority DESC,created_at ASC
                    FOR UPDATE SKIP LOCKED LIMIT 1""", params)
            row = cur.fetchone()
            if row is None:
                conn.commit()
                return None
            claim_id = str(uuid.uuid4())
            attempt = int(row[3]) + 1
            expires = now + max(1, min(int(lease_seconds), 3600))
            cur.execute(
                """UPDATE orchestrator_tasks SET status='processing',attempt=%s,
                   worker_id=%s,claim_id=%s,lease_expires_at=%s,updated_at=%s
                   WHERE task_id=%s""",
                (attempt, worker_id, claim_id, expires, now, row[0]))
            conn.commit()
            return QueueClaimRecord(str(row[0]),str(row[1]),json.loads(row[2]),worker_id,
                                    claim_id,attempt,expires,int(row[4]),str(row[5]))
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def ack(self, claim: QueueClaimRecord) -> bool:
        conn = self.connection_factory()
        try:
            cur = conn.cursor()
            cur.execute(
                """UPDATE orchestrator_tasks SET status='completed',worker_id=NULL,
                   claim_id=NULL,lease_expires_at=NULL,updated_at=%s
                   WHERE task_id=%s AND status='processing'
                   AND worker_id=%s AND claim_id=%s""",
                (int(time.time()),claim.task_id,claim.worker_id,claim.claim_id))
            conn.commit()
            return cur.rowcount == 1
        finally:
            conn.close()

    def nack(self, claim: QueueClaimRecord, error: str, *,
             retry_at: int | None = None, dead_letter: bool = False) -> str:
        conn = self.connection_factory()
        try:
            cur = conn.cursor()
            now = int(time.time())
            cur.execute(
                """SELECT attempt,max_attempts FROM orchestrator_tasks
                   WHERE task_id=%s AND status='processing' AND worker_id=%s AND claim_id=%s
                   FOR UPDATE""", (claim.task_id,claim.worker_id,claim.claim_id))
            row = cur.fetchone()
            if row is None:
                conn.commit()
                return "stale"
            status = "dead" if dead_letter or int(row[0]) >= int(row[1]) else "queued"
            cur.execute(
                """UPDATE orchestrator_tasks SET status=%s,worker_id=NULL,claim_id=NULL,
                   lease_expires_at=NULL,available_at=%s,last_error=%s,updated_at=%s
                   WHERE task_id=%s AND status='processing'
                     AND worker_id=%s AND claim_id=%s""",
                (status, now if retry_at is None else int(retry_at), str(error)[:2048],
                 now, claim.task_id, claim.worker_id, claim.claim_id)
            conn.commit()
            return status
        finally:
            conn.close()


class NATSJetStreamAdapter:
    """Transport-neutral JetStream seam.

    The injected request function performs the provider-specific NATS protocol
    operations. This keeps the core dependency-free while making publish,
    acknowledgement, negative acknowledgement, and termination explicit.
    """

    def __init__(self, request: Callable[[str, dict[str, Any]], Any]) -> None:
        self.request = request

    def enqueue(self, stream: str, subject: str, envelope: dict[str, Any], *,
                msg_id: str, headers: dict[str, str] | None = None) -> Any:
        return self.request("publish", {
            "stream": stream, "subject": subject, "msg_id": msg_id,
            "headers": headers or {}, "payload": envelope,
        })

    def ack(self, consumer: str, sequence: int) -> Any:
        return self.request("ack", {"consumer": consumer, "sequence": int(sequence)})

    def nak(self, consumer: str, sequence: int, *, delay_seconds: int = 0) -> Any:
        return self.request("nak", {"consumer": consumer, "sequence": int(sequence),
                                    "delay_seconds": max(0, int(delay_seconds))})

    def term(self, consumer: str, sequence: int) -> Any:
        return self.request("term", {"consumer": consumer, "sequence": int(sequence)})
