import { DurableObject } from "cloudflare:workers";

const RETENTION_SECONDS = 7 * 24 * 60 * 60;

const CREATE_SQL = `
  CREATE TABLE IF NOT EXISTS lease_state (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    owner_id TEXT NOT NULL,
    lease_until INTEGER NOT NULL,
    retention_until INTEGER NOT NULL,
    attempts INTEGER NOT NULL
  )
`;

export class ExecutionLease extends DurableObject {
  constructor(ctx, env) {
    super(ctx, env);
    this.ctx.storage.sql.exec(CREATE_SQL);
  }

  async acquire(subject, ownerId, now, ttlSeconds, maxAttempts, initialAttempts) {
    const row = this.ctx.storage.sql
      .exec(
        "SELECT id, owner_id, lease_until, retention_until, attempts FROM lease_state WHERE id = 1",
      )
      .toArray()[0];

    const previousAttempts = row ? row.attempts : Math.max(0, initialAttempts || 0);
    if (previousAttempts > maxAttempts) {
      return {
        ok: false,
        budget_exhausted: true,
        attempts: previousAttempts,
        max_attempts: maxAttempts,
      };
    }

    if (row && row.lease_until > now && row.owner_id && row.owner_id !== ownerId) {
      return {
        ok: false,
        conflict: true,
        lease_until: row.lease_until,
        attempts: previousAttempts,
      };
    }

    const leaseUntil = now + ttlSeconds;
    const retentionUntil = Math.max(
      row ? row.retention_until : 0,
      leaseUntil + RETENTION_SECONDS,
      now + RETENTION_SECONDS,
    );

    this.ctx.storage.sql.exec(
      `INSERT INTO lease_state
         (id, owner_id, lease_until, retention_until, attempts)
       VALUES (1, ?, ?, ?, ?)
       ON CONFLICT(id) DO UPDATE SET
         owner_id = excluded.owner_id,
         lease_until = excluded.lease_until,
         retention_until = excluded.retention_until,
         attempts = excluded.attempts`,
      ownerId,
      leaseUntil,
      retentionUntil,
      previousAttempts,
    );
    await this.ctx.storage.setAlarm(leaseUntil * 1000);

    return {
      ok: true,
      lease_until: leaseUntil,
      retention_until: retentionUntil,
      attempts: previousAttempts,
    };
  }

  async reserve(subject, ownerId, now, count, maxAttempts, ttlSeconds) {
    const row = this.ctx.storage.sql
      .exec(
        "SELECT id, owner_id, lease_until, retention_until, attempts FROM lease_state WHERE id = 1",
      )
      .toArray()[0];

    if (!row || row.owner_id !== ownerId || row.lease_until <= now) {
      return { ok: false, lease_lost: true };
    }

    const current = row.attempts;
    if (current + count > maxAttempts) {
      return {
        ok: false,
        budget_exhausted: true,
        used_attempts: current,
        max_attempts: maxAttempts,
      };
    }

    const next = current + count;
    const leaseUntil = now + ttlSeconds;
    const retentionUntil = Math.max(
      row.retention_until,
      leaseUntil + RETENTION_SECONDS,
    );

    this.ctx.storage.sql.exec(
      `UPDATE lease_state
          SET attempts = ?, lease_until = ?, retention_until = ?
        WHERE id = 1 AND owner_id = ? AND lease_until > ?`,
      next,
      leaseUntil,
      retentionUntil,
      ownerId,
      now,
    );
    await this.ctx.storage.setAlarm(leaseUntil * 1000);

    return {
      ok: true,
      start_attempt: current + 1,
      used_attempts: next,
      max_attempts: maxAttempts,
      lease_until: leaseUntil,
      retention_until: retentionUntil,
    };
  }

  async release(subject, ownerId, now) {
    const row = this.ctx.storage.sql
      .exec(
        "SELECT owner_id, lease_until, retention_until, attempts FROM lease_state WHERE id = 1",
      )
      .toArray()[0];

    if (!row || row.owner_id !== ownerId) {
      return { ok: true, released: false };
    }

    const retentionUntil = Math.max(
      row.retention_until,
      now + RETENTION_SECONDS,
    );

    this.ctx.storage.sql.exec(
      `UPDATE lease_state
          SET owner_id = '', lease_until = 0, retention_until = ?
        WHERE id = 1 AND owner_id = ?`,
      retentionUntil,
      ownerId,
    );
    await this.ctx.storage.setAlarm(retentionUntil * 1000);

    return {
      ok: true,
      released: true,
      retention_until: retentionUntil,
      attempts: row.attempts,
    };
  }

  async alarm() {
    const now = Math.floor(Date.now() / 1000);
    const row = this.ctx.storage.sql
      .exec(
        "SELECT owner_id, lease_until, retention_until, attempts FROM lease_state WHERE id = 1",
      )
      .toArray()[0];

    if (!row) return;

    if (row.lease_until > now) {
      await this.ctx.storage.setAlarm(row.lease_until * 1000);
      return;
    }

    if (row.retention_until > now) {
      this.ctx.storage.sql.exec(
        "UPDATE lease_state SET owner_id = '', lease_until = 0 WHERE id = 1",
      );
      await this.ctx.storage.setAlarm(row.retention_until * 1000);
      return;
    }

    this.ctx.storage.sql.exec("DELETE FROM lease_state WHERE id = 1");
    await this.ctx.storage.deleteAlarm();
  }
}
