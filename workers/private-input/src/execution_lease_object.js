import { DurableObject } from "cloudflare:workers";

const CREATE_SQL = `
  CREATE TABLE IF NOT EXISTS lease_state (
    subject TEXT PRIMARY KEY,
    owner_id TEXT NOT NULL,
    lease_until INTEGER NOT NULL,
    attempts INTEGER NOT NULL
  )
`;

export class ExecutionLease extends DurableObject {
  constructor(ctx, env) {
    super(ctx, env);
    this.ctx.storage.sql.exec(CREATE_SQL);
  }

  async acquire(subject, ownerId, now, ttlSeconds) {
    const row = this.ctx.storage.sql
      .exec(
        "SELECT subject, owner_id, lease_until, attempts FROM lease_state WHERE subject = ?",
        subject,
      )
      .toArray()[0];

    if (row && row.lease_until > now && row.owner_id !== ownerId) {
      return {
        ok: false,
        conflict: true,
        lease_until: row.lease_until,
        attempts: row.attempts,
      };
    }

    const previousAttempts = row ? row.attempts : 0;
    const leaseUntil = now + ttlSeconds;
    this.ctx.storage.sql.exec(
      `INSERT INTO lease_state (subject, owner_id, lease_until, attempts)
       VALUES (?, ?, ?, ?)
       ON CONFLICT(subject) DO UPDATE SET
         owner_id = excluded.owner_id,
         lease_until = excluded.lease_until,
         attempts = excluded.attempts`,
      subject,
      ownerId,
      leaseUntil,
      previousAttempts,
    );
    await this.ctx.storage.setAlarm(leaseUntil * 1000);
    return {
      ok: true,
      lease_until: leaseUntil,
      attempts: previousAttempts,
    };
  }

  async reserve(subject, ownerId, now, count, maxAttempts) {
    const row = this.ctx.storage.sql
      .exec(
        "SELECT subject, owner_id, lease_until, attempts FROM lease_state WHERE subject = ?",
        subject,
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
    this.ctx.storage.sql.exec(
      "UPDATE lease_state SET attempts = ? WHERE subject = ? AND owner_id = ?",
      next,
      subject,
      ownerId,
    );
    return {
      ok: true,
      start_attempt: current + 1,
      used_attempts: next,
      max_attempts: maxAttempts,
    };
  }

  async release(subject, ownerId) {
    const row = this.ctx.storage.sql
      .exec(
        "SELECT owner_id FROM lease_state WHERE subject = ?",
        subject,
      )
      .toArray()[0];
    if (!row || row.owner_id !== ownerId) {
      return { ok: true, released: false };
    }
    this.ctx.storage.sql.exec(
      "DELETE FROM lease_state WHERE subject = ? AND owner_id = ?",
      subject,
      ownerId,
    );
    await this.ctx.storage.deleteAlarm();
    return { ok: true, released: true };
  }

  async alarm() {
    this.ctx.storage.sql.exec(
      "DELETE FROM lease_state WHERE lease_until <= ?",
      Math.floor(Date.now() / 1000),
    );
    await this.ctx.storage.deleteAlarm();
  }
}
