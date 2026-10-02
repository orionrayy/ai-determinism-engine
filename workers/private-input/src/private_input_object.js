import { DurableObject } from "cloudflare:workers";

const CREATE_SQL = `
  CREATE TABLE IF NOT EXISTS inputs (
    input_ref TEXT PRIMARY KEY,
    execution_id TEXT NOT NULL,
    intent_fingerprint TEXT NOT NULL,
    input_digest TEXT NOT NULL,
    expires_at INTEGER NOT NULL,
    payload_json TEXT NOT NULL
  )
`;

export class PrivateInput extends DurableObject {
  constructor(ctx, env) {
    super(ctx, env);
    this.ctx.storage.sql.exec(CREATE_SQL);
  }

  async putInput(envelope) {
    const inputRef = envelope.input_ref;
    const payloadJson = JSON.stringify(envelope.payload);
    const existing = this.ctx.storage.sql
      .exec(
        `SELECT input_ref, execution_id, intent_fingerprint, input_digest,
                expires_at, payload_json
           FROM inputs WHERE input_ref = ?`,
        inputRef,
      )
      .toArray()[0];

    if (existing) {
      const identical =
        existing.execution_id === envelope.execution_id &&
        existing.intent_fingerprint === envelope.intent_fingerprint &&
        existing.input_digest === envelope.input_digest &&
        existing.payload_json === payloadJson;
      if (!identical) return { ok: false, conflict: true };
      return {
        ok: true,
        created: false,
        input_ref: inputRef,
        execution_id: envelope.execution_id,
        intent_fingerprint: envelope.intent_fingerprint,
        input_digest: envelope.input_digest,
        expires_at: envelope.expires_at,
      };
    }

    this.ctx.storage.sql.exec(
      `INSERT INTO inputs
         (input_ref, execution_id, intent_fingerprint, input_digest, expires_at, payload_json)
       VALUES (?, ?, ?, ?, ?, ?)`,
      inputRef,
      envelope.execution_id,
      envelope.intent_fingerprint,
      envelope.input_digest,
      envelope.expires_at,
      payloadJson,
    );
    await this.ctx.storage.setAlarm(envelope.expires_at * 1000);
    return {
      ok: true,
      created: true,
      input_ref: inputRef,
      execution_id: envelope.execution_id,
      intent_fingerprint: envelope.intent_fingerprint,
      input_digest: envelope.input_digest,
      expires_at: envelope.expires_at,
    };
  }

  async getInput(inputRef) {
    const row = this.ctx.storage.sql
      .exec(
        `SELECT input_ref, execution_id, intent_fingerprint, input_digest,
                expires_at, payload_json
           FROM inputs WHERE input_ref = ?`,
        inputRef,
      )
      .toArray()[0];

    if (!row) return null;

    const now = Math.floor(Date.now() / 1000);
    if (row.expires_at <= now) {
      this.ctx.storage.sql.exec("DELETE FROM inputs WHERE input_ref = ?", inputRef);
      await this.ctx.storage.deleteAlarm();
      return null;
    }

    let payload;
    try {
      payload = JSON.parse(row.payload_json);
    } catch {
      throw new Error("stored_payload_invalid_json");
    }

    return {
      ok: true,
      input_ref: row.input_ref,
      execution_id: row.execution_id,
      intent_fingerprint: row.intent_fingerprint,
      input_digest: row.input_digest,
      expires_at: row.expires_at,
      payload,
    };
  }

  async deleteInput(inputRef) {
    this.ctx.storage.sql.exec("DELETE FROM inputs WHERE input_ref = ?", inputRef);
    await this.ctx.storage.deleteAlarm();
    return { ok: true, input_ref: inputRef, deleted: true };
  }

  async alarm() {
    this.ctx.storage.sql.exec(
      "DELETE FROM inputs WHERE expires_at <= ?",
      Math.floor(Date.now() / 1000),
    );
    await this.ctx.storage.deleteAlarm();
  }
}
