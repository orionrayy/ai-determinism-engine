import pathlib
import sqlite3
import unittest

class TelegramSchemaTests(unittest.TestCase):
    def test_schema_creates_and_supports_core_records(self):
        schema = pathlib.Path(__file__).with_name("telegram_schema.sql").read_text(encoding="utf-8")
        with sqlite3.connect(":memory:") as db:
            db.executescript(schema)
            db.execute("INSERT INTO telegram_consents(principal_key,policy_version,consented_at) VALUES(?,?,?)", ("p1", "2026-10-05", 1))
            db.execute("INSERT INTO telegram_sessions(chat_key,encrypted_session,created_at,updated_at,expires_at) VALUES(?,?,?,?,?)", ("c1", "ciphertext", 1, 1, 10))
            db.execute("INSERT INTO telegram_workflows(workflow_id,principal_key,created_at,last_seen_at,expires_at) VALUES(?,?,?,?,?)", ("wf-1", "p1", 1, 1, 100))
            db.execute("INSERT INTO telegram_inbox(event_id,principal_key,status,claim_token,created_at,updated_at,expires_at) VALUES(?,?,?,?,?,?,?)", ("e1", "p1", "processing", "claim-1", 1, 1, 100))
            db.execute("UPDATE telegram_inbox SET status='completed',workflow_id=? WHERE event_id=?", ("wf-1", "e1"))
            db.execute("INSERT INTO telegram_audit(event_id,principal_key,chat_key,action,workflow_id,intent_digest,created_at,expires_at) VALUES(?,?,?,?,?,?,?,?)", ("a1", "p1", "c1", "run", "wf-1", "a" * 64, 2, 100))
            db.commit()
            self.assertEqual(db.execute("SELECT COUNT(*) FROM telegram_inbox").fetchone()[0], 1)
            self.assertEqual(db.execute("SELECT workflow_id FROM telegram_inbox").fetchone()[0], "wf-1")
            self.assertEqual(db.execute("SELECT principal_key FROM telegram_workflows WHERE workflow_id='wf-1'").fetchone()[0], "p1")
            self.assertEqual(db.execute("SELECT expires_at FROM telegram_audit WHERE event_id='a1'").fetchone()[0], 100)
            self.assertEqual(db.execute("SELECT COUNT(*) FROM telegram_audit").fetchone()[0], 1)

if __name__ == "__main__":
    unittest.main()