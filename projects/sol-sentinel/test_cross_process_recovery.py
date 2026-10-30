from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROCESS_SCRIPT = r"""
import json
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from depth_execution import (
    BookLevel, DepthSnapshot, OrderRequest, OrderState, OrderType, Side,
    SQLitePaperExecutionLedger, TimeInForce,
)

db_path, phase = sys.argv[1], sys.argv[2]
buy_id = sys.argv[3] if len(sys.argv) > 3 else ""
sell_id = sys.argv[4] if len(sys.argv) > 4 else ""
t0 = datetime(2026, 10, 30, 4, 0, tzinfo=timezone.utc)

def request(client_id, side, quantity, created_at):
    return OrderRequest(
        client_id, "INDODAX", "SOL/IDR", side, Decimal(quantity),
        OrderType.MARKET, None, TimeInForce.GTC, created_at, None,
        Decimal("1"), Decimal("1"), Decimal("10"),
        Decimal("0.01"), Decimal("30"),
    )

def snapshot(offset_seconds, bids, asks):
    return DepthSnapshot(
        "INDODAX", "SOL/IDR", t0 + timedelta(seconds=offset_seconds),
        tuple(BookLevel(*level) for level in bids),
        tuple(BookLevel(*level) for level in asks),
    )

ledger = SQLitePaperExecutionLedger(db_path, initial_cash_idr=Decimal("10000"))

if phase == "A":
    buy_id = ledger.create_order(request("cross-run-buy", Side.BUY, "5", t0))
    first_buy = ledger.process_snapshot(
        buy_id, snapshot(1, [(100, 2)], [(101, 2)]), as_of=t0 + timedelta(seconds=2)
    )
    assert first_buy.state is OrderState.PARTIALLY_FILLED_SIMULATED
    assert first_buy.cumulative_filled_quantity == Decimal("2")
    assert first_buy.remaining_quantity == Decimal("3")

    sell_id = ledger.create_order(request(
        "cross-run-sell", Side.SELL, "2", t0 + timedelta(seconds=3)
    ))
    first_sell = ledger.process_snapshot(
        sell_id, snapshot(4, [(100, 1)], [(101, 2)]), as_of=t0 + timedelta(seconds=5)
    )
    assert first_sell.state is OrderState.PARTIALLY_FILLED_SIMULATED
    assert first_sell.cumulative_filled_quantity == Decimal("1")
    assert first_sell.remaining_quantity == Decimal("1")
    portfolio = ledger.portfolio()
    assert portfolio["sol_quantity"] == Decimal("1")
    report = ledger.recover()
    assert report["consistent"], report["issues"]
    assert report["order_count"] == 2 and report["fill_count"] == 2
    print(json.dumps({"buy_id": buy_id, "sell_id": sell_id, "phase": "A",
                      "open_orders": len(ledger.list_open_orders()),
                      "portfolio": {k: str(v) for k, v in portfolio.items()},
                      "fill_count": report["fill_count"]}))

elif phase == "B":
    report = ledger.recover()
    assert report["consistent"], report["issues"]
    assert report["order_count"] == 2 and report["fill_count"] == 2
    assert ledger.get_order(buy_id)["state"] == OrderState.PARTIALLY_FILLED_SIMULATED.value
    assert ledger.get_order(sell_id)["state"] == OrderState.PARTIALLY_FILLED_SIMULATED.value
    assert len(ledger.list_open_orders()) == 2
    resumed_snapshot = snapshot(6, [(100, 2)], [(101, 3)])
    buy_result = ledger.process_snapshot(
        buy_id, resumed_snapshot, as_of=t0 + timedelta(seconds=8)
    )
    sell_result = ledger.process_snapshot(
        sell_id, resumed_snapshot, as_of=t0 + timedelta(seconds=8)
    )
    assert buy_result.cumulative_filled_quantity == Decimal("5")
    assert buy_result.remaining_quantity == Decimal("0")
    assert sell_result.cumulative_filled_quantity == Decimal("2")
    assert sell_result.remaining_quantity == Decimal("0")
    report = ledger.recover()
    assert report["consistent"], report["issues"]
    assert report["fill_count"] == 4 and report["order_count"] == 2
    print(json.dumps({"phase": "B", "fill_count": report["fill_count"],
                      "portfolio": report["replayed_portfolio"],
                      "open_orders": len(ledger.list_open_orders())}))

elif phase == "C":
    report = ledger.recover()
    assert report["consistent"], report["issues"]
    assert report["fill_count"] == 4 and report["order_count"] == 2
    buy = ledger.get_order(buy_id)
    sell = ledger.get_order(sell_id)
    assert buy["state"] == OrderState.FILLED_SIMULATED.value
    assert sell["state"] == OrderState.FILLED_SIMULATED.value
    assert Decimal(buy["filled_qty"]) == Decimal("5")
    assert Decimal(buy["remaining_qty"]) == Decimal("0")
    assert Decimal(sell["filled_qty"]) == Decimal("2")
    assert Decimal(sell["remaining_qty"]) == Decimal("0")
    portfolio = ledger.portfolio()
    assert portfolio["sol_quantity"] == Decimal("3")
    assert portfolio["fees_paid_idr"] > 0
    assert ledger.list_open_orders() == []

    replay_snapshot = snapshot(6, [(100, 2)], [(101, 3)])
    buy_replay = ledger.process_snapshot(
        buy_id, replay_snapshot, as_of=t0 + timedelta(seconds=9)
    )
    sell_replay = ledger.process_snapshot(
        sell_id, replay_snapshot, as_of=t0 + timedelta(seconds=9)
    )
    assert buy_replay.duplicate and sell_replay.duplicate
    final_report = ledger.recover()
    assert final_report["consistent"], final_report["issues"]
    assert final_report["fill_count"] == 4
    with sqlite3.connect(db_path) as db:
        assert db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert db.execute("PRAGMA foreign_key_check").fetchall() == []
    print(json.dumps({"phase": "C", "fill_count": final_report["fill_count"],
                      "event_count": final_report["event_count"],
                      "portfolio": {k: str(v) for k, v in portfolio.items()},
                      "duplicate_replay": True, "integrity": "ok"}))
else:
    raise SystemExit("invalid test phase")
"""


class CrossProcessRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="sol-sentinel-cross-process-")
        self.db_path = Path(self.temp.name) / "non-flat-paper.sqlite3"

    def tearDown(self):
        self.temp.cleanup()

    def run_phase(self, phase: str, *order_ids: str) -> dict:
        env = os.environ.copy()
        project_dir = str(Path(__file__).resolve().parent)
        env["PYTHONPATH"] = project_dir + os.pathsep + env.get("PYTHONPATH", "")
        result = subprocess.run(
            [sys.executable, "-c", PROCESS_SCRIPT, str(self.db_path), phase, *order_ids],
            capture_output=True, text=True, timeout=15, check=False, env=env,
        )
        self.assertEqual(
            result.returncode, 0,
            f"phase {phase} failed\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}",
        )
        return json.loads(result.stdout)

    def test_nonflat_partial_buy_sell_residuals_recover_across_three_processes(self):
        phase_a = self.run_phase("A")
        self.assertEqual(phase_a["phase"], "A")
        self.assertEqual(phase_a["open_orders"], 2)
        self.assertEqual(phase_a["fill_count"], 2)
        self.assertEqual(phase_a["portfolio"]["sol_quantity"], "1")

        order_ids = (phase_a["buy_id"], phase_a["sell_id"])
        phase_b = self.run_phase("B", *order_ids)
        self.assertEqual(phase_b["phase"], "B")
        self.assertEqual(phase_b["fill_count"], 4)
        self.assertEqual(phase_b["open_orders"], 0)

        phase_c = self.run_phase("C", *order_ids)
        self.assertEqual(phase_c["phase"], "C")
        self.assertEqual(phase_c["fill_count"], 4)
        self.assertEqual(phase_c["portfolio"]["sol_quantity"], "3")
        self.assertTrue(phase_c["duplicate_replay"])
        self.assertEqual(phase_c["integrity"], "ok")


if __name__ == "__main__":
    unittest.main()
