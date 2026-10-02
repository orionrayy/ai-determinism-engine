import unittest

from federation_scheduler import (
    DEFAULT_MAX_BATCHES_PER_WORKFLOW,
    DEFAULT_MAX_TASKS_PER_WORKFLOW,
    FEDERATION_SLOTS,
    FederationQuotaError,
    can_reserve,
    federation_slot,
    refund,
    reserve,
)


class FederationSchedulerTests(unittest.TestCase):
    def workflow(self):
        return {
            "max_federation_batches": DEFAULT_MAX_BATCHES_PER_WORKFLOW,
            "max_federation_tasks": DEFAULT_MAX_TASKS_PER_WORKFLOW,
            "federation_batches_used": 0,
            "federation_tasks_used": 0,
        }

    def test_slot_is_stable_and_bounded(self):
        first = federation_slot("fed-example")
        second = federation_slot("fed-example")
        self.assertEqual(first, second)
        self.assertGreaterEqual(first, 0)
        self.assertLess(first, FEDERATION_SLOTS)

    def test_quota_reservation_and_refund(self):
        workflow = self.workflow()
        allowed, reason = can_reserve(workflow, 4)
        self.assertTrue(allowed)
        self.assertEqual(reason, "")
        reserve(workflow, 4)
        self.assertEqual(workflow["federation_batches_used"], 1)
        self.assertEqual(workflow["federation_tasks_used"], 4)
        refund(workflow, 4)
        self.assertEqual(workflow["federation_batches_used"], 0)
        self.assertEqual(workflow["federation_tasks_used"], 0)

    def test_max_batches_prevents_monopoly(self):
        workflow = self.workflow()
        for _ in range(DEFAULT_MAX_BATCHES_PER_WORKFLOW):
            reserve(workflow, 4)
        allowed, reason = can_reserve(workflow, 1)
        self.assertFalse(allowed)
        self.assertEqual(reason, "max_batches")
        with self.assertRaises(FederationQuotaError):
            reserve(workflow, 1)

    def test_max_tasks_prevents_overrun(self):
        workflow = self.workflow()
        reserve(workflow, 4)
        reserve(workflow, 4)
        reserve(workflow, 4)
        allowed, reason = can_reserve(workflow, 5)
        self.assertFalse(allowed)
        self.assertEqual(reason, "batch_size")
        allowed, reason = can_reserve(workflow, 4)
        self.assertTrue(allowed)
        reserve(workflow, 4)
        allowed, reason = can_reserve(workflow, 1)
        self.assertFalse(allowed)
        self.assertEqual(reason, "max_batches")

    def test_invalid_task_count_is_bounded(self):
        workflow = self.workflow()
        allowed, reason = can_reserve(workflow, 0)
        self.assertFalse(allowed)
        self.assertEqual(reason, "batch_size")
        allowed, reason = can_reserve(workflow, 5)
        self.assertFalse(allowed)
        self.assertEqual(reason, "batch_size")


if __name__ == "__main__":
    unittest.main()
