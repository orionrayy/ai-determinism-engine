import unittest

from authority import (
    DISTRIBUTED_CONTROL_PLANE,
    GIT_DURABLE,
    authority_mode_for_workflow,
    infer_legacy_authority,
    require_runtime_authority,
)


class AuthorityTests(unittest.TestCase):
    def test_live_workflow_uses_distributed_authority_when_available(self):
        self.assertEqual(
            authority_mode_for_workflow(True, True),
            DISTRIBUTED_CONTROL_PLANE,
        )

    def test_dry_run_is_always_git_durable(self):
        self.assertEqual(
            authority_mode_for_workflow(False, True),
            GIT_DURABLE,
        )

    def test_live_workflow_without_control_plane_is_git_durable(self):
        self.assertEqual(
            authority_mode_for_workflow(True, False),
            GIT_DURABLE,
        )

    def test_legacy_control_plane_execution_infers_distributed_authority(self):
        workflow = {
            "live": True,
            "control_plane": {"enabled": True},
        }
        self.assertEqual(
            infer_legacy_authority(workflow),
            DISTRIBUTED_CONTROL_PLANE,
        )

    def test_legacy_effect_record_infers_distributed_authority(self):
        workflow = {
            "live": True,
            "executions": {
                "e1": {"durability_authority": "control_plane"},
            },
        }
        self.assertEqual(
            infer_legacy_authority(workflow),
            DISTRIBUTED_CONTROL_PLANE,
        )

    def test_legacy_live_workflow_defaults_to_git_durable(self):
        self.assertEqual(
            infer_legacy_authority({"live": True}),
            GIT_DURABLE,
        )

    def test_distributed_workflow_cannot_downgrade_to_git_only(self):
        workflow = {
            "live": True,
            "authority_mode": DISTRIBUTED_CONTROL_PLANE,
        }
        with self.assertRaises(RuntimeError):
            require_runtime_authority(
                workflow,
                control_plane_configured=False,
                control_plane_active=False,
            )

    def test_distributed_workflow_requires_active_control_plane(self):
        workflow = {
            "live": True,
            "authority_mode": DISTRIBUTED_CONTROL_PLANE,
        }
        with self.assertRaises(RuntimeError):
            require_runtime_authority(
                workflow,
                control_plane_configured=True,
                control_plane_active=False,
            )

    def test_distributed_workflow_passes_when_control_plane_active(self):
        workflow = {
            "live": True,
            "authority_mode": DISTRIBUTED_CONTROL_PLANE,
        }
        require_runtime_authority(
            workflow,
            control_plane_configured=True,
            control_plane_active=True,
        )

    def test_git_workflow_does_not_require_control_plane(self):
        workflow = {
            "live": True,
            "authority_mode": GIT_DURABLE,
        }
        require_runtime_authority(
            workflow,
            control_plane_configured=False,
            control_plane_active=False,
        )


if __name__ == "__main__":
    unittest.main()
