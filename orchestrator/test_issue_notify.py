import os
import unittest
from unittest.mock import patch

import issue_notify


class IssueNotifyTests(unittest.TestCase):
    def test_missing_configuration_is_noop(self):
        workflow = {'id': 'wf_test', 'trigger_issue': 123}
        with patch.dict(os.environ, {}, clear=True):
            issue_notify.post_issue_status(workflow, 'hello', lambda **kwargs: None)

    def test_configured_notification_calls_github(self):
        workflow = {'id': 'wf_test', 'trigger_issue': 123}
        calls = []
        def fake_http(url, **kwargs):
            calls.append((url, kwargs))
            return {'status_code': 201}
        with patch.dict(os.environ, {
            'GITHUB_TOKEN': 'token',
            'GITHUB_REPOSITORY': 'owner/repo',
        }, clear=False):
            issue_notify.post_issue_status(workflow, 'hello', fake_http)
        self.assertEqual(len(calls), 1)
        self.assertIn('/issues/123/comments', calls[0][0])


if __name__ == '__main__':
    unittest.main()