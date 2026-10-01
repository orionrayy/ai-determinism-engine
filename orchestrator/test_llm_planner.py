import unittest

from llm_planner import _object_from_text


class PlannerTests(unittest.TestCase):
    def test_extracts_plain_json(self):
        value = _object_from_text('{"nodes": []}')
        self.assertEqual(value, {"nodes": []})

    def test_extracts_fenced_json(self):
        value = _object_from_text('```json\n{"nodes": [{"id": "n01"}]}\n```')
        self.assertEqual(value["nodes"][0]["id"], "n01")

    def test_rejects_non_json(self):
        with self.assertRaises(ValueError):
            _object_from_text('no json here')


if __name__ == '__main__':
    unittest.main()