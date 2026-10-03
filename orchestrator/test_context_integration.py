import unittest
from orchestrator import Node, build_node_context

class ContextIntegrationTests(unittest.TestCase):
    def test_build_node_context_uses_global_budget(self):
        deps = []
        for i in range(12):
            deps.append({
                'id': f'n{i}', 'capability': 'analyze', 'tool': 'gemini',
                'status': 'completed', 'output': {'blob': 'x' * 9000},
                'error': {}, 'contract': {}, 'agent_role': 'analyst',
            })
        node = Node(id='consumer', capability='analyze', tool='gemini',
                    depends_on=[d['id'] for d in deps],
                    input={'goal': 'consume bounded context', 'repair_feedback': {}},
                    contract={'required_fields': ['result']})
        nodes = [Node(id=d['id'], capability=d['capability'], tool=d['tool'], depends_on=[],
                      status=d['status'], output=d['output'], error=d['error'],
                      contract=d['contract'], agent_role=d['agent_role']) for d in deps]
        nodes.append(node)
        context = build_node_context(nodes, node)
        raw = __import__('json').dumps(context, ensure_ascii=False, sort_keys=True).encode()
        self.assertLessEqual(len(raw), 48 * 1024)
        self.assertIn('context_budget', context)

if __name__ == '__main__':
    unittest.main()