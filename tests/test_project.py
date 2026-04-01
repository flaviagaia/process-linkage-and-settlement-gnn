from __future__ import annotations

import unittest
from pathlib import Path

from src.modeling import run_pipeline


class ProcessLinkageSettlementGnnTestCase(unittest.TestCase):
    def test_pipeline_contract(self) -> None:
        project_dir = Path(__file__).resolve().parents[1]
        summary = run_pipeline(project_dir)
        self.assertEqual(summary["process_count"], 10)
        self.assertGreater(summary["node_count"], 10)
        self.assertGreater(summary["edge_count"], 10)
        self.assertGreater(summary["linked_process_groups"], 0)
        self.assertGreater(summary["macro_f1"], 0.4)


if __name__ == "__main__":
    unittest.main()
