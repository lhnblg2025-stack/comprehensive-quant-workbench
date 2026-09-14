import sys
import tempfile
import unittest
from pathlib import Path
from quant_system.task_dag import TaskDAG
from quant_system.overfitting_tests import deflated_sharpe_ratio
from scripts.snapshot_storage import write_snapshot, archive_snapshots, snapshot_paths

class ExtendedTests(unittest.TestCase):
    def test_failed_parent_skips_child(self):
        dag = TaskDAG()
        dag.add_task('parent', [sys.executable, '-c', 'raise SystemExit(1)'])
        dag.add_task('child', [sys.executable, '-c', 'raise SystemExit(0)'], deps=('parent',))
        result = dag.run()
        self.assertEqual(result['child']['status'], 'skipped_dep_failed')

    def test_snapshot_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for i in range(1, 4):
                write_snapshot(root / f'intraday_chain_2026-01-0{i}.json', {'synthetic': i})
            self.assertEqual(len(archive_snapshots(root, keep=1)), 2)
            paths = snapshot_paths(root, 'intraday_chain_*.json')
            self.assertEqual(len(paths), 3)
            self.assertIn('synthetic', paths[0].read_text())

    def test_multiple_testing_penalty(self):
        one, _ = deflated_sharpe_ratio(.1, 1, 252)
        many, _ = deflated_sharpe_ratio(.1, 100, 252)
        self.assertLess(many, one)
