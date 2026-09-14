import unittest
from unittest.mock import patch
import numpy as np
from quant_platform.demo import prices, run
from quant_system.config import StrategyConfig
from quant_system.signals import generate_signals
from quant_system.metrics_calculator import MetricsCalculator

class PublicTests(unittest.TestCase):
    def test_no_future_signal(self):
        original = prices()
        changed = original.copy()
        changed.loc[201:, 'close'] *= 2
        a = generate_signals(original, StrategyConfig())
        b = generate_signals(changed, StrategyConfig())
        self.assertTrue(a.signal.iloc[:201].equals(b.signal.iloc[:201]))

    def test_offline_demo(self):
        with patch('socket.socket.connect', side_effect=AssertionError('Network forbidden')):
            result = run()
        self.assertEqual(result['data_kind'], 'synthetic')
        self.assertNotEqual(result['result'].get('status'), 'insufficient_data')
        self.assertIn('equity_curve', result['result'])

    def test_sharpe(self):
        self.assertEqual(MetricsCalculator.sharpe([0, 0, 0]), 0)
        self.assertTrue(np.isfinite(MetricsCalculator.sharpe([0.01, -0.01, 0.02])))

if __name__ == '__main__':
    unittest.main()
