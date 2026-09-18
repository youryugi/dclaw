import sys
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flume_validation import (  # noqa: E402
    centerline_axis,
    compare_series,
    first_persistent_exceedance,
    front_position,
    historical_front_envelope,
    initial_pile_and_gate,
    tracked_front_position,
)


class ValidationCalculationTests(unittest.TestCase):
    def test_persistent_arrival_rejects_single_spike(self):
        time = np.arange(-1.0, 2.0, 0.01)
        depth = np.zeros_like(time)
        depth[np.argmin(abs(time - 0.2))] = 1.0
        depth[(time >= 0.7) & (time < 1.0)] = 0.1
        self.assertAlmostEqual(
            first_persistent_exceedance(time, depth, 0.02, 0.05), 0.7, places=10
        )

    def test_centerline_axis_points_uphill(self):
        x = np.linspace(10, 0, 50)
        y = np.linspace(0, 20, 50)
        z = np.linspace(100, 110, 50)
        origin, axis = centerline_axis(x, y, z)
        along = (np.column_stack((x, y)) - origin) @ axis
        self.assertGreater(np.corrcoef(along, z)[0, 1], 0.99)

    def test_gate_is_downslope_edge_of_main_pile(self):
        s = np.arange(0.0, 10.0, 0.5)
        h = np.zeros_like(s)
        h[(s >= 6.0) & (s <= 9.0)] = 1.0
        pile, gate = initial_pile_and_gate(s, h)
        self.assertEqual(gate, 6.0)
        self.assertEqual(int(pile.sum()), 7)

    def test_front_requires_spatial_support(self):
        x = np.arange(-2.0, 9.0)
        h = np.zeros((2, x.size))
        h[0, (x >= -1) & (x <= 3)] = 0.2
        h[0, x == 8] = 0.5  # isolated saltating return
        h[1, (x >= -1) & (x <= 6)] = 0.2
        front = front_position(x, h, threshold_m=0.03, support_bins=2)
        np.testing.assert_allclose(front, [3.0, 6.0])

    def test_tracked_front_rejects_supported_noise_ahead(self):
        time = np.arange(4.0)
        x = np.arange(-1.0, 15.0)
        h = np.zeros((4, x.size))
        for i, front in enumerate([2, 4, 6, 8]):
            h[i, (x >= -1) & (x <= front)] = 0.2
            h[i, (x >= 12) & (x <= 14)] = 0.2
        tracked = tracked_front_position(time, x, h, max_speed_m_s=3.0, support_bins=2)
        np.testing.assert_allclose(tracked, [2, 4, 6, 8])

    def test_comparison_metrics(self):
        metrics = compare_series([0, 1, 2], [0, 2, 2])
        self.assertEqual(metrics.n, 3)
        self.assertAlmostEqual(metrics.mae, 1 / 3)
        self.assertAlmostEqual(metrics.bias, 1 / 3)

    def test_historical_front_envelope_fills_later_gaps(self):
        front = historical_front_envelope([np.nan, 1.0, 3.0, np.nan, 2.0, 4.0])
        self.assertTrue(np.isnan(front[0]))
        np.testing.assert_allclose(front[1:], [1.0, 3.0, 3.0, 3.0, 4.0])


if __name__ == "__main__":
    unittest.main()
