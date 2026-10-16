import inspect
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from hlft import (Config, baseline_predictions, check_splits, distance_gradients,
                  fit, inverse_target, project_ball, read_observations,
                  regression_metrics, train_epoch, transform_target)


class HLFTTests(unittest.TestCase):
    def test_distance_gradients_match_finite_differences(self):
        a, b = np.array([.12, -.06, .03]), np.array([-.09, .1, .05])
        distance, ga, gb = distance_gradients(a, b)
        self.assertGreater(distance, 0.)
        for vector, gradient, first in [(a, ga, True), (b, gb, False)]:
            for index in range(len(vector)):
                plus, minus = vector.copy(), vector.copy()
                plus[index] += 1e-6
                minus[index] -= 1e-6
                high = distance_gradients(plus, b)[0] if first else distance_gradients(a, plus)[0]
                low = distance_gradients(minus, b)[0] if first else distance_gradients(a, minus)[0]
                self.assertAlmostEqual((high - low) / 2e-6, gradient[index], places=6)

    def test_complete_loss_gradient_matches_finite_difference(self):
        a, b, c = np.array([.12, -.06]), np.array([-.09, .1]), np.array([.02, .04])
        target, scale = 1.4, 2.0
        ab, ga, _ = distance_gradients(a, b)
        ac, gc, _ = distance_gradients(a, c)
        bc = distance_gradients(b, c)[0]
        tangent = np.tanh(ab + ac + bc)
        prediction = scale * (1 - tangent)
        gradient = 2 * (target - prediction) * scale * (1 - tangent ** 2) * (ga + gc)
        def loss(x):
            total = distance_gradients(x, b)[0] + distance_gradients(x, c)[0] + bc
            return (target - scale * (1 - np.tanh(total))) ** 2
        for k in range(len(a)):
            plus, minus = a.copy(), a.copy()
            plus[k] += 1e-6
            minus[k] -= 1e-6
            self.assertAlmostEqual((loss(plus) - loss(minus)) / 2e-6, gradient[k], places=6)

    def test_projection_preserves_direction_and_stays_inside_ball(self):
        x = np.array([-3., -4.])
        projected = project_ball(x)
        self.assertLess(np.linalg.norm(projected), 1)
        np.testing.assert_allclose(projected, x / 5 * (1 - 1e-5))

    def test_coincident_points_are_finite(self):
        distance, ga, gb = distance_gradients(np.zeros(3), np.zeros(3))
        self.assertEqual(distance, 0)
        self.assertTrue(np.isfinite(ga).all() and np.isfinite(gb).all())

    def test_transform_roundtrip(self):
        values = np.array([0., .2, 15., 100.])
        np.testing.assert_allclose(inverse_target(transform_target(values, 'log10'), 'log10'), values)

    def test_reader_handles_no_final_newline_and_zero_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'data.txt'
            path.write_text('1::2::3::0\n2::3::4::12.34')
            rows, metadata = read_observations(path)
            self.assertEqual(rows[0, 3], 12.34)
            self.assertEqual(metadata['zero_rows'], 1)
            self.assertEqual(len(read_observations(path, keep_zero=True)[0]), 2)

    def test_split_overlap_raises(self):
        a = np.array([[0., 0., 0., 5.]])
        with self.assertRaises(ValueError):
            check_splits({'train': a, 'test': a.copy()})

    def test_duplicate_coordinates_raise(self):
        a = np.array([[0., 0., 0., 5.], [0., 0., 0., 6.]])
        with self.assertRaises(ValueError):
            check_splits({'train': a})

    def test_zero_safe_metrics(self):
        measured = regression_metrics(np.array([0., 2.]), np.array([1., 3.]))
        self.assertEqual(measured['mape_percent'], 50.)
        self.assertEqual(measured['mape_excluded_zero_targets'], 1)
        self.assertEqual(measured['rmse'], 1)

    def test_sensor_mean_falls_back_for_unseen_sensor(self):
        train = np.array([[0, 0, 0, 2.], [0, 1, 0, 4.]])
        heldout = np.array([[0, 2, 0, 8.], [1, 2, 0, 9.]])
        baseline = baseline_predictions(train, heldout, 2)
        np.testing.assert_allclose(baseline['sensor_mean'], [3., 3.])

    def test_small_step_reduces_training_loss(self):
        s, d, t = np.array([[.12, -.06]]), np.array([[-.09, .1]]), np.array([[.02, .04]])
        def loss():
            total = distance_gradients(s[0], d[0])[0] + distance_gradients(s[0], t[0])[0] + distance_gradients(d[0], t[0])[0]
            return (1.4 - 2 * (1 - np.tanh(total))) ** 2
        before = loss()
        train_epoch(s, d, t, np.array([[0., 0., 0., 1.4]]), 2., 1e-5, np.array([0]))
        self.assertLess(loss(), before)

    def test_fitting_api_has_no_test_values(self):
        self.assertEqual(list(inspect.signature(fit).parameters), ['train', 'valid', 'shape', 'config'])


if __name__ == '__main__':
    unittest.main()
