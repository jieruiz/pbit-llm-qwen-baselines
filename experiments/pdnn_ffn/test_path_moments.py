import unittest
import torch
from measure_path_moments import path_moments


class PathMomentTests(unittest.TestCase):
    def test_bias_variance_identity_and_native_averages(self):
        torch.manual_seed(23)
        paths = torch.randn(32, 2, 5).to(torch.bfloat16)
        target = torch.randn(2, 5)
        result = path_moments(paths, target)
        # Exact finite-sample decomposition; distinguishes R from R-1 correction.
        self.assertAlmostEqual(result["empirical_single_path_mse"], result["finite_sample_mean_mse"] + 31 / 32 * result["single_path_variance_mse"], places=5)
        for count in (4, 16):
            outputs = []
            for offset in range(0, 32, count):
                total = paths[offset].clone()
                for index in range(1, count):
                    total = total + paths[offset + index]
                outputs.append(total / count)
            expected = (torch.stack(outputs).float() - target).square().mean().item()
            self.assertAlmostEqual(result[f"empirical_n{count}_mse"], expected, places=6)

    def test_deterministic_paths(self):
        result = path_moments(torch.full((16, 2, 3), 2.), torch.ones(2, 3))
        self.assertEqual(result["single_path_variance_mse"], 0)
        self.assertEqual(result["corrected_bias_mse"], 1)
        self.assertEqual(result["predicted_n4_mse"], 1)


if __name__ == "__main__":
    unittest.main()
