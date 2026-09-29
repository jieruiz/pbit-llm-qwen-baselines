import unittest
import torch
from full_path_pdnn_ffn import FullPathPBitFFN, FullPathPBitFFNConfig
from run_depth_comparison import SHAPES, parameter_count


class DepthTests(unittest.TestCase):
    def test_parameter_budget(self):
        baseline = parameter_count(SHAPES[2])
        self.assertEqual(baseline, 8722048)
        for depth, hidden in SHAPES.items():
            model = FullPathPBitFFN(FullPathPBitFFNConfig(hidden_sizes=hidden, coding='binary'))
            count = sum(p.numel() for p in model.parameters())
            self.assertEqual(count, parameter_count(hidden))
            self.assertEqual(len(model.projections), depth)
            self.assertLess(abs(count / baseline - 1), 0.002)

    def test_binary_boundaries_and_gradients_at_all_depths(self):
        for depth in (2, 3, 4):
            model = FullPathPBitFFN(FullPathPBitFFNConfig(
                input_size=3, hidden_sizes=(8,) * (depth-1), output_size=3,
                coding='binary', input_temperature=0.125, hidden_temperature=0.5))
            states, outputs = [], []
            handles = [p.register_forward_pre_hook(lambda m, a: states.append(a[0].detach()))
                       for p in model.projections]
            handles.append(model.projections[-1].register_forward_hook(
                lambda m, a, out: outputs.append(out.detach())))
            model.set_sample_count(4)
            output = model(torch.randn(4, 3))
            self.assertEqual(len(states), depth * 4)
            self.assertEqual(len(outputs), 4)
            for state in states:
                self.assertTrue(((state == 0) | (state == 1)).all())
            torch.testing.assert_close(output.detach(), sum(outputs) / 4)
            output.square().sum().backward()
            for p in model.parameters():
                self.assertIsNotNone(p.grad)
                self.assertTrue(torch.isfinite(p.grad).all())
            for handle in handles:
                handle.remove()


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
