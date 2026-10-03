import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F
from full_path_pdnn_ffn import FullPathPBitFFNConfig, build_full_path_student, load_full_path_checkpoint


def make_student():
    return build_full_path_student(FullPathPBitFFNConfig(
        input_size=3, hidden_sizes=(5,), output_size=2, coding="binary",
        learnable_encoding=True, learnable_thresholds=False, input_encoding="continuous_raw"))


class ContinuousInputTests(unittest.TestCase):
    def test_raw_input_hidden_binary_and_final_path_average(self):
        student = make_student()
        x = torch.tensor([[4.25, -8.5, .37], [-2.7, .53, 12.3]])
        calls = []
        linear = F.linear
        def capture(state, weight, bias=None):
            calls.append(state.detach().clone())
            return linear(state, weight, bias)
        student.set_sample_count(4)
        torch.manual_seed(15)
        with patch("torch.nn.functional.linear", side_effect=capture):
            actual = student(x)
        self.assertEqual(len(calls), 8)
        for index in range(0, 8, 2):
            torch.testing.assert_close(calls[index], x)
            self.assertTrue(bool(((calls[index+1] == 0) | (calls[index+1] == 1)).all()))
        torch.manual_seed(15)
        expected = sum(student.sampled_path_forward(x) for _ in range(4)) / 4
        torch.testing.assert_close(actual, expected)

    def test_upstream_gradient_matches_hidden_probability_ste(self):
        student = make_student()
        x = torch.tensor([[3.2, -2.1, .37]], requires_grad=True)
        student.set_sample_count(4)
        actual_grad = torch.autograd.grad(student(x).sum(), x)[0]
        p = torch.sigmoid(student.projections[0](x) / student.effective_hidden_temperature(0))
        expected_grad = torch.autograd.grad(student.projections[-1](p).sum(), x)[0]
        torch.testing.assert_close(actual_grad, expected_grad)
        self.assertTrue(bool((actual_grad.abs() > 0).all()))

    def test_checkpoint_and_exact_single_hidden_layer_expectation(self):
        student = make_student()
        self.assertFalse(hasattr(student, "input_bound"))
        self.assertFalse(hasattr(student, "input_temperature_raw"))
        x = torch.randn(2, 3)
        p = torch.sigmoid(student.projections[0](x) / student.effective_hidden_temperature(0))
        # Enumerate all hidden states: the mean-field is exact for this one
        # stochastic hidden layer followed by a linear readout, at fixed input.
        expectation = torch.zeros(2, 2)
        for code in range(32):
            state = torch.tensor([(code >> bit) & 1 for bit in range(5)]).float()
            probability = torch.where(state.bool(), p, 1-p).prod(-1)
            expectation = expectation + probability[:, None] * student.projections[-1](state)
        torch.testing.assert_close(student.mean_field_forward(x), expectation)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            torch.save({"student_config": student.checkpoint_config(),
                        "student_state_dict": student.state_dict()}, path)
            restored, _ = load_full_path_checkpoint(str(path), device="cpu")
            torch.testing.assert_close(restored.mean_field_forward(x), expectation)


if __name__ == "__main__":
    unittest.main()
