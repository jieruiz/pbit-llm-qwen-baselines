"""Architectural invariants for binary gated FFNs; run directly with Python."""
from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from full_path_pdnn_ffn import (
    FullPathPBitFFNConfig, GatedPBitFFN, build_full_path_student,
    load_full_path_checkpoint, student_type_name,
)


def make_model():
    return build_full_path_student(FullPathPBitFFNConfig(
        input_size=5, hidden_sizes=(7,), output_size=3, coding="binary",
        input_temperature=0.8, hidden_temperature=0.8,
        learnable_encoding=True, learnable_thresholds=False,
        architecture="gated_dual_rail",
    ))


class GatedTests(unittest.TestCase):
    def test_binary_inputs_shared_readout_and_gradients(self):
        torch.manual_seed(31)
        model = make_model()
        model.set_sample_count(4)
        calls = []
        linear = F.linear

        def capture(state, weight, bias=None):
            calls.append((state.detach().clone(), weight, bias))
            return linear(state, weight, bias)

        with patch("torch.nn.functional.linear", side_effect=capture):
            output = model(torch.randn(64, 5))
            output.square().mean().backward()
        self.assertEqual(len(calls), 16)
        for state, _, _ in calls:
            self.assertTrue(bool(((state == 0) | (state == 1)).all()))
        for offset in range(0, len(calls), 4):
            gate, value, positive, negative = calls[offset:offset + 4]
            torch.testing.assert_close(gate[0], value[0])
            self.assertIs(positive[1], negative[1])
            self.assertIsNotNone(positive[2])
            self.assertIsNone(negative[2])
            self.assertTrue(bool((positive[0] * negative[0] == 0).all()))
        for name, parameter in model.named_parameters():
            self.assertNotIn("threshold", name)
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(bool(torch.isfinite(parameter.grad).all()), name)
            self.assertGreater(parameter.grad.abs().sum().item(), 0, name)
        for temperature in (model.effective_input_temperature(), model.effective_hidden_temperature(0), model.effective_value_temperature()):
            self.assertGreater(temperature.item(), model.config.minimum_temperature)

    def test_truth_table_signed_equivalence_and_bias_once(self):
        model = make_model()
        with torch.no_grad():
            model.projections[-1].bias.fill_(0.37)
        gate = torch.tensor([0., 0., 1., 1.])[:, None].expand(4, 7)
        value = torch.tensor([0., 1., 0., 1.])[:, None].expand(4, 7)
        positive, negative = gate * value, gate * (1 - value)
        actual = model.dual_rail_readout(positive, negative)
        expected = F.linear(gate * (2 * value - 1), model.projections[-1].weight, model.projections[-1].bias)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual[:2], torch.full((2, 3), 0.37))

    def test_only_final_outputs_averaged(self):
        model = make_model()
        inputs = torch.randn(4, 5)
        model.set_sample_count(4)
        torch.manual_seed(4)
        expected = sum(model.sampled_path_forward(inputs) for _ in range(4)) / 4
        torch.manual_seed(4)
        torch.testing.assert_close(model(inputs), expected)

    def test_roundtrip_and_legacy_defaults(self):
        inputs = torch.randn(4, 5)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "student.pt"
            for model in (make_model(), build_full_path_student(FullPathPBitFFNConfig(input_size=5, hidden_sizes=(7,), output_size=3, learnable_encoding=True))):
                config = asdict(model.config)
                if not isinstance(model, GatedPBitFFN):
                    config.pop("architecture")
                    config.pop("learnable_thresholds")
                torch.save({"student_type": student_type_name(model), "student_config": config, "student_state_dict": model.state_dict()}, path)
                restored, _ = load_full_path_checkpoint(str(path), device="cpu")
                torch.testing.assert_close(restored(inputs), model(inputs))
                self.assertEqual(type(restored), type(model))
                model.set_sample_count(4)
                restored.set_sample_count(4)
                torch.manual_seed(19)
                expected = model(inputs)
                torch.manual_seed(19)
                torch.testing.assert_close(restored(inputs), expected)


if __name__ == "__main__":
    unittest.main()
