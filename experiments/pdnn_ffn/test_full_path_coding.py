"""Small CPU checks for coding, complete paths, gradients and old checkpoints."""
import tempfile
import unittest
from pathlib import Path

import torch

from full_path_pdnn_ffn import FullPathPBitFFN, FullPathPBitFFNConfig, load_full_path_checkpoint


class CodingTests(unittest.TestCase):
    def model(self, coding="binary"):
        return FullPathPBitFFN(FullPathPBitFFNConfig(
            input_size=3, hidden_sizes=(5, 4), output_size=2, coding=coding,
        )).double()

    def test_every_matrix_receives_bits_and_only_outputs_are_averaged(self):
        for coding, allowed in (("binary", (0, 1)), ("bipolar", (-1, 1))):
            model = self.model(coding).eval()
            states, readouts = [], []
            handles = [p.register_forward_pre_hook(lambda m, a: states.append(a[0].detach()))
                       for p in model.projections]
            handles.append(model.projections[-1].register_forward_hook(
                lambda m, a, out: readouts.append(out.detach())))
            model.set_sample_count(4)
            with torch.no_grad():
                out = model(torch.randn(2, 3, dtype=torch.float64))
            self.assertEqual(len(states), 12)
            self.assertEqual(len(readouts), 4)
            for state in states:
                self.assertTrue(((state == allowed[0]) | (state == allowed[1])).all())
            torch.testing.assert_close(out, sum(readouts) / 4)
            for handle in handles:
                handle.remove()

    def test_sigmoid_sampling_and_ste(self):
        model = self.model()
        fields = torch.tensor([-1., 0., 1.], dtype=torch.float64, requires_grad=True)
        mean = model.hidden_mean(fields)
        torch.testing.assert_close(mean, torch.sigmoid(fields))
        model._sample_state(mean).sum().backward()
        torch.testing.assert_close(fields.grad, mean.detach() * (1 - mean.detach()))
        torch.manual_seed(23)
        with torch.no_grad():
            draws = model._sample_state(mean.detach().expand(100000, 3))
        torch.testing.assert_close(draws.mean(0), mean.detach(), atol=0.005, rtol=0)

    def test_affine_recoding_equivalence(self):
        # s=2b-1; sigmoid uses half the tanh temperature. Every affine layer
        # must also become W'=2W, bias'=bias-W*1 for equivalent complete paths.
        bipolar, binary = self.model("bipolar"), self.model("binary")
        binary.config.input_temperature = bipolar.config.input_temperature / 2
        binary.config.hidden_temperature = bipolar.config.hidden_temperature / 2
        with torch.no_grad():
            for src, dst in zip(bipolar.projections, binary.projections):
                dst.weight.copy_(2 * src.weight)
                dst.bias.copy_(src.bias - src.weight.sum(dim=1))
        x = torch.randn(20, 3, dtype=torch.float64)
        for count in (0, 1, 4):
            bipolar.set_sample_count(count)
            binary.set_sample_count(count)
            torch.manual_seed(41)
            a = bipolar(x)
            torch.manual_seed(41)
            b = binary(x)
            torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-12)

    def test_legacy_checkpoint_defaults_to_bipolar(self):
        model = self.model("bipolar").float()
        config = model.checkpoint_config()
        del config["coding"]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "legacy.pt"
            torch.save({"student_config": config, "student_state_dict": model.state_dict()}, path)
            loaded, _ = load_full_path_checkpoint(str(path), device="cpu")
        self.assertEqual(loaded.config.coding, "bipolar")
        x = torch.randn(2, 3)
        torch.manual_seed(4)
        model.set_sample_count(4)
        expected = model(x)
        torch.manual_seed(4)
        loaded.set_sample_count(4)
        torch.testing.assert_close(loaded(x), expected, atol=0, rtol=0)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
