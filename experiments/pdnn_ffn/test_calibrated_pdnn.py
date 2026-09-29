import tempfile
import unittest
from pathlib import Path

import torch

from calibrated_pdnn_ffn import CalibratedPBitFFN, TYPE, load_checkpoint, validate_training_mode
from full_path_pdnn_ffn import FullPathPBitFFN, FullPathPBitFFNConfig


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3)
        self.base = FullPathPBitFFN(FullPathPBitFFNConfig(5, (9,), 5, .25, 1.))
        self.student = CalibratedPBitFFN(self.base.config)
        self.student.projections.load_state_dict(self.base.projections.state_dict())

    def test_identity_for_both_precisions_and_sampling(self):
        for dtype in (torch.float32, torch.bfloat16):
            self.base.to(dtype=dtype); self.student.to(dtype=dtype)
            x = torch.randn(2, 3, 5, dtype=dtype)
            for n in (0, 4):
                self.base.set_sample_count(n); self.student.set_sample_count(n)
                torch.manual_seed(19); a = self.base(x)
                torch.manual_seed(19); b = self.student(x)
                self.assertTrue(torch.equal(a, b))

    def test_binary_inputs_and_readout_average(self):
        self.student.set_sample_count(4)
        inputs, outputs = [], []
        handles = [p.register_forward_pre_hook(lambda m, a: inputs.append(a[0].detach()))
                   for p in self.student.projections]
        handles.append(self.student.projections[-1].register_forward_hook(
            lambda m, a, y: outputs.append(y.detach())))
        y = self.student(torch.randn(2, 5))
        self.assertEqual(len(inputs), 8)
        self.assertTrue(all(torch.all((x == -1) | (x == 1)) for x in inputs))
        self.assertTrue(torch.equal(y, sum(outputs) / 4))
        for h in handles: h.remove()

    def test_gradients_reach_fields_and_weights(self):
        self.student.set_sample_count(4)
        self.student(torch.randn(12, 5) * .2).square().mean().backward()
        for name, p in self.student.named_parameters():
            self.assertIsNotNone(p.grad, name)
            self.assertTrue(torch.isfinite(p.grad).all(), name)
            self.assertGreater(p.grad.abs().sum().item(), 0, name)

    def test_checkpoint_roundtrip_and_rng(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            path = Path(directory) / 'student.pt'
            for model, kind in ((self.base, 'full_path_bipolar_pdnn_v1'), (self.student, TYPE)):
                torch.save({'student_config': model.checkpoint_config(), 'student_type': kind,
                            'student_state_dict': model.state_dict()}, path)
                rng = torch.get_rng_state().clone()
                loaded, _ = load_checkpoint(path, device='cpu', enable_calibration=True)
                self.assertTrue(torch.equal(rng, torch.get_rng_state()))
                x = torch.randn(3, 5)
                self.assertTrue(torch.equal(model(x), loaded(x)))

    def test_reject_mixed_encoding_variants(self):
        self.base.config.coding = 'binary'
        with self.assertRaises(ValueError):
            CalibratedPBitFFN(self.base.config)
        self.base.config.coding = 'bipolar'
        self.base.config.learnable_encoding = True
        with self.assertRaises(ValueError):
            CalibratedPBitFFN(self.base.config)

    def test_training_mode_cannot_mislabel_control(self):
        validate_training_mode(self.base, False)
        validate_training_mode(self.student, True)
        with self.assertRaises(ValueError):
            validate_training_mode(self.student, False)


if __name__ == '__main__':
    unittest.main()
