import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import torch
from torch.nn import functional as F
from full_path_pdnn_ffn import FullPathPBitFFNConfig, build_full_path_student, load_full_path_checkpoint
from multibit_input_ffn import quantize_codes, decode_codes, bitplanes


def model(bits=4, mode="stochastic"):
    result = build_full_path_student(FullPathPBitFFNConfig(
        input_size=3, hidden_sizes=(7,), output_size=2, coding="binary",
        learnable_encoding=True, learnable_thresholds=False,
        input_encoding=mode, input_bits=bits))
    result.input_bound.copy_(torch.tensor([1., 2., 3.]))
    return result


class MultiBitTests(unittest.TestCase):
    def test_grid_endpoints_and_bit_reconstruction(self):
        for bits in (1, 2, 4):
            codes = torch.arange(2**bits)
            reconstructed = sum(b * 2**k for k, b in enumerate(bitplanes(codes, bits, torch.float32)))
            torch.testing.assert_close(reconstructed, codes.float())
            decoded = decode_codes(codes, torch.tensor(2.), bits)
            self.assertEqual(decoded[0].item(), -2.)
            self.assertEqual(decoded[-1].item(), 2.)
            torch.testing.assert_close(quantize_codes(decoded, torch.tensor(2.), bits, False), codes)

    def test_stochastic_mean_variance_and_clipping(self):
        torch.manual_seed(42)
        x = torch.tensor([-.73, .33, 2.5]).expand(80000, 3)
        bound = torch.tensor([1., 1., 2.])
        for bits in (1, 2, 4):
            decoded = decode_codes(quantize_codes(x, bound, bits, True), bound, bits)
            torch.testing.assert_close(decoded.mean(0), torch.tensor([-.73, .33, 2.]), atol=.015, rtol=0)
            delta = 2*bound/(2**bits-1)
            self.assertTrue(bool((decoded.var(0) <= delta.square()/4 + .005).all()))

    def test_bitplane_projection_and_weight_gradient_match_decoded_matmul(self):
        student = model(4, "deterministic")
        x = torch.randn(10, 3)
        projection = student.projections[0]
        actual = student.quantized_first_field(x)
        decoded = decode_codes(quantize_codes(x, student.input_bound, 4, False), student.input_bound, 4)
        expected = F.linear(decoded, projection.weight, projection.bias)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
        grad1 = torch.autograd.grad(actual.square().sum(), projection.weight, retain_graph=True)[0]
        grad2 = torch.autograd.grad(expected.square().sum(), projection.weight)[0]
        torch.testing.assert_close(grad1, grad2, atol=1e-4, rtol=1e-4)

    def test_binary_matrix_inputs_parameters_and_final_average(self):
        counts = []
        for bits in (1,2,4):
            student=model(bits)
            counts.append(sum(p.numel() for p in student.parameters()))
            x=torch.randn(8,3)
            calls=[]
            linear=F.linear
            def capture(state, weight, bias=None):
                calls.append(state.detach())
                return linear(state,weight,bias)
            student.set_sample_count(4)
            torch.manual_seed(3)
            with patch('torch.nn.functional.linear',side_effect=capture):
                actual=student(x)
            self.assertEqual(len(calls),4*(bits+1))
            self.assertTrue(all(bool(((v==0)|(v==1)).all()) for v in calls))
            torch.manual_seed(3)
            expected=sum(student.sampled_path_forward(x) for _ in range(4))/4
            torch.testing.assert_close(actual,expected)
            actual.square().mean().backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in student.parameters()))
        self.assertEqual(len(set(counts)),1)

    def test_checkpoint_roundtrip_and_input_gradient(self):
        student=model()
        x=torch.randn(5,3,requires_grad=True)
        student.set_sample_count(1)
        student(x).square().mean().backward()
        self.assertTrue(bool(torch.isfinite(x.grad).all()))
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'student.pt'
            torch.save({'student_config':student.checkpoint_config(),'student_state_dict':student.state_dict()},path)
            restored,_=load_full_path_checkpoint(str(path),device='cpu')
            torch.testing.assert_close(student.input_bound,restored.input_bound)
            torch.testing.assert_close(student.mean_field_forward(x),restored.mean_field_forward(x))


if __name__=='__main__':
    unittest.main()
