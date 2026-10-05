import unittest
import torch
from multithreshold_and_ffn import AndConfig, MultiThresholdAndFFN
from and_variance_loss import normalized_output_variance, VarianceCollector


def make():
    torch.manual_seed(73)
    module = MultiThresholdAndFFN(AndConfig(3, 2, 3, 4))
    with torch.no_grad():
        for bank in (module.gate_bank, module.value_bank):
            bank.coefficient.normal_(0, .3)
            bank.offset.normal_(0, .2)
            bank.threshold.normal_(0, .4)
    module.set_sample_count(4)
    module.binary_readout = False
    return module


class VarianceTests(unittest.TestCase):
    def test_four_path_variance_against_empirical_averages(self):
        m = make()
        x = torch.tensor([[.3, -.7, .2]])
        with torch.no_grad():
            p, q = m.probabilities(x)
            mean, variance = m.conditional_moments(x)
            p, q = p.expand(80000, -1, -1), q.expand(80000, -1, -1)
            paths = m.factorized_readout(m.sample(p), m.sample(q))
            averages = paths.reshape(20000, 4, 3).mean(1)
            torch.testing.assert_close(averages.mean(0), mean[0], atol=.003, rtol=.03)
            torch.testing.assert_close(averages.var(0), variance[0]/4, atol=.001, rtol=.05)
            torch.testing.assert_close(normalized_output_variance(m, x, 2), variance.mean()/8)

    def test_variance_derivative_with_finite_difference(self):
        m = make()
        x = torch.randn(2, 3)
        normalized_output_variance(m, x, .7).backward()
        for parameter in (m.gate_projection.weight, m.value_projection.weight,
                          m.gate_bank.threshold, m.value_bank.coefficient, m.readout.weight):
            index = (0,) * parameter.ndim
            analytic = parameter.grad[index].item()
            with torch.no_grad():
                old = parameter[index].item()
                parameter[index] = old + .002
                high = normalized_output_variance(m, x, .7).item()
                parameter[index] = old - .002
                low = normalized_output_variance(m, x, .7).item()
                parameter[index] = old
            self.assertAlmostEqual(analytic, (high-low)/.004, delta=2e-5)

    def test_hook_preserves_rng_and_forward(self):
        m = make()
        x = torch.randn(2, 3)
        torch.manual_seed(91)
        plain = m(x)
        rng = torch.get_rng_state().clone()
        collector = VarianceCollector({12:m}, {'12':.7})
        torch.manual_seed(91)
        collector.begin()
        captured = m(x)
        penalty = collector.loss()
        self.assertTrue(torch.equal(plain, captured))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertGreater(penalty.item(), 0)
        collector.close()

    def test_layers_are_equally_weighted_and_input_gradient_flows(self):
        a, b = make(), make()
        x = torch.randn(2, 3, requires_grad=True)
        collector = VarianceCollector({1:a,4:b}, {'1':.7,'4':1.3})
        collector.begin()
        a(x); b(x)
        penalty = collector.loss()
        expected = (normalized_output_variance(a,x,.7)+normalized_output_variance(b,x,1.3))/2
        torch.testing.assert_close(penalty, expected)
        penalty.backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertGreater(x.grad.abs().sum().item(), 0)
        collector.close()

    def test_reject_invalid_scales_and_budget(self):
        m = make()
        for scale in (0,-1,float('nan'),float('inf')):
            with self.assertRaises(ValueError):
                normalized_output_variance(m,torch.randn(2,3),scale)
        with self.assertRaises(ValueError):
            normalized_output_variance(m,torch.randn(2,3),1,16)
        with self.assertRaises(ValueError):
            VarianceCollector({1:m},{'2':1})


if __name__ == '__main__':
    unittest.main()
