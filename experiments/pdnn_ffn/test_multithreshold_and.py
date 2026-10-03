import copy
import itertools
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import torch
from torch.nn import functional as F
from multithreshold_and_ffn import AndConfig, MultiThresholdAndFFN, load_checkpoint


def make_model(bits=2):
    torch.manual_seed(73)
    m = MultiThresholdAndFFN(AndConfig(3, 2, 2, bits))
    for bank in (m.gate_bank, m.value_bank):
        with torch.no_grad():
            bank.coefficient.normal_()
            bank.offset.normal_()
            bank.threshold.normal_()
            bank.scale.copy_(torch.tensor([.7, 1.3]))
    return m


class AndTests(unittest.TestCase):
    def test_binary_readout_and_factorization(self):
        m = make_model()
        gate, value = torch.randint(2, (5,2,2)).float(), torch.randint(2, (5,2,2)).float()
        seen = []
        linear = F.linear
        def audit(state, weight, bias=None):
            if state.ndim > 1:
                seen.append(state)
                self.assertTrue(bool(((state == 0) | (state == 1)).all()))
            return linear(state, weight, bias)
        with patch("torch.nn.functional.linear", side_effect=audit):
            expanded = m.expanded_readout(gate, value)
        self.assertEqual(len(seen), 8)
        torch.testing.assert_close(expanded, m.factorized_readout(gate, value), atol=2e-6, rtol=2e-6)

    def test_exact_moments_by_enumeration(self):
        m = make_model()
        x = torch.tensor([[.3,-.8,.4]])
        p,q = m.probabilities(x)
        expectation = torch.zeros(1,2)
        second = torch.zeros(1,2)
        # Enumerate all 2^(2 channels * 2 banks * 2 bits) states.
        for bits in itertools.product((0.,1.), repeat=8):
            gate = torch.tensor(bits[:4]).reshape(1,2,2)
            value = torch.tensor(bits[4:]).reshape(1,2,2)
            probability = torch.where(gate.bool(), p, 1-p).prod() * torch.where(value.bool(), q, 1-q).prod()
            y = m.expanded_readout(gate, value)
            expectation += probability*y
            second += probability*y.square()
        mean, variance = m.conditional_moments(x)
        torch.testing.assert_close(mean, expectation, atol=3e-6, rtol=3e-6)
        torch.testing.assert_close(variance, second-expectation.square(), atol=3e-6, rtol=3e-6)

    def test_factorized_and_expanded_training_gradients(self):
        first, second = make_model(), make_model()
        x = torch.tensor([[.4,.7,-.3]])
        for m, expanded in ((first, False),(second,True)):
            p,q = m.probabilities(x)
            torch.manual_seed(9)
            b,c = m.sample(p),m.sample(q)
            y = m.expanded_readout(b,c) if expanded else m.factorized_readout(b,c)
            y.square().sum().backward()
        for a,b in zip(first.parameters(),second.parameters()):
            torch.testing.assert_close(a.grad, b.grad, atol=1e-5, rtol=1e-5)

    def test_final_average_rng_and_checkpoint(self):
        m = make_model()
        x = torch.randn(3,3)
        m.set_sample_count(4)
        torch.manual_seed(19)
        result = m(x)
        p,q = m.probabilities(x)
        torch.manual_seed(19)
        expected = sum(m.expanded_readout(m.sample(p),m.sample(q)) for _ in range(4))/4
        torch.testing.assert_close(result,expected)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"model.pt"
            torch.save({"student_type":"multithreshold_and_v1", "student_config":m.checkpoint_config(),
                        "student_state_dict":m.state_dict()},path)
            restored,_ = load_checkpoint(path,"cpu")
            restored.set_sample_count(4)
            torch.manual_seed(19)
            torch.testing.assert_close(restored(x),result)


if __name__ == "__main__":
    unittest.main()
