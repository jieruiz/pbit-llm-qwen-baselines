"""Integration checks for a chain of AND FFNs and joint checkpoint metadata."""
import argparse
import tempfile
from pathlib import Path
import unittest
import torch
from torch import nn
from multithreshold_and_ffn import AndConfig,MultiThresholdAndFFN,load_checkpoint
from train_joint_multithreshold_and import load_training_student
from train_joint_full_path_distillation import save_students


class JointAndTests(unittest.TestCase):
    def test_joint_gradients_and_saved_layer_identity(self):
        torch.manual_seed(371)
        modules={}
        sources={}
        for layer in (1,4):
            m=MultiThresholdAndFFN(AndConfig(3,4,3,2))
            with torch.no_grad():
                m.gate_bank.coefficient.fill_(.5)
                m.value_bank.coefficient.fill_(.5)
                m.value_bank.offset.fill_(-.5)
            m.binary_readout=False
            m.set_sample_count(4)
            modules[layer]=m
            sources[layer]={"student_type":"multithreshold_and_v1","training_args":{"layer":layer}}
        x=torch.randn(2,3)
        output=x
        for m in modules.values():
            output=output+m(output)
        output.square().mean().backward()
        for m in modules.values():
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()))
            self.assertGreater(m.gate_projection.weight.grad.abs().sum().item(),0)
        with tempfile.TemporaryDirectory() as tmp:
            save_students(Path(tmp),"best",modules,sources,argparse.Namespace(checkpoints=[]),7,{"perplexity":12.})
            for layer,m in modules.items():
                path=Path(tmp)/f"best_layer{layer}.pt"
                restored,payload=load_checkpoint(path,"cpu")
                self.assertEqual(payload["training_args"]["layer"],layer)
                self.assertEqual(payload["phase"],"joint_end_to_end_distillation")
                self.assertEqual(payload["step"],7)
                m.set_sample_count(0)
                torch.testing.assert_close(restored(x),m(x))
                training_student,_=load_training_student(path,"cpu")
                self.assertFalse(training_student.binary_readout)


if __name__=="__main__":
    unittest.main()
