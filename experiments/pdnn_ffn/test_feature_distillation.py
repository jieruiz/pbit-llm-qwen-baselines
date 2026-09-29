import unittest

import torch
from torch import nn

from feature_distillation import FFNFeatureDistillation, normalized_feature_loss


class FeatureTests(unittest.TestCase):
    def test_normalized_loss_and_teacher_detach(self):
        teacher = torch.tensor([1.0, 3.0], requires_grad=True)
        student = torch.tensor([2.0, 1.0], requires_grad=True)
        loss = normalized_feature_loss(student, teacher)
        self.assertAlmostEqual(loss.item(), 0.5)
        loss.backward()
        torch.testing.assert_close(student.grad, torch.tensor([0.2, -0.4]))
        self.assertIsNone(teacher.grad)
        torch.testing.assert_close(loss, normalized_feature_loss(student * 5, teacher * 5))

    def test_hooks_preserve_student_gradients(self):
        teacher = nn.Sequential(nn.Linear(3, 3), nn.Linear(3, 3))
        student = nn.Sequential(nn.Linear(3, 3), nn.Linear(3, 3))
        capture = FFNFeatureDistillation(dict(enumerate(teacher)), dict(enumerate(student)))
        x = torch.randn(4, 3)
        with torch.no_grad():
            teacher(x)
        student(x)
        capture.loss().backward()
        for p in student.parameters():
            self.assertIsNotNone(p.grad)
            self.assertTrue(torch.isfinite(p.grad).all())
        self.assertTrue(all(p.grad is None for p in teacher.parameters()))
        capture.clear()
        with self.assertRaises(RuntimeError):
            capture.loss()
        capture.close()
        teacher(x)
        student(x)
        self.assertEqual(capture.student, {})

    def test_zero_weight_is_no_extra_gradient(self):
        s = torch.tensor([0.5, 1.0], requires_grad=True)
        t = torch.tensor([0.1, 0.2])
        base = s.square().sum()
        grad = torch.autograd.grad(base + 0.0 * normalized_feature_loss(s, t), s)[0]
        torch.testing.assert_close(grad, 2 * s)

    def test_shape_and_layer_guards(self):
        with self.assertRaises(ValueError):
            normalized_feature_loss(torch.zeros(2), torch.zeros(3))
        with self.assertRaises(ValueError):
            FFNFeatureDistillation({}, {})
        self.assertTrue(torch.isfinite(normalized_feature_loss(torch.ones(2), torch.zeros(2))))


if __name__ == "__main__":
    unittest.main()
