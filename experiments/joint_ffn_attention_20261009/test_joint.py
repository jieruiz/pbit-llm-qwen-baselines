import json,unittest
from unittest.mock import patch
import torch
from sampling import ising_counts,iid_counts
class Tests(unittest.TestCase):
    def test_ising_distribution_without_softmax(self):
        g=torch.Generator(device='cuda').manual_seed(4)
        z=torch.tensor([0.,-1.,-2.,-torch.inf],device='cuda').expand(1500,-1)
        expected=z[0].softmax(-1)
        with patch.object(torch.Tensor,'softmax',side_effect=AssertionError('softmax must not drive Ising')):
            c,_=ising_counts(z,128,16,4,g)
        self.assertTrue(torch.equal(c.sum(-1),torch.full((1500,),128.,device='cuda')))
        torch.testing.assert_close(c.mean(0)/128,expected,atol=.007,rtol=0)
        self.assertTrue((c[:,-1]==0).all())
    def test_single_valid_category(self):
        z=torch.tensor([[-torch.inf,2.,-torch.inf]],device='cuda')
        c,_=ising_counts(z,32,generator=torch.Generator(device='cuda').manual_seed(2))
        torch.testing.assert_close(c,torch.tensor([[0.,32.,0.]],device='cuda'))
    def test_iid_mask_and_counts(self):
        z=torch.tensor([[1.,2.,-torch.inf]],device='cuda').expand(1000,-1)
        c=iid_counts(z,64,torch.Generator(device='cuda').manual_seed(3))
        self.assertTrue((c[:,-1]==0).all());self.assertTrue((c.sum(-1)==64).all())
        torch.testing.assert_close(c.mean(0)/64,z[0].softmax(-1),atol=.008,rtol=0)
if __name__=='__main__':unittest.main(verbosity=2)
