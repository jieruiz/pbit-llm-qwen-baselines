import unittest
from types import SimpleNamespace
import torch
from transformers.models.qwen2.modeling_qwen2 import repeat_kv
from attention import compact_attention,Controller
from metrics import cache,build

class Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)
    def test_compact_exact_with_ragged_candidates(self):
        q=torch.randn(2,14,1,64,device='cuda',dtype=torch.bfloat16)
        k=torch.randn(2,2,137,64,device='cuda',dtype=torch.bfloat16);v=torch.randn_like(k)
        mask=torch.rand(2,14,3,device='cuda')>.5;mask[:,:,0]=True
        out,st=compact_attention(q,k,v,mask)
        z=q.float()@repeat_kv(k,7).float().transpose(-1,-2)/8
        valid=mask.repeat_interleave(64,-1)[...,:137].unsqueeze(-2)
        expected=(z.masked_fill(~valid,-torch.inf).softmax(-1)@repeat_kv(v,7).float()).to(q.dtype)
        torch.testing.assert_close(out,expected,atol=.004,rtol=.01)
        self.assertEqual(int(st['selected']),int(valid.sum()))
    def test_sampled_pv_converges_and_excludes_dropped_blocks(self):
        q=torch.randn(64,2,1,64,device='cuda');k=torch.randn(64,1,128,64,device='cuda')
        v=torch.randn_like(k);v[:,:,64:]=1e6
        mask=torch.tensor([True,False],device='cuda').expand(64,2,2)
        expected,_=compact_attention(q,k,v,mask)
        out,_=compact_attention(q,k,v,mask,4096,torch.Generator(device='cuda').manual_seed(2))
        self.assertLess((out-expected).square().mean().item(),.001)
        self.assertLess(out.abs().max().item(),2.)
    def test_summary_incremental(self):
        a=SimpleNamespace();k=torch.randn(2,2,63,64,device='cuda')
        for _ in range(5):
            actual=cache(a,k,'max4');expected=build(k,4,False)
            torch.testing.assert_close(actual[0],expected[0])
            k=torch.cat((k,torch.randn(2,2,1,64,device='cuda')),2)
    def test_all_mandatory_short_prefix(self):
        c=Controller('screen_exact',2048,0,0)
        mask=c.select(SimpleNamespace(),torch.randn(1,14,1,64,device='cuda'),torch.randn(1,2,100,64,device='cuda'))
        self.assertTrue(bool(mask.all()))

if __name__=='__main__':unittest.main(verbosity=2)
