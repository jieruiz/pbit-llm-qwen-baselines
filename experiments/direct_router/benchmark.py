"""Serial warmed-up GPU timings, separate from concurrent accuracy runs."""
import json
import os
from pathlib import Path
from types import SimpleNamespace
import torch
import triton
from torch.nn import functional as F
from transformers.models.qwen2.modeling_qwen2 import repeat_kv
from attention import Controller
from sparse import sparse_attention


@torch.inference_mode()
def main():
    torch.manual_seed(17)
    output=[]
    for b,l in ((16,2048),(4,8192),(1,2048),(1,8192)):
        snapshot=torch.load(f'artifacts/snapshot_ctx{l}.pt',map_location='cuda',weights_only=True)
        q=snapshot['q'][:b].contiguous();k=snapshot['k'][:b].contiguous();v=snapshot['v'][:b].contiguous()
        # Matches Qwen's repeat_kv SDPA path; replication belongs in timing.
        def dense():return F.scaled_dot_product_attention(q,repeat_kv(k,7),repeat_kv(v,7),is_causal=False)
        dense(); base=triton.testing.do_bench(dense,warmup=100,rep=300)
        for reps in (1,4):
            ctl=Controller('learned',reps,.5,0)
            attn=SimpleNamespace(layer_idx=12,num_key_value_groups=7)
            # Preinitialize summaries, as a decode cache would. Do NOT scan full K in timing.
            ctl.select(attn,q,k)
            initial=attn._direct_summary
            def select():
                # Repeated benchmark of a fixed decode step: restore sums without a full scan.
                # At this step use prefix L-1 plus the final key, so update cost is included.
                attn._direct_summary=(l-1,initial[1])
                attn._direct_summary[1][:,:,-1]-=k[:,:,-1].float()
                return ctl.select(attn,q,k)
            mask=select()
            def full():return sparse_attention(q,k,v,select())
            sparse_attention(q,k,v,mask);full()
            sparse_ms=triton.testing.do_bench(lambda:sparse_attention(q,k,v,mask),warmup=100,rep=300)
            route_ms=triton.testing.do_bench(select,warmup=100,rep=300)
            full_ms=triton.testing.do_bench(full,warmup=100,rep=300)
            output.append({'batch':b,'context':l,'reps':reps,'dense_SDPA_ms':base,'sparse_kernel_ms':sparse_ms,'selector_with_reset_ms':route_ms,'selector_plus_sparse_with_reset_ms':full_ms,'selected_block_fraction':mask.float().mean().item()})
            print(output[-1],flush=True)
    calibrated=bool(os.environ.get('ROUTER_CALIBRATED'))
    dest='results/benchmark_calibrated.json' if calibrated else 'results/benchmark_uncalibrated.json'
    Path(dest).write_text(json.dumps({'gpu':torch.cuda.get_device_name(),'calibrated':calibrated,'data':'Real first-decode Q/K/V captured at layer12 in mean-router evaluation; budget0.5; first example for batch1','timing':'Serial CUDA event benchmark; warmed up; summary reset subtract overhead included; no summaries rebuilt from full keys; sparse-only benchmark uses fixed sampled mask. Baseline is Qwen repeat_kv + SDPA, not a native-GQA best-possible kernel. CPU dispatch is included between GPU events.','results':output},indent=2))


if __name__=='__main__':main()
