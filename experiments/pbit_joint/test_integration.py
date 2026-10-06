"""Check actual checkpoint, RNG isolation and composition before the screen."""
import hashlib
import json
from pathlib import Path
from types import MethodType
import torch
from transformers import AutoModelForCausalLM
from evaluate import prepare_student,chunked_ffn,Controller,install


def main():
    torch.backends.cuda.matmul.allow_tf32=False
    model=AutoModelForCausalLM.from_pretrained('models/Qwen2.5-0.5B',local_files_only=True,
        torch_dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval()
    root=Path('results/multithreshold_and_twenty_layer_20261002')
    expected=json.loads((root/'evaluation/joint/n16_seed0.json').read_text())['checkpoints']
    identity=[]
    for e in expected:
        p=root/'joint'/f"best_layer{e['layer']}.pt"
        digest=hashlib.sha256(p.read_bytes()).hexdigest();assert digest==e['sha256']
        identity.append({'layer':e['layer'],'sha256':digest})
    student,_=prepare_student(root/'joint/best_layer12.pt')
    student.set_sample_count(4)
    x=torch.randn(2,20,896,device='cuda',dtype=torch.bfloat16)
    with torch.inference_mode():
        student.set_sample_count(0)
        original=student(x)
        student._joint_original_forward=student.forward;student._joint_chunk=7
        student.forward=MethodType(chunked_ffn,student)
        chunked=student(x)
        torch.testing.assert_close(original,chunked,atol=.008,rtol=.008)
        student.set_sample_count(4)
        student._joint_chunk=256
    model.model.layers[12].mlp=student
    ids=torch.randint(100,12000,(2,24),device='cuda')
    def run():
        torch.manual_seed(917)
        with torch.inference_mode():
            prefix=model.model(ids[:,:20],use_cache=True)
            prefix_hidden=prefix.last_hidden_state.clone()
            cache=prefix.past_key_values
            logits=[]
            for t in range(20,23):
                out=model.model(ids[:,t:t+1],past_key_values=cache,use_cache=True)
                cache=out.past_key_values
                logits.append(model.lm_head(out.last_hidden_state).float())
        return prefix_hidden,torch.cat(logits,1),torch.cuda.get_rng_state()
    p,a,rng=run()
    ctl=Controller('sdpa',512,917);install(model,ctl)
    p2,b,rng2=run()
    assert torch.equal(a,b) and torch.equal(p,p2) and torch.equal(rng,rng2)
    ctl.mode='tree'
    p3,c,rng3=run()
    assert torch.equal(p,p3),'sampled attention altered prefill for fixed FFN stream'
    assert torch.equal(rng,rng3),'attention consumed FFN random stream'
    assert torch.isfinite(c).all() and not torch.equal(a,c)
    assert model.model.layers[12].mlp is student
    result={'status':'passed','checkpoints_verified':identity,'sdpa_delegation_bitwise':True,
        'same_FFN_prefill_bitwise_across_attention_modes':True,'FFN_RNG_stream_unchanged_by_attention':True,
        'chunked_mean_field_close':True,'combined_logits_finite_and_changed':True,
        'note':'One real stochastic FFN integration plus identity checks for all20; formal smoke tests all20.'}
    path=Path('results/pbit_joint_20261006/integration.json');path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
