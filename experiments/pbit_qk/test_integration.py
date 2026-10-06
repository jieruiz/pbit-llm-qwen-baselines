"""Validate cache/projection/FFN plumbing against original SDPA decoding."""
import json
import torch
from transformers import AutoModelForCausalLM
from attention import Controller,install


def main():
    torch.manual_seed(47)
    torch.backends.cuda.matmul.allow_tf32=False
    model=AutoModelForCausalLM.from_pretrained('models/Qwen2.5-0.5B',local_files_only=True,
        torch_dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval()
    ids=torch.randint(100,12000,(2,72),device='cuda')
    ffn_ids=[id(layer.mlp) for layer in model.model.layers]
    def run():
        logits=[]
        with torch.inference_mode():
            pref=model.model(ids[:,:63],use_cache=True)
            prefix=pref.last_hidden_state.clone()
            cache=pref.past_key_values
            for i in range(63,71):
                out=model.model(ids[:,i:i+1],past_key_values=cache,use_cache=True)
                cache=out.past_key_values
                logits.append(model.lm_head(out.last_hidden_state).float())
        return prefix,torch.cat(logits,dim=1)
    prefix,reference=run()
    ctl=Controller('dense',128,0)
    install(model,ctl)
    prefix_new,new=run()
    assert torch.equal(prefix,prefix_new), 'prefill changed'
    assert ffn_ids==[id(layer.mlp) for layer in model.model.layers], 'MLP object replaced'
    a=reference.log_softmax(-1); b=new.log_softmax(-1)
    kl=(a.exp()*(a-b)).sum(-1).mean().item()
    maxdiff=(reference-new).abs().max().item()
    assert kl < .002, (kl,maxdiff)
    # sdpa mode delegates every call and must match exactly.
    ctl.mode='sdpa'
    _,delegated=run()
    assert torch.equal(reference,delegated), 'SDPA delegation changed logits'
    for mode in ('iid','stratified'):
        ctl.mode=mode
        ctl.samples=4
        ctl.generator.manual_seed(42)
        p1,stochastic=run()
        ctl.generator.manual_seed(42)
        p2,repeated=run()
        assert torch.equal(prefix,p1) and torch.equal(p1,p2)
        assert torch.isfinite(stochastic).all()
        assert torch.equal(stochastic,repeated), 'seed reproducibility'
        assert not torch.equal(reference,stochastic), 'sampling inactive'
        assert ffn_ids==[id(layer.mlp) for layer in model.model.layers]
    print(json.dumps({'status':'passed','dense_vs_original_mean_KL':kl,'max_logit_difference':maxdiff,
        'prefill_bitwise_identical':True,'sdpa_delegation_bitwise_identical':True,'original_ffn_objects_retained':True,
        'stochastic_cache_smoke_and_seed_reproducibility':True,
        'note':'FP32 manual attention differs slightly from fused BF16 SDPA; both baselines evaluated'}))


if __name__=='__main__':
    main()
