"""Train-only range calibration and fixed-teacher projection error audit."""
import argparse
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from train_full_path_distillation import FFNCapture, teacher_examples, random_batch
from multibit_input_ffn import quantize_codes, decode_codes


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output-dir',required=True)
    parser.add_argument('--layer',type=int,default=12)
    args=parser.parse_args()
    output=Path(args.output_dir)
    output.mkdir(parents=True,exist_ok=True)
    tokenizer=AutoTokenizer.from_pretrained('models/Qwen2.5-0.5B',local_files_only=True)
    tokens=tokenizer(Path('data/wikitext-2/wiki.train.raw').read_text(),add_special_tokens=False,return_tensors='pt').input_ids[0]
    teacher=AutoModelForCausalLM.from_pretrained('models/Qwen2.5-0.5B',local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval()
    teacher.requires_grad_(False)
    capture=FFNCapture()
    if args.layer < 0 or args.layer >= len(teacher.model.layers):
        raise ValueError(f'invalid decoder layer: {args.layer}')
    mlp=teacher.model.layers[args.layer].mlp
    hook=mlp.register_forward_hook(capture)
    generator=torch.Generator().manual_seed(42017)
    values=[]
    with torch.inference_mode():
        for _ in range(32):
            x,_=teacher_examples(teacher,capture,random_batch(tokens[:-65536],4,256,generator).cuda())
            values.append(x.reshape(-1,x.shape[-1]).float())
        train_inputs=torch.cat(values)
        bounds=torch.quantile(train_inputs.abs(),.999,dim=0).clamp_min(1e-4)
        calibration={'layer':args.layer,'seed':42017,'calibration_tokens':32768,'excluded_validation_tokens':65536,
                     'rule':'per-channel abs quantile 0.999, symmetric range, frozen',
                     'input_bound':bounds.cpu().tolist(),'training_clipped_fraction':(train_inputs.abs()>bounds).float().mean().item()}
        (output/'calibration.json').write_text(json.dumps(calibration,indent=2)+'\n')
        del train_inputs,values
        probes=[]
        generator=torch.Generator().manual_seed(27183)
        for _ in range(4):
            x,_=teacher_examples(teacher,capture,random_batch(tokens[-65536:],1,256,generator).cuda())
            probes.append(x.reshape(-1,x.shape[-1]).float())
        x=torch.cat(probes)
        clipped=torch.maximum(torch.minimum(x,bounds),-bounds)
        weights=torch.cat([mlp.gate_proj.weight.float(),mlp.up_proj.weight.float()])
        target=F.linear(x,weights)
        target_power=target.square().mean()
        input_power=x.square().mean()
        proj_clip_bias=F.linear(clipped-x,weights).square().mean()
        input_clip_bias=(clipped-x).square().mean()
        rows=[]
        for bits in (1,2,4):
            step=2*bounds/(2**bits-1)
            position=(clipped+bounds)/step
            fraction=position-position.floor()
            variance=step.square()*fraction*(1-fraction)
            projection_variance=(variance.mean(0)*weights.square().sum(0)).sum()/weights.shape[0]
            for mode in ('stochastic','deterministic'):
                if mode=='stochastic':
                    input_bias=input_clip_bias
                    projection_bias=proj_clip_bias
                    input_var=variance.mean()
                    proj_var=projection_variance
                else:
                    quantized=decode_codes(quantize_codes(x,bounds,bits,False),bounds,bits)
                    input_bias=(quantized-x).square().mean()
                    projection_bias=F.linear(quantized-x,weights).square().mean()
                    input_var=x.new_zeros(())
                    proj_var=input_var
                rows.append({'mode':mode,'bits':bits,'input_bias_nmse':(input_bias/input_power).item(),
                             'input_single_variance_nmse':(input_var/input_power).item(),
                             'teacher_projection_bias_nmse':(projection_bias/target_power).item(),
                             'teacher_projection_single_variance_nmse':(proj_var/target_power).item(),
                             'teacher_projection_n4_nmse':((projection_bias+proj_var/4)/target_power).item()})
        audit={'probe_tokens':1024,'input_distribution':f'held-out original teacher layer{args.layer} inputs',
               'clipped_fraction':(x.abs()>bounds).float().mean().item(),
               'continuous_clipped_projection_nmse':(proj_clip_bias/target_power).item(),
               'rows':rows,'note':'Analytical unbiased adjacent-rounding moments for frozen teacher gate/up matrices, prior to nonlinearities; not a full-FFN quality claim.'}
        (output/'projection_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    hook.remove()


if __name__=='__main__':
    main()
