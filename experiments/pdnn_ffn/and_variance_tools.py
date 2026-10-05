"""Train-only fixed scales, validation-only lambda pilot, and held-out probes."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from multithreshold_and_ffn import load_checkpoint
from and_variance_loss import normalized_output_variance
from rebuild_and_variance_initial import LAYERS, write, sha
from train_full_path_distillation import FFNCapture, teacher_examples, random_batch


def backbone(model):
    result = AutoModelForCausalLM.from_pretrained(model,local_files_only=True,
        torch_dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval()
    result.requires_grad_(False)
    return result


def corpus(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    tokens = tokenizer(Path(args.train_text).read_text(encoding='utf-8'),
                       add_special_tokens=False,return_tensors='pt').input_ids[0]
    return tokens[:-65536], tokens[-65536:]


def calibrate(args):
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    teacher = backbone(args.model)
    train, _ = corpus(args)
    sums = {layer:0. for layer in LAYERS}
    counts = {layer:0 for layer in LAYERS}
    handles=[]
    for layer in LAYERS:
        def hook(module, inputs, out, layer=layer):
            sums[layer] += out.float().square().sum().item()
            counts[layer] += out.numel()
        handles.append(teacher.model.layers[layer].mlp.register_forward_hook(hook))
    generator=torch.Generator().manual_seed(719)
    with torch.no_grad(), torch.autocast('cuda',dtype=torch.bfloat16):
        for _ in range(32):
            teacher.model(input_ids=random_batch(train,4,256,generator).cuda(),use_cache=False)
    for handle in handles:
        handle.remove()
    powers={str(layer):sums[layer]/counts[layer] for layer in LAYERS}
    assert all(value>0 for value in powers.values())
    write(output, {'powers':powers,'calibration_tokens':32768,'batch_seed':719,
        'split':'train excluding final 65536 tokens','source':'unmodified Qwen FFN mean squared output',
        'train_text_sha256':sha(args.train_text),'frozen':True})


def pilot(args):
    output=Path(args.output)
    output.mkdir(parents=True,exist_ok=False)
    train,validation=corpus(args)
    teacher=backbone(args.model)
    capture=FFNCapture()
    handle=teacher.model.layers[12].mlp.register_forward_hook(capture)
    power=json.loads(Path(args.scales).read_text())['powers']['12']
    fixed=[]
    for start in torch.linspace(0,validation.numel()-256,16).round().long().tolist():
        x,y=teacher_examples(teacher,capture,validation[start:start+256].unsqueeze(0).cuda())
        fixed.append((x,y))
    summaries=[]
    for weight in (0.,.01,.1,1.):
        label=f'lambda_{weight:g}'
        directory=output/label
        directory.mkdir()
        with torch.random.fork_rng(devices=[]):
            student,payload=load_checkpoint(args.checkpoint)
        assert payload['training_args']['layer']==12 and student.config.bits==4
        student.binary_readout=False
        student.set_sample_count(4)
        torch.manual_seed(0)
        generator=torch.Generator().manual_seed(17)
        optimizer=torch.optim.AdamW(student.parameters(),lr=1e-5,weight_decay=.01)
        records=[]
        def save(step,metric):
            saved={**payload,'student_state_dict':{k:v.detach().cpu() for k,v in student.state_dict().items()},
                   'phase':'and_variance_local_pilot','step':step,'pilot_lambda':weight,'pilot_validation':metric}
            torch.save(saved,directory/'best.pt')
        @torch.no_grad()
        def validate(step):
            metrics=[]
            for x,y in fixed:
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    mean,var=student.conditional_moments(x)
                bias=(mean-y.float()).square().mean()/power
                variance=var.mean()/(4*power)
                metrics.append({'bias_nmse':bias.item(),'n4_variance_nmse':variance.item(),
                                'expected_n4_nmse':(bias+variance).item()})
            result={key:sum(m[key] for m in metrics)/len(metrics) for key in metrics[0]}
            record={'event':'validation','step':step,**result}
            records.append(record)
            print(label,json.dumps(record),flush=True)
            return result
        best=validate(0);best_step=0;save(0,best)
        begin=time.monotonic()
        for step in range(1,601):
            x,y=teacher_examples(teacher,capture,random_batch(train,4,256,generator).cuda())
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                prediction=student(x)
                penalty=normalized_output_variance(student,x,power) if weight else prediction.new_zeros(())
            mse=(prediction.float()-y.float()).square().mean()/y.float().square().mean().clamp_min(1e-12)
            cosine=1-F.cosine_similarity(prediction.float(),y.float(),dim=-1).mean()
            loss=mse+.05*cosine+weight*penalty
            if not torch.isfinite(loss):raise FloatingPointError('pilot objective is non-finite')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(),1.)
            optimizer.step()
            if step%200==0:
                metric=validate(step)
                if metric['expected_n4_nmse']<best['expected_n4_nmse']:
                    best,best_step=metric,step;save(step,best)
        summary={'lambda':weight,'best_step':best_step,'best_validation':best,
                 'seconds':time.monotonic()-begin,'initial_checkpoint_sha256':sha(args.checkpoint)}
        write(directory/'summary.json',summary)
        (directory/'metrics.jsonl').write_text(''.join(json.dumps(m)+'\n' for m in records),encoding='utf-8',newline='\n')
        summaries.append(summary)
        del student,optimizer
        torch.cuda.empty_cache()
    control=summaries[0]['best_validation']
    admissible=[s for s in summaries[1:] if s['best_validation']['bias_nmse']<=1.1*control['bias_nmse']]
    selected=min(admissible,key=lambda s:s['best_validation']['expected_n4_nmse']) if admissible else summaries[1]
    result={'selected_lambda':selected['lambda'],'selection':'minimum expected-N4-NMSE among bias<=110% control; fallback .01',
            'fallback_used':not bool(admissible),'control':control,'candidates':summaries,
            'selected_variance_reduction':1-selected['best_validation']['n4_variance_nmse']/control['n4_variance_nmse'],
            'test_used':False,'pilot_weight_reused_for_main':False}
    write(output/'selection.json',result)
    handle.remove()


def probe(args):
    model=backbone(args.model)
    _,tokens=corpus(args)
    powers=json.loads(Path(args.scales).read_text())['powers']
    starts=torch.linspace(0,tokens.numel()-256,16).round().long().tolist()
    fixed={layer:[] for layer in LAYERS}
    teacher_handles=[]
    for layer in LAYERS:
        def capture_teacher(module,inputs,output,layer=layer):
            fixed[layer].append((inputs[0].detach(),output.detach()))
        teacher_handles.append(model.model.layers[layer].mlp.register_forward_hook(capture_teacher))
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        for start in starts:
            model.model(input_ids=tokens[start:start+256].unsqueeze(0).cuda(),use_cache=False)
    for handle in teacher_handles:handle.remove()
    students={};originals={};metadata=[]
    for path in args.checkpoints:
        with torch.random.fork_rng(devices=[]):
            student,payload=load_checkpoint(path)
        layer=int(payload['training_args']['layer'])
        if layer in students:raise ValueError('duplicate layer')
        assert student.config.bits==4
        originals[layer]=model.model.layers[layer].mlp
        student.gate_projection.to(dtype=torch.bfloat16)
        student.value_projection.to(dtype=torch.bfloat16)
        student.set_sample_count(4)
        student.binary_readout=True
        model.model.layers[layer].mlp=student
        students[layer]=student
        metadata.append({'layer':layer,'path':str(path),'sha256':sha(path)})
    assert set(students)==set(LAYERS)
    records={layer:[] for layer in students};handles=[]
    for layer,student in students.items():
        def hook(module,inputs,output,layer=layer):
            x=inputs[0]
            mean,var=module.conditional_moments(x)
            target=originals[layer](x)
            p,q=module.probabilities(x)
            fixed_power=powers[str(layer)]
            records[layer].append({'n4_variance_nmse':(var.mean()/(4*fixed_power)).item(),
                'same_input_bias_nmse':((mean-target.float()).square().mean()/fixed_power).item(),
                'output_power_to_reference':(output.float().square().mean()/fixed_power).item(),
                'gate_saturation':((p<.01)|(p>.99)).float().mean().item(),
                'value_saturation':((q<.01)|(q>.99)).float().mean().item()})
        handles.append(student.register_forward_hook(hook))
    torch.manual_seed(12345)
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        for start in starts:
            model.model(input_ids=tokens[start:start+256].unsqueeze(0).cuda(),use_cache=False)
    for handle in handles:handle.remove()
    means={str(layer):{key:sum(x[key] for x in values)/len(values) for key in values[0]}
           for layer,values in records.items()}
    common_means={}
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        for layer,student in students.items():
            values=[]
            for x,target in fixed[layer]:
                mean,var=student.conditional_moments(x)
                power=powers[str(layer)]
                values.append({'n4_variance_nmse':(var.mean()/(4*power)).item(),
                    'bias_nmse':((mean-target.float()).square().mean()/power).item()})
            common_means[str(layer)]={key:sum(x[key] for x in values)/len(values) for key in values[0]}
    write(Path(args.output), {'layers':means,'common_teacher_inputs':common_means,
         'checkpoints':metadata,'windows':16,'sequence_length':256,
         'seed':12345,'split':'held-out final 65536 train tokens',
         'input_distribution':'student rollout with expanded binary four-path readout; original FFN only logs same-input bias',
         'reference_scales_sha256':sha(args.scales)})


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['calibrate','pilot','probe'])
    parser.add_argument('--model',required=True)
    parser.add_argument('--train-text',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--scales')
    parser.add_argument('--checkpoint')
    parser.add_argument('--checkpoints',nargs='+')
    args=parser.parse_args()
    if args.action!='calibrate' and not args.scales:parser.error('--scales is required')
    if args.action=='pilot' and not args.checkpoint:parser.error('--checkpoint is required')
    if args.action=='probe' and not args.checkpoints:parser.error('--checkpoints is required')
    globals()[args.action](args)


if __name__=='__main__':
    main()
