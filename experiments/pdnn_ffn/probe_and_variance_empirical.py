"""Independent post-training check of actual expanded N4 output variance."""
import argparse
import json
from pathlib import Path
import time
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from evaluate_multithreshold_and import prepare_student
from train_full_path_distillation import FFNCapture,teacher_examples
from rebuild_and_variance_initial import write,sha


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',required=True)
    parser.add_argument('--train-text',required=True)
    parser.add_argument('--control',required=True)
    parser.add_argument('--variance',required=True)
    parser.add_argument('--scales',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    tokenizer=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    tokens=tokenizer(Path(args.train_text).read_text(encoding='utf-8'),add_special_tokens=False,
                     return_tensors='pt').input_ids[0][-65536:]
    teacher=AutoModelForCausalLM.from_pretrained(args.model,local_files_only=True,
        torch_dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval()
    teacher.requires_grad_(False)
    capture=FFNCapture()
    hook=teacher.model.layers[12].mlp.register_forward_hook(capture)
    inputs,target=teacher_examples(teacher,capture,tokens[:128].unsqueeze(0).cuda())
    hook.remove()
    power=json.loads(Path(args.scales).read_text())['powers']['12']
    results={}
    for condition,path in [('control',args.control),('variance',args.variance)]:
        module,payload=prepare_student(path)
        assert payload['training_args']['layer']==12 and module.config.bits==4
        module.set_sample_count(4)
        started=time.monotonic()
        with torch.no_grad():
            mean,variance=module.conditional_moments(inputs)
            groups=[]
            for repeat in range(128):
                torch.manual_seed(83160+repeat)
                groups.append(module(inputs).float())
            empirical_variance,empirical_mean=torch.var_mean(torch.stack(groups),dim=0,unbiased=True)
            predicted=variance.mean()/4
            observed=empirical_variance.mean()
            results[condition]={'checkpoint_sha256':sha(path),'checkpoint_step':payload['step'],
                'analytical_n4_variance_nmse':(predicted/power).item(),
                'empirical_n4_variance_nmse':(observed/power).item(),
                'empirical_over_analytical':(observed/predicted).item(),
                'empirical_mean_vs_analytical_nmse':((empirical_mean-mean).square().mean()/power).item(),
                'analytical_bias_nmse':((mean-target.float()).square().mean()/power).item(),
                'elapsed_seconds':time.monotonic()-started}
        del module,groups
    write(Path(args.output),{'layer':12,'tokens':128,'groups':128,'samples_per_group':4,
        'split':'first128 tokens of held-out train tail','seeds':'83160+repeat, paired across A/B',
        'readout':'actual expanded binary AND, FP32 accumulation, BF16 FFN output',
        'results':results,'used_for_selection':False,
        'note':'Finite Monte Carlo diagnostic; local conditional variance, not whole-model variance or a quality confidence interval.'})


if __name__=='__main__':main()
