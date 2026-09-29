"""Held-out student-distribution diagnostics; teacher signals are logging only."""
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from calibrated_pdnn_ffn import load_checkpoint


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--train-text', required=True)
    p.add_argument('--checkpoints', nargs='+', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    torch.manual_seed(12345)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tokens = tokenizer(Path(args.train_text).read_text(encoding='utf-8'),
                       add_special_tokens=False, return_tensors='pt').input_ids[0][-65536:]
    kw = dict(local_files_only=True, torch_dtype=torch.bfloat16, attn_implementation='sdpa')
    teacher = AutoModelForCausalLM.from_pretrained(args.model, **kw).cuda().eval()
    model = AutoModelForCausalLM.from_pretrained(args.model, **kw).cuda().eval()
    students, accum, records = {}, {}, {}
    for checkpoint in args.checkpoints:
        student, payload = load_checkpoint(checkpoint)
        layer = int(payload['training_args']['layer'])
        student.to(dtype=torch.bfloat16).eval(); student.set_sample_count(4)
        model.model.layers[layer].mlp = student
        students[layer] = student
        accum[layer] = dict(input_saturated=0., input_count=0, hidden_saturated=0.,
                            hidden_count=0, error_power=0., target_power=0., output_power=0.)
        if hasattr(student, 'input_log_scale'):
            records[layer] = {name: {'min': float(v.float().min()), 'max': float(v.float().max()),
                                     'mean': float(v.float().mean())}
                              for name, v in student.named_parameters() if not name.startswith('projections.')}
        else:
            records[layer] = {}

    def field_hook(layer):
        def hook(module, inputs, output):
            mean = students[layer].hidden_mean(output)
            accum[layer]['hidden_saturated'] += (mean.abs() > .99).sum().item()
            accum[layer]['hidden_count'] += mean.numel()
        return hook

    def ffn_hook(layer):
        def hook(module, inputs, output):
            x = inputs[0]
            mean = module.input_mean(x)
            accum[layer]['input_saturated'] += (mean.abs() > .99).sum().item()
            accum[layer]['input_count'] += mean.numel()
            # Original FFN evaluated on student x; never alters the student output.
            target = teacher.model.layers[layer].mlp(x).float()
            y = output.float()
            accum[layer]['error_power'] += (y - target).square().sum().item()
            accum[layer]['target_power'] += target.square().sum().item()
            accum[layer]['output_power'] += y.square().sum().item()
        return hook

    handles = []
    for layer, student in students.items():
        handles += [student.register_forward_hook(ffn_hook(layer)),
                    student.projections[0].register_forward_hook(field_hook(layer))]
    for start in torch.linspace(0, tokens.numel() - 256, 16).round().long().tolist():
        model(input_ids=tokens[start:start+256].unsqueeze(0).cuda(), use_cache=False)
    for handle in handles: handle.remove()
    result = {'windows': 16, 'sequence_length': 256, 'seed': 12345,
              'split': 'held-out final 65536 train tokens', 'layers': {}}
    for layer, a in accum.items():
        result['layers'][layer] = {
            'input_saturation_fraction': a['input_saturated'] / a['input_count'],
            'hidden_saturation_fraction': a['hidden_saturated'] / a['hidden_count'],
            'same_student_input_ffn_nmse': a['error_power'] / a['target_power'],
            'output_rms_ratio': (a['output_power'] / a['target_power']) ** .5,
            'calibration_parameters': records[layer]}
    Path(args.output).write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
