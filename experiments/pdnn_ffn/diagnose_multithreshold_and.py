"""Validate exact fixed-input moments and binary/factorized readout equivalence."""
import argparse
import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from evaluate_multithreshold_and import prepare_student
from train_full_path_distillation import FFNCapture, teacher_examples, random_batch


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--checkpoint",required=True)
    p.add_argument("--output",required=True)
    args=p.parse_args()
    student,payload=prepare_student(args.checkpoint)
    teacher=AutoModelForCausalLM.from_pretrained("models/Qwen2.5-0.5B",local_files_only=True,
        torch_dtype=torch.bfloat16,attn_implementation="sdpa").cuda().eval()
    tokenizer=AutoTokenizer.from_pretrained("models/Qwen2.5-0.5B",local_files_only=True)
    tokens=tokenizer(Path("data/wikitext-2/wiki.train.raw").read_text(),add_special_tokens=False,return_tensors="pt").input_ids[0][-65536:]
    capture=FFNCapture()
    hook=teacher.model.layers[payload["training_args"]["layer"]].mlp.register_forward_hook(capture)
    generator=torch.Generator().manual_seed(27183)
    records=[]
    with torch.inference_mode():
        for batch in range(4):
            x,t=teacher_examples(teacher,capture,random_batch(tokens,1,256,generator).cuda())
            mean,variance=student.conditional_moments(x)
            power=t.float().square().mean()
            bias=(mean-t.float()).square().mean()/power
            var=variance.mean()/power
            prob,q=student.probabilities(x)
            torch.manual_seed(28183+batch)
            outputs=[]
            differences=[]
            for i in range(128):
                b,c=student.sample(prob),student.sample(q)
                y=student.factorized_readout(b,c)
                outputs.append(y)
                if i<4:
                    expanded=student.expanded_readout(b,c)
                    differences.append((expanded-y).abs().max().item())
            outputs=torch.stack(outputs)
            empirical_var=outputs.var(0,unbiased=True).mean()/power
            empirical_bias=((outputs.mean(0)-t.float()).square().mean()/power)-empirical_var/128
            gate=student.gate_projection(x).float()
            value=student.value_projection(x).float()
            gm=student.gate_bank.decode(prob)
            vm=student.value_bank.decode(q)
            gate_target=torch.nn.functional.silu(gate)
            records.append({"bias_nmse":bias.item(),"single_path_variance_nmse":var.item(),
                "expected_n4_nmse":(bias+var/4).item(),"expected_n16_nmse":(bias+var/16).item(),
                "empirical_single_path_variance_nmse":empirical_var.item(),
                "empirical_corrected_bias_nmse":empirical_bias.item(),
                "expanded_factorized_max_absolute_difference":max(differences),
                "gate_branch_nmse":((gm-gate_target).square().mean()/gate_target.square().mean()).item(),
                "value_branch_nmse":((vm-value).square().mean()/value.square().mean()).item()})
    result={"checkpoint":args.checkpoint,"phase":payload["phase"],"bits":student.config.bits,
            "probe":"1024 held-out teacher-input tokens; 128 iid paths each; no test data",
            "means":{k:sum(r[k] for r in records)/len(records) for k in records[0]},"records":records,
            "notes":"Exact moments condition on fixed input and independent banks/channels; include shared-AND covariance. Real arithmetic, before BF16 final cast."}
    output=Path(args.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result["means"]),flush=True)
    hook.remove()


if __name__=="__main__":
    main()
