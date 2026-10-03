"""Teacher initialization, train-only branch fitting, and local FFN distillation."""
import argparse
import json
import time
from pathlib import Path
import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from multithreshold_and_ffn import AndConfig, MultiThresholdAndFFN
from train_full_path_distillation import FFNCapture, teacher_examples, random_batch, set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bits", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="models/Qwen2.5-0.5B")
    parser.add_argument("--train-text", default="data/wikitext-2/wiki.train.raw")
    parser.add_argument("--layer", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--calibration-tokens", type=int, default=32768)
    parser.add_argument("--fit-steps", type=int, default=500)
    parser.add_argument("--mean-steps", type=int, default=2000)
    parser.add_argument("--sample-steps", type=int, default=6000)
    parser.add_argument("--validation-tokens", type=int, default=65536)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if (output/"train_metrics.jsonl").exists():
        raise FileExistsError(output)
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tokens = tokenizer(Path(args.train_text).read_text(), add_special_tokens=False, return_tensors="pt").input_ids[0]
    train, validation = tokens[:-args.validation_tokens], tokens[-args.validation_tokens:]
    teacher = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True,
        torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    teacher.requires_grad_(False)
    mlp = teacher.model.layers[args.layer].mlp
    config = AndConfig(teacher.config.hidden_size, teacher.config.intermediate_size,
                       teacher.config.hidden_size, args.bits)
    student = MultiThresholdAndFFN(config).cuda()
    student.initialize_teacher(mlp)
    student.binary_readout = False  # algebraically equivalent, verified by tests
    capture = FFNCapture()
    hook = mlp.register_forward_hook(capture)
    log = (output/"train_metrics.jsonl").open("w")
    def record(data):
        log.write(json.dumps(data)+"\n")
        log.flush()
        print(json.dumps(data),flush=True)
    def save(name, phase, step):
        torch.save({"student_type":"multithreshold_and_v1", "student_config":student.checkpoint_config(),
                    "student_state_dict":{k:v.detach().cpu() for k,v in student.state_dict().items()},
                    "training_args":vars(args), "phase":phase, "step":step}, output/name)
    record({"event":"start", "args":vars(args), "parameters":sum(p.numel() for p in student.parameters()),
            "train_tokens":train.numel(), "validation_tokens":validation.numel(),
            "gpu":torch.cuda.get_device_name(), "training_readout":"factorized, same hard bits and STE algebra",
            "inference_readout":"expanded binary and AND features, FP32 tied readout"})
    begin = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    calibration_generator = torch.Generator().manual_seed(719)
    gate_fields, value_fields = [], []
    with torch.no_grad():
        for _ in range(args.calibration_tokens//1024):
            x,_ = teacher_examples(teacher,capture,random_batch(train,4,256,calibration_generator).cuda())
            with torch.autocast("cuda",dtype=torch.bfloat16):
                gate_fields.append(student.gate_projection(x).flatten(0,1).detach())
                value_fields.append(student.value_projection(x).flatten(0,1).detach())
        gate_fields, value_fields = torch.cat(gate_fields),torch.cat(value_fields)
        student.gate_bank.initialize(gate_fields,True)
        student.value_bank.initialize(value_fields,False)
    fit_optimizer = torch.optim.Adam(list(student.gate_bank.parameters())+list(student.value_bank.parameters()),lr=.015)
    fit_generator = torch.Generator().manual_seed(811)
    for step in range(args.fit_steps):
        indices = torch.randint(gate_fields.shape[0],(256,),generator=fit_generator).cuda()
        a,u = gate_fields[indices].float(),value_fields[indices].float()
        s = student.gate_bank.decode(student.gate_bank.probabilities(a))
        v = student.value_bank.decode(student.value_bank.probabilities(u))
        # Equal per-channel standardized MSE; calibration never sees validation or test.
        loss = ((s-F.silu(a))/student.gate_bank.scale).square().mean() + ((v-u)/student.value_bank.scale).square().mean()
        fit_optimizer.zero_grad()
        loss.backward()
        fit_optimizer.step()
        if (step+1)%100 == 0:
            record({"event":"branch_fit","step":step+1,"loss":loss.item()})
    del gate_fields, value_fields, fit_optimizer
    torch.cuda.empty_cache()
    save("student_fitted.pt","branch_fit",args.fit_steps)
    fitting_seconds = time.perf_counter()-begin

    @torch.no_grad()
    def validate(step):
        sums = {}
        vg = torch.Generator().manual_seed(927)
        with torch.random.fork_rng(devices=[0]):
            torch.manual_seed(1729)
            for _ in range(4):
                x,t = teacher_examples(teacher,capture,random_batch(validation,4,256,vg).cuda())
                with torch.autocast("cuda",dtype=torch.bfloat16):
                    mean,var = student.conditional_moments(x)
                    a,u = student.gate_projection(x),student.value_projection(x)
                p,q = student.gate_bank.probabilities(a),student.value_bank.probabilities(u)
                power=t.float().square().mean().clamp_min(1e-12)
                bias=(mean-t.float()).square().mean()/power
                variance=var.mean()/power
                values={"bias_nmse":bias.item(),"single_path_variance_nmse":variance.item(),
                        "expected_n4_nmse":(bias+variance/4).item(),
                        "gate_saturation":((p<.01)|(p>.99)).float().mean().item(),
                        "value_saturation":((q<.01)|(q>.99)).float().mean().item()}
                for key,value in values.items():
                    sums[key]=sums.get(key,0)+value/4
        record({"event":"validation","step":step,**sums})
        return sums
    validate(0)
    optimizer = torch.optim.AdamW(student.parameters(),lr=3e-4,weight_decay=.01)
    generator = torch.Generator().manual_seed(args.seed+17)
    training_start = time.perf_counter()
    for step in range(1,args.mean_steps+args.sample_steps+1):
        sampled=step>args.mean_steps
        student.set_sample_count(4 if sampled else 0)
        if step==args.mean_steps+1:
            for group in optimizer.param_groups:
                group["lr"]=1e-4
        x,t = teacher_examples(teacher,capture,random_batch(train,4,256,generator).cuda())
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda",dtype=torch.bfloat16):
            y=student(x)
        mse=(y.float()-t.float()).square().mean()/t.float().square().mean().clamp_min(1e-12)
        cosine=1-F.cosine_similarity(y.float().flatten(0,1),t.float().flatten(0,1),dim=-1).mean()
        loss=mse+.05*cosine
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(),1.)
        optimizer.step()
        if step==1 or step%100==0:
            record({"event":"train","step":step,"phase":"sampled" if sampled else "mean_field",
                    "loss":loss.item(),"nmse":mse.item(),"elapsed_seconds":time.perf_counter()-training_start})
        if step%1000==0 or step==args.mean_steps+args.sample_steps:
            diagnostics=validate(step)
        if step==args.mean_steps:
            save("student_mean.pt","mean_field",step)
    save("student_sampled.pt","sampled",args.sample_steps)
    summary={"event":"complete","fitting_seconds":fitting_seconds,
             "training_seconds":time.perf_counter()-training_start,"elapsed_seconds":time.perf_counter()-begin,
             "peak_allocated_bytes":torch.cuda.max_memory_allocated(),"final_validation":diagnostics,
             "parameters":sum(p.numel() for p in student.parameters()),"steps":args.mean_steps+args.sample_steps}
    (output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    record(summary)
    hook.remove()
    log.close()


if __name__=="__main__":
    main()
