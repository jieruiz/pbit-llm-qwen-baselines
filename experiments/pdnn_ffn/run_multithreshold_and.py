"""Layer-12 K=L=1/2/4 teacher-initialized AND-bank experiment."""
import hashlib
from pathlib import Path
import torch
from run_continuous_ten_layer import run,write,pooled

ROOT=Path("results/multithreshold_and_layer12_20261002")
EXP=Path("experiments/pdnn_ffn")
BITS=(1,2,4)


def checkpoint(bits,stage):
    return ROOT/f"k{bits}"/f"student_{stage}.pt"


def train(bits,gpu):
    write(ROOT/"progress"/f"k{bits}.json",{"stage":"training","gpu":gpu})
    output=ROOT/f"k{bits}"
    if not (output/"summary.json").exists():
        run("train_multithreshold_and.py",["--bits",bits,"--output-dir",output],gpu,ROOT/"logs"/f"train_k{bits}.log")
    for stage in ("fitted","mean","sampled"):
        path=checkpoint(bits,stage)
        payload=torch.load(path,map_location="cpu",weights_only=False)
        if payload["student_config"]["bits"]!=bits or payload["training_args"]["layer"]!=12:
            raise RuntimeError("checkpoint mismatch")
        expected={"fitted":500,"mean":2000,"sampled":6000}[stage]
        if payload["step"]!=expected:
            raise RuntimeError("checkpoint duration mismatch")
        write(ROOT/"checkpoints"/f"k{bits}_{stage}.json",{"path":str(path),
            "sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"phase":payload["phase"],
            "step":payload["step"],"config":payload["student_config"]})
    write(ROOT/"progress"/f"k{bits}.json",{"stage":"trained","gpu":gpu})


def evaluate(bits,stage,count,seed,gpu):
    output=ROOT/"evaluation"/f"k{bits}"/stage/f"n{count}_seed{seed}.json"
    if not output.exists():
        run("evaluate_multithreshold_and.py",["--model","models/Qwen2.5-0.5B",
            "--checkpoint",checkpoint(bits,stage),"--text-file","data/wikitext-2/wiki.test.raw",
            "--output",output,"--sample-count",count,"--seed",seed],gpu,
            ROOT/"logs"/f"eval_k{bits}_{stage}_n{count}_seed{seed}.log")


def diagnose(bits,stage,gpu):
    output=ROOT/"moments"/f"k{bits}_{stage}.json"
    if not output.exists():
        run("diagnose_multithreshold_and.py",["--checkpoint",checkpoint(bits,stage),"--output",output],
            gpu,ROOT/"logs"/f"moments_k{bits}_{stage}.log")


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    sources=("multithreshold_and_ffn.py","train_multithreshold_and.py","evaluate_multithreshold_and.py",
             "diagnose_multithreshold_and.py","test_multithreshold_and.py","run_multithreshold_and.py",
             "train_full_path_distillation.py","evaluate_full_path_perplexity.py","run_continuous_ten_layer.py")
    hashes={name:hashlib.sha256((EXP/name).read_bytes()).hexdigest() for name in sources}
    if (ROOT/"source_sha256.json").exists():
        import json
        if json.loads((ROOT/"source_sha256.json").read_text())!=hashes:
            raise RuntimeError("execution sources changed")
    write(ROOT/"source_sha256.json",hashes)
    write(ROOT/"protocol.json",{"layer":12,"bits_per_branch":list(BITS),"hidden_width":4864,
        "init":"original Qwen gate/up/down weights; affine biases zero; train-only .995 abs quantile scales",
        "branch_fit":"500 steps on 32768 train-only tokens, per-channel standardized MSE",
        "mean_steps":2000,"sample_steps":6000,"training_samples":4,"seed":0,
        "learning_rates":[.0003,.0001],"batch":4,"sequence":256,"loss":"NMSE + 0.05 cosine loss",
        "trainable":"all three matrices, biases, per-channel bank thresholds, temperatures, coefficients, offsets",
        "input":"raw floating; no input quantization",
        "inference":"binary single-bit and AND terms, shared tied output weights; output/path accumulation FP32",
        "comparison":"K=1/2/4 same matrices and schedule; bank parameter counts differ slightly; old serial is historical only",
        "evaluation":"full WT2 test 299078 tokens, scored299077, context2048 stride1024; final fixed6000 sample step",
        "selection":"no test checkpoint selection; fitted/mean/final checkpoints all prespecified"})
    write(ROOT/"status.json",{"stage":"training"})
    pooled([(k,) for k in BITS],train,[0,1,2])
    write(ROOT/"status.json",{"stage":"diagnostics"})
    pooled([(k,s) for k in BITS for s in ("fitted","mean","sampled")],diagnose,list(range(8)))
    write(ROOT/"status.json",{"stage":"evaluation"})
    tasks=[]
    for k in BITS:
        for stage in ("fitted","mean","sampled"):
            pairs=((0,0),(4,0),(4,1),(4,2),(16,0)) if stage=="sampled" else ((0,0),(4,0))
            tasks.extend((k,stage,n,seed) for n,seed in pairs)
    pooled(tasks,evaluate,list(range(8)))
    write(ROOT/"status.json",{"stage":"complete","evaluations":27,"diagnostics":9})


if __name__=="__main__":
    try:
        main()
    except Exception as error:
        write(ROOT/"status.json",{"stage":"failed","error":repr(error)})
        raise
