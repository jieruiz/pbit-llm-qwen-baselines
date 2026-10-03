"""Twenty-layer K=4 AND-bank composition and joint adaptation."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path
import subprocess
import torch
from run_continuous_ten_layer import run,write,read,pooled

ROOT=Path("results/multithreshold_and_twenty_layer_20261002")
PREVIOUS=Path("results/multithreshold_and_layer12_20261002")
EXP=Path("experiments/pdnn_ffn")
LAYERS=(1,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22)
PAIRS=((0,0),(1,0),(2,0),(4,0),(4,1),(4,2),(16,0))
SOURCES=("multithreshold_and_ffn.py","train_multithreshold_and.py","evaluate_multithreshold_and.py",
    "train_joint_multithreshold_and.py","train_joint_full_path_distillation.py",
    "evaluate_multi_layer_multithreshold_and.py","evaluate_multi_layer_full_path_perplexity.py",
    "evaluate_full_path_perplexity.py","train_full_path_distillation.py","run_continuous_ten_layer.py",
    "run_multithreshold_twenty_layer.py")
PROTOCOL={"layers":list(LAYERS),"retained_original_layers":[0,2,3,23],"bits_per_branch":4,
    "hidden_width":4864,"training_seed":0,"reused_layer":12,"source_experiment":str(PREVIOUS),
    "local_training":"same 500 branch fitting +2000 mean-field +6000 N4 as single-layer experiment",
    "joint_start":"20 independent final local checkpoints; no prior ten-layer joint adaptation",
    "joint_steps":1000,"joint_samples":4,"joint_learning_rate":1e-5,"joint_kd_weight":.8,
    "joint_microbatch":1,"joint_accumulation":4,"sequence_length":256,
    "selection":"minimum held-out train-tail validation PPL; no test selection; existing joint protocol",
    "input":"floating BF16, no input quantization","readout":"expanded binary AND terms for all sampled PPL",
    "training_readout":"algebraically equivalent factorized sampled implementation; same STE",
    "evaluation_pairs":[list(p) for p in PAIRS],"evaluation_stages":["independent","joint"],
    "test":"299078 WT2 test tokens, scored299077; context2048 stride1024",
    "historical_comparison":"old raw/sigmoid 20-layer models differ in parameter count, initialization, precision and inherited joint schedule"}


def checkpoint(layer,stage="independent"):
    if stage=="joint":
        return ROOT/"joint"/f"best_layer{layer}.pt"
    if layer==12:
        return PREVIOUS/"k4"/"student_sampled.pt"
    return ROOT/"individual"/f"layer{layer}"/"student_sampled.pt"


def verify(layer,stage="independent"):
    path=checkpoint(layer,stage)
    payload=torch.load(path,map_location="cpu",weights_only=False)
    config=payload["student_config"]
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    if (payload["student_type"]!="multithreshold_and_v1" or config["bits"]!=4
        or config["hidden_size"]!=4864 or payload["training_args"]["layer"]!=layer):
        raise RuntimeError(f"checkpoint identity mismatch: {path}")
    if stage=="joint":
        expected=read(ROOT/"joint"/"summary.json")["best_step"]
        if payload["phase"]!="joint_end_to_end_distillation" or payload["step"]!=expected:
            raise RuntimeError("joint step mismatch")
    else:
        if payload["phase"]!="sampled" or payload["step"]!=6000:
            raise RuntimeError("local duration mismatch")
        if layer==12 and digest!=read(PREVIOUS/"checkpoints"/"k4_sampled.json")["sha256"]:
            raise RuntimeError("inherited checkpoint changed")
    write(ROOT/"checkpoints"/stage/f"layer{layer}.json",{"path":str(path),"sha256":digest,
        "config":config,"phase":payload["phase"],"step":payload["step"],"reused":layer==12 and stage!="joint"})


def individual(layer,gpu):
    output=checkpoint(layer).parent
    write(ROOT/"progress"/f"layer{layer}.json",{"stage":"training","gpu":gpu})
    if not (output/"summary.json").exists():
        if output.exists():
            raise RuntimeError(f"partial local run needs explicit recovery: {output}")
        run("train_multithreshold_and.py",["--bits",4,"--layer",layer,"--output-dir",output],gpu,
            ROOT/"logs"/f"train_layer{layer}.log")
    verify(layer)
    write(ROOT/"progress"/f"layer{layer}.json",{"stage":"complete","gpu":gpu})


def evaluate(stage,count,seed,gpu):
    output=ROOT/"evaluation"/stage/f"n{count}_seed{seed}.json"
    if not output.exists():
        run("evaluate_multi_layer_multithreshold_and.py",["--model","models/Qwen2.5-0.5B",
            "--checkpoints",*[checkpoint(layer,stage) for layer in LAYERS],
            "--text-file","data/wikitext-2/wiki.test.raw","--output",output,
            "--sample-count",count,"--seed",seed],gpu,ROOT/"logs"/f"eval_{stage}_n{count}_seed{seed}.log")


def train_joint(gpu):
    output=ROOT/"joint"
    if not (output/"summary.json").exists():
        if output.exists():
            raise RuntimeError(f"partial joint training exists: {output}")
        run("train_joint_multithreshold_and.py",["--model","models/Qwen2.5-0.5B",
            "--checkpoints",*[checkpoint(layer) for layer in LAYERS],
            "--train-text","data/wikitext-2/wiki.train.raw","--output-dir",output,
            "--steps",1000,"--sample-count",4,"--seed",0],gpu,ROOT/"logs"/"joint.log")
    for layer in LAYERS:
        verify(layer,"joint")


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--resume",action="store_true")
    args=parser.parse_args()
    memory=subprocess.check_output(["nvidia-smi","--query-gpu=index,memory.used","--format=csv,noheader,nounits"],text=True)
    if any(int(line.split(",")[1])>500 for line in memory.splitlines()):
        raise RuntimeError(f"GPU occupied: {memory}")
    hashes={name:hashlib.sha256((EXP/name).read_bytes()).hexdigest() for name in SOURCES}
    for name,expected in read(PREVIOUS/"source_sha256.json").items():
        if name in hashes and hashes[name]!=expected:
            raise RuntimeError(f"single-layer source changed: {name}")
    if args.resume:
        if read(ROOT/"protocol.json")!=PROTOCOL or read(ROOT/"source_sha256.json")!=hashes:
            raise RuntimeError("resume sources or protocol changed")
    else:
        ROOT.mkdir(parents=True,exist_ok=False)
        write(ROOT/"protocol.json",PROTOCOL)
        write(ROOT/"source_sha256.json",hashes)
    verify(12)
    write(ROOT/"status.json",{"stage":"individual_training"})
    pooled([(layer,) for layer in LAYERS if layer!=12],individual,list(range(8)))
    # Each GPU has one job. Evaluation does not modify local checkpoints.
    write(ROOT/"status.json",{"stage":"joint_training_and_independent_evaluation"})
    with ThreadPoolExecutor(max_workers=2) as pool:
        joint=pool.submit(train_joint,0)
        evaluation=pool.submit(pooled,[("independent",n,s) for n,s in PAIRS],evaluate,list(range(1,8)))
        joint.result()
        evaluation.result()
    write(ROOT/"status.json",{"stage":"joint_evaluation"})
    pooled([("joint",n,s) for n,s in PAIRS],evaluate,list(range(8)))
    write(ROOT/"status.json",{"stage":"complete","new_local_trainings":19,"reused_local_trainings":1,
        "joint_steps":1000,"full_test_evaluations":14})


if __name__=="__main__":
    try:
        main()
    except Exception as error:
        if ROOT.exists():
            write(ROOT/"status.json",{"stage":"failed","error":repr(error)})
        raise
