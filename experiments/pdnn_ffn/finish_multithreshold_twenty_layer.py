"""Overlap final evaluation with the still-running independent N=16 evaluation.

Only replaces our waiting orchestration process after training has completed.
No training/evaluation worker is stopped, no settings/checkpoints are changed.
"""
import argparse
import hashlib
import os
from pathlib import Path
import signal
import subprocess
import time
from run_multithreshold_twenty_layer import ROOT,LAYERS,PAIRS,verify,evaluate
from run_continuous_ten_layer import pooled,read,write


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--controller-pid",type=int,required=True)
    parser.add_argument("--independent-worker-pid",type=int,required=True)
    args=parser.parse_args()
    summary=read(ROOT/"joint"/"summary.json")
    if summary["event"]!="complete" or summary["steps"]!=1000:
        raise RuntimeError("joint training has not completed")
    controller=Path(f"/proc/{args.controller_pid}")
    cmd=controller.joinpath("cmdline").read_bytes().replace(b"\0",b" ").decode()
    if "python experiments/pdnn_ffn/run_multithreshold_twenty_layer.py" not in cmd or controller.joinpath("cwd").resolve()!=Path.cwd():
        raise RuntimeError(f"unexpected controller identity: {cmd}")
    worker=Path(f"/proc/{args.independent_worker_pid}")
    worker_cmd=worker.joinpath("cmdline").read_bytes().replace(b"\0",b" ").decode()
    if ("evaluate_multi_layer_multithreshold_and.py" not in worker_cmd or str(ROOT/"evaluation"/"independent"/"n16_seed0.json") not in worker_cmd):
        raise RuntimeError("unexpected independent evaluation worker")
    memory=subprocess.check_output(["nvidia-smi","--query-gpu=index,memory.used","--format=csv,noheader,nounits"],text=True)
    usage={int(line.split(",")[0]):int(line.split(",")[1]) for line in memory.splitlines()}
    if any(usage[gpu]>500 for gpu in range(7)):
        raise RuntimeError(f"final evaluation GPUs still occupied: {usage}")
    for n,seed in PAIRS:
        if n!=16 and not (ROOT/"evaluation"/"independent"/f"n{n}_seed{seed}.json").exists():
            raise RuntimeError("independent short evaluations incomplete")
    for layer in LAYERS:
        verify(layer,"joint")
    write(ROOT/"completion_scheduler.json",{"reason":"Overlap long independent/final N16 evaluations after all training completed",
        "original_controller_pid":args.controller_pid,"original_controller_command":cmd,
        "preserved_independent_worker_pid":args.independent_worker_pid,
        "final_evaluation_gpus":list(range(7)),"preserved_independent_gpu":7,
        "script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "training_or_evaluation_workers_stopped":False})
    os.kill(args.controller_pid,signal.SIGTERM)
    write(ROOT/"status.json",{"stage":"joint_evaluation_with_independent_n16_still_running"})
    pooled([("joint",n,seed) for n,seed in PAIRS],evaluate,list(range(7)))
    pending=ROOT/"evaluation"/"independent"/"n16_seed0.json"
    while not pending.exists():
        if not worker.exists():
            raise RuntimeError("independent N16 worker exited without result")
        time.sleep(10)
    read(pending)
    write(ROOT/"status.json",{"stage":"complete","new_local_trainings":19,"reused_local_trainings":1,
        "joint_steps":1000,"full_test_evaluations":14})


if __name__=="__main__":
    main()
