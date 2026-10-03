"""Prespecified supplemental N=1/2 sweep of all three FINAL checkpoints.

Motivation after primary experiment: K^2 binary-readout cost makes low-N
quality an unresolved deployment question. No retraining or checkpoint selection.
"""
import hashlib
from pathlib import Path
from run_continuous_ten_layer import pooled,write
from run_multithreshold_and import ROOT,BITS,evaluate


if __name__=="__main__":
    write(ROOT/"sample_budget_protocol.json",{"counts":[1,2],"seeds":[0,1,2],
        "bits":list(BITS),"checkpoint":"fixed final student_sampled.pt for all arms",
        "motivation":"cost-quality sweep after primary N4/N16 experiment; no retraining/test selection",
        "script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    write(ROOT/"sample_budget_status.json",{"stage":"evaluation"})
    try:
        pooled([(k,"sampled",n,seed) for k in BITS for n in (1,2) for seed in (0,1,2)],evaluate,list(range(8)))
    except Exception as error:
        write(ROOT/"sample_budget_status.json",{"stage":"failed","error":repr(error)})
        raise
    write(ROOT/"sample_budget_status.json",{"stage":"complete","additional_evaluations":18})
