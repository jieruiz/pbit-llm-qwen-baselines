"""Frozen original baseline and protected-parameter audit."""
import hashlib, importlib.util
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent/'qwen35_baseline_20261013'
spec=importlib.util.spec_from_file_location('original_qwen35_baseline',BASE/'baseline.py')
baseline=importlib.util.module_from_spec(spec);spec.loader.exec_module(baseline)

def protected_hash(model):
    h=hashlib.sha256()
    for name,p in model.named_parameters():
        if '.mlp.' in name:continue
        h.update(name.encode());h.update(p.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return h.hexdigest()

def audit_sources():
    files=[ROOT/f for f in ['common.py','exact_ffn.py','pv_sampling.py','calibrate.py','experiment.py','verify.py']]
    files.append(BASE/'baseline.py')
    return {str(p):baseline.sha(p) for p in files}
