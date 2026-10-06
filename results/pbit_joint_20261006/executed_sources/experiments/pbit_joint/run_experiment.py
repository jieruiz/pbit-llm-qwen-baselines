"""Predeclared matched factorial inference evaluation; no retraining."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path('results/pbit_joint_20261006')
EXP=Path('experiments/pbit_joint')
CHECKPOINTS='results/multithreshold_and_twenty_layer_20261002/joint'
MANIFEST='results/multithreshold_and_twenty_layer_20261002/evaluation/joint/n16_seed0.json'


def write(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(obj,indent=2)+'\n');temp.replace(path)


def jobs():
    planned=[]
    for context,examples in ((2048,32),(8192,16)):
        common={'context':context,'examples':examples,'batch':16 if context==2048 else 4,'decode':128,'samples':512,'ffn_chunk':256}
        for mode in ('sdpa','dense'):
            planned.append({**common,'ffn':'original','mode':mode,'seed':0})
        for seed in range(3):
            planned.append({**common,'ffn':'original','mode':'tree','seed':seed})
        for n in (4,16):
            for mode in ('sdpa','tree'):
                for seed in range(3):
                    planned.append({**common,'ffn':'and','ffn_samples':n,'mode':mode,'seed':seed})
            # Same-manual-arithmetic control for the main combined configuration.
            if n==16:
                for seed in range(3):
                    planned.append({**common,'ffn':'and','ffn_samples':n,'mode':'dense','seed':seed})
        # Conditional-mean FFNs diagnose finite-FFN-sampling contribution.
        planned.append({**common,'ffn':'and','ffn_samples':0,'mode':'sdpa','seed':0})
        for seed in range(3):
            planned.append({**common,'ffn':'and','ffn_samples':0,'mode':'tree','seed':seed})
    return planned


def name(j):
    return f"ctx{j['context']}_{j['ffn']}_n{j.get('ffn_samples',0)}_{j['mode']}_s{j['samples']}_seed{j['seed']}"


def command(j,out):
    cmd=[sys.executable,str(EXP/'evaluate.py')]
    for k,v in j.items():cmd += ['--'+k.replace('_','-'),str(v)]
    if j['ffn']=='and':cmd += ['--checkpoint-dir',CHECKPOINTS,'--checkpoint-manifest',MANIFEST]
    return cmd+['--output',str(out)]


def main():
    if (ROOT/'protocol.json').exists():raise FileExistsError('Refuse duplicate experiment launch')
    checks=json.loads((ROOT/'integration.json').read_text())
    assert checks['status']=='passed'
    snapshot=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    gpus=[int(row.split(',')[0]) for row in snapshot.splitlines() if int(row.split(',')[1])<500 and int(row.split(',')[2])<5]
    if not gpus:raise RuntimeError('No idle GPUs')
    planned=jobs()
    write(ROOT/'protocol.json',{'jobs':planned,'gpus':gpus,'gpu_snapshot':snapshot,'started':time.time(),
        'primary':'20 AND FFNs, K4/N16, all24 exact-softmax tree-PV S512, three inference seeds',
        'controls':'original; attention-only; FFN-only; combined N4/N16; N16 manual dense; mean-field FFN diagnostic',
        'checkpoint_selection':'Historical Oct2 joint validation-best step800, confirmed by user; no new model selection/training',
        'prefill':'Exact attention, selected FFN active. Thus FFN changes cached prefix as well as decode.',
        'randomness':'FFN and attention own separate generators; FFN streams paired across attention modes',
        'targets':'same fixed suffix positions as prior attention work; all conditions batch16 at2k / batch4 at8k, chunk256',
        'precision':'BF16 backbone, FP32 banks/expanded binary FFN readout and manual PV',
        'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in EXP.glob('*.py')}})
    write(ROOT/'status.json',{'stage':'running','jobs':len(planned)})
    def worker(gpu,queue):
        for j in queue:
            tag=name(j);out=ROOT/'evaluation'/(tag+'.json')
            if out.exists():raise FileExistsError(out)
            cmd=command(j,out)
            write(ROOT/'commands'/(tag+'.json'),{'gpu':gpu,'command':cmd})
            write(ROOT/'progress'/(tag+'.json'),{'stage':'running','gpu':gpu,'started':time.time()})
            log=ROOT/'logs'/(tag+'.log');log.parent.mkdir(exist_ok=True)
            with log.open('w') as f:
                subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True,
                    env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'OMP_NUM_THREADS':'4','TOKENIZERS_PARALLELISM':'false'})
            write(ROOT/'progress'/(tag+'.json'),{'stage':'complete','gpu':gpu,'finished':time.time()})
            print(json.dumps({'done':tag}),flush=True)
    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futures=[pool.submit(worker,g,planned[i::len(gpus)]) for i,g in enumerate(gpus)]
        for f in futures:f.result()
    write(ROOT/'status.json',{'stage':'complete','evaluations':len(planned),'finished':time.time()})


if __name__=='__main__':
    try:main()
    except Exception as e:
        write(ROOT/'failure.json',{'error':repr(e),'time':time.time()})
        raise
