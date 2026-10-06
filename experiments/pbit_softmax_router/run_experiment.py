"""Matched 2k/8k screen on idle GPUs; results are resumable, never overwritten."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path('results/pbit_softmax_router_20261006')
EXP=Path('experiments/pbit_softmax_router')


def write(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2)+'\n');tmp.replace(path)


def jobs():
    output=[]
    for ctx,examples,batch in ((2048,32,16),(8192,16,4)):
        common={'context':ctx,'examples':examples,'batch':batch,'decode':128}
        def add(mode,**kwargs): output.append({**common,'mode':mode,**kwargs})
        add('sdpa');add('dense')
        for samples in (64,128,512):
            for seed in (0,1): add('iid',samples=samples,seed=seed)
        for samples in (128,512):
            for gap in (1.,4.):
                for seed in (0,1): add('ising',samples=samples,spacing=gap,burn=4.,seed=seed)
        add('ising',samples=1024,spacing=4.,burn=4.,seed=0)
        if ctx==2048:
            for burn in (0.,16.): add('ising',samples=128,spacing=1.,burn=burn,seed=0)
        for mode in ('block_mean','block_moment','block_paper','oracle','oracle_topk'):
            for ratio in (.125,.25,.5): add(mode,ratio=ratio,seed=0)
        for mode in ('block_mean','block_moment'): add(mode,ratio=.25,rank=16,seed=0)
    return output


def jobname(j):
    return f"ctx{j['context']}_{j['mode']}_s{j.get('samples',128)}_t{j.get('spacing',1.)}_w{j.get('burn',4.)}_r{j.get('ratio',.25)}_d{j.get('rank',64)}_seed{j.get('seed',0)}"


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    if not (ROOT/'diagnostics.json').exists(): raise RuntimeError('Run diagnostics first')
    if json.loads((ROOT/'diagnostics.json').read_text())['status']!='passed':raise RuntimeError('Diagnostics failed')
    memory=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
    gpus=[int(x.split(',')[0]) for x in memory.splitlines() if int(x.split(',')[1])<500]
    if not gpus: raise RuntimeError('No idle GPU')
    planned=jobs()
    with (ROOT/'integration.json').open('w') as out:
        subprocess.run([sys.executable,str(EXP/'test_integration.py')],stdout=out,check=True,
            env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpus[0]),'OMP_NUM_THREADS':'4'})
    write(ROOT/'protocol.json',{'jobs':planned,'gpus':gpus,'targets':'same starts and suffix lengths as previous attention experiments',
        'ffn':'original Qwen; all24 decode attention only; exact prefill; no training',
        'ising':'hard-penalty-limit CTMC with unit per-bit update clocks, initial all-zero/max-reference state, no annealing',
        'router':'untrained block-summary adaptation; mandatory first and last two blocks; Bernoulli inclusion 1-(1-p)^K',
        'precision':'BF16 base/cache, FP32 manual reference attention. Logical savings only, not a GPU speed benchmark.',
        'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in EXP.glob('*.py')}})
    write(ROOT/'status.json',{'stage':'running','total':len(planned),'started':time.time()})
    def worker(gpu,queue):
        for j in queue:
            name=jobname(j); dest=ROOT/'evaluation'/(name+'.json')
            if dest.exists(): continue
            cmd=[sys.executable,str(EXP/'evaluate.py')]
            for key,val in j.items():cmd.extend(['--'+key.replace('_','-'),str(val)])
            cmd.extend(['--output',str(dest)])
            write(ROOT/'commands'/(name+'.json'),{'command':cmd,'gpu':gpu})
            write(ROOT/'progress'/(name+'.json'),{'stage':'running','start':time.time(),'gpu':gpu})
            log=ROOT/'logs'/(name+'.log');log.parent.mkdir(exist_ok=True)
            with log.open('w') as out:
                subprocess.run(cmd,stdout=out,stderr=subprocess.STDOUT,check=True,
                    env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'OMP_NUM_THREADS':'4','TOKENIZERS_PARALLELISM':'false'})
            write(ROOT/'progress'/(name+'.json'),{'stage':'complete','gpu':gpu,'end':time.time()})
    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futures=[pool.submit(worker,gpu,planned[i::len(gpus)]) for i,gpu in enumerate(gpus)]
        for f in futures:f.result()
    write(ROOT/'status.json',{'stage':'complete','evaluations':len(planned),'finished':time.time()})


if __name__=='__main__':
    try:main()
    except Exception as e:
        write(ROOT/'failure.json',{'error':repr(e),'time':time.time()})
        raise
