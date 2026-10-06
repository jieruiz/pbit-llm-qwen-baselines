"""Follow-up: find stratified fidelity boundary after poor B<=16 results."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path('results/pbit_qk_20261006')
EXP=Path('experiments/pbit_qk')


def write(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2)+'\n'); tmp.replace(path)


def main():
    if json.loads((ROOT/'status.json').read_text())['stage']!='complete':
        raise RuntimeError('Initial experiment must finish first')
    if (ROOT/'extension_protocol.json').exists(): raise RuntimeError('Already launched')
    memory=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
    gpus=[int(line.split(',')[0]) for line in memory.splitlines() if int(line.split(',')[1])<500]
    if not gpus: raise RuntimeError('No idle GPUs')
    jobs=[(ctx,examples,batch,B,seed) for ctx,examples,batch in ((2048,32,16),(8192,16,4))
          for B in (32,64,128) for seed in (0,1,2)]
    write(ROOT/'extension_protocol.json',{'reason':'B<=16 strongly degrades QK PPL; extend stratified budget to locate fidelity/access tradeoff',
        'samples':[32,64,128],'mode':'stratified','jobs':jobs,'gpus':gpus,
        'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in EXP.glob('*.py')}})
    write(ROOT/'extension_status.json',{'stage':'evaluating','total':len(jobs)})
    def worker(gpu,queue):
        for ctx,examples,batch,B,seed in queue:
            name=f'ctx{ctx}_stratified_b{B}_seed{seed}'
            cmd=[sys.executable,str(EXP/'evaluate.py'),'--context',str(ctx),'--examples',str(examples),
                 '--decode','128','--batch',str(batch),'--mode','stratified','--samples',str(B),
                 '--seed',str(seed),'--output',str(ROOT/'evaluation'/(name+'.json'))]
            write(ROOT/'commands'/(name+'.json'),{'command':cmd,'gpu':gpu})
            write(ROOT/'progress'/(name+'.json'),{'stage':'running','gpu':gpu,'started':time.time()})
            with (ROOT/'logs'/(name+'.log')).open('w') as out:
                subprocess.run(cmd,check=True,stdout=out,stderr=subprocess.STDOUT,
                    env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'OMP_NUM_THREADS':'4','TOKENIZERS_PARALLELISM':'false'})
            write(ROOT/'progress'/(name+'.json'),{'stage':'complete','gpu':gpu})
    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futures=[pool.submit(worker,gpu,jobs[i::len(gpus)]) for i,gpu in enumerate(gpus)]
        for f in futures: f.result()
    write(ROOT/'extension_status.json',{'stage':'complete','evaluations':len(jobs)})


if __name__=='__main__': main()
