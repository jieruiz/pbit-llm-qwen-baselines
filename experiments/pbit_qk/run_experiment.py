"""52 matched evaluations: two contexts, two encodings, four B, three seeds."""
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
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(obj,indent=2)+'\n'); temp.replace(path)


def main():
    if (ROOT/'protocol.json').exists():
        raise RuntimeError('Experiment already launched; do not overwrite')
    memory=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
    available=[int(line.split(',')[0]) for line in memory.splitlines() if int(line.split(',')[1])<500]
    if not available: raise RuntimeError('No idle GPUs')
    jobs=[]
    for context,examples,batch in ((2048,32,16),(8192,16,4)):
        for mode in ('sdpa','dense'): jobs.append((context,examples,batch,mode,4,0))
        for seed in (0,1,2):
            for mode in ('iid','stratified'):
                for B in (2,4,8,16): jobs.append((context,examples,batch,mode,B,seed))
    write(ROOT/'protocol.json',{'date':'2026-10-06','model':'Qwen2.5-0.5B Base',
        'scope':'all 24 decode-time QK; dense PV, original FFNs/projections/RoPE/prefill; no training',
        'encoding':'per head signed Bernoulli abs(q)/max(abs(q)); B counts averaged; K continuous',
        'stratified':'exact compressed count law floor(B*p)+Bernoulli(frac(B*p)); independent features',
        'contexts':[2048,8192],'decode_steps':128,'seeds':[0,1,2],'samples':[2,4,8,16],
        'diagnostics':'same-state score relative L2, attention KL/TV/argmax, head and GQA-union key features',
        'limitations':'ideal probability sampler; dense GPU arithmetic; no acceleration claim; no physical device model',
        'jobs':[list(j) for j in jobs],'gpus':available,
        'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in EXP.glob('*.py')}})
    write(ROOT/'status.json',{'stage':'testing','total':len(jobs)})
    for script in ('test_sampling.py','test_integration.py'):
        with (ROOT/script.replace('.py','.json')).open('w') as out:
            subprocess.run([sys.executable,str(EXP/script)],check=True,stdout=out,
                env={**os.environ,'CUDA_VISIBLE_DEVICES':str(available[0]),'OMP_NUM_THREADS':'4'})
    write(ROOT/'status.json',{'stage':'evaluating','total':len(jobs)})
    def worker(gpu,queue):
        for context,examples,batch,mode,B,seed in queue:
            name=f'ctx{context}_{mode}_b{B}_seed{seed}'
            result=ROOT/'evaluation'/(name+'.json')
            command=[sys.executable,str(EXP/'evaluate.py'),'--context',str(context),'--examples',str(examples),
                '--decode','128','--batch',str(batch),'--mode',mode,'--samples',str(B),'--seed',str(seed),'--output',str(result)]
            write(ROOT/'commands'/(name+'.json'),{'command':command,'gpu':gpu})
            write(ROOT/'progress'/(name+'.json'),{'stage':'running','gpu':gpu,'started':time.time()})
            log=ROOT/'logs'/(name+'.log'); log.parent.mkdir(parents=True,exist_ok=True)
            try:
                with log.open('w') as out:
                    subprocess.run(command,check=True,stdout=out,stderr=subprocess.STDOUT,
                        env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'OMP_NUM_THREADS':'4','TOKENIZERS_PARALLELISM':'false'})
            except Exception:
                write(ROOT/'progress'/(name+'.json'),{'stage':'failed','gpu':gpu})
                raise
            write(ROOT/'progress'/(name+'.json'),{'stage':'complete','gpu':gpu})
    with ThreadPoolExecutor(max_workers=len(available)) as pool:
        futures=[pool.submit(worker,gpu,jobs[i::len(available)]) for i,gpu in enumerate(available)]
        for f in futures: f.result()
    write(ROOT/'status.json',{'stage':'complete','evaluations':len(jobs)})


if __name__=='__main__':
    try: main()
    except Exception as e:
        write(ROOT/'failure.json',{'error':str(e),'time':time.time()})
        raise
