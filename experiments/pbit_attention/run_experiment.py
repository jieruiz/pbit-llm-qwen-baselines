"""No training: 25 matched cached-decode evaluations on otherwise original Qwen."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path('results/pbit_attention_20261003')
EXP=Path('experiments/pbit_attention')


def write(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(obj,indent=2)+'\n'); temp.replace(path)


def main():
    if (ROOT/'protocol.json').exists():
        raise RuntimeError('Experiment already launched; inspect before any recovery')
    memory=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
    if any(int(line.split(',')[1])>500 for line in memory.splitlines()):
        raise RuntimeError('GPUs not idle')
    jobs=[]
    for mode in ('sdpa','dense'):
        jobs.append((2048,32,128,16,mode,128,0))
    for seed in (0,1,2):
        for n in (32,128,512):
            for mode in ('tree','independent'):
                jobs.append((2048,32,128,16,mode,n,seed))
    for mode,n in (('dense',128),('tree',128),('independent',128),('tree',512),('independent',512)):
        jobs.append((8192,16,128,4,mode,n,0))
    write(ROOT/'protocol.json',{'date':'2026-10-03','model':'Qwen2.5-0.5B Base BF16',
        'attention':'all24; only AV changed; exact QK/softmax','ffn':'all original Qwen2MLP',
        'training':'none','sampling':'IID categorical via binary p-bit tree versus independent Bernoulli per position',
        'independent_simulation':'Binomial(S,p) exactly compresses S independent Bernoulli rounds; no count renormalization',
        'context2048':'32 evenly spread WT2-test spans; exact prefix2047 then128 sampled teacher-forced decode queries;4096 scored tokens',
        'context8192':'16 evenly spread spans; exact prefix8191 then128 queries;2048 scored tokens',
        'inference_seeds_main':[0,1,2],'samples':[32,128,512],'jobs':[list(j) for j in jobs],
        'readout':'FP32 count/S @ V arithmetic reference; no sparse GPU acceleration claim',
        'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in EXP.glob('*.py')}})
    write(ROOT/'status.json',{'stage':'testing'})
    for script in ('test_sampling.py','test_integration.py'):
        dest=ROOT/(script.replace('.py','.json'))
        with dest.open('w') as out:
            subprocess.run([sys.executable,str(EXP/script)],check=True,stdout=out,
                env={**os.environ,'CUDA_VISIBLE_DEVICES':'0','OMP_NUM_THREADS':'4'})
    write(ROOT/'status.json',{'stage':'evaluating','total_jobs':len(jobs)})
    def worker(gpu,queue):
        for context,examples,decode,batch,mode,n,seed in queue:
            name=f'ctx{context}_{mode}_s{n}_seed{seed}'
            result=ROOT/'evaluation'/(name+'.json')
            command=[sys.executable,str(EXP/'evaluate.py'),'--context',str(context),'--examples',str(examples),
                '--decode',str(decode),'--batch',str(batch),'--mode',mode,'--samples',str(n),'--seed',str(seed),'--output',str(result)]
            write(ROOT/'commands'/(name+'.json'),{'command':command,'gpu':gpu})
            write(ROOT/'progress'/(name+'.json'),{'stage':'running','gpu':gpu,'start':time.time()})
            log=ROOT/'logs'/(name+'.log'); log.parent.mkdir(parents=True,exist_ok=True)
            with log.open('w') as out:
                subprocess.run(command,check=True,stdout=out,stderr=subprocess.STDOUT,
                    env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'OMP_NUM_THREADS':'4','TOKENIZERS_PARALLELISM':'false'})
            write(ROOT/'progress'/(name+'.json'),{'stage':'complete','gpu':gpu})
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures=[pool.submit(worker,gpu,jobs[gpu::8]) for gpu in range(8)]
        for future in futures:
            future.result()
    write(ROOT/'status.json',{'stage':'complete','evaluations':len(jobs)})


if __name__=='__main__':
    main()
