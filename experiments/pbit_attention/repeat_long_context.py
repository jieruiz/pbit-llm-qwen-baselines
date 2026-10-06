"""Additional inference seeds prompted by the uncertain single-seed 8k ranking."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import os
from pathlib import Path
import subprocess
import sys
from run_experiment import ROOT,EXP,write


def main():
    if (ROOT/'long_context_repeat_protocol.json').exists():
        raise RuntimeError('Repeat already launched')
    jobs=[(mode,n,seed) for seed in (1,2) for n in (128,512) for mode in ('tree','independent')]
    jobs.append(('sdpa',128,0))
    memory=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
    if any(int(line.split(',')[1])>500 for line in memory.splitlines()):
        raise RuntimeError('GPUs not idle')
    write(ROOT/'long_context_repeat_protocol.json',{'reason':'Single-seed 8k ranking differed from 2k; add seeds1/2 for both methods/budgets, and original SDPA baseline',
        'jobs':jobs,'context':8192,'examples':16,'decode':128,'batch':4,
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    write(ROOT/'repeat_status.json',{'stage':'running','jobs':len(jobs)})
    def worker(gpu,queue):
        for mode,n,seed in queue:
            name=f'ctx8192_{mode}_s{n}_seed{seed}'
            command=[sys.executable,str(EXP/'evaluate.py'),'--context','8192','--examples','16','--decode','128','--batch','4',
                '--mode',mode,'--samples',str(n),'--seed',str(seed),'--output',str(ROOT/'evaluation'/(name+'.json'))]
            write(ROOT/'commands'/(name+'.json'),{'command':command,'gpu':gpu})
            with (ROOT/'logs'/(name+'.log')).open('w') as out:
                subprocess.run(command,check=True,stdout=out,stderr=subprocess.STDOUT,
                    env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'OMP_NUM_THREADS':'4','TOKENIZERS_PARALLELISM':'false'})
    with ThreadPoolExecutor(max_workers=8) as pool:
        jobs_running=[pool.submit(worker,gpu,jobs[gpu::8]) for gpu in range(8)]
        for job in jobs_running:
            job.result()
    write(ROOT/'repeat_status.json',{'stage':'complete','jobs':len(jobs)})


if __name__=='__main__':
    main()
