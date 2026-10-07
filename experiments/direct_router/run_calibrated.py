from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    memory=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
    gpus=[int(s.split(',')[0]) for s in memory.splitlines() if int(s.split(',')[1])<500]
    assert gpus
    jobs=[(ctx,ex,batch,reps,seed) for ctx,ex,batch in ((2048,32,16),(8192,16,4)) for reps in (1,4) for seed in (0,1,2)]
    Path('results/calibrated').mkdir(exist_ok=True)
    def worker(gpu,queue):
        for ctx,ex,batch,reps,seed in queue:
            name=f'ctx{ctx}_r{reps}_seed{seed}';dest=f'results/calibrated/{name}.json'
            if Path(dest).exists():continue
            cmd=[sys.executable,'experiments/direct_router/evaluate.py','--mode','learned','--context',str(ctx),'--examples',str(ex),'--batch',str(batch),'--reps',str(reps),'--ratio','.5','--seed',str(seed),'--output',dest]
            with open(f'results/calibrated/{name}.log','w') as log:
                subprocess.run(cmd,env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'OMP_NUM_THREADS':'4','TOKENIZERS_PARALLELISM':'false','ROUTER_CALIBRATED':'1'},stdout=log,stderr=subprocess.STDOUT,check=True)
            print(name,flush=True)
    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futures=[pool.submit(worker,gpu,jobs[i::len(gpus)]) for i,gpu in enumerate(gpus)]
        for f in futures:f.result()
    Path('results/calibrated/status.json').write_text(json.dumps({'status':'complete','jobs':len(jobs)}))


if __name__=='__main__':main()
