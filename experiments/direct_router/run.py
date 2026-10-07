"""Run preregistered budgets, seeds and matched original-FFN controls."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    gpus = [int(s.split(',')[0]) for s in subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True).splitlines() if int(s.split(',')[1]) < 500]
    assert gpus, 'No idle GPUs'
    jobs=[]
    for ctx,ex,batch in ((2048,32,16),(8192,16,4)):
        base={'context':ctx,'examples':ex,'batch':batch,'decode':128}
        jobs.append({**base,'mode':'sdpa','reps':1,'ratio':.5,'seed':0})
        for backend in ('sparse','dense'):
            jobs.append({**base,'mode':'all','reps':1,'ratio':.5,'seed':0,'backend':backend})
        for mode,reps,ratios in (('mean',1,(.25,.5)),('learned',1,(.125,.25,.5)),('learned',4,(.125,.25,.5))):
            for ratio in ratios:
                for seed in (0,1,2):
                    jobs.append({**base,'mode':mode,'reps':reps,'ratio':ratio,'seed':seed})
    Path('results/evaluation').mkdir(exist_ok=True)
    Path('results/protocol.json').write_text(json.dumps({'jobs':jobs,'gpus':gpus,'training':'Router only; train-tail validation; fixed budgets; best validation BCE checkpoint; no test tuning','FFN':'original frozen','stage':'running'},indent=2))
    def worker(gpu,queue):
        for j in queue:
            name=f"ctx{j['context']}_{j['mode']}_r{j['reps']}_budget{j['ratio']}_seed{j['seed']}_{j.get('backend','sparse')}"
            dest=Path('results/evaluation')/(name+'.json')
            if dest.exists():continue
            cmd=[sys.executable,'experiments/direct_router/evaluate.py']
            for k,v in j.items():cmd.extend(['--'+k,str(v)])
            cmd.extend(['--output',str(dest)])
            env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'OMP_NUM_THREADS':'4','TOKENIZERS_PARALLELISM':'false'}
            if j['mode']=='mean' and j['ratio']==.5 and j['seed']==0:
                env['ROUTER_SNAPSHOT']=f"artifacts/snapshot_ctx{j['context']}.pt"
            with (Path('results')/(name+'.log')).open('w') as log:
                subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            print(json.dumps({'complete':name,'gpu':gpu}),flush=True)
    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futures=[pool.submit(worker,gpu,jobs[i::len(gpus)]) for i,gpu in enumerate(gpus)]
        for f in futures:f.result()
    Path('results/status.json').write_text(json.dumps({'stage':'complete','jobs':len(jobs),'finished':time.time()}))


if __name__=='__main__':main()
