"""Confirm conservative 16-clock burn after real-logit transient inspection."""
from concurrent.futures import ThreadPoolExecutor
import os
import subprocess
import sys
sys.path.insert(0,'experiments/pbit_softmax_router')
from run_experiment import ROOT,EXP,write,jobname

jobs=[]
for ctx,examples,batch in ((2048,32,16),(8192,16,4)):
    for seed in range(5):jobs.append({'context':ctx,'examples':examples,'batch':batch,'decode':128,
        'mode':'ising','samples':512,'spacing':4.,'burn':16.,'seed':seed})
memory=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
gpus=[int(x.split(',')[0]) for x in memory.splitlines() if int(x.split(',')[1])<500]
if not gpus:raise RuntimeError('No idle GPU')
write(ROOT/'burn_protocol.json',{'jobs':jobs,'gpus':gpus,'reason':'20 real score rows show max TV < 8.7e-6 at burn16. Confirm recommended burn16, gap4, S512.'})
write(ROOT/'burn_status.json',{'stage':'running','total':len(jobs)})
def worker(gpu,queue):
    for j in queue:
        name=jobname(j);dest=ROOT/'evaluation'/(name+'.json')
        if dest.exists():continue
        cmd=[sys.executable,str(EXP/'evaluate.py')]
        for k,v in j.items():cmd.extend(['--'+k,str(v)])
        cmd.extend(['--output',str(dest)])
        write(ROOT/'commands'/(name+'.json'),{'command':cmd,'gpu':gpu,'phase':'burn_confirmation'})
        with (ROOT/'logs'/(name+'.log')).open('w') as out:
            subprocess.run(cmd,stdout=out,stderr=subprocess.STDOUT,check=True,
                env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'OMP_NUM_THREADS':'4','TOKENIZERS_PARALLELISM':'false'})
with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
    futures=[pool.submit(worker,g,jobs[i::len(gpus)]) for i,g in enumerate(gpus)]
    for f in futures:f.result()
write(ROOT/'burn_status.json',{'stage':'complete','evaluations':len(jobs)})
