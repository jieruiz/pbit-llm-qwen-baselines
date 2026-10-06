"""Five-seed sampling confirmation and three-seed router confirmation."""
from concurrent.futures import ThreadPoolExecutor
import os
import subprocess
import sys
import time
from run_experiment import ROOT,EXP,write,jobname


def main():
    planned=[]
    for ctx,examples,batch in ((2048,32,16),(8192,16,4)):
        common={'context':ctx,'examples':examples,'batch':batch,'decode':128}
        for mode,samples,gap in (('iid',128,1.),('iid',512,1.),('iid',1024,1.),
                                 ('ising',128,4.),('ising',512,1.),('ising',512,4.),('ising',1024,4.)):
            for seed in range(5):
                j={**common,'mode':mode,'samples':samples,'spacing':gap,'burn':4.,'seed':seed}
                if not (ROOT/'evaluation'/(jobname(j)+'.json')).exists(): planned.append(j)
        for mode,ratio in (('block_mean',.25),('block_mean',.5),('oracle',.25)):
            for seed in (1,2): planned.append({**common,'mode':mode,'ratio':ratio,'seed':seed})
    memory=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
    gpus=[int(x.split(',')[0]) for x in memory.splitlines() if int(x.split(',')[1])<500]
    if not gpus:raise RuntimeError('No idle GPU')
    write(ROOT/'repeat_protocol.json',{'jobs':planned,'gpus':gpus,'reason':'Resolve seed variability in sampling and selected promising routers; same test targets.'})
    write(ROOT/'repeat_status.json',{'stage':'running','total':len(planned),'started':time.time()})
    def worker(gpu,queue):
        for j in queue:
            name=jobname(j);dest=ROOT/'evaluation'/(name+'.json')
            if dest.exists():continue
            cmd=[sys.executable,str(EXP/'evaluate.py')]
            for key,val in j.items():cmd.extend(['--'+key.replace('_','-'),str(val)])
            cmd.extend(['--output',str(dest)])
            write(ROOT/'commands'/(name+'.json'),{'command':cmd,'gpu':gpu,'phase':'repeat'})
            with (ROOT/'logs'/(name+'.log')).open('w') as out:
                subprocess.run(cmd,stdout=out,stderr=subprocess.STDOUT,check=True,
                    env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'OMP_NUM_THREADS':'4','TOKENIZERS_PARALLELISM':'false'})
    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futures=[pool.submit(worker,gpu,planned[i::len(gpus)]) for i,gpu in enumerate(gpus)]
        for f in futures:f.result()
    write(ROOT/'repeat_status.json',{'stage':'complete','evaluations':len(planned),'finished':time.time()})


if __name__=='__main__': main()
