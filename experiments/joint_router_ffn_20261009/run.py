import concurrent.futures as cf
import hashlib,json,os,queue,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parent;R=ROOT/'results'
def write(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj,indent=2)+'\n');tmp.replace(path)
def job(name,script,args,gpu):
    command=[sys.executable,'-u',str(ROOT/script),*map(str,args)]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    write(R/'jobs'/f'{name}.json',dict(status='running',command=command,gpu=gpu))
    log=R/'logs'/f'{name}.log';log.parent.mkdir(parents=True,exist_ok=True);start=time.time()
    with log.open('w') as f:rc=subprocess.call(command,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
    write(R/'jobs'/f'{name}.json',dict(status='complete' if rc==0 else 'failed',returncode=rc,gpu=gpu,command=command,seconds=time.time()-start))
    if rc:raise RuntimeError(f'{name} failed: {log}')
def evaluate(case,gpu,smoke=False):
    ffn,mode,context,seed=case;name=f'{ffn}_{mode}_c{context}_s{seed}'
    stage='smoke' if smoke else 'test';examples=1 if smoke else (32 if context==2048 else 16)
    args=['--ffn',ffn,'--mode',mode,'--context',context,'--seed',seed,'--examples',examples,
        '--decode',4 if smoke else 128,'--batch',4,'--output',R/stage/(name+'.json')]
    job(stage+'_'+name,'evaluate.py',args,gpu)
def main():
    R.mkdir(exist_ok=False)
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    free=[int(r.split(',')[0]) for r in rows.splitlines() if 1<=int(r.split(',')[0])<=6 and int(r.split(',')[1])<500 and int(r.split(',')[2])<5]
    if not free:raise RuntimeError('No free GPU among1..6')
    cases=[('original','sdpa',ctx,0) for ctx in (2048,8192)]
    cases += [('fixed',mode,ctx,seed) for ctx in (2048,8192) for seed in (0,1,2)
        for mode in ('all_exact','all_sampled','screen_exact','screen_sampled')]
    write(R/'protocol.json',dict(ffn='current fixed all24, W8, magnitude12+sign, T4096/output3072, BF16 interface/residual',
        attention='all24 decode only; original accurate prefill attention; Q/K/V/O projections original',
        selector='max4 64-token blocks,16-token submeans; per-query score mean/std; old moment.pt 30% target; first+last2 mandatory; independent Bernoulli, no annealing',
        variants='screen_exact: accurate candidate Softmax/PV; screen_sampled: accurate candidate Softmax and512 IID V gathers averaged',
        controls='all_exact and all_sampled with SAME prefill/batch/tokens/seeds; original SDPA',
        test='2k32x128=4096 scored;8k16x128=2048; batch4; three seeds; no test tuning, no training',
        cost='compact QK before multiplication, padded rows charged; full KV storage retained; no hardware claim',cases=cases,gpus=free))
    write(R/'source_sha256.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('*.py')})
    write(R/'calibration_sha256.json',{n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ('calibration_ffn.json','moment.pt')})
    write(R/'status.json',dict(stage='checks'));job('units','test_router.py',[],free[0])
    for mode in ('all_exact','all_sampled','screen_exact','screen_sampled'):evaluate(('fixed',mode,2048,0),free[0],True)
    write(R/'status.json',dict(stage='test',jobs=len(cases),gpus=free))
    q=queue.Queue()
    for case in cases:q.put(case)
    def worker(gpu):
        while True:
            try:case=q.get_nowait()
            except queue.Empty:return
            evaluate(case,gpu)
    with cf.ThreadPoolExecutor(max_workers=len(free)) as pool:
        for f in cf.as_completed([pool.submit(worker,g) for g in free]):f.result()
    write(R/'status.json',dict(stage='complete',test_jobs=len(cases)))
    job('report','report.py',[],free[0])
if __name__=='__main__':
    try:main()
    except Exception:
        if R.exists():write(R/'status.json',dict(stage='failed',error=traceback.format_exc()))
        raise
