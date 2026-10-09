import concurrent.futures as cf
import hashlib,json,os,queue,subprocess,sys,time,traceback
from pathlib import Path
HERE=Path(__file__).resolve().parent;OUT=HERE/'results'
def read(p):return json.loads(p.read_text())
def write(p,x):
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def spec(ffn='fixed',qcycles=256,attn='ising',seed=0):
    return dict(ffn=ffn,qcycles=qcycles,attn=attn,seed=seed)
def name(s):return f"{s['ffn']}_q{s['qcycles']}_{s['attn']}_s{s['seed']}"
def job(n,args,gpu,script='evaluate.py'):
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    command=[sys.executable,'-u',str(HERE/script),*map(str,args)]
    write(OUT/'jobs'/f'{n}.json',{'status':'running','gpu':gpu,'command':command})
    log=OUT/'logs'/f'{n}.log';log.parent.mkdir(parents=True,exist_ok=True);start=time.time()
    with log.open('w') as f:rc=subprocess.call(command,cwd=HERE,env=env,stdout=f,stderr=subprocess.STDOUT)
    write(OUT/'jobs'/f'{n}.json',{'status':'complete' if rc==0 else 'failed','gpu':gpu,'command':command,'returncode':rc,'seconds':time.time()-start})
    if rc:raise RuntimeError(n+' failed; '+str(log))
def evaluate(s,gpu,split='validation'):
    output=OUT/split/(name(s)+'.json')
    args=['--ffn',s['ffn'],'--qcycles',s['qcycles'],'--attn',s['attn'],'--seed',s['seed'],
        '--split',split,'--context',512 if split=='validation' else 2048,
        '--decode',64 if split=='validation' else 128,'--examples',8 if split=='validation' else 32,
        '--batch',4,'--samples',512,'--chunk',32,'--output',output]
    job(split+'_'+name(s),args,gpu);return output
def pooled(specs,gpus,split='validation'):
    q=queue.Queue()
    for s in specs:q.put(s)
    def worker(gpu):
        while True:
            try:s=q.get_nowait()
            except queue.Empty:return
            evaluate(s,gpu,split)
    with cf.ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        for f in cf.as_completed([pool.submit(worker,g) for g in gpus]):f.result()
def main():
    os.chdir(HERE);OUT.mkdir(exist_ok=False)
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    rows=[list(map(int,r.split(','))) for r in rows.splitlines()]
    gpus=[i for i,m,u in rows if i!=0 and m<500 and u<5]
    if not gpus:raise RuntimeError('No free GPUs outside0')
    variants=[spec('original',0,'sdpa'),spec('original',0,'dense'),
        spec('fixed',0,'sdpa'),spec('fixed',0,'dense'),spec('fixed',256,'dense'),
        spec('original',0,'ising'),spec('fixed',0,'ising'),spec('fixed',0,'iid')]
    variants += [spec('fixed',q,a) for q in (256,1024) for a in ('ising','iid')]
    write(OUT/'protocol.json',{'model':'Qwen2.5-0.5B Base','training':False,
        'ffn':'CURRENT fixed group_max all24,W8,12magnitude+sign,T4096,interval256,burn25%; BF16 residual/interface',
        'qk':'fixed learned q_proj/k_proj ONLY; W10,input9magnitude+sign,dyadic scales; QK dot remains deterministic',
        'attention':'all24 layers, BOTH prefill and decode sampled, causal mask, physical K/V cache from modified prefix',
        'ising':'hard-exclusion continuous-time heat-bath,reference maximum score,burn16,spacing4,samples512; sigmoid birth rates, no softmax probabilities drive Ising',
        'iid':'exact numerical softmax then512 IID categorical positions, sparse mathematical PV via dense counts@V reference',
        'validation':'last65536 train tokens,8evenly-spaced spans,context512,decode64,batch4,512scored; disjoint old calibration first8192',
        'test':'WT2 test,32evenly-spaced spans,context2048,decode128,batch4,4096scored; NOT full WT2 PPL',
        'selection':'choose q256 if joint Ising validation <=1.05*fixedFFN+dense validation; else q1024 if passes; else lower joint Ising validation PPL. Budget diagnostic only.',
        'fallback':'always measure same-Q exactsoftmax+sampledPV and no-Q exactsoftmax+sampledPV; recommend fallback if Ising adds>5% beyond FFN-only, report total loss separately',
        'validation_specs':variants,'repeats':'test seeds1/2 for fixedFFN sdpa, selected jointIsing, selected jointIID, fixedFFN IID without Q',
        'generation':'baseline,FFN-only,jointIsing,jointIID,no-Q IID fixed prompts48tokens',
        'limits':['FFN ideal sigmoid; Ising ideal hard constraint not finite circuitry','original RMSNorm/RoPE/residual maintained',
            'IID and Ising software count compression, not bit-level RTL or sparse CUDA','no energy/area claim','subset cached PPL cannot compare to full11.6527baseline']})
    write(OUT/'resources.json',{'snapshot':rows,'gpus':gpus})
    write(OUT/'source_sha256.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('*.py')})
    write(OUT/'calibration_sha256.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('calibration*.json')})
    write(OUT/'status.json',{'stage':'unit_and_integration'});job('units',[],gpus[0],'test_joint.py');job('integration',[],gpus[0],'integration.py')
    write(OUT/'status.json',{'stage':'validation','jobs':len(variants)});pooled(variants,gpus)
    base=read(OUT/'validation'/(name(spec('fixed',0,'dense'))+'.json'))['perplexity']
    candidates=[(q,read(OUT/'validation'/(name(spec(qcycles=q))+'.json'))['perplexity']) for q in (256,1024)]
    eligible=[(q,p) for q,p in candidates if p<=base*1.05]
    chosen=min(eligible)[0] if eligible else min(candidates,key=lambda x:x[1])[0]
    write(OUT/'selection.json',{'qcycles':chosen,'ffn_only_validation_ppl':base,'joint_candidates':candidates,
        'ising_within_incremental5pct':bool(eligible),'test_used':False})
    if chosen!=256:evaluate(spec('fixed',chosen,'dense'),gpus[0])
    cases=[spec('original',0,'sdpa'),spec('original',0,'dense'),spec('fixed',0,'sdpa'),spec('fixed',0,'dense'),
        spec('fixed',chosen,'dense'),spec('original',0,'ising'),spec('fixed',0,'ising'),
        spec('fixed',chosen,'ising'),spec('fixed',chosen,'iid'),spec('fixed',0,'iid')]
    repeated=[spec('fixed',0,'sdpa'),spec('fixed',chosen,'ising'),spec('fixed',chosen,'iid'),spec('fixed',0,'iid')]
    cases += [dict(s,seed=seed) for s in repeated for seed in (1,2)]
    write(OUT/'status.json',{'stage':'test','jobs':len(cases),'selected_qcycles':chosen});pooled(cases,gpus,'test')
    write(OUT/'status.json',{'stage':'generation'});job('generation',[],gpus[0],'generation.py')
    write(OUT/'status.json',{'stage':'complete','validation_jobs':len(list((OUT/'validation').glob('*.json'))),'test_jobs':len(cases)})
if __name__=='__main__':
    try:main()
    except Exception:
        if OUT.exists():write(OUT/'status.json',{'stage':'failed','error':traceback.format_exc()})
        raise
