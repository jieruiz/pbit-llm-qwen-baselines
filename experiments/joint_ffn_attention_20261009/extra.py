import hashlib,json,subprocess,time,traceback
from run import HERE,OUT,write,read,pooled,spec
try:
    write(OUT/'supplement_protocol.json',{'reason':'512-token screening showed FFN+SDPA and FFN+manual-dense vary despite near-identical nonstochastic attention; repeat exact-attention controls to avoid attributing single-seed variation to Ising.',
        'test_used_to_choose':False,'extra_cases':'fixedFFN+manual dense, and fixedFFN+selected Q/K+manual dense, seeds1/2',
        'source_sha256':hashlib.sha256((HERE/'extra.py').read_bytes()).hexdigest()})
    write(OUT/'supplement_status.json',{'stage':'waiting_for_main_queue_dispatched'})
    while True:
        if (OUT/'status.json').exists() and read(OUT/'status.json')['stage']=='failed':raise RuntimeError('main failed')
        if len(list((OUT/'jobs').glob('test_*.json')))>=18:
            rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
            free=[int(r.split(',')[0]) for r in rows.splitlines() if int(r.split(',')[0]) not in (0,1) and int(r.split(',')[1])<500 and int(r.split(',')[2])<5]
            if len(free)>=2:break
        time.sleep(5)
    q=read(OUT/'selection.json')['qcycles']
    cases=[spec('fixed',qc,'dense',seed) for qc in (0,q) for seed in (1,2)]
    write(OUT/'supplement_status.json',{'stage':'test','gpus':free[:2],'cases':cases})
    pooled(cases,free[:2],'test')
    write(OUT/'supplement_status.json',{'stage':'complete','extra_tests':4})
except Exception:
    write(OUT/'supplement_status.json',{'stage':'failed','error':traceback.format_exc()});raise
