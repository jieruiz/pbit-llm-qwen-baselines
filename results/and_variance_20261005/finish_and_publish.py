"""Finish the authorized fixed protocol, verify artifacts and update draft PR4.

No hyperparameter or checkpoint selection is performed by this utility.
"""
from pathlib import Path
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request

stage=Path(__file__).resolve().parent
repo=stage.parents[1]/'pbit-llm-qwen-baselines'
root='/home/Weican_Chen/projects/pbit-qwen-and-variance-20261005'
branch='codex/and-variance-20261005'
result=repo/'results/and_variance_20261005'
python=sys.executable
ssh=['C:/Windows/System32/OpenSSH/ssh.exe','-i','C:/Users/A/.ssh/id_ed25519_lerobot_gpu',
     '-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','ConnectTimeout=15',
     '-o','ServerAliveInterval=10','-o','ServerAliveCountMax=3','-p','21570','Weican_Chen@10.129.164.156']
scp=['C:/Windows/System32/OpenSSH/scp.exe','-i','C:/Users/A/.ssh/id_ed25519_lerobot_gpu',
     '-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','ConnectTimeout=15','-P','21570']
git=['C:/Program Files/Git/cmd/git.exe','-c',f'safe.directory={repo.as_posix()}',
     '-c','credential.username=amlink1219','-c','http.proxy=http://127.0.0.1:7890']

def command(argv,timeout=90,cwd=None):
    run=subprocess.run(argv,cwd=cwd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=timeout)
    if run.returncode:
        raise RuntimeError(f'Command failed ({run.returncode}): {argv[0]}: '+run.stderr.decode('utf-8',errors='replace')[-1800:])
    return run.stdout

def remote(text,retry=True):
    attempts=3 if retry else 1
    for attempt in range(attempts):
        try:
            return command([*ssh,text]).decode('utf-8')
        except (subprocess.TimeoutExpired,RuntimeError):
            if attempt==attempts-1:raise
            time.sleep(5)

def save(path,value):
    path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8',newline='\n')

def wait_job(job):
    started=time.monotonic()
    while time.monotonic()-started<12*3600:
        queue=remote(f"/opt/slurm/21.08.8/bin/squeue -h -j {int(job)} -o '%T %M'").strip()
        if queue:
            time.sleep(30)
            continue
        state=remote(f'/opt/slurm/21.08.8/bin/scontrol show job {int(job)}')
        if 'JobState=COMPLETED' not in state or 'ExitCode=0:0' not in state:
            raise RuntimeError(f'Job {job} did not complete successfully: '+state)
        return
    raise TimeoutError(f'Job {job} monitor timeout')

def download(name,destination):
    assert re.fullmatch(r'[A-Za-z0-9_./-]+',name) and '..' not in Path(name).parts
    command([*scp,f'Weican_Chen@10.129.164.156:{root}/{name}',str(destination)],timeout=3600)

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def main():
    begin=time.monotonic()
    while not (stage/'run-identity.json').is_file():
        if time.monotonic()-begin>4*3600:raise TimeoutError('Initializer/formal submission timeout')
        time.sleep(30)
    identity=json.loads((stage/'run-identity.json').read_text(encoding='utf-8'))
    assert identity['root']==root
    print('WAIT_FORMAL',identity['formal_job'],flush=True)
    wait_job(identity['formal_job'])
    status=json.loads(remote(f'cat {root}/execution-status.json'))
    assert status['application_exit_code']==0 and status['run_complete']
    # Guard the only new GPU submission with the same per-user scheduler lock.
    supplement=stage/'supplement-jobs.json'
    if supplement.exists():
        jobs=json.loads(supplement.read_text(encoding='utf-8'))
    else:
        script=f'''set -euo pipefail
[ "$(id -un)" = Weican_Chen ]
[ "$(realpath ~)" = /home/Weican_Chen ]
[ "$(realpath {root})" = {root} ]
[ -f {root}/results/RUN_COMPLETE ]
exec 9>/home/Weican_Chen/.pbit-gpu-submit.lock
flock -x 9
queue=$(/opt/slurm/21.08.8/bin/squeue -h -u Weican_Chen -o '%b')
if [[ "$queue" == *gpu* ]]; then echo 'Personal GPU queue occupied' >&2; exit 1; fi
bash -n {root}/empirical.slurm
bash -n {root}/archive.slurm
/opt/slurm/21.08.8/bin/sbatch --parsable {root}/empirical.slurm
/opt/slurm/21.08.8/bin/sbatch --parsable {root}/archive.slurm
'''
        submitted=remote("bash -s <<'PBIT_FINISH'\n"+script+'PBIT_FINISH\n',retry=False)
        numbers=[int(line) for line in submitted.splitlines() if re.fullmatch(r'\d+',line)]
        assert len(numbers)==2,submitted
        jobs={'empirical':numbers[0],'archive':numbers[1]};save(supplement,jobs)
    print('SUPPLEMENT_JOBS',json.dumps(jobs),flush=True)
    wait_job(jobs['empirical']);wait_job(jobs['archive'])
    for name in ('results-review.zip','result-manifest.json','empirical_variance.json',
                 'best-checkpoints-manifest.json','checkpoint-archive-status.json'):
        download(name,stage/name)
    for phase,job in [('initial',1073),('formal',identity['formal_job']),*jobs.items()]:
        for suffix in ('out','err'):download(f'logs/{phase}-{job}.{suffix}',stage/f'{phase}-{job}.{suffix}')
    formal_out=(stage/f"formal-{identity['formal_job']}.out").read_text(encoding='utf-8')
    hashes=re.findall(r'RESULT_ARCHIVE_SHA256 ([a-f0-9]{64})',formal_out)
    assert len(hashes)==1 and digest(stage/'results-review.zip')==hashes[0]
    command([python,'-X','utf8',str(stage/'finalize.py')],timeout=180)
    for name in ('empirical_variance.json','best-checkpoints-manifest.json','checkpoint-archive-status.json'):
        shutil.copy2(stage/name,result/name)
    logs=result/'job-logs';logs.mkdir(exist_ok=True)
    for pattern in ('initial-1073.*',f"formal-{identity['formal_job']}.*",f"empirical-{jobs['empirical']}.*",f"archive-{jobs['archive']}.*"):
        for path in stage.glob(pattern):shutil.copy2(path,logs/path.name)
    empirical=json.loads((stage/'empirical_variance.json').read_text(encoding='utf-8'))
    assert empirical['groups']==128 and empirical['samples_per_group']==4 and not empirical['used_for_selection']
    data=json.loads((result/'comparison.json').read_text(encoding='utf-8'))
    a,b=data['aggregates']['control'],data['aggregates']['variance']
    status_text=f'''# 固定 K=4、N=4 方差约束：已完成

重训初始化作业1073；正式作业{identity['formal_job']}；补充采样核验作业{jobs['empirical']}；CPU制品归档作业{jobs['archive']}，均COMPLETED、退出码0。
六组正式训练各2000步，26次完整测试已完成；两个源码快照及全部原始结果SHA256核验，局部实际N4采样核验已记录。

- A：三组训练seed的N4平均PPL {a['ppl_mean']:.6f}。
- B：同预算N4平均PPL {b['ppl_mean']:.6f}，lambda={data['selected_lambda']:g}。
- 相对改善 {100*data['relative_ppl_improvement']:.3f}%，配对胜出 {data['paired_wins']}/3；预定目标{'达到' if data['prespecified_target_met'] else '未达到'}。
- 所有组使用同方法重训的共同初始化；不是师弟原checkpoint的字节复现。三个seed仅重复继续训练阶段。
- [原始结果报告](RESULTS_ZH.md)、[比较数据](comparison.json)、[核验记录](VERIFICATION_ZH.md)、[实际N4采样核验](empirical_variance.json)。
- 140个权重的远端归档位于 `{root}/best-checkpoints.zip`，身份见best-checkpoints-manifest.json；权重不上传Git。

A日志variance_penalty=0表示该项未启用、未计算，不表示A的物理输出方差为零。
'''
    (result/'STATUS.md').write_text(status_text,encoding='utf-8',newline='\n')
    verification={'formal_archive_sha256':hashes[0],'formal_job':identity['formal_job'],
                  'supplement_jobs':jobs,'supplement_files':[{ 'file':name,'sha256':digest(stage/name)}
                  for name in ('empirical_variance.json','best-checkpoints-manifest.json','checkpoint-archive-status.json')],
                  'publication_script_sha256':digest(Path(__file__))}
    save(result/'publication-verification.json',verification)
    shutil.copy2(Path(__file__),result/'finish_and_publish.py')
    # Preserve every file listed in the original manifest; derived notes are separate.
    manifest=json.loads((result/'result-manifest.json').read_text(encoding='utf-8'))
    for row in manifest['files']:
        relative=row['file'][8:] if row['file'].startswith('results/') else row['file']
        path=result/relative
        assert path.resolve().is_relative_to(result.resolve())
        assert digest(path)==row['sha256'] and path.stat().st_size==row['bytes'],relative
    # Verify actual GCM identity immediately before any authenticated GitHub write.
    os.environ['GIT_TERMINAL_PROMPT']='0';os.environ['GCM_INTERACTIVE']='Never'
    creds=subprocess.run([*git,'-c','credential.interactive=never','credential','fill'],
        input=b'protocol=https\nhost=github.com\nusername=amlink1219\n\n',stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    if creds.returncode:raise RuntimeError('Credential lookup failed')
    fields=dict(line.split('=',1) for line in creds.stdout.decode().splitlines() if '=' in line)
    token=fields['password']
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({'https':'http://127.0.0.1:7890'}))
    def api(path,method='GET',payload=None):
        req=urllib.request.Request('https://api.github.com'+path,method=method,
            data=json.dumps(payload).encode() if payload else None,
            headers={'Authorization':'Bearer '+token,'Accept':'application/vnd.github+json',
                     'User-Agent':'Codex-pbit-research','Content-Type':'application/json'})
        with opener.open(req,timeout=60) as response:return json.load(response)
    user=api('/user');assert user['login']=='amlink1219' and user['id']==75301230
    target=api('/repos/jieruiz/pbit-llm-qwen-baselines');assert target['permissions']['push']
    assert command([*git,'branch','--show-current'],cwd=repo).decode().strip()==branch
    assert not command([*git,'diff','--cached','--name-only'],cwd=repo).strip()
    command([*git,'fetch','origin',branch],cwd=repo)
    command([*git,'merge-base','--is-ancestor',f'origin/{branch}','HEAD'],cwd=repo)
    command([*git,'add','results/and_variance_20261005'],cwd=repo)
    command([*git,'diff','--cached','--check'],cwd=repo)
    command([*git,'-c','user.name=amlink1219','-c','user.email=75301230+amlink1219@users.noreply.github.com',
             'commit','-m','Record fixed-N4 variance ablation results and verified artifacts'],cwd=repo)
    command([*git,'push','origin',branch],cwd=repo,timeout=180)
    head=command([*git,'rev-parse','HEAD'],cwd=repo).decode().strip()
    body='Fixed K=4 and inference N=4; all six paired 2000-step runs and 26 full tests completed. Only B adds normalized analytical conditional variance.\n\n'+status_text+'\n\n'+(result/'RESULTS_ZH.md').read_text(encoding='utf-8')
    pr=api('/repos/jieruiz/pbit-llm-qwen-baselines/pulls/4','PATCH',{'body':body})
    assert pr['head']['sha']==head and pr['draft'] and not pr['merged']
    save(stage/'PUBLISHED.json',{'head':head,'pr':pr['html_url'],'comparison':data,'account':user['login']})
    print('PUBLISHED',head,pr['html_url'],flush=True)

if __name__=='__main__':main()
