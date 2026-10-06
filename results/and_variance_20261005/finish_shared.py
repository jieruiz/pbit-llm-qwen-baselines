"""Archive the already verified experiment under the designated Baidu root."""
from pathlib import Path
import json
import re
import shutil
import subprocess
import sys
import time

stage=Path(__file__).resolve().parent
shared=Path('E:/BaiduSyncdisk/p-bit/pbit-qwen').resolve()
approved=Path('E:/BaiduSyncdisk/p-bit').resolve()
destination=shared/'results/and_variance_20261005'
root='/home/Weican_Chen/projects/pbit-qwen-and-variance-20261005'
ssh=['C:/Windows/System32/OpenSSH/ssh.exe','-i','C:/Users/A/.ssh/id_ed25519_lerobot_gpu',
     '-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','ConnectTimeout=15',
     '-o','ServerAliveInterval=10','-o','ServerAliveCountMax=3','-p','21570','Weican_Chen@10.129.164.156']
scp=['C:/Windows/System32/OpenSSH/scp.exe','-i','C:/Users/A/.ssh/id_ed25519_lerobot_gpu',
     '-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','ConnectTimeout=15',
     '-o','ServerAliveInterval=10','-o','ServerAliveCountMax=3','-P','21570']

def run(argv,timeout):
    completed=subprocess.run(argv,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=timeout)
    if completed.returncode:
        raise RuntimeError(f'Command failed {completed.returncode}: '+completed.stderr.decode('utf-8',errors='replace')[-1800:])
    return completed.stdout

def main():
    begin=time.monotonic()
    while not (stage/'PUBLISHED.json').is_file():
        if time.monotonic()-begin>16*3600:raise TimeoutError('Verified publication did not finish')
        time.sleep(30)
    publication=json.loads((stage/'PUBLISHED.json').read_text(encoding='utf-8'))
    assert publication['account']=='amlink1219'
    status=json.loads((stage/'checkpoint-archive-status.json').read_text(encoding='utf-8'))
    jobs=json.loads((stage/'supplement-jobs.json').read_text(encoding='utf-8'))
    assert int(status['archive_job'])==jobs['archive']
    assert status['checkpoints']==140 and status['formal_best']==120 and status['common_initializer']==20
    assert re.fullmatch(r'[a-f0-9]{64}',status['archive_sha256'])
    assert shared.is_dir() and shared.is_relative_to(approved)
    assert destination.resolve().is_relative_to(shared)
    assert not destination.exists(), 'Destination already exists; inspect before any overwrite'
    assert shutil.disk_usage(shared).free>=status['bytes']+256*1024*1024, 'Insufficient local disk space'
    identity=run([*ssh,f'id -un\nrealpath ~\nrealpath {root}'],90).decode().splitlines()
    assert identity==['Weican_Chen','/home/Weican_Chen',root],identity
    destination.mkdir(parents=True,exist_ok=False)
    partial=destination/'best-checkpoints.zip.part'
    assert partial.resolve().is_relative_to(shared)
    print('CHECKPOINT_DOWNLOAD',status['bytes'],str(destination),flush=True)
    run([*scp,f'Weican_Chen@10.129.164.156:{root}/best-checkpoints.zip',str(partial)],9000)
    assert partial.stat().st_size==status['bytes']
    archive=destination/'best-checkpoints.zip'
    assert archive.resolve().is_relative_to(shared) and not archive.exists()
    partial.rename(archive)
    verified=run([sys.executable,'-X','utf8',str(stage/'save_shared.py')],1800)
    print(verified.decode('utf-8'),flush=True)
    record=json.loads((destination/'local-verification.json').read_text(encoding='utf-8'))
    assert record['verified_checkpoints']==140 and record['archive_sha256']==status['archive_sha256']
    completion={'directory':str(destination),'checkpoints':140,'archive_sha256':status['archive_sha256'],
                'publication_head':publication['head'],'archive_job':jobs['archive']}
    (stage/'SHARED_COMPLETE.json').write_text(json.dumps(completion,indent=2)+'\n',encoding='utf-8',newline='\n')
    print('SHARED_COMPLETE',json.dumps(completion),flush=True)

if __name__=='__main__':main()
